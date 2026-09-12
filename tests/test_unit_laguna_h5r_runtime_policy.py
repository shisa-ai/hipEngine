from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_CANDIDATE = "swa_context_rows_qrow4_cached_exact_spans"
_RETAINED = "swa_context_rows_qrow4_m128_c256_exact_spans"
_H5M = "swa_context_rows_qrow4_sourcequal_exact_spans"


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


def test_h5r_preappend_role_metadata_is_source_default_and_runner_gated() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner

    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == {
        _ROLE: _CANDIDATE
    }
    assert hip_gfx1100.LAGUNA_PREFILL_KV_PREAPPEND is True
    assert not hasattr(hip_gfx1151, "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS")
    rows_source = inspect.getsource(
        laguna_gguf_runner.LagunaGGUFResidentSession._run_layer_rows
    )
    init_source = inspect.getsource(
        laguna_gguf_runner.LagunaGGUFResidentSession.__init__
    )
    assert ".can_preappend_attention_prefill(" in rows_source
    assert "self._swa_prefill_package_default = swa_prefill_variant is None" in (
        init_source
    )
    assert "prefill_preappend_package_default=(" in init_source


def test_h5r_scoped_preappend_policy_routes_only_safe_swa_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS",
        {_ROLE: _CANDIDATE},
        raising=False,
    )
    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS",
        {},
        raising=False,
    )
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[str] = []

    def resolve(layer: str, variant: str):
        calls.append(f"resolve:{layer}:{variant}")
        return lambda *args, **kwargs: calls.append(f"launch:{variant}")

    cache._resolve = resolve

    def dispatch(layer_id: int, start: int, rows: int) -> bool:
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

    try:
        assert cache.prefill_preappend_role_scoped is True
        assert cache.prefill_preappend_role_variants == {_ROLE: _CANDIDATE}
        for start in (0, 128, 256, 384):
            before = len(calls)
            assert dispatch(1, start, 128)
            launches = [
                call for call in calls[before:] if call.startswith("launch:")
            ]
            assert launches == ["launch:swa_f32_rows_spans", f"launch:{_CANDIDATE}"]

        for layer_id, start, rows, expected in (
            (1, 64, 128, _RETAINED),
            (1, 512, 128, _H5M),
            (1, 0, 64, _RETAINED),
            (0, 0, 128, "global_context_rows_spans"),
        ):
            before = len(calls)
            assert not dispatch(layer_id, start, rows)
            launches = [
                call for call in calls[before:] if call.startswith("launch:")
            ]
            assert launches[0] == f"launch:{expected}"
    finally:
        cache.free()
    assert runtime.allocations == {}


def test_h5r_preappend_role_policy_fails_closed_on_selector_and_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS",
        {_ROLE: _CANDIDATE},
        raising=False,
    )
    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS",
        {},
        raising=False,
    )

    explicit_runtime = _FakeRuntime()
    explicit = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=explicit_runtime,
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    try:
        explicit.prepare_rows(tuple(range(128)))
        assert explicit.prefill_preappend_role_scoped is True
        assert explicit.prefill_preappend_role_variants == {}
        assert not explicit.can_preappend_attention_prefill(1, 128)
        explicit.discard_rows()
    finally:
        explicit.free()
    assert explicit_runtime.allocations == {}

    monkeypatch.setattr(module, "is_registered", lambda key: False)
    missing_runtime = _FakeRuntime()
    missing = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=missing_runtime,
    )
    try:
        missing.prepare_rows(tuple(range(128)))
        assert missing.prefill_preappend_role_scoped is True
        assert missing.prefill_preappend_role_variants == {}
        assert not missing.can_preappend_attention_prefill(1, 128)
        missing.discard_rows()
    finally:
        missing.free()
    assert missing_runtime.allocations == {}


def test_h5r_preappend_role_metadata_rejects_malformed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    for malformed, message in (
        (17, "must be a mapping"),
        ({"unknown": _CANDIDATE}, "unsupported Laguna preappend role"),
        ({_ROLE: ""}, "non-empty variants"),
        ({_ROLE: _H5M}, "unsupported variant"),
    ):
        monkeypatch.setattr(
            hip_gfx1100,
            "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS",
            malformed,
            raising=False,
        )
        runtime = _FakeRuntime()
        with pytest.raises(ValueError, match=message):
            module.allocate_laguna_kv_cache(
                _production_config(),
                context_length=4096,
                backend="hip_gfx1100",
                runtime=runtime,
            )
        assert runtime.malloc_calls == 0


def test_h5r_role_scope_does_not_change_legacy_gfx1151_preappend() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        cache.prepare_rows(tuple(range(128)))
        assert cache.prefill_preappend_role_scoped is False
        assert cache.prefill_preappend_role_variants == {}
        assert cache.can_preappend_attention_prefill(0, 128)
        assert cache.can_preappend_attention_prefill(1, 128)
        cache.discard_rows()
    finally:
        cache.free()
    assert runtime.allocations == {}
