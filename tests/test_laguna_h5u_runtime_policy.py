from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


_GLOBAL_ROLE = "global_m128_c4096_start0_384_exact"
_GLOBAL_CANDIDATE = "global_context_rows_cached_exact_spans"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_SWA_CANDIDATE = "swa_context_rows_qrow4_cached_exact_spans"
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


def _install_candidate_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_PREFILL_GLOBAL_PREAPPEND_ROLE_VARIANTS",
        {_GLOBAL_ROLE: _GLOBAL_CANDIDATE},
        raising=False,
    )


def test_h5u_global_preappend_metadata_is_default_off_and_runner_gated() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner

    assert hip_gfx1100.LAGUNA_PREFILL_GLOBAL_PREAPPEND_ROLE_VARIANTS == {}
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == {
        _SWA_ROLE: _SWA_CANDIDATE
    }
    assert not hasattr(hip_gfx1151, "LAGUNA_PREFILL_GLOBAL_PREAPPEND_ROLE_VARIANTS")
    rows_source = inspect.getsource(
        laguna_gguf_runner.LagunaGGUFResidentSession._run_layer_rows
    )
    init_source = inspect.getsource(
        laguna_gguf_runner.LagunaGGUFResidentSession.__init__
    )
    assert ".can_preappend_attention_prefill(" in rows_source
    assert "self._global_prefill_package_default = global_prefill_variant is None" in (
        init_source
    )
    assert "prefill_global_preappend_package_default=(" in init_source


def test_h5u_scoped_policy_routes_safe_global_and_retained_h5r_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import laguna_kv as module

    _install_candidate_map(monkeypatch)
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def resolve(layer: str, variant: str):
        calls.append((f"resolve:{layer}:{variant}", {}))

        def launch(*args: object, **kwargs: object) -> None:
            del args
            calls.append((f"launch:{variant}", dict(kwargs)))

        return launch

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
        assert cache.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _GLOBAL_CANDIDATE,
            _SWA_ROLE: _SWA_CANDIDATE,
        }
        for start in (0, 128, 256, 384):
            before = len(calls)
            assert dispatch(0, start, 128)
            launches = [
                call for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert [call[0] for call in launches] == [
                "launch:global_f32_rows_spans",
                f"launch:{_GLOBAL_CANDIDATE}",
            ]
            assert launches[-1][1]["start_position"] == start

        before = len(calls)
        assert dispatch(1, 256, 128)
        assert [
            call[0] for call in calls[before:] if call[0].startswith("launch:")
        ] == ["launch:swa_f32_rows_spans", f"launch:{_SWA_CANDIDATE}"]

        for layer_id, start, rows, expected in (
            (0, 512, 128, _GLOBAL_RETAINED),
            (0, 0, 64, _GLOBAL_RETAINED),
            (1, 64, 128, _SWA_RETAINED),
        ):
            before = len(calls)
            assert not dispatch(layer_id, start, rows)
            launches = [
                call[0] for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert launches[0] == f"launch:{expected}"
    finally:
        cache.free()
    assert runtime.allocations == {}


def test_h5u_global_and_h5r_swa_selectors_and_registration_fail_closed_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import laguna_kv as module

    _install_candidate_map(monkeypatch)

    def assert_routes(cache, *, global_expected: bool, swa_expected: bool) -> None:
        cache.position = -1
        cache.prepare_rows(tuple(range(128)))
        assert cache.can_preappend_attention_prefill(0, 128) is global_expected
        assert cache.can_preappend_attention_prefill(1, 128) is swa_expected
        cache.discard_rows()

    global_runtime = _FakeRuntime()
    explicit_global = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=global_runtime,
        global_prefill_variant="global_context_rows_qrow2_online_spans",
    )
    try:
        assert explicit_global.prefill_preappend_role_variants == {
            _SWA_ROLE: _SWA_CANDIDATE
        }
        assert_routes(explicit_global, global_expected=False, swa_expected=True)
    finally:
        explicit_global.free()
    assert global_runtime.allocations == {}

    swa_runtime = _FakeRuntime()
    explicit_swa = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=swa_runtime,
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    try:
        assert explicit_swa.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _GLOBAL_CANDIDATE
        }
        assert_routes(explicit_swa, global_expected=True, swa_expected=False)
    finally:
        explicit_swa.free()
    assert swa_runtime.allocations == {}

    original_is_registered = module.is_registered
    monkeypatch.setattr(
        module,
        "is_registered",
        lambda key: (
            False
            if key.variant == _GLOBAL_CANDIDATE
            else original_is_registered(key)
        ),
    )
    missing_runtime = _FakeRuntime()
    missing = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=missing_runtime,
    )
    try:
        assert missing.prefill_preappend_role_variants == {
            _SWA_ROLE: _SWA_CANDIDATE
        }
        assert_routes(missing, global_expected=False, swa_expected=True)
    finally:
        missing.free()
    assert missing_runtime.allocations == {}


def test_h5u_global_role_metadata_rejects_malformed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    for malformed, message in (
        (17, "must be a mapping"),
        ({"unknown": _GLOBAL_CANDIDATE}, "unsupported Laguna global preappend role"),
        ({_SWA_ROLE: _SWA_CANDIDATE}, "unsupported Laguna global preappend role"),
        ({_GLOBAL_ROLE: ""}, "non-empty variants"),
        ({_GLOBAL_ROLE: _GLOBAL_RETAINED}, "unsupported variant"),
    ):
        monkeypatch.setattr(
            hip_gfx1100,
            "LAGUNA_PREFILL_GLOBAL_PREAPPEND_ROLE_VARIANTS",
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


def test_h5u_role_scope_keeps_gfx1151_legacy_global_and_swa_preappend() -> None:
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
