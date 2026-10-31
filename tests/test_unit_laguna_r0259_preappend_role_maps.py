"""R0259: bounded backend preappend role maps as the supported policy form.

The maps are declaration data on the backend package. Two scopes exist: the
base preappend map (SWA role only) and the dense-initial map (global and SWA
roles). This module pins the semantics a normalized selector would have to
reproduce: absence versus an empty map, independent global/SWA overrides, the
per-scope candidate sets, the bounded start/row gate, and the operand
difference between the plain and score-replay variants.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_BASE = "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS"
_DENSE = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H5R = "swa_context_rows_qrow4_cached_exact_spans"
_H5M = "swa_context_rows_qrow4_sourcequal_exact_spans"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6Z_GLOBAL = (
    "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
    "exact_spans"
)
_H6A_SWA = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6W_SWA = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_SWA_PREFILL_ROLE = "qrow4_m128_c256_exact"
_SWA_PACKAGE_PREFILL = "swa_context_rows_qrow4_m128_c256_exact_spans"
_GLOBAL_RETAINED = "global_context_rows_spans"
_SWA_RETAINED = "swa_context_rows_qrow4_m128_c256_exact_spans"
_SCORE_SCRATCH_PTR = 0x40000000
_SCORE_SCRATCH_AVAILABLE = 150_994_944
_SCORE_SCRATCH_NBYTES = 12_582_912


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


def _allocate(**kwargs):
    from hipengine.runtime import laguna_kv as module

    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
        **kwargs,
    )
    return cache, runtime


def _record_launches(cache) -> list[tuple[str, tuple, dict]]:
    calls: list[tuple[str, tuple, dict]] = []

    def resolve(layer: str, variant: str):
        del layer

        def launch(*args: object, **kwargs: object) -> None:
            calls.append((variant, args, dict(kwargs)))

        return launch

    cache._resolve = resolve
    return calls


def _route(
    cache,
    calls: list[tuple[str, tuple, dict]],
    layer_id: int,
    positions: tuple[int, ...],
    rows: int,
    *,
    row_offset: int = 0,
) -> tuple[bool, list[str]]:
    cache.position = positions[0] - 1
    cache.prepare_rows(positions)
    sliced = row_offset != 0 or rows != len(positions)
    slice_kwargs = (
        {
            "row_offset": row_offset,
            "row_positions_ptr": 0x50000000 + row_offset * 8,
        }
        if sliced
        else {}
    )
    qualified = cache.can_preappend_attention_prefill(
        layer_id,
        rows,
        row_offset=row_offset,
    )
    if qualified:
        cache.append_rows(layer_id, 0x2000, 0x3000, rows, **slice_kwargs)
        cache.attend_prefill_cached(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
            **slice_kwargs,
        )
    else:
        cache.attend_prefill(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
            **slice_kwargs,
        )
        cache.append_rows(layer_id, 0x2000, 0x3000, rows, **slice_kwargs)
    cache.discard_rows()
    return qualified, [variant for variant, _, _ in calls]


def test_r0259_absent_and_empty_role_maps_are_distinct_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.delattr(hip_gfx1100, _BASE, raising=False)
    monkeypatch.delattr(hip_gfx1100, _DENSE, raising=False)
    absent, absent_runtime = _allocate()
    try:
        assert absent.prefill_preappend_role_scoped is False
        assert absent.prefill_preappend_role_variants == {}
        absent.position = -1
        absent.prepare_rows(tuple(range(128)))
        assert absent.can_preappend_attention_prefill(0, 128)
        assert absent.can_preappend_attention_prefill(1, 128)
        absent.discard_rows()
    finally:
        absent.free()
    assert absent_runtime.allocations == {}

    monkeypatch.setattr(hip_gfx1100, _BASE, {}, raising=False)
    monkeypatch.setattr(hip_gfx1100, _DENSE, {}, raising=False)
    empty, empty_runtime = _allocate()
    try:
        assert empty.prefill_preappend_role_scoped is True
        assert empty.prefill_preappend_role_variants == {}
        empty.position = -1
        empty.prepare_rows(tuple(range(128)))
        assert not empty.can_preappend_attention_prefill(0, 128)
        assert not empty.can_preappend_attention_prefill(1, 128)
        empty.discard_rows()
    finally:
        empty.free()
    assert empty_runtime.allocations == {}


def test_r0259_role_map_scopes_resolve_and_merge_by_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.delattr(hip_gfx1100, _BASE, raising=False)
    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE,
        {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6W_SWA},
        raising=False,
    )
    dense_only, dense_runtime = _allocate()
    try:
        assert dense_only.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _H6N_GLOBAL,
            _SWA_ROLE: _H6W_SWA,
        }
    finally:
        dense_only.free()
    assert dense_runtime.allocations == {}

    monkeypatch.setattr(hip_gfx1100, _BASE, {_SWA_ROLE: _H5R}, raising=False)
    monkeypatch.delattr(hip_gfx1100, _DENSE, raising=False)
    base_only, base_runtime = _allocate()
    try:
        assert base_only.prefill_preappend_role_variants == {_SWA_ROLE: _H5R}
    finally:
        base_only.free()
    assert base_runtime.allocations == {}

    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE,
        {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6A_SWA},
        raising=False,
    )
    merged, merged_runtime = _allocate()
    try:
        assert merged.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _H6Z_GLOBAL,
            _SWA_ROLE: _H6A_SWA,
        }
    finally:
        merged.free()
    assert merged_runtime.allocations == {}


def test_r0259_global_and_swa_selection_drop_one_role_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.setattr(hip_gfx1100, _BASE, {_SWA_ROLE: _H5R}, raising=False)
    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE,
        {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6A_SWA},
        raising=False,
    )

    explicit_global, global_runtime = _allocate(
        global_prefill_variant="global_context_rows_qrow2_online_spans",
    )
    try:
        assert explicit_global.prefill_preappend_role_variants == {
            _SWA_ROLE: _H6A_SWA
        }
    finally:
        explicit_global.free()
    assert global_runtime.allocations == {}

    # The runner's package-default lever drops the same role without an
    # explicit variant request, so the rollback does not depend on a variant
    # name that the backend package may later change.
    overridden_global, override_runtime = _allocate(
        prefill_global_preappend_package_default=False,
    )
    try:
        assert overridden_global.prefill_preappend_role_variants == {
            _SWA_ROLE: _H6A_SWA
        }
    finally:
        overridden_global.free()
    assert override_runtime.allocations == {}

    explicit_swa, swa_runtime = _allocate(
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    try:
        assert explicit_swa.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _H6N_GLOBAL
        }
    finally:
        explicit_swa.free()
    assert swa_runtime.allocations == {}

    overridden_swa, overridden_swa_runtime = _allocate(
        prefill_preappend_package_default=False,
    )
    try:
        assert overridden_swa.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _H6N_GLOBAL
        }
    finally:
        overridden_swa.free()
    assert overridden_swa_runtime.allocations == {}

    # An explicit request that names the package default is still explicit for
    # the preappend gate, while the SWA prefill role map keys on the resolved
    # variant. The two role-map mechanisms disagree on this one request, so a
    # unified selector has to pick a side rather than preserve both.
    explicit_default_swa, explicit_default_runtime = _allocate(
        swa_prefill_variant=_SWA_PACKAGE_PREFILL,
    )
    try:
        assert explicit_default_swa.prefill_preappend_role_variants == {
            _GLOBAL_ROLE: _H6N_GLOBAL
        }
        assert explicit_default_swa.swa_prefill_role_variants == {
            _SWA_PREFILL_ROLE: _H5M
        }
    finally:
        explicit_default_swa.free()
    assert explicit_default_runtime.allocations == {}

    both, both_runtime = _allocate(
        global_prefill_variant="global_context_rows_qrow2_online_spans",
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    try:
        assert both.prefill_preappend_role_scoped is True
        assert both.prefill_preappend_role_variants == {}
        assert not both.can_preappend_attention_prefill(0, 128)
        assert not both.can_preappend_attention_prefill(1, 128)
    finally:
        both.free()
    assert both_runtime.allocations == {}


def test_r0259_role_scopes_reject_cross_scope_roles_and_variants_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    for scope, malformed, message in (
        (
            _BASE,
            {_GLOBAL_ROLE: _H6N_GLOBAL},
            "unsupported Laguna preappend role "
            f"{_GLOBAL_ROLE!r}",
        ),
        (
            _DENSE,
            {_GLOBAL_ROLE: _H5M},
            f"unsupported variant {_H5M!r} for Laguna dense-initial "
            f"preappend role {_GLOBAL_ROLE!r}",
        ),
    ):
        monkeypatch.setattr(hip_gfx1100, _BASE, {}, raising=False)
        monkeypatch.setattr(hip_gfx1100, _DENSE, {}, raising=False)
        monkeypatch.setattr(hip_gfx1100, scope, malformed, raising=False)
        runtime = _FakeRuntime()
        with pytest.raises(ValueError, match=message):
            module.allocate_laguna_kv_cache(
                _production_config(),
                context_length=4096,
                backend="hip_gfx1100",
                runtime=runtime,
            )
        assert runtime.malloc_calls == 0


def test_r0259_gate_is_bounded_by_start_row_and_ring_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.setattr(hip_gfx1100, _BASE, {_SWA_ROLE: _H5R}, raising=False)
    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE,
        {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6A_SWA},
        raising=False,
    )
    cache, runtime = _allocate()
    calls = _record_launches(cache)
    try:
        for layer_id, write_variant, candidate in (
            (0, "global_f32_rows_spans", _H6N_GLOBAL),
            (1, "swa_f32_rows_spans", _H6A_SWA),
        ):
            for row_offset in (0, 128, 256, 384):
                calls.clear()
                qualified, launches = _route(
                    cache,
                    calls,
                    layer_id,
                    tuple(range(512)),
                    128,
                    row_offset=row_offset,
                )
                assert qualified
                assert launches == [write_variant, candidate]

        # A chunk that starts at the SWA ring capacity wraps, so every tile of
        # that chunk must attend before appending.
        for layer_id, retained in (
            (0, _GLOBAL_RETAINED),
            (1, _H5M),
        ):
            for row_offset in (0, 128, 256, 384):
                calls.clear()
                qualified, launches = _route(
                    cache,
                    calls,
                    layer_id,
                    tuple(range(512, 1024)),
                    128,
                    row_offset=row_offset,
                )
                assert not qualified
                assert launches == [retained, (
                    "global_f32_rows_spans"
                    if layer_id == 0
                    else "swa_f32_rows_spans"
                )]

        # Complete global attention widths are safe for the primitive and are
        # still outside the bounded role policy.
        for rows in (256, 512):
            cache.position = -1
            cache.prepare_rows(tuple(range(rows)))
            assert cache.can_preappend_prefill(0, rows)
            assert not cache.can_preappend_attention_prefill(0, rows)
            cache.discard_rows()

        # The SWA ring tail that exactly fills the window is admitted; the
        # partial ring tail that would wrap is not.
        for start, rows, expected in (
            (384, 128, True),
            (448, 128, False),
            (512, 128, False),
        ):
            calls.clear()
            qualified, _ = _route(
                cache,
                calls,
                1,
                tuple(range(start, start + rows)),
                rows,
            )
            assert qualified is expected

        for start in (64, 192):
            calls.clear()
            qualified, launches = _route(
                cache,
                calls,
                1,
                tuple(range(start, start + 128)),
                128,
            )
            assert not qualified
            assert launches[0] == _SWA_RETAINED
    finally:
        cache.free()
    assert runtime.allocations == {}


def test_r0259_score_replay_roles_fall_back_without_changing_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.setattr(hip_gfx1100, _BASE, {_SWA_ROLE: _H5R}, raising=False)
    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE,
        {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6W_SWA},
        raising=False,
    )

    unbound, unbound_runtime = _allocate()
    calls = _record_launches(unbound)
    try:
        assert not unbound.prefill_score_scratch_bound
        qualified, launches = _route(
            unbound,
            calls,
            0,
            tuple(range(256, 384)),
            128,
        )
        assert qualified
        assert launches == ["global_f32_rows_spans", _H6N_GLOBAL]
        variant, args, kwargs = calls[-1]
        assert variant == _H6N_GLOBAL
        assert args[6] != _SCORE_SCRATCH_PTR
        assert "score_scratch_nbytes" not in kwargs
        assert kwargs["start_position"] == 256
    finally:
        unbound.free()
    assert unbound_runtime.allocations == {}

    bound, bound_runtime = _allocate()
    calls = _record_launches(bound)
    try:
        malloc_calls = bound_runtime.malloc_calls
        bound.bind_prefill_score_scratch(
            _SCORE_SCRATCH_PTR,
            _SCORE_SCRATCH_AVAILABLE,
        )
        assert bound_runtime.malloc_calls == malloc_calls
        assert bound.prefill_score_scratch_bound
        qualified, launches = _route(
            bound,
            calls,
            0,
            tuple(range(256, 384)),
            128,
        )
        assert qualified
        assert launches == ["global_f32_rows_spans", _H6Z_GLOBAL]
        _, args, kwargs = calls[-1]
        assert args[6] == _SCORE_SCRATCH_PTR
        assert kwargs["score_scratch_nbytes"] == _SCORE_SCRATCH_NBYTES
        assert kwargs["start_position"] == 256

        # The SWA score-replay role only applies at the late starts; the
        # earlier starts keep the dense-initial cached variant on the same
        # append-before-attention route.
        calls.clear()
        qualified, launches = _route(bound, calls, 1, tuple(range(128)), 128)
        assert qualified
        assert launches == ["swa_f32_rows_spans", _H6A_SWA]
    finally:
        bound.free()
    assert bound_runtime.allocations == {}


def test_r0259_base_scope_role_is_not_dense_initial_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100

    monkeypatch.setattr(hip_gfx1100, _BASE, {_SWA_ROLE: _H5R}, raising=False)
    monkeypatch.delattr(hip_gfx1100, _DENSE, raising=False)
    cache, runtime = _allocate()
    try:
        assert cache.prefill_preappend_role_variants == {_SWA_ROLE: _H5R}
        cache._dense_initial_metadata_valid = False
        cache.position = -1
        cache.prepare_rows(tuple(range(128)))
        assert cache.can_preappend_attention_prefill(1, 128)
        assert not cache.can_preappend_attention_prefill(0, 128)
        cache.discard_rows()
    finally:
        cache.free()
    assert runtime.allocations == {}
