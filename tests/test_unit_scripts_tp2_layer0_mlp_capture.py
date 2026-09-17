"""CPU tests for the layer-0 dense-MLP capture descriptors and FP32 reference.

No GPU, no model: every descriptor is built from synthetic buffers and every
arithmetic claim is checked against the independent numpy reference in
``scripts/tp2_layer0_mlp_capture.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tp2_layer0_capture import (
    CaptureError,
    CapturePlanError,
    descriptor,
    validate_capture_order,
)
from scripts.tp2_layer0_mlp_capture import (
    BF16,
    F32,
    PERSISTENT_LIFETIME,
    bf16_round,
    bf16_round_bits,
    bf16_ulp,
    build_mlp_capture_plan,
    concatenate_rank_rows,
    describe_resident_mlp,
    describe_shard_mlp,
    reference_full_width_mlp,    reference_sharded_mlp,
    reference_exact_partials,
    resident_mlp_producers,
    resident_mlp_widths,
    shard_mlp_lifetimes,
    shard_mlp_producers,
    shard_mlp_widths,
    shard_slices,
    silu,
    verify_shard_ownership,
)


class _Buffer:
    """The ``DeviceBuffer`` surface the descriptors read."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


# -- bf16 rounding ----------------------------------------------------------


def test_bf16_round_is_round_to_nearest_even() -> None:
    # 1 + 2**-9 is exactly halfway between two bf16 values; ties go to even.
    halfway_odd = np.float32(1.0) + np.float32(2.0**-9)
    halfway_even = np.float32(1.0) + np.float32(3.0) * np.float32(2.0**-9)
    assert float(bf16_round(np.array([halfway_odd], dtype=np.float32))[0]) == 1.0
    assert float(
        bf16_round(np.array([halfway_even], dtype=np.float32))[0]
    ) == np.float32(1.0) + np.float32(2.0**-7)


def test_bf16_round_preserves_exactly_representable_values() -> None:
    values = np.array([0.0, -0.0, 1.0, -2.5, 512.0], dtype=np.float32)
    assert np.array_equal(bf16_round(values), values)


def test_bf16_round_bits_widens_back_to_the_same_bits() -> None:
    rng = np.random.default_rng(20260917)
    values = (rng.standard_normal(4096).astype(np.float32) * 37.0).astype(np.float32)
    bits = bf16_round_bits(values)
    assert bits.dtype == np.uint16
    assert np.array_equal((bits.astype(np.uint32) << 16).view(np.float32), bf16_round(values))


def test_bf16_ulp_is_the_local_spacing() -> None:
    # bf16 has 7 explicit mantissa bits, so the spacing at 1.0 is 2**-7.
    assert float(bf16_ulp(np.float32(1.0))) == float(np.float32(2.0**-7))
    assert float(bf16_ulp(np.float32(2.0))) == float(np.float32(2.0**-6))
    assert float(bf16_ulp(np.float32(16.75))) == float(np.float32(2.0**-3))
    assert float(bf16_ulp(np.float32(0.0))) == 0.0


def test_silu_matches_the_direct_formula() -> None:
    values = np.array([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float32)
    expected = values * (np.float32(1.0) / (np.float32(1.0) + np.exp(-values)))
    assert np.allclose(silu(values), expected, rtol=0, atol=1e-6)
    assert float(silu(values)[2]) == 0.0


# -- ownership --------------------------------------------------------------


def test_shard_slices_tile_the_axis_exactly() -> None:
    ranges = shard_slices(17408, ranks=2)
    assert ranges == ((0, 8704), (8704, 17408))
    verify_shard_ownership(ranges, total=17408)
    assert shard_slices(6, ranks=3) == ((0, 2), (2, 4), (4, 6))


def test_shard_slices_refuse_a_non_divisible_axis() -> None:
    with pytest.raises(CaptureError, match="does not split"):
        shard_slices(17409, ranks=2)


def test_verify_shard_ownership_rejects_a_gap_and_an_overlap() -> None:
    with pytest.raises(CaptureError, match="not contiguous"):
        verify_shard_ownership(((0, 4), (5, 8)), total=8)
    with pytest.raises(CaptureError, match="not contiguous"):
        verify_shard_ownership(((0, 4), (3, 8)), total=8)
    with pytest.raises(CaptureError, match="cover"):
        verify_shard_ownership(((0, 4), (4, 7)), total=8)


# -- the reference schedules ------------------------------------------------


def _weights(rows: int, hidden: int, ffn: int, seed: int = 7) -> tuple:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, hidden)) * 2.0).astype(np.float32)
    gate = (rng.standard_normal((ffn, hidden)) * 0.05).astype(np.float32)
    up = (rng.standard_normal((ffn, hidden)) * 0.05).astype(np.float32)
    down = (rng.standard_normal((hidden, ffn)) * 0.05).astype(np.float32)
    return x, gate, up, down


def test_full_width_reference_shapes() -> None:
    x, gate, up, down = _weights(8, 16, 32)
    out = reference_full_width_mlp(x, gate, up, down)
    assert out["gate"].shape == (8, 32)
    assert out["up"].shape == (8, 32)
    assert out["intermediate"].shape == (8, 32)
    assert out["down"].shape == (8, 16)
    assert np.array_equal(out["intermediate"], bf16_round(silu(out["gate"]) * out["up"]))
    assert np.array_equal(out["down"], bf16_round(out["intermediate"] @ down.T))


def test_full_width_reference_rejects_mismatched_shapes() -> None:
    x, gate, up, down = _weights(4, 8, 16)
    with pytest.raises(CaptureError, match="gate .* and up"):
        reference_full_width_mlp(x, gate, up[:8], down)
    with pytest.raises(CaptureError, match="down shape"):
        reference_full_width_mlp(x, gate, up, down[:, :8])


def test_shard_intermediates_reconstruct_the_full_width_activation_exactly() -> None:
    """Splitting the output-feature axis must not change a single activation bit.

    Each rank computes ``bf16(silu(bf16(x @ gate_slice.T)) * bf16(x @ up_slice.T))``
    on its own column slice, so concatenating the shards has to reproduce the
    full-width activation bit-for-bit. A difference here would be an ownership
    or layout bug, not a rounding effect.
    """

    x, gate, up, down = _weights(64, 512, 8704 * 2)
    full = reference_full_width_mlp(x, gate, up, down)
    sharded = reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype=F32)
    rebuilt = concatenate_rank_rows(sharded["intermediates"], ranks=2)
    assert np.array_equal(rebuilt, full["intermediate"])
    for rank, (start, stop) in enumerate(sharded["output_ranges"]):
        assert np.array_equal(sharded["intermediates"][rank], full["intermediate"][:, start:stop])


def test_the_reduction_itself_adds_no_rounding() -> None:
    """``reduced`` must be exactly the f32 sum of the staged partials."""

    x, gate, up, down = _weights(16, 64, 256)
    sharded = reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype=BF16)
    exact = sharded["partials"][0].astype(np.float32) + sharded["partials"][1].astype(
        np.float32
    )
    assert np.array_equal(sharded["reduced_f32"], exact)
    assert np.array_equal(sharded["reduced_bf16"], bf16_round(exact))


def test_bf16_partials_cost_more_than_f32_partials_and_more_than_one_ulp() -> None:
    """The shipped bf16 partial boundary is a measurable extra rounding.

    This is the characterization fixture for the layer-0 MLP half: the sharded
    schedule with f32 partials differs from the full-width f32 down projection
    only by f32 reassociation, while the shipped bf16 partials add a rounding
    that is visible at the bf16 ULP of the partial. It pins the claim, so a
    silent schedule change (for example moving the group to f32 staging) makes
    it stale rather than quietly re-baselining.
    """

    x, gate, up, down = _weights(64, 512, 17408)
    full = reference_full_width_mlp(x, gate, up, down)
    f32_partials = reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype=F32)
    bf16_partials = reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype=BF16)

    f32_error = np.abs(f32_partials["reduced_f32"] - full["down_f32"])
    bf16_error = np.abs(bf16_partials["reduced_f32"] - full["down_f32"])
    assert float(f32_error.max()) < float(bf16_error.max())
    # Against the *unrounded* full-width f32 down projection: f32 partials
    # differ only by reassociation, bf16 partials by the partial rounding.
    for partial, rank in zip(bf16_partials["partials"], range(2)):
        exact = f32_partials["partials"][rank]
        limit = np.array([float(bf16_ulp(np.float32(v))) / 2.0 for v in exact.ravel()])
        assert np.all(np.abs(partial - exact).ravel() <= limit + 1e-6)
    # ...but the summed deviation is of the order of the bf16 ULP of the
    # partials themselves (two independently rounded partials add), not of the
    # f32-reassociation size.
    partial_scale = float(
        np.abs(np.asarray(f32_partials["partials"][0], dtype=np.float32)).max()
    )
    assert float(bf16_error.max()) >= 0.5 * float(bf16_ulp(np.float32(partial_scale)))
    assert float(bf16_error.max()) > 10.0 * float(f32_error.max())


def test_exact_partials_slice_the_activation_and_the_weight_together() -> None:
    """Regression: rank *i* contracts its own activation columns.

    The captured defect this pins: a reference that contracts the *full-width*
    activation against a shard-width down slice has the wrong inner dimension
    (``ffn`` against ``ffn / ranks``), so it raises instead of producing the
    rank's partial.
    """

    x, gate, up, down = _weights(5, 4, 8)
    sharded = reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype=F32)
    activation = concatenate_rank_rows(sharded["intermediates"], ranks=2)

    ranges = shard_slices(8, ranks=2)
    partials = reference_exact_partials(activation, down, ranks=2)
    assert [tuple(partial.shape) for partial in partials] == [(5, 4), (5, 4)]
    for (start, stop), partial, want in zip(ranges, partials, sharded["partials"]):
        assert np.array_equal(
            partial, activation[:, start:stop] @ down[:, start:stop].T
        )
        assert np.allclose(partial, want, rtol=1e-6, atol=1e-7)

    # Only the rank's own columns participate: zeroing the other rank's
    # activation columns leaves each partial bit-identical.
    for index, (start, stop) in enumerate(ranges):
        masked = activation.copy()
        masked[:, :start] = 0.0
        masked[:, stop:] = 0.0
        assert np.array_equal(
            reference_exact_partials(masked, down, ranks=2)[index], partials[index]
        )


def test_exact_partials_fail_closed_on_a_width_mismatch_and_a_bad_rank_count() -> None:
    with pytest.raises(CaptureError, match="does not match"):
        reference_exact_partials(
            np.zeros((2, 8), np.float32), np.zeros((4, 6), np.float32), ranks=2
        )
    with pytest.raises(CaptureError, match="does not split"):
        reference_exact_partials(
            np.zeros((2, 8), np.float32), np.zeros((4, 8), np.float32), ranks=3
        )


def test_sharded_reference_rejects_an_unknown_partial_dtype_and_bad_ranks() -> None:
    x, gate, up, down = _weights(4, 8, 16)
    with pytest.raises(CaptureError, match="partial_dtype"):
        reference_sharded_mlp(x, gate, up, down, ranks=2, partial_dtype="f16")
    with pytest.raises(CaptureError, match="does not split"):
        reference_sharded_mlp(x, gate, up, down, ranks=3)
    with pytest.raises(CaptureError, match="expected 3 rank arrays"):
        concatenate_rank_rows([np.zeros((1, 2))], ranks=3)


# -- descriptors ------------------------------------------------------------

_HIDDEN = 5120
_PER_RANK_FFN = 8704
_FFN = 17408
_ROWS = 64


def _resident_buffers() -> dict[str, _Buffer]:
    return {
        "post_norm": _Buffer(0x1000, _ROWS * _HIDDEN * 2),
        "residual": _Buffer(0x2000, _ROWS * _HIDDEN * 2),
        "ffn_intermediate": _Buffer(0x3000, _ROWS * _FFN * 2),
        "ffn_down": _Buffer(0x5000, _ROWS * _HIDDEN * 2),
        "out": _Buffer(0x4000, _ROWS * _HIDDEN * 2),
    }


def _resident_lifetimes() -> dict[str, tuple]:
    return {
        "post_norm": (("linear", 7, 15), ("full", 7, 15)),
        "residual": (("linear", 7, 17), ("full", 7, 17)),
        "ffn_intermediate": (("linear", 10, 13), ("full", 10, 13)),
        "ffn_down": (("linear", 12, 17), ("full", 12, 17)),
        "out": (("persistent", 0, 1),),
    }


def test_resident_widths_and_producers_cover_the_same_fields() -> None:
    widths = resident_mlp_widths(hidden_size=_HIDDEN, ffn_size=_FFN)
    declared = {name for producer in resident_mlp_producers() for name in producer.buffers}
    assert set(widths) == declared
    assert widths["ffn_intermediate"] == (BF16, _FFN)
    assert widths["post_norm"] == (BF16, _HIDDEN)


def test_shard_widths_and_producers_cover_every_rank_field() -> None:
    widths = shard_mlp_widths(hidden_size=_HIDDEN, per_rank_ffn=_PER_RANK_FFN)
    producers = shard_mlp_producers((0, 1))
    declared = {name for producer in producers for name in producer.buffers}
    expected = {f"{name}@{rank}" for name in widths for rank in (0, 1)}
    assert declared == expected
    assert widths["down_partial"] == (BF16, _HIDDEN)
    assert widths["reduced"] == (F32, _HIDDEN)
    assert widths["act"] == (BF16, _PER_RANK_FFN)


def test_describe_resident_mlp_uses_the_producer_widths_and_rows() -> None:
    descriptors = describe_resident_mlp(
        buffers=_resident_buffers(),
        rows=_ROWS,
        hidden_size=_HIDDEN,
        ffn_size=_FFN,
        lifetimes=_resident_lifetimes(),
    )
    by_name = {item.name: item for item in descriptors}
    assert set(by_name) == set(resident_mlp_widths(hidden_size=_HIDDEN, ffn_size=_FFN))
    assert by_name["ffn_intermediate"].shape == (_ROWS, _FFN)
    assert by_name["ffn_intermediate"].nbytes == _ROWS * _FFN * 2
    assert by_name["out"].producer == "residual_add"
    assert by_name["ffn_down"].producer == "down"
    assert by_name["ffn_intermediate"].producer == "gate_up_silu"


def test_describe_resident_mlp_fails_closed_on_an_over_read() -> None:
    buffers = _resident_buffers()
    buffers["ffn_intermediate"] = _Buffer(0x3000, _ROWS * _FFN * 2 - 2)
    with pytest.raises(CaptureError, match="capture would read"):
        describe_resident_mlp(
            buffers=buffers,
            rows=_ROWS,
            hidden_size=_HIDDEN,
            ffn_size=_FFN,
            lifetimes=_resident_lifetimes(),
        )


def test_describe_resident_mlp_fails_closed_on_a_missing_field() -> None:
    buffers = _resident_buffers()
    del buffers["residual"]
    with pytest.raises(CaptureError, match="no 'residual' buffer"):
        describe_resident_mlp(
            buffers=buffers,
            rows=_ROWS,
            hidden_size=_HIDDEN,
            ffn_size=_FFN,
            lifetimes=_resident_lifetimes(),
        )


def _shard_lifetimes() -> dict[str, tuple]:
    return shard_mlp_lifetimes(
        scratch_lifetimes={
            "post_norm": (("linear", 7, 15), ("full", 7, 15)),
            "residual": (("linear", 7, 17), ("full", 7, 17)),
        }
    )


def test_describe_shard_mlp_covers_both_ranks_with_rank_local_lifetimes() -> None:
    shared = _shard_lifetimes()
    buffers_by_rank = {
        rank: {
            "post_norm": _Buffer(0x1000 + rank, _ROWS * _HIDDEN * 2),
            "residual": _Buffer(0x2000 + rank, _ROWS * _HIDDEN * 2),
            "gate": _Buffer(0x3000 + rank, _ROWS * _PER_RANK_FFN * 2),
            "up": _Buffer(0x4000 + rank, _ROWS * _PER_RANK_FFN * 2),
            "act": _Buffer(0x5000 + rank, _ROWS * _PER_RANK_FFN * 2),
            "down_partial": _Buffer(0x6000 + rank, _ROWS * _HIDDEN * 2),
            "reduced": _Buffer(0x7000 + rank, _ROWS * _HIDDEN * 4),
            "cast": _Buffer(0x8000 + rank, _ROWS * _HIDDEN * 2),
            "out": _Buffer(0x9000 + rank, _ROWS * _HIDDEN * 2),
        }
        for rank in (0, 1)
    }
    descriptors = describe_shard_mlp(
        buffers_by_rank=buffers_by_rank,
        rows=_ROWS,
        hidden_size=_HIDDEN,
        per_rank_ffn=_PER_RANK_FFN,
        default_lifetimes=shared,
    )
    by_name = {item.name: item for item in descriptors}
    assert len(by_name) == 2 * len(shard_mlp_widths(hidden_size=_HIDDEN, per_rank_ffn=_PER_RANK_FFN))
    assert by_name["down_partial@1"].shape == (_ROWS, _HIDDEN)
    assert by_name["down_partial@1"].dtype == BF16
    assert by_name["reduced@0"].dtype == F32
    assert by_name["act@1"].width == _PER_RANK_FFN
    assert by_name["reduced@0"].producer == "staged_reduce"
    assert by_name["cast@1"].producer == "cast_reduced"
    assert by_name["out@1"].producer == "residual_add"
    assert by_name["gate@0"].ptr != by_name["gate@1"].ptr


def test_shard_producers_reject_duplicate_ranks() -> None:
    with pytest.raises(CaptureError, match="duplicate ranks"):
        shard_mlp_producers((0, 0))


def test_the_single_rank_producer_order_matches_the_suffixed_one() -> None:
    """A one-rank recorder's unsuffixed names must tile the same fields."""

    from scripts.tp2_layer0_mlp_capture import shard_mlp_producers_single_rank

    widths = shard_mlp_widths(hidden_size=_HIDDEN, per_rank_ffn=_PER_RANK_FFN)
    single = {name for producer in shard_mlp_producers_single_rank() for name in producer.buffers}
    assert single == set(widths)
    assert [p.name for p in shard_mlp_producers_single_rank()] == [
        p.name for p in shard_mlp_producers((0,))
    ]


def test_shard_widths_reject_an_unknown_partial_dtype() -> None:
    with pytest.raises(CaptureError, match="partial_dtype"):
        shard_mlp_widths(hidden_size=_HIDDEN, per_rank_ffn=_PER_RANK_FFN, partial_dtype="f16")


def test_shard_lifetimes_require_the_scratch_fields_and_persist_the_rest() -> None:
    lifetimes = _shard_lifetimes()
    assert lifetimes["post_norm"] == (("linear", 7, 15), ("full", 7, 15))
    assert lifetimes["residual"] == (("linear", 7, 17), ("full", 7, 17))
    for name in ("gate", "up", "act", "down_partial", "reduced", "cast", "out"):
        assert lifetimes[name] == PERSISTENT_LIFETIME
    with pytest.raises(CaptureError, match="no arena lifetime"):
        shard_mlp_lifetimes(scratch_lifetimes={"post_norm": (("linear", 7, 15),)})


# -- capture plan -----------------------------------------------------------


class _StubWeight:
    """The ``GGUFDeviceWeight`` surface dispatch resolution reads."""

    backend = "hip_gfx1100"

    def __init__(self, layout: str) -> None:
        self.spec = type(
            "_Spec", (), {"layout": layout, "quant_key": layout, "allocations": {"tiles"}}
        )()

    def allocation(self, name: str | None = None):
        class _Tensor:
            ptr = 0x1000

        class _Allocation:
            tensor = _Tensor()
            buffer = type("_Buffer", (), {"nbytes": 1 << 20, "ptr": 0x1000})()

        return _Allocation()


def test_the_bulk_down_residual_fusion_fails_closed_for_a_t16_down_projection() -> None:
    """Why the resident layer-0 MLP has its own bf16 ``ffn_down`` plane.

    ``launch_gguf_linear_residual``'s rows>4 branch is registered for
    ``dense_bf16`` weights only, so for the T16 layout this model loads it
    returns False *before* any registry or capability work and the resident
    helper falls through to a plain down projection plus ``gguf_bf16_add``.
    That is what makes ``ffn_down`` capturable on the teacher and what makes the
    two routes comparable at the down boundary.
    """

    import hipengine.runtime.gguf_linear as gguf_linear

    weight = _StubWeight("gguf_q6_k_t16_qmicro_planar_v1")
    assert (
        gguf_linear.launch_gguf_linear_residual(
            weight, 0x2000, 0x3000, 0x4000, 64, 17408, 5120, runtime=None
        )
        is False
    )
    assert (
        gguf_linear.launch_gguf_linear_residual(
            weight, 0x2000, 0x3000, 0x4000, 1, 17408, 5120, runtime=None
        )
        is False
    )


def _selected_leaf(monkeypatch, layout: str, *, rows: int, in_features: int, out_features: int, output_dtype: str = BF16, use_wmma: bool = True) -> tuple[str, bool]:
    """The leaf ``launch_gguf_linear`` actually resolves for one shape.

    The real launcher rewrites the decode-shape variant to a ``wmma_prefill_*``
    leaf that ``resolve_gguf_linear_dispatch`` alone does not name, so this
    captures the key at the launcher's own ``resolve`` call instead of guessing
    from the surface table.
    """

    import hipengine.runtime.gguf_linear as gguf_linear
    from hipengine.kernels.registry import KernelKey, is_registered

    captured: dict = {}

    class _Stop(Exception):
        pass

    def _capture(**kwargs):
        captured["key"] = kwargs
        raise _Stop

    monkeypatch.setattr(gguf_linear, "resolve", _capture)
    weight = _StubWeight(layout)
    try:
        with gguf_linear.wmma_prefill_session(use_wmma):
            gguf_linear.launch_gguf_linear(
                weight,
                0x2000,
                0x3000,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
                output_dtype=output_dtype,
                stream=0,
                runtime=None,
            )
    except _Stop:
        pass
    key = captured["key"]
    kernel_key = KernelKey(key["backend"], key["layer"], key["quant"], key["variant"])
    return str(key["variant"]), bool(is_registered(kernel_key))


def test_full_width_and_shard_down_shapes_select_the_same_wmma_prefill_leaf(monkeypatch) -> None:
    """The K-split is the only down-projection difference at rows>1.

    Both the full-width ``(17408 -> 5120)`` down projection and the shard
    ``(8704 -> 5120)`` slice resolve to the same ``t16_wmma_prefill_*`` leaf, so
    the shard route's extra rounding is the bf16 partial boundary and not a
    different kernel family.
    """

    layout = "gguf_q6_k_t16_qmicro_planar_v1"
    full = _selected_leaf(monkeypatch, layout, rows=64, in_features=17408, out_features=5120)
    shard = _selected_leaf(monkeypatch, layout, rows=64, in_features=8704, out_features=5120)
    assert full == shard == ("t16_wmma_prefill_bf16_bf16_out", True)


def test_the_t16_f32_output_wmma_prefill_leaf_is_unregistered(monkeypatch) -> None:
    """There is no f32-partial route for this quant/shape under the bulk context.

    The bulk schedule enters ``resident_prefill_dispatch_session`` with WMMA
    prefill on, which rewrites the registered ``t16_gemv_decode_bf16_f32_out``
    leaf to ``t16_wmma_prefill_bf16_f32_out``; that leaf is not registered, so a
    diagnostic arm that switched the group to ``staging_dtype='f32'`` would
    silently fall back to the CPU reference instead of measuring a kernel. The
    bf16 partial boundary therefore cannot be swapped out for f32 at this shape.
    """

    layout = "gguf_q6_k_t16_qmicro_planar_v1"
    variant, registered = _selected_leaf(
        monkeypatch, layout, rows=64, in_features=8704, out_features=5120, output_dtype=F32
    )
    assert (variant, registered) == ("t16_wmma_prefill_bf16_f32_out", False)
    variant, registered = _selected_leaf(
        monkeypatch,
        layout,
        rows=64,
        in_features=8704,
        out_features=5120,
        output_dtype=F32,
        use_wmma=False,
    )
    assert (variant, registered) == ("t16_gemv_decode_bf16_f32_out", True)


def test_mlp_capture_plan_orders_each_field_after_its_producer() -> None:
    descriptors = describe_resident_mlp(
        buffers=_resident_buffers(),
        rows=_ROWS,
        hidden_size=_HIDDEN,
        ffn_size=_FFN,
        lifetimes=_resident_lifetimes(),
    )
    plan = build_mlp_capture_plan(descriptors, resident_mlp_producers())
    assert [step.producer for step in plan] == [
        "post_norm_residual",
        "gate_up_silu",
        "down",
        "residual_add",
    ]
    assert plan[0].buffers == ("post_norm", "residual")
    assert plan[1].buffers == ("ffn_intermediate",)
    assert plan[2].buffers == ("ffn_down",)
    assert plan[3].buffers == ("out",)


def test_mlp_capture_plan_rejects_reading_a_field_after_its_aliasing_writer() -> None:
    """A dense gate/up plane is dead once the down projection reads it.

    ``ffn_gate_up`` is live over ``(linear, 9, 12)`` and ``ffn_down`` over
    ``(linear, 12, 17)``: the intervals are stage-disjoint, so the allocator is
    free to hand the down output the gate/up bytes. A capture that reads
    ``ffn_gate_up`` at or after the down writer returns the down tensor, and the
    plan has to fail closed instead.
    """

    descriptors = (
        descriptor(
            name="ffn_gate_up",
            producer="gate_up",
            ptr=0x5000,
            rows=_ROWS,
            width=2 * _FFN,
            dtype=BF16,
            allocated_nbytes=_ROWS * 2 * _FFN * 2,
            lifetime=(("linear", 9, 12),),
        ),
        descriptor(
            name="ffn_down",
            producer="down",
            ptr=0x5000,
            rows=_ROWS,
            width=_HIDDEN,
            dtype=BF16,
            allocated_nbytes=_ROWS * _HIDDEN * 2,
            lifetime=(("linear", 12, 17),),
        ),
    )
    # The end-of-layer read: both fields read after every writer has run.
    with pytest.raises(CapturePlanError, match="aliasing writer"):
        validate_capture_order(
            descriptors,
            capture_index={"ffn_gate_up": 2, "ffn_down": 2},
            write_index={"ffn_gate_up": 0, "ffn_down": 1},
            route="linear",
        )
    # Reading each one right after its own producer is accepted.
    validate_capture_order(
        descriptors,
        capture_index={"ffn_gate_up": 0, "ffn_down": 1},
        write_index={"ffn_gate_up": 0, "ffn_down": 1},
        route="linear",
    )
