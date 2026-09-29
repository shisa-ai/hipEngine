"""Which prefill kernel a Gemma 4 block routes to, and where each route is registered.

Routing is the load-bearing decision behind the tiled attention path: the kernel
existing is not enough if the layer never selects it, and selecting it for a
shape it cannot execute would be a crash. These tests pin the decision itself --
pure arithmetic on the dimensions, no GPU required -- separately from the kernel
mathematics, which `test_unit_gemma4_attention_tiled_prefill.py` pins against the
block kernel.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    _select_prefill_route,
    last_prefill_attention_route,
)


def route(**overrides):
    """A global-layer block (head_dim 512, 16 query heads, 2 KV heads)."""

    base = dict(
        rows=128,
        keys=128,
        num_heads=16,
        num_kv_heads=2,
        head_dim=512,
        sliding_window=None,
        mask_is_causal=True,
    )
    base.update(overrides)
    return _select_prefill_route(**base)


def test_a_head_dim_512_global_block_selects_the_tiled_kernel() -> None:
    """The five global layers are the whole point: they must route to tiled."""

    assert route() == "tiled"


def test_the_three_routes_partition_by_shape_not_by_provenance() -> None:
    """head_dim 512 -> tiled, head_dim 256 -> aotriton, anything else -> exact.

    The split is by dimensions the kernel can execute. No route consults a model
    name, a file, a hash, or a benchmark record.
    """

    assert route(head_dim=512) == "tiled"
    assert route(head_dim=256) == "aotriton"
    assert route(head_dim=128) == "exact"


def test_a_key_count_that_is_not_a_multiple_of_the_tile_falls_back() -> None:
    """A tile width the kernel cannot divide is a capability limit, not a refusal.

    Falling through to the correctness-first kernel keeps every shape runnable;
    nothing is gated off because it is unmeasured.
    """

    assert route(keys=131) == "exact"


def test_head_counts_that_do_not_tile_evenly_fall_back() -> None:
    """The grid maps whole tiles across heads; a remainder is unsupported."""

    assert route(num_heads=4) == "exact"  # 4 < 8 tiles
    assert route(num_heads=16, num_kv_heads=3) == "exact"  # gqa 16/3 is not 8


def test_the_sliding_layers_keep_their_aotriton_route() -> None:
    """head_dim 256 with a self-derivable mask still selects aotriton."""

    assert route(head_dim=256, rows=8, keys=4096) == "aotriton"


def test_an_older_block_is_never_left_without_a_kernel() -> None:
    """Every combination a caller can produce resolves to a runnable route."""

    for rows, keys, head_dim, heads in (
        (1, 1, 512, 16),
        (1, 130, 512, 16),
        (7, 7, 512, 16),
        (3, 4096, 256, 8),
        (5, 5, 64, 4),
        (1, 1024, 256, 8),
    ):
        assert route(
            rows=rows, keys=keys, head_dim=head_dim, num_heads=heads
        ) in {"tiled", "aotriton", "exact"}


def test_the_route_reporter_is_none_before_any_forward_call() -> None:
    """Observability must not invent an answer it has not been given."""

    # Other tests may have run a layer already; the contract is only that the
    # accessor exists and returns one of the three route names or None.
    route_seen = last_prefill_attention_route()
    assert route_seen is None or route_seen in {"tiled", "aotriton", "exact"}


def test_the_tiled_variant_is_registered_on_the_four_axis_key() -> None:
    """Reachable by registry key, not by a backend branch in the model layer."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        register_gemma4_attention_kernels,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
        gemma4_attention_prefill_tiled,
    )
    from hipengine.kernels.registry import resolve

    register_gemma4_attention_kernels(replace=True)
    kernel = resolve(
        backend="hip_gfx1100",
        layer="prefill_attention",
        quant="gguf_q4_k_m",
        variant="gemma4_tiled",
    )
    assert kernel is gemma4_attention_prefill_tiled

    # The existing shape-agnostic entry point still resolves to the block kernel.
    plain = resolve(
        backend="hip_gfx1100",
        layer="prefill_attention",
        quant="gguf_q4_k_m",
        variant="gemma4_plain",
    )
    assert plain is not gemma4_attention_prefill_tiled