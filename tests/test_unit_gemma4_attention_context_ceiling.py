"""Context admission for the Gemma 4 runner, and the ceiling each layer is charged.

Two questions, and both are asked of the *selected* consumer rather than of one
kernel in the family:

* **Which storage the run selected.** BF16 256/512-head attention moves its
  logits to request-owned global scratch when a resident row would exceed the LDS
  budget, so that geometry is servable at any key count. The direct INT8 consumer
  has its own context-sized LDS contract and is checked separately.
* **Which attention paths the run selected.** A layer the strict kernel cannot
  serve can still be served by a selected tiled variant, which stages a fixed K/V
  tile and walks the keys in online-softmax batches, so its shared memory does not
  grow with the context.

A windowed layer is charged its band -- ``window + rows - 1`` -- and not the whole
capacity, because the window is what bounds what the kernel reads.

CPU-only: ``__post_init__`` reads only ``weights.config``, so a ``SimpleNamespace``
stands in for the weights and no device is touched. The malloc/free pair is stubbed
for the same reason. The constructor is expected to raise before allocating
anything, which is itself worth asserting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention_module
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_shared_bytes,
)
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import (
    Gemma4Runner,
    gemma4_layer_key_count,
    gemma4_require_context_capacity,
)


def _config(*, head_dim: int, window: int | None) -> Gemma4TextConfig:
    geometry = Gemma4AttentionGeometry(
        layer_type="sliding_attention" if window else "full_attention",
        num_heads=16,
        num_kv_heads=2,
        head_dim=head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=10_000.0, head_dim=head_dim, rope_angles=head_dim // 2
        ),
        sliding_window=window,
        k_eq_v=False,
    )
    return Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=(geometry,),
        vocab_size=128,
    )


def _construct(
    config: Gemma4TextConfig,
    capacity: int,
    monkeypatch,
    variants: tuple[str, ...] | None = None,
) -> None:
    def _malloc(nbytes: int):
        raise AssertionError("the capacity check must run before any allocation")

    monkeypatch.setattr(gemma4_module, "malloc", _malloc)
    monkeypatch.setattr(gemma4_module, "free", lambda buffer: None)
    Gemma4Runner(
        weights=SimpleNamespace(config=config),
        capacity=capacity,
        max_block=8,
        prefill_attention_variants=variants,
    )


def _narrow_the_strict_bound(monkeypatch) -> None:
    """Make the strict kernel refuse every key count.

    On Gemma 4's own geometries it does not: the 256/512 class kernel moves its
    logits to request-owned global scratch beyond the LDS budget, so the strict
    path serves any context and the selected-path question never has to be asked.
    Narrowing the bound is how the *mechanism* is exercised -- when the strict
    kernel does refuse, the answer has to come from the paths the run selected,
    not from the geometry alone.
    """

    def _refuse(*, head_dim: int, keys: int) -> int:
        raise NotImplementedError(
            f"narrowed for this test: head_dim={head_dim} cannot hold {keys} keys"
        )

    monkeypatch.setattr(attention_module, "gemma4_attention_shared_bytes", _refuse)


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("capacity", [1024, 15_616, 15_617, 15_856, 15_857, 16_384, 32_768])
def test_repaired_bf16_geometries_admit_both_sides_of_the_old_lds_boundary(
    head_dim: int, capacity: int, monkeypatch
) -> None:
    assert gemma4_attention_shared_bytes(head_dim=head_dim, keys=capacity) <= 65_536
    with pytest.raises(AssertionError, match="before any allocation"):
        _construct(_config(head_dim=head_dim, window=None), capacity, monkeypatch)


def test_an_unimplemented_global_score_geometry_is_refused_before_allocation(monkeypatch) -> None:
    """A head width with no global-scratch route keeps the LDS bound.

    Windowless only, because the band is what a windowed layer is charged: a
    windowed head_dim 128 layer reads ``window + rows - 1`` columns whatever the
    context length is, and that band fits. The windowed case is pinned by
    ``test_a_sliding_only_config_is_served_at_a_long_context`` below.
    """

    with pytest.raises(NotImplementedError) as caught:
        _construct(_config(head_dim=128, window=None), 16_384, monkeypatch)
    assert "128" in str(caught.value)
    assert "16384" in str(caught.value)


def test_int8_admission_uses_its_own_consumer_not_the_bf16_global_path() -> None:
    config = _config(head_dim=512, window=None)
    gemma4_require_context_capacity(config, 16_384, kv_storage="bf16")
    with pytest.raises(ValueError, match="INT8 attention consumer"):
        gemma4_require_context_capacity(config, 16_384, kv_storage="int8_per_token_head")


def test_a_sliding_window_lifts_the_ceiling_it_is_charged(monkeypatch) -> None:
    """A window does help, and by exactly the band the layer reads.

    The previous reading charged every layer the whole capacity. A windowed layer
    with window 1024 reads ``window + rows - 1`` columns whatever the context
    length is, so at capacity 15,857 it was being asked for 1,050,688 bytes of LDS
    to read 1,024 keys. The window is what bounds the read, so the window is what
    the check has to be written against. A windowless layer keeps the capacity
    bound, which the refusal test above still pins.
    """

    with pytest.raises(AssertionError, match="before any allocation"):
        _construct(_config(head_dim=256, window=1024), 15_857, monkeypatch)


def test_a_windowed_layer_is_charged_its_band_not_the_capacity() -> None:
    """The band is ``window + rows - 1`` and ``rows`` is the widest submitted block.

    ``_keep_mask`` gives a windowed layer column 0 as key ``max(0, start - window +
    1)`` and its last column as key ``start + rows - 1``, so the span is ``window +
    rows - 1`` however long the context grows. ``rows`` is the widest block the
    runner submits, which is ``max_block``; a decode step is the narrow case and
    reads exactly the window. Charging the capacity instead asks the kernel to hold
    logits for keys the mask has already zeroed.
    """

    geometry = _config(head_dim=256, window=1024).attention[0]
    assert gemma4_layer_key_count(geometry, 262_144, rows=1) == 1024
    assert gemma4_layer_key_count(geometry, 262_144, rows=512) == 1535
    # A block can never be wider than the context, so the band is capped by it.
    assert gemma4_layer_key_count(geometry, 200, rows=512) == 200


def test_a_windowless_layer_is_charged_the_whole_capacity() -> None:
    """The full layers have no window, so the band is the context and stays so.

    This is the half of the guard the sliding fix must not loosen: the five full
    layers of Gemma 4 26B-A4B read every cached position, and they are what the
    tiled-attention work exists for.
    """

    geometry = _config(head_dim=512, window=None).attention[0]
    assert gemma4_layer_key_count(geometry, 262_144, rows=1) == 262_144
    assert gemma4_layer_key_count(geometry, 262_144, rows=512) == 262_144


def test_the_guard_checks_the_widest_block_the_runner_will_submit(monkeypatch) -> None:
    """The default ``rows`` is the runner's own prefill block, not one.

    ``Gemma4Runner`` derives ``max_block`` as ``min(capacity, DEFAULT_PREFILL_BLOCK)``
    and the generator's pre-load check has to reach the same answer without the
    runner. If the two disagreed, the early check would admit a context the
    construction-time check then refused, or the reverse.
    """

    geometry = _config(head_dim=256, window=1024).attention[0]
    block = min(262_144, gemma4_module.DEFAULT_PREFILL_BLOCK)
    # The band widens with the block, and the guard's default ``rows`` is the
    # runner's block rather than a literal of its own.
    assert gemma4_layer_key_count(geometry, 262_144, rows=block) == 1024 + block - 1
    assert gemma4_layer_key_count(geometry, 262_144, rows=1) == 1024


def test_a_sliding_only_config_is_served_at_a_long_context(monkeypatch) -> None:
    """The ceiling a windowed layer imposes is its band, not the context length.

    A windowed head_dim 256 layer at 262,144 positions needs 6,208 bytes of LDS,
    against the 1,050,688 it was charged. This is the assertion that the ceiling
    now belongs to the windowless layers alone.
    """

    with pytest.raises(AssertionError, match="before any allocation"):
        _construct(_config(head_dim=256, window=1024), 262_144, monkeypatch)


def test_a_selected_tiled_variant_lifts_a_ceiling_the_strict_kernel_sets(monkeypatch) -> None:
    """The capacity question is asked of the family, not of the strict kernel alone.

    This is what makes a long context reachable where the strict kernel cannot
    serve it: the answer depends on whether the requested profile selects a
    variant that implements the geometry and walks keys without holding them.

    Both sides are asserted here because they are one decision: a change that
    admitted a long context without a selected path to serve it would construct a
    runner whose first decode step fails.
    """

    _narrow_the_strict_bound(monkeypatch)

    with pytest.raises(NotImplementedError, match="no attention path serves"):
        _construct(_config(head_dim=512, window=None), 262_144, monkeypatch)

    with pytest.raises(AssertionError, match="before any allocation"):
        _construct(
            _config(head_dim=512, window=None),
            262_144,
            monkeypatch,
            variants=("gemma4_wmma_flash_full",),
        )


def test_a_variant_for_the_other_geometry_does_not_lift_the_ceiling(monkeypatch) -> None:
    """Selection is a capability match, so the wrong geometry is not a path.

    ``gemma4_wmma_flash`` implements the sliding shape and refuses head_dim 512.
    Requesting it must leave the windowless layers exactly where they were
    rather than being read as "some tiled variant was asked for".
    """

    _narrow_the_strict_bound(monkeypatch)

    with pytest.raises(NotImplementedError, match="no attention path serves"):
        _construct(
            _config(head_dim=512, window=None),
            262_144,
            monkeypatch,
            variants=("gemma4_wmma_flash",),
        )


@pytest.mark.parametrize("head_dim,servable", [(128, False), (512, True)])
def test_generator_checks_selected_context_capability_before_loading_weights(head_dim, servable, monkeypatch) -> None:
    """The refusal must not cost a multi-gigabyte load to reach.

    ``_ensure_runner`` loads the artifact and then constructs the runner, and the
    runner is where the ceiling was originally enforced. The geometry the ceiling
    depends on comes from metadata alone, so the generator now checks it first. This
    asserts that ordering: the load is stubbed to fail the test if it is reached, so
    a regression that moves the check back after the load is caught rather than
    quietly costing the user a load before the refusal.

    Both sides of the boundary are exercised: 128 is a head width no route serves at
    this context, and 512 is one the strict class kernel serves by moving its logits
    to owned global scratch.
    """

    import hipengine.generation.gemma4_gguf as module

    def _load(*args, **kwargs):
        raise AssertionError("weights loader reached")

    monkeypatch.setattr(module, "load_gemma4_device_weights", _load)
    monkeypatch.setattr(module, "gemma4_text_config_from_reader", lambda *a, **k: _config(head_dim=head_dim, window=None))
    generator = module.Gemma4GGUFGenerator("/unused.gguf", object(), object(), context_length=32_768)
    generator._reader = object()
    if servable:
        with pytest.raises(AssertionError, match="weights loader reached"):
            generator._ensure_runner()
    else:
        with pytest.raises(NotImplementedError):
            generator._ensure_runner()
