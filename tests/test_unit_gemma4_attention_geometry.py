"""CPU launch-planning limits for the gfx1100 Gemma attention kernel."""

import pytest


@pytest.mark.parametrize("keys", [1, 8192, 15615, 15616])
def test_supported_context_geometry(keys):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_shared_bytes

    assert gemma4_attention_shared_bytes(head_dim=512, keys=keys) == (512 + keys + 256) * 4


def test_oversized_attention_is_rejected_before_build_or_launch(monkeypatch):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as module

    def unexpected(**kwargs):
        raise AssertionError("an unsupported launch reached the compiler")

    monkeypatch.setattr(module, "build_gemma4_attention", unexpected)
    with pytest.raises(NotImplementedError, match="shared memory"):
        module.gemma4_attention_prefill_f32(
            1, 2, 3, 4, 5, tokens=1, keys=16640,
            num_heads=1, num_kv_heads=1, head_dim=768, scale=1.0,
        )


def test_reduction_workspace_uses_power_of_two_threads():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_shared_bytes

    # The reported requirement covers whichever route a launch may take. At a
    # narrow head the batched route's floor dominates: the key-class kernel holds
    # the logits plus a 256-lane partial per 256-thread group plus one max slot
    # per warp, so keys + 512 + 16 floats, independent of head_dim.
    assert gemma4_attention_shared_bytes(head_dim=6, keys=3) == (3 + 256 * 2 + 16) * 4
    # At a wide head the block kernel's own requirement dominates instead, with
    # its thread count rounded up to a power of two.
    assert gemma4_attention_shared_bytes(head_dim=512, keys=3) == (512 + 3 + 256) * 4


def test_runner_names_unsupported_context_before_allocating(monkeypatch):
    from types import SimpleNamespace
    from hipengine.runtime import gemma4 as module

    # ``sliding_window`` is part of the geometry the guard reads: it is what
    # decides whether a layer is charged the whole capacity or only its band.
    # ``num_heads``/``num_kv_heads`` go with it, because whether a tiled variant
    # can serve a layer the strict kernel cannot is a question about the head
    # geometry. ``Gemma4AttentionGeometry`` carries all four fields, so a stub
    # standing in for one has to carry them too rather than relying on the guard
    # not looking.
    #
    # 768 is the head width that still has no route: the 256/512 class kernel
    # moves its logits to owned global scratch beyond the LDS budget, and every
    # other width is bounded by it.
    config = SimpleNamespace(
        attention=(
            SimpleNamespace(
                head_dim=768, sliding_window=None, num_heads=16, num_kv_heads=2
            ),
        )
    )
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported geometry reached allocation")
    monkeypatch.setattr(module, "malloc", unexpected)
    with pytest.raises(NotImplementedError, match="tiled attention"):
        module.Gemma4Runner(SimpleNamespace(config=config), capacity=16384)
