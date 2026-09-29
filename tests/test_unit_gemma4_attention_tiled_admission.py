"""Capability rules for the tiled Gemma 4 prefill kernel.

Pure shape arithmetic, so it runs without a GPU: each rule the HIP entry point
enforces by return code is re-checked here as a named exception, and each
*accepted* shape is checked just as hard. The point of the file is that a caller
learns the kernel cannot take a shape *before* allocating buffers, and that a
shape it does take is described by arithmetic rather than by a list of
previously-measured configurations.
"""

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
    Gemma4AttentionTiledUnsupported,
    gemma4_attention_tiled_admits,
)

# The real global-layer geometry: 16 query heads, 2 KV heads, head_dim 512.
_GEOMETRY = dict(tokens=512, num_heads=16, num_kv_heads=2, head_dim=512, keys=1024)


def _admits(**overrides):
    arguments = {**_GEOMETRY, **overrides}
    gemma4_attention_tiled_admits(**arguments)
    return arguments


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # A shape that fits every rule, varied along each dimension so acceptance
        # is not an artefact of one lucky combination.
        ({}, "global-layer-geometry"),
        (dict(tokens=1), "single-token"),
        (dict(tokens=7), "token-count-not-a-multiple-of-the-head-tile"),
        (dict(keys=128), "keys-equal-to-the-key-tile"),
        (dict(keys=8192), "keys-many-tiles"),
        (dict(num_heads=8, num_kv_heads=1, tokens=64, keys=256), "one-kv-head"),
        (dict(num_heads=64, num_kv_heads=8, tokens=64, keys=256), "eight-kv-heads"),
        # The rule is on the GQA ratio, not on either head count alone: 24/3 = 8
        # is admitted while 24/1 = 24 is not.
        (dict(num_heads=24, num_kv_heads=3, tokens=64, keys=256), "gqa-ratio-eight"),
    ],
)
def test_admits_shapes_the_kernel_can_execute(overrides, expected) -> None:
    # Absence of an exception is the assertion; `expected` exists so the failure
    # output names which dimension's rule was being probed.
    _admits(**overrides)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        # The capability rules themselves, each refused before any allocation.
        (dict(head_dim=256), "head_dim must be 512"),
        (dict(head_dim=0), "head_dim must be positive"),
        (dict(tokens=0), "tokens must be positive"),
        (dict(keys=0), "keys must be positive"),
        (dict(keys=100), "multiple of 128"),
        (dict(keys=64), "multiple of 128"),
        (dict(num_heads=4), "multiple of 8"),
        (dict(num_heads=12, num_kv_heads=2), "multiple of 8"),
        # 16/2 = 8 passes; a GQA ratio that is not a multiple of the head tile
        # fails the z-decoding rule even though both head counts are valid
        # multiples of their own divisors.
        (dict(num_heads=16, num_kv_heads=4), "multiple of 8"),
        (dict(num_heads=16, num_kv_heads=8), "multiple of 8"),
        (dict(num_heads=16, num_kv_heads=3), "multiple of num_kv_heads"),
    ],
)
def test_refuses_shapes_outside_the_templates_capability(overrides, needle) -> None:
    with pytest.raises(Gemma4AttentionTiledUnsupported, match=needle):
        _admits(**overrides)


def test_refusal_names_the_constraint_not_the_measurement() -> None:
    """The error states what the kernel cannot execute, never what is unmeasured.

    Absence of a benchmark row is not a reason to refuse a shape, so no message
    here may mention qualification, measurement, or benchmark status.
    """

    with pytest.raises(Gemma4AttentionTiledUnsupported) as raised:
        _admits(head_dim=128)

    message = str(raised.value).lower()
    for forbidden in ("measured", "benchmark", "qualified", "untested", "not tested"):
        assert forbidden not in message