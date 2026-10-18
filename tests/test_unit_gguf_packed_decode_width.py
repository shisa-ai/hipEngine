"""Packed decode width policy for one resident session's KV storage dtype.

A resident session caps the packed decode width only where one packed batch
cannot describe the work: a direct INT8 stack that is missing its retained leaf,
or a hybrid INT8/BF16 layer layout whose layers take two different attention
leaves. A BF16 stack has no retained INT8 planes anywhere, so every
full-attention layer takes the standard batch leaf, one packed batch describes
the whole group, and the session keeps the declared standard width.
"""

from __future__ import annotations

from hipengine.core.dtype import DType
from hipengine.runtime.qwen35_gguf_runner import (
    _GGUF_STANDARD_PACKED_DECODE_ROWS,
    _gguf_packed_decode_max_rows,
)

# The inputs the resident session constructor resolves for a BF16 KV policy:
# ``direct_rows`` keeps its 1 initializer because the direct INT8 route
# resolution is INT8-only, ``_retained_decode_kernel`` stays None, and
# ``_gguf_int8_bf16_full_attention_layer_indices`` returns () for any storage
# dtype other than int8_per_token_head.
_BF16_SESSION_INPUTS = {
    "kv_storage_dtype": DType.BF16,
    "direct_rows": 1,
    "retained_decode_kernel": None,
    "bf16_full_attention_layer_indices": (),
}


def _retained_kernel() -> None:
    """Stand-in for the bound retained INT8 decode leaf."""


def test_bf16_session_keeps_the_declared_packed_decode_width() -> None:
    """A BF16 stack is never capped; it packs up to the declared standard width."""

    assert _GGUF_STANDARD_PACKED_DECODE_ROWS == 8
    assert _gguf_packed_decode_max_rows(**_BF16_SESSION_INPUTS) == 8


def test_bf16_width_ignores_int8_only_inputs() -> None:
    """Neither the INT8 route resolution nor a stale layer list narrows BF16.

    The constructor's BF16 inputs are the ones above, but the width must stay
    the standard one even if a caller hands the policy a retained leaf or a
    BF16-mirror layer list: those describe an INT8 stack, and a BF16 session
    never runs the retained leaf.
    """

    assert (
        _gguf_packed_decode_max_rows(
            kv_storage_dtype=DType.BF16,
            direct_rows=4,
            retained_decode_kernel=_retained_kernel,
            bf16_full_attention_layer_indices=(0, 1, 2, 3),
        )
        == 8
    )


def test_int8_without_the_retained_leaf_stays_capped_at_one() -> None:
    """An INT8 stack with no retained leaf still has no packed INT8 batch."""

    assert (
        _gguf_packed_decode_max_rows(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            direct_rows=4,
            retained_decode_kernel=None,
            bf16_full_attention_layer_indices=(),
        )
        == 1
    )


def test_int8_hybrid_layout_stays_capped_at_one() -> None:
    """A hybrid INT8/BF16 stack cannot share one packed batch."""

    assert (
        _gguf_packed_decode_max_rows(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            direct_rows=4,
            retained_decode_kernel=_retained_kernel,
            bf16_full_attention_layer_indices=(0, 1, 2, 3, 4, 5, 6, 7),
        )
        == 1
    )
    assert (
        _gguf_packed_decode_max_rows(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            direct_rows=4,
            retained_decode_kernel=_retained_kernel,
            bf16_full_attention_layer_indices=(0,),
        )
        == 1
    )


def test_int8_uniform_layout_keeps_the_declared_direct_width() -> None:
    """A uniform INT8 stack keeps the width the declaration qualifies."""

    assert (
        _gguf_packed_decode_max_rows(
            kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            direct_rows=4,
            retained_decode_kernel=_retained_kernel,
            bf16_full_attention_layer_indices=(),
        )
        == 4
    )
