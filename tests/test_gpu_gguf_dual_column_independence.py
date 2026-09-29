"""P6 premise: a fused projection's output columns must be independent.

Punchlist P6 proposes concatenating q/k/v into one resident weight and issuing
a single ``gemma4_project`` call, then splitting the result for the consumers
that still need separate buffers (route (b), chosen over route (a) in #41
because route (a) would widen the attention, norm, rotary and cache-append ABI).

That is only sound if **adding, resizing or removing a neighbouring projection
leaves this projection's own output columns bit-identical**. Each output column
is its own reduction over ``in_features``, so it should -- but "should" is not
measurement, and a kernel that tiles columns, shares loads across them, or
picks a different variant for a different total width would break it. That
failure would be silent in production: the fused pipeline would still run and
still return plausible numbers, just not the same ones.

**Why this fixture makes the test possible.** ``_build_compact_fixture`` builds
the two halves independently -- ``qweight_a`` with ``offset=0`` and
``qweight_b`` with ``offset=3`` -- so ``qweight_a`` depends only on
``out_features_a``, ``in_features``, ``counts`` and ``quant``, and is identical
across builds that differ **solely** in ``out_features_b``. The activation is
pinned by ``seed``. So the first half's weights and inputs are held constant
while only the second half's width varies, which isolates exactly the coupling
the fusion would introduce.

Both outcomes are useful: if the columns hold still, P6's arithmetic premise is
established on the real kernel; if they move, the row's premise is refuted
before eight files are touched.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

_QUANT = "gguf_q5_k"


def _hip_available() -> bool:
    """Explicit HIP guard so no-ROCm CI and publish runners skip, not fail."""
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _run(counts, in_features, out_features_a, out_features_b):
    """Build one fixture and return (kernel output, CPU reference)."""
    from tests.test_gpu_gguf_q4_k_q8_1_selected_prefill import (
        _run_q8_1_ds4_mmq32_selected_dual_gpu,
    )
    from tests.test_gpu_gguf_q4_k_selected_wmma_prefill import _build_compact_fixture

    fixture = _build_compact_fixture(
        quant=_QUANT,
        counts=counts,
        in_features=in_features,
        out_features_a=out_features_a,
        out_features_b=out_features_b,
        dtype="bf16",
        seed=23,
    )
    actual = _run_q8_1_ds4_mmq32_selected_dual_gpu(fixture, quant=_QUANT)
    return actual, fixture.reference


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "neighbour_widths"),
    [
        pytest.param([0, 17, 31], 512, 32, (64, 32, 96), id="uneven-neighbour-widths"),
        pytest.param([4, 0, 5], 256, 32, (32, 64), id="empty-expert-neighbour-widths"),
    ],
)
def test_first_half_columns_do_not_move_when_only_the_second_half_width_changes(
    counts, in_features, out_features_a, neighbour_widths
) -> None:
    """The kernel output for the first half must be bit-identical across runs
    whose only difference is how wide the second half is.

    Bit-equality, not ``allclose``: the weights and the activation are held
    constant by construction, so any movement is coupling between columns, not
    rounding. ``allclose`` here would hide precisely the failure the test
    exists to catch.
    """
    runs = []
    for out_features_b in neighbour_widths:
        actual, _reference = _run(counts, in_features, out_features_a, out_features_b)
        assert actual.shape[1] == out_features_a + out_features_b, (
            f"expected {out_features_a + out_features_b} columns at neighbour "
            f"width {out_features_b}, got {actual.shape[1]}"
        )
        runs.append((out_features_b, np.ascontiguousarray(actual[:, :out_features_a])))

    anchor_width, anchor = runs[0]
    for out_features_b, columns in runs[1:]:
        np.testing.assert_array_equal(
            anchor,
            columns,
            err_msg=(
                f"first half's {out_features_a} columns changed when only the "
                f"neighbour width moved {anchor_width} -> {out_features_b}. "
                f"Columns are coupled, so a fused q/k/v projection would not "
                f"reproduce the unfused one and P6's premise fails."
            ),
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("counts", "in_features", "out_features_a", "out_features_b"),
    [
        pytest.param([0, 17, 31], 512, 32, 64, id="uneven-halves"),
        pytest.param([4, 0, 5], 256, 32, 32, id="even-halves"),
    ],
)
def test_each_half_agrees_with_its_own_reference_at_every_width(
    counts, in_features, out_features_a, out_features_b
) -> None:
    """Independence alone is not enough: the columns must also stay correct.

    A kernel that ignored its neighbour but quietly lost accuracy in one
    configuration would pass the test above. This pairs it with the CPU oracle
    at the same thresholds the leaf's own numeric gate uses, so the premise
    rests on correctness as well as on stability.
    """
    from tests.test_gpu_gguf_q4_k_q8_1_selected_prefill import _max_softmax_kl
    from tests.test_gpu_gguf_q4_k_selected_wmma_prefill import _TOLERANCE_BF16

    actual, reference = _run(counts, in_features, out_features_a, out_features_b)

    np.testing.assert_allclose(actual, reference, **_TOLERANCE_BF16)
    assert _max_softmax_kl(reference, actual) <= 0.05
    top1 = float(
        np.mean(np.argmax(reference, axis=-1) == np.argmax(actual, axis=-1))
    )
    assert top1 >= 0.9, f"top-1 agreement {top1} fell below 0.9"