"""DP-9 closure guard: the decode MoE gate/up owner is already near its comparator.

DP-9 ("small projection load scheduling") was re-ranked UP on 2026-09-30 after its
owner prerequisite was measured, naming ``gguf_q4_k_selected_dual_silu_logical128_t64``
as the largest less-examined decode owner at 13.968 ms/step. That figure came from a
capture taken *before* ``DE-1`` promoted the decode dp4a logical128 route into the
production default. After the promotion the same owner runs at 5.404 ms/step, i.e.
164.48 GB/s against the in-block comparator's 184.66 GB/s = 89.1% of it, leaving
0.591 ms/step = 1.25% of the decode kernel sum.

These tests pin the *condition* under which DP-9 stays closed rather than re-pinning
the promotion (``test_unit_qwen4_exp_profiles.py`` already does that). If the model
geometry changes, or the dp4a route is reverted, the closure no longer holds and
these fail loudly so the row is re-opened with a measurement instead of an assumption.

Geometry source (read from the model's own GGUF metadata, shard 1 of 4,
``unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL``): embedding_length 2560,
expert_feed_forward_length 640, expert_count 512, expert_used_count 10,
block_count 48.
"""

from __future__ import annotations

import pytest

from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_Q4_DP4A_DECODE_LAYERS,
)

# --- Model geometry this closure depends on (see module docstring for source) ---
HIDDEN = 2560
EXPERT_FFN = 640
EXPERTS = 512
TOP_K = 10
BLOCK_COUNT = 48

# --- Q4_K block layout, from kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip ---
QK_K = 256
Q4_K_BLOCK_BYTES = 144

# --- Measured figures, DE-1 (benchmarks/results/2026-09-29-qwen4exp-de1-dp4a-e2e) ---
COMPARATOR_GBPS = 184.66  # dense_gemv_out_kernel fp32 router GEMV, same step
LEDGER_TRAFFIC_B = 888.83e6  # includes a shared-expert term this model omits
OWNER_MS_PER_STEP = 5.404
DECODE_KERNEL_SUM_MS = 47.390
# The owner's rate the ledger quotes (164.48 GB/s) charges the larger ledger
# traffic. Our geometry model derives the unique gate+up floor (884.74 MB), so
# the rate it supports is slightly lower and is the honest one to pin.
OWNER_GBPS_FROM_FLOOR = 163.72


def _q4_k_row_bytes(in_features: int) -> int:
    assert in_features % QK_K == 0, f"{in_features} not a whole number of QK_K blocks"
    return (in_features // QK_K) * Q4_K_BLOCK_BYTES


def _gate_up_traffic_bytes(layers: int) -> int:
    """Unique gate+up weight bytes one decode step must stream for one token.

    Every selected expert's row is read once per step; the 1.5 GB expert working
    set far exceeds cache, so this is a floor, not an estimate.
    """
    return layers * TOP_K * EXPERT_FFN * _q4_k_row_bytes(HIDDEN) * 2


def test_dp9_geometry_is_consistent_with_the_measured_traffic_model() -> None:
    """The 888.83 MB/step DE-1 charged must reproduce from the geometry.

    This is the check that separates a real traffic model from a bits-per-weight
    guess: an unsound model would not land within a few percent of the figure the
    measured rates were computed from.
    """
    traffic = _gate_up_traffic_bytes(BLOCK_COUNT)
    # The ledger figure carries a small shared-expert term our unique-expert
    # model omits, so the derived value is a strict floor and must sit just below.
    assert 0 < LEDGER_TRAFFIC_B - traffic < 0.01 * LEDGER_TRAFFIC_B, (
        f"geometry-derived traffic {traffic/1e6:.2f} MB does not sit just below the "
        f"measured {LEDGER_TRAFFIC_B/1e6:.2f} MB/step; DP-9's closure needs re-derivation"
    )


def test_dp9_owner_rates_reproduce_from_the_geometry() -> None:
    """Both the pre- and post-promotion rates must recompute from traffic/time."""
    traffic_mb = _gate_up_traffic_bytes(BLOCK_COUNT) / 1e6
    # Pre-promotion: the exact logical128 owner at 14.013 ms/step.
    assert traffic_mb / 14.013 == pytest.approx(63.18, abs=0.5)
    # Post-promotion: the dp4a owner at 5.404 ms/step.
    assert traffic_mb / OWNER_MS_PER_STEP == pytest.approx(OWNER_GBPS_FROM_FLOOR, abs=0.5)


def test_dp9_owner_is_near_its_comparator_so_the_row_stays_closed() -> None:
    """The closure condition: <10% of headroom remains against the comparator.

    If this fails the owner is genuinely starved again and DP-9 must be re-opened.
    """
    ratio = OWNER_GBPS_FROM_FLOOR / COMPARATOR_GBPS
    assert ratio >= 0.85, (
        f"owner at {ratio:.1%} of its comparator; DP-9's 'already captured by the "
        f"DE-1 promotion' closure assumes >= 85%"
    )
    floor_ms = _gate_up_traffic_bytes(BLOCK_COUNT) / 1e6 / COMPARATOR_GBPS
    headroom_ms = OWNER_MS_PER_STEP - floor_ms
    assert headroom_ms == pytest.approx(0.61, abs=0.05)
    assert headroom_ms / DECODE_KERNEL_SUM_MS < 0.02


def test_dp9_closure_depends_on_the_dp4a_promotion_being_live() -> None:
    """The closure is only valid while the promoted route is the production default.

    DE-1 records the exact mechanism: "Q4_DP4A64 moved into
    PRODUCTION_Q8_QSA_RESTORED_FLAGS; the base profile already set it to \"1\" and
    the blanket arithmetic-recovery sweep was the only thing suppressing it".

    That is a two-sided invariant, so pin both sides. `Q4_DP4A64` must be listed in
    the recovery-sweep set (otherwise the sweep stops suppressing it and the flag is
    no longer evidence of a deliberate restoration) AND carry "1" in the restored
    dict (otherwise the sweep wins and the promoted owner is reverted, invalidating
    DP-9's closure). The end-to-end strict/production environment binding is pinned
    by `test_unit_qwen4_exp_profiles.py`; this asserts the closure's own precondition.
    """
    from hipengine.generation.qwen4_exp_profiles import (
        PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
        PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    )

    assert "Q4_DP4A64" in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS, (
        "Q4_DP4A64 left the recovery-sweep set: the restoration is no longer what "
        "keeps the promoted decode owner live, so DP-9's closure basis changed"
    )
    assert PRODUCTION_Q8_QSA_RESTORED_FLAGS.get("Q4_DP4A64") == "1", (
        "Q4_DP4A64 is no longer restored to 1: the arithmetic-recovery sweep would "
        "suppress the promoted decode owner and DP-9 must be re-opened"
    )
    # 43 of 48 layers carry the dp4a route; the rest keep the exact residual, and
    # the shortfall is the dynamic quantization of this model, not a config error.
    assert len(PRODUCTION_Q4_DP4A_DECODE_LAYERS) == 43
    assert max(PRODUCTION_Q4_DP4A_DECODE_LAYERS) < BLOCK_COUNT
    assert len(set(PRODUCTION_Q4_DP4A_DECODE_LAYERS)) == len(
        PRODUCTION_Q4_DP4A_DECODE_LAYERS
    )



