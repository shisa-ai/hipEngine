"""MT-1 verifier-body composition boundary (CPU, no GPU required).

MT-1 asks for one target pass over the draft rows that uses the c1 decode
arithmetic *per row*, so every row equals what serial decode would have
produced. The q8_0 dense-linear half of that pass already exists and is
width-invariant (``test_q8_0_coltile8_rowbatch4_is_width_invariant_to_serial_c1``),
and the head half is already batched
(``Qwen4ExpGGUFResidentModelRunner.verify_target_block_deferred_head``).

What is missing is the **body**: a rows>1 target pass whose attention, recurrent
(GDN), MoE, and PLE ops reproduce the c1 decode arithmetic per row. These tests
pin one hard composition boundary at its source: the strict decode QSA mixer
refuses any row count other than one, so no decode-arithmetic body can use it at
the widths (2..8) a verifier consumes.

The rows>1 alternative that *does* exist (``run_qwen4_exp_qsa_prefill_token_mixer``)
is a different function with different arithmetic and is reached only from the
prefill layer. Composing the prefill owners into a complete body is bit-exact
against serial c1 on synthetic layer fixtures, but on the real target the
composed body diverges (hidden/logits) even at row 0 - the divergence is in the
rows>1 production dispatch, not the layer structure. See
``benchmarks/results/2026-10-07-qwen4exp-mt1-verifier-body-boundary/``.
"""

from __future__ import annotations

import pytest

from hipengine.runtime.qwen4_exp_runner import (
    run_qwen4_exp_dense_qsa_token_mixer,
)

#: Draft widths a width-invariant verifier must consume. MT-1 is scoped to
#: rows<=8; width 1 is excluded because the batched dense owner is *slower* at
#: width 1 (0.80x) and a verifier only batches rows>=2.
_VERIFIER_WIDTHS = (2, 3, 4, 5, 6, 7, 8)


@pytest.mark.parametrize("rows", _VERIFIER_WIDTHS)
def test_decode_qsa_mixer_rejects_rows_above_one(rows: int) -> None:
    """The strict c1 QSA mixer is rows==1-only.

    This is the concrete blocker for MT-1's complete verifier body: a rows>1
    decode-arithmetic pass cannot even reach the mixer, so the body has no
    attention primitive. The guard fires before any GPU state is touched, so
    this test needs no HIP runtime.
    """

    with pytest.raises(ValueError, match="requires rows == 1"):
        run_qwen4_exp_dense_qsa_token_mixer(
            0,
            {},
            attention_state=None,
            scratch=None,
            position=0,
            rows=rows,
            hidden=8,
            query_heads=2,
            kv_heads=1,
            head_dim=4,
            rotary_dim=4,
            theta=100.0,
        )


def test_decode_qsa_mixer_rows_one_is_not_rejected_by_the_width_guard() -> None:
    """Pin that the guard is specifically ``rows != 1``.

    At ``rows=1`` the mixer proceeds past the width guard and fails later on the
    (deliberately absent) device state. Asserting the *guard message* is what
    localizes the blocker to the width check rather than to argument handling.
    """

    with pytest.raises(Exception) as excinfo:
        run_qwen4_exp_dense_qsa_token_mixer(
            0,
            {},
            attention_state=None,
            scratch=None,
            position=0,
            rows=1,
            hidden=8,
            query_heads=2,
            kv_heads=1,
            head_dim=4,
            rotary_dim=4,
            theta=100.0,
        )
    assert "requires rows == 1" not in str(excinfo.value)
