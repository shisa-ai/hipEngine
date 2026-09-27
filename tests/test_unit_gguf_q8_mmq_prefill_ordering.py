"""Ordering guard for the raw-Q8 MMQ prefill gate.

The Q8_0 MMQ chain and the bf16 WMMA prefill owner are both registered for the
same shapes. Which one a shape gets is decided by where each sits in
``resolve_gguf_linear_dispatch``'s rewrite chain, not by a priority field, and
the chain applies ``_wmma_prefill_dispatch`` first. The MMQ gate recognises only
the un-rewritten ``prefill_bf16_bf16_out`` name, so a WMMA-claimed shape never
reaches it.

That ordering is load-bearing. The Gemma 4 policy
(``GEMMA4_Q8_MMQ_MIN_ROWS``) was calibrated against the *exact* owner, which is
what a shape runs when WMMA prefill is off; measured against the WMMA owner at
rows=512 the chain is 1.44x slower (258.6 -> 373.2 ms for a 512-token prefill,
campaign iteration 154). So moving the MMQ gate ahead of the WMMA rewrite would
silently regress the default path by 44%, and this file is what makes that
failure loud instead of silent.

CPU-only: it exercises the rewrite functions and the policy directly, with no
device work.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    Q8MMQPrefillPolicy,
    q8_mmq_d4x3_nbytes,
)
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear

# One of Gemma 4's dense Q8_0 projections at the runner's block size.
_ROWS = 512
_IN = 2816
_OUT = 4096

# The map Gemma's runner carries; kept here as literals so a change to the
# production constant has to be reflected deliberately in this guard too.
_GEMMA4_MIN_ROWS = {
    (2816, 2112): 512,
    (2816, 2048): 512,
    (4096, 2816): 512,
    (2816, 4096): 512,
    (8192, 2816): 512,
    (2816, 8192): 512,
}


def _policy(min_rows: dict[tuple[int, int], int] | None = None) -> Q8MMQPrefillPolicy:
    return Q8MMQPrefillPolicy(
        min_rows=_GEMMA4_MIN_ROWS if min_rows is None else min_rows,
        max_rows=4096,
        risk_threshold=1.0e-5,
        max_out_features=8192,
    )


def _session(policy: Q8MMQPrefillPolicy) -> gguf_linear._Q8MMQPrefillSession:
    """A session with valid-looking pointers and a workspace that fits."""

    return gguf_linear._Q8MMQPrefillSession(
        workspace_ptr=1,
        workspace_nbytes=q8_mmq_d4x3_nbytes(_ROWS, _IN),
        risk_count_ptr=2,
        risk_count_nbytes=4,
        risk_indices_ptr=3,
        risk_indices_nbytes=_ROWS * _OUT * 4,
        library=None,
        policy=policy,
    )


def _raw_q8_dispatch() -> gguf_linear.GGUFLinearDispatch:
    """The dispatch the rows>1 alias produces before any rewrite."""

    return gguf_linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out"),
        "raw",
    )


def test_the_wmma_rewrite_claims_the_variant_the_mmq_gate_reads():
    """The premise of the ordering: WMMA rewrites the name MMQ matches on."""

    rewritten = gguf_linear._wmma_prefill_dispatch(
        _raw_q8_dispatch(), rows=_ROWS, in_features=_IN, use_wmma=True
    )
    assert rewritten.key.variant == "wmma_prefill_bf16_bf16_out"
    assert rewritten.abi == "wmma_raw"


def test_mmq_gate_leaves_a_wmma_claimed_shape_alone():
    """In the shipped order the MMQ chain must not take a WMMA-claimed shape.

    This is the guard. Measured against the WMMA owner the chain is 1.44x
    slower at this shape, so a reorder that let it through would regress the
    default prefill path by 44%.
    """

    dispatch = _raw_q8_dispatch()
    token = gguf_linear._q8_mmq_prefill_session.set(_session(_policy()))
    try:
        assert _policy()(_ROWS, _IN, _OUT), "the shape must be admitted, or this proves nothing"
        after_wmma = gguf_linear._wmma_prefill_dispatch(
            dispatch, rows=_ROWS, in_features=_IN, use_wmma=True
        )
        after_mmq = gguf_linear._q8_mmq_prefill_dispatch(
            after_wmma, rows=_ROWS, in_features=_IN, out_features=_OUT
        )
        assert after_mmq is after_wmma
        assert after_mmq.key.variant == "wmma_prefill_bf16_bf16_out"
        assert after_mmq.abi == "wmma_raw"
    finally:
        gguf_linear._q8_mmq_prefill_session.reset(token)


def test_mmq_gate_takes_the_shape_when_wmma_prefill_is_off():
    """The route the Gemma policy was calibrated against is still reachable.

    With ``use_wmma`` false the rewrite is a no-op, the un-rewritten name
    survives, and the chain takes over -- which is the configuration the map's
    ``min_rows`` values were measured in.
    """

    dispatch = _raw_q8_dispatch()
    token = gguf_linear._q8_mmq_prefill_session.set(_session(_policy()))
    try:
        untouched = gguf_linear._wmma_prefill_dispatch(
            dispatch, rows=_ROWS, in_features=_IN, use_wmma=False
        )
        assert untouched is dispatch
        after_mmq = gguf_linear._q8_mmq_prefill_dispatch(
            untouched, rows=_ROWS, in_features=_IN, out_features=_OUT
        )
        assert after_mmq is not untouched
        assert after_mmq.key.variant == "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out"
        assert after_mmq.abi == "raw_mmq_d4x3"
    finally:
        gguf_linear._q8_mmq_prefill_session.reset(token)


def test_the_gate_is_inert_without_a_session():
    """No session, no rewrite: the gate is a model-plugin decision."""

    dispatch = _raw_q8_dispatch()
    after_mmq = gguf_linear._q8_mmq_prefill_dispatch(
        dispatch, rows=_ROWS, in_features=_IN, out_features=_OUT
    )
    assert after_mmq is dispatch


@pytest.mark.parametrize("shape", sorted(_GEMMA4_MIN_ROWS))
def test_gemma_policy_admits_its_shapes_only_at_the_block_size(shape):
    """Every mapped shape is admitted at 512 rows and refused below it."""

    in_features, out_features = shape
    policy = _policy()
    assert policy(_ROWS, in_features, out_features)
    for rows in (1, 64, 256, 511):
        assert not policy(rows, in_features, out_features)
    for rows in (513, 1024, 4096):
        assert policy(rows, in_features, out_features)
    assert not policy(_ROWS, in_features, out_features + 1)
