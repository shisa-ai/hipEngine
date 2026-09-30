"""Family bucketing for the Gemma 4 census rollup.

The classifier is first-match-wins over (family, regex) pairs. The t16
gate_up owners added by the D1 repack landing carry neither the legacy
``q4_k_selected`` substring nor a ``gate_up`` marker, so without patterns for
them the rollup reports ``moe.gate_up`` at 0.0000 ms on zero launches while
the real cost sits in ``other`` - which is what the first post-landing
census did.
"""

from __future__ import annotations

from scripts.gemma4_family_census import ENGINE_FAMILIES, family_of

DECODE_T16 = (
    "void (anonymous namespace)::qk_t16_selected_direct_gemv_kernel"
    "<unsigned short, 4, false, false, false>(unsigned short const*, ...)"
)
PREFILL_T16 = (
    "(anonymous namespace)::gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_"
    "prefill_compact32_kernel<1, false>(...)"
)
LEGACY_DECODE = (
    "void (anonymous namespace)::gguf_q4_k_selected_prefill_out_kernel"
    "<unsigned short, unsigned short>(...)"
)
LEGACY_PREFILL_RAW = (
    "(anonymous namespace)::gguf_q4_k_selected_dual_q8_1_ds4_mmq32_"
    "prefill_compact32_kernel<1>(...)"
)
DENSE_Q5_DIRECT = (
    "void (anonymous namespace)::qk_t16_selected_direct_gemv_kernel"
    "<unsigned short, 5, false, false, true>(...)"
)


def test_t16_gate_up_owners_bucket_to_gate_up():
    assert family_of(ENGINE_FAMILIES, DECODE_T16) == "moe.gate_up"
    assert family_of(ENGINE_FAMILIES, PREFILL_T16) == "moe.gate_up"


def test_legacy_gate_up_owners_still_bucket_to_gate_up():
    assert family_of(ENGINE_FAMILIES, LEGACY_DECODE) == "moe.gate_up"
    assert family_of(ENGINE_FAMILIES, LEGACY_PREFILL_RAW) == "moe.gate_up"


def test_dense_q5_direct_instance_is_not_claimed_by_gate_up():
    # The shared t16 template also serves dense single-expert launches at
    # qtype 5; the gate_up pattern must key the qtype-4 instantiation only.
    assert family_of(ENGINE_FAMILIES, DENSE_Q5_DIRECT) != "moe.gate_up"


def test_unrelated_owners_keep_their_buckets():
    assert family_of(ENGINE_FAMILIES, "q5_1_selected_gemv_bf16_bf16_kernel(...)") == "moe.down"
    assert family_of(ENGINE_FAMILIES, "gemma4_rmsnorm_kernel<...>(...)") == "norm"
