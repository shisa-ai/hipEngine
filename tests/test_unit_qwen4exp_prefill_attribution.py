"""Bucket assignment for the per-prefill kernel attribution.

Every case here is a kernel name taken verbatim from one of the four recorded
``rocprofv3`` traces, not an invented example. The bucket function exists to put
hipEngine's owner-oriented kernels and the llama.cpp family's flat kernel names
side by side, so a name that lands in the wrong bucket silently moves time
between the rows of a published comparison.
"""

from __future__ import annotations

import pytest

from scripts.qwen4exp_per_prefill_attribution import bucket

# (kernel name, expected bucket). Names are copied from the traces under
# /tmp/comparators-20260915 and /tmp/hipengine-gap-20260915.
CASES: tuple[tuple[str, str], ...] = (
    # --- hipEngine -----------------------------------------------------
    (
        "void (anonymous namespace)::gguf_k_prefill_out_coltile_rowbatch_kernel"
        "<float, float, 8, 8, 4>(float const*, unsigned short const*)",
        "dense_matmul",
    ),
    (
        "void (anonymous namespace)::gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_kernel"
        "<unsigned short const*, long const*, long const*, float*, float const*, long, long, long, long)",
        "moe_gate_up",
    ),
    (
        "(anonymous namespace)::q5_1_selected_wmma_iu8_risk_prefill_kernel"
        "(unsigned short const*, long const*, long const*, float*, float const*)",
        "moe_down",
    ),
    (
        "(anonymous namespace)::q5_1_selected_sparse_exact_repair_row_publish_kernel"
        "(unsigned short const*, long const*)",
        "moe_down",
    ),
    (
        "void (anonymous namespace)::q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32_kernel"
        "<true>(float const*, unsigned short const*)",
        "gr_read",
    ),
    (
        "void (anonymous namespace)::qwen35_paged_full_attn_decode_context_tensor_batch_kernel"
        "<true, 4>(float const*, float const*)",
        "attention",
    ),
    (
        "void (anonymous namespace)::qsa_sparse_attention_h256_wave_rows_f32_kernel"
        "<true, 4>(float const*, float const*)",
        "attention",
    ),
    (
        "void (anonymous namespace)::qwen4_exp_gdn_prefill_f32_kernel<true, true>"
        "(float const*, float const*)",
        "gdn",
    ),
    (
        "void (anonymous namespace)::qwen35_router_logits_f32_token_tile_dense_exact_kernel<4>"
        "(float const*, float const*)",
        "router_index",
    ),
    (
        "void (anonymous namespace)::grouped_rmsnorm_bf16_f32_kernel"
        "(unsigned short const*, float const*)",
        "elementwise_norm",
    ),
    # --- pwilkin strix-halo (MMB family) --------------------------------
    (
        "void (anonymous namespace)::mmb_dense_kernel<128, 128, 32, 64, 1>"
        "(unsigned char const*, unsigned short const*)",
        "dense_matmul",
    ),
    (
        "void (anonymous namespace)::mmb_dense_kernel<128, 256, 64, 64, 2>"
        "(unsigned char const*, unsigned short const*)",
        "dense_matmul",
    ),
    (
        "void (anonymous namespace)::mmb_f32split_kernel<128, 128, 32, 64, true>"
        "(float const*, float const*)",
        "dense_matmul",
    ),
    (
        "void (anonymous namespace)::mmb_routed_glu_kernel<64, 32, 16, 16, 44>"
        "(unsigned char const*, unsigned char const*)",
        "moe_gate_up",
    ),
    (
        "void (anonymous namespace)::mmb_routed_kernel<128, 32, 32, 16, 39>"
        "(unsigned char const*, unsigned long, unsigned short const*)",
        "moe_down",
    ),
    (
        "void gated_delta_net_tiled_cuda<128, 8, 8, 16, false>"
        "(float const*, float const*, float const*, float const*)",
        "gdn",
    ),
    (
        "hc_combine_norm_f32_b256(float const*, float const*, float const*, float const*, float*)",
        "gr_read",
    ),
    (
        "void (anonymous namespace)::hc_gate_mix_kernel<4, 1>"
        "(unsigned char const*, unsigned short const*)",
        "gr_read",
    ),
    (
        "void mm_ids_helper<10>(int const*, int*, int*, int*, int, int, int, int, int, bool)",
        "router_index",
    ),
    (
        "moe_weighted_reduction_f32_v4(float const*, float const*, float const*, float*)",
        "moe_down",
    ),
    # --- halo-box / upstream (classic llama.cpp family) -----------------
    (
        "void mul_mat_q<(ggml_type)7, 128, false>(char const*, int const*, int const*, int const*)",
        "dense_matmul",
    ),
    (
        "void mul_mat_q<(ggml_type)8, 128, true>(char const*, int const*, int const*, int const*)",
        "dense_matmul",
    ),
    (
        "void mul_mat_q_routed_compact<(ggml_type)12, 48, false>"
        "(char const*, int const*, int const*, int const*)",
        "moe_gate_up",
    ),
    # Plain upstream routes the Q4_K experts through the un-suffixed MMQ kernel.
    (
        "void mul_mat_q<(ggml_type)12, 128, false>(char const*, int const*, int const*, int const*)",
        "moe_gate_up",
    ),
    # The rocBLAS/Tensile GEMM. Regression: the name is lowercased before
    # matching, so a mixed-case pattern never fires and this landed in
    # "unattributed".
    (
        "Cijk_Alik_Bljk_SB_MT32x32x8_SN_1LDSB0_AMAS0_BL1_BS1_EPS0_GLVWA1_GLVWB1"
        "_GRVW1_GSU1_GSUASB_ISA1151_IU1_K1_K1",
        "dense_matmul",
    ),
    # Activation packing. Regression: `mmq` matched before `quantize`, so this
    # was counted as a matmul.
    (
        "void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>"
        "(float const*, int const*, void*, long, long, long, long)",
        "quantize_pack",
    ),
    (
        "void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, long)",
        "quantize_pack",
    ),
    (
        "void concat_transposed_src1_dim0<unsigned int>(char const*, char const*, char*, long, long)",
        "quantize_pack",
    ),
    (
        "void flash_attn_ext_f16<256, 256, 16, 4, false, false, false>"
        "(char const*, char const*, char const*, char const*)",
        "attention",
    ),
    (
        "void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>"
        "(float const*, float const*, float*)",
        "elementwise_norm",
    ),
    (
        "void unary_gated_op_kernel<&(op_sigmoid(float)), float>(float const*, float const*, float*)",
        "elementwise_norm",
    ),
    (
        "void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long, float)",
        "elementwise_norm",
    ),
    (
        "void gated_delta_net_cuda<128, false, false>(float const*, float const*, float const*)",
        "gdn",
    ),
    # A QSA kernel with a digit suffix. Regression: `qsa_` requires an
    # underscore and `\\battn` needs a word boundary, so neither matched and
    # this landed in "unattributed".
    ("void qsa3_attn_kernel(float const*, float const*, float*)", "attention"),
    # The wide-row dense tile family, promoted to the Q8_0 dense prefill default
    # at layers 16-47 on 2026-09-17. Regression: no `dense_gemv` / `gemm` /
    # `wmma` substring, so it landed in "unattributed".
    (
        "void (anonymous namespace)::q8_0_dense_wide_kernel<128, 256, 64, 64, false>"
        "(float const*, unsigned char const*, float*, int, int, int)",
        "dense_matmul",
    ),
)


@pytest.mark.parametrize(("name", "expected"), CASES)
def test_bucket_assigns_real_kernel_names(name: str, expected: str) -> None:
    assert bucket(name) == expected


def test_every_case_is_unique() -> None:
    names = [name for name, _ in CASES]
    assert len(names) == len(set(names)), "duplicate kernel name in the fixture"


def test_unknown_kernel_is_reported_not_guessed() -> None:
    assert bucket("void some_future_kernel_family(float const*)") == "unattributed"
