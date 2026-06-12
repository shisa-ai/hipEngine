/*
 * MoE C1 dispatcher: bundles ~11 extern-C kernel launchers into one Python →
 * C call to cut 10 ctypes ABI transitions per MoE layer.  See
 * docs/MTP.md M13.C deferral and docs/DFLASH.md "M14.dispatch.1-beta design
 * notes" for the design rationale.
 *
 * This TU contains no HIP kernel code.  It calls existing extern-C launcher
 * symbols (`hipengine_qwen35_router_topk_shared_out_fp16`,
 * `hipengine_paro_rotate1_fp16`, …) via typed function pointers supplied by
 * the Python caller, in the same sequence Python's `run_moe_c1_fp16` would
 * have launched them.  Numerically identical to the unfused Python path.
 *
 * Two entry points cover the verifier's batched path:
 *
 *   hipengine_moe_c1_dispatch_linear_fp16(...)
 *     30/40 layers (linear_attention layer_type).  Shared expert uses the
 *     fused-W4 prefill kernels: paro_rotate2 + awq_fusedw4_prefill_dual +
 *     silu_mul_separate + paro_rotate1 + awq_fusedw4_prefill.
 *
 *   hipengine_moe_c1_dispatch_full_fp16(...)
 *     10/40 layers (full_attention layer_type at tokens ≤
 *     small_batch_decode_threshold).  Shared expert uses the small-batch GEMV
 *     path: paro_rotate2 + gemv_awq_dual_pack8_transposed +
 *     silu_mul_dual_rotate_out (fuses down rotate) + gemv_awq_pack8_transposed.
 *
 * Both entry points share the rest of the pipeline: router, selected gate-up
 * (paro_rotate1 + gemv_awq_selected_dual_pack8_transposed), activate-down
 * (silu_mul_dual_rotate_out), selected down (gemv_awq_selected_pack8_transposed),
 * and combine (weighted_sum_shared_gate_combine_residual_batch).
 *
 * Plain C++ in a .hip file — hipcc compiles it as straight host C++; no
 * device code.  Lives in a .hip file so the existing build_hip plumbing
 * picks it up unchanged.
 */

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Match `_check_launch` semantics: return non-zero error code = failure.
typedef int32_t hipengine_err_t;

// ---------- Function-pointer typedefs (must match Python wrappers) ----------

// router_topk_shared_out_fp16:
//   hidden, combined_weight, logits, selected, routing,
//   tokens, hidden_size, num_rows, num_experts, top_k, threads, stream
typedef hipengine_err_t (*router_topk_shared_out_fp16_fn)(
    void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// paro_rotate1_fp16:
//   x, out, pairs, theta, scales, tokens, hidden, group_size, krot, stream
typedef hipengine_err_t (*paro_rotate1_fp16_fn)(
    void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t,
    void*);

// paro_rotate2_fp16:
//   x, out0, out1, pairs0, pairs1, theta0, theta1, scales0, scales1,
//   tokens, hidden, group_size, krot, stream
typedef hipengine_err_t (*paro_rotate2_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_selected_dual_pack8_transposed_fp16:
//   x, selected, qw_a, qz_a, s_a, qw_b, qz_b, s_b, out,
//   x_rows, rows, in_features, out_packed_a, out_packed_b, num_experts,
//   group_size, threads, stream
typedef hipengine_err_t (*gemv_awq_selected_dual_pack8_transposed_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16:
//   x, rotated, selected, pairs, theta, channel_scales,
//   qw_a, qz_a, s_a, qw_b, qz_b, s_b, out, barrier,
//   x_rows, rows, in_features, out_packed_a, out_packed_b, num_experts,
//   group_size, krot, threads, barrier_count_target, barrier_ready_value, stream
typedef hipengine_err_t (*gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16_fn)(
    void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_selected_pack8_transposed_fp16 (strided, "selected_single"):
//   x, selected, qw, qz, s, out,
//   rows, in_features, out_packed, num_experts, group_size, threads, stream
typedef hipengine_err_t (*gemv_awq_selected_pack8_transposed_fp16_fn)(
    void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// silu_mul_dual_rotate_out_fp16:
//   gate_up, pairs, theta, scales, out,
//   rows, features, group_size, krot, stream
typedef hipengine_err_t (*silu_mul_dual_rotate_out_fp16_fn)(
    void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16:
//   gate_up, down_input, selected, pairs, theta, channel_scales, qw, qz, scales,
//   out, barrier, rows, in_features, out_packed, num_experts, group_size, krot,
//   threads, barrier_count_target, barrier_ready_value, stream
typedef hipengine_err_t (*gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*,
    void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// silu_mul_separate_out_fp16:
//   gate, up, out, rows, features, threads, stream
typedef hipengine_err_t (*silu_mul_separate_out_fp16_fn)(
    void*, void*, void*,
    int64_t, int64_t, int64_t,
    void*);

// silu_mul_pair_rotate_out_fp16:
//   gate, up, pairs, theta, scales, out, rows, features, group_size, krot, stream
typedef hipengine_err_t (*silu_mul_pair_rotate_out_fp16_fn)(
    void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t,
    void*);

// awq_fusedw4_prefill_dual_fp16:
//   x_a, x_b, qw_a, qz_a, s_a, qw_b, qz_b, s_b, out_a, out_b,
//   rows, in_features, out_packed_a, out_packed_b, group_size, tile_m, tile_n,
//   stream
typedef hipengine_err_t (*awq_fusedw4_prefill_dual_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_dual_pack8_output_tiled_split_transposed_fp16:
//   x_a, x_b, qw_a, qz_a, s_a, qw_b, qz_b, s_b, out_a, out_b,
//   rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream
typedef hipengine_err_t (*gemv_awq_dual_pack8_output_tiled_split_transposed_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// awq_fusedw4_prefill_fp16:
//   x, qw, qz, s, out,
//   rows, in_features, out_packed, group_size, tile_m, tile_n, stream
typedef hipengine_err_t (*awq_fusedw4_prefill_fp16_fn)(
    void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_pack8_transposed_fp16 ("pack8_single"):
//   x, qw, qz, s, out, rows, in_features, out_packed, group_size, threads,
//   stream
typedef hipengine_err_t (*gemv_awq_pack8_transposed_fp16_fn)(
    void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16:
//   x, qw, qz, s, selected, weights, gate_logits, residual, out,
//   rows, in_features, out_packed, group_size, rows_per_token, gate_stride,
//   threads, stream
typedef hipengine_err_t (*gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// gemv_awq_dual_pack8_transposed_fp16 (full-attn shared expert packed gate||up):
//   x_a, x_b, qw_a, qz_a, s_a, qw_b, qz_b, s_b, out,
//   rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream
typedef hipengine_err_t (*gemv_awq_dual_pack8_transposed_fp16_fn)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w:
//   expert, weights, shared, gate_logits, residual, out,
//   tokens, top_k, features, gate_stride, threads, stream
typedef hipengine_err_t (*combine_batch_fp16_fn)(
    void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    void*);

// ---------- Function table -----------------------------------------------

struct MoeC1Fns {
    void* router_topk_shared_out_fp16;
    void* paro_rotate1_fp16;
    void* paro_rotate2_fp16;
    void* gemv_awq_selected_dual_pack8_transposed_fp16;
    void* gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16;
    void* gemv_awq_selected_pack8_transposed_fp16;
    void* gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16;
    void* silu_mul_dual_rotate_out_fp16;
    void* silu_mul_separate_out_fp16;
    void* silu_mul_pair_rotate_out_fp16; // linear-attn reduced-DAG default
    void* awq_fusedw4_prefill_dual_fp16;       // linear-attn only
    void* gemv_awq_dual_pack8_output_tiled_split_transposed_fp16; // linear-attn opt-in
    void* awq_fusedw4_prefill_fp16;            // linear-attn only
    void* gemv_awq_pack8_multi_row_transposed_fp16;        // linear-attn shared down opt-in
    void* gemv_awq_pack8_multi_row_decode_transposed_fp16; // linear-attn shared down opt-in
    void* gemv_awq_pack8_output_tiled_transposed_fp16;     // linear-attn shared down opt-in
    void* gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16; // linear-attn reduced-DAG opt-in
    void* gemv_awq_pack8_output_tiled_combine_residual_full_transposed_fp16; // full-attn reduced-DAG default
    void* gemv_awq_dual_pack8_transposed_fp16; // full-attn only
    void* gemv_awq_pack8_transposed_fp16;      // full-attn only
    void* combine_batch_fp16;
};

// ---------- Args struct: per-layer constants + per-call variables ---------

struct MoeC1Args {
    // ---- Per-call variables (refilled each invocation) ----
    void* hidden;
    void* residual;
    void* out;
    int64_t tokens;
    void* stream;
    int64_t selected_rotate_barrier_target;
    int64_t selected_rotate_barrier_epoch;
    int64_t selected_down_barrier_target;
    int64_t selected_down_barrier_epoch;

    // ---- Layer-constant scratch pointers ----
    void* router_logits;
    void* selected_experts;
    void* routing_weights;
    void* gate_up_input;       // selected gate-up rotated input
    void* selected_rotate_fuse_barrier;
    void* gate_up;             // selected dual gate*up packed output
    void* down_input;          // after silu+rotate
    void* down_out;            // selected down output
    void* shared_gate_input;   // shared gate rotated input
    void* shared_up_input;     // shared up rotated input
    void* shared_gate_out;     // (linear only) shared gate output
    void* shared_up_out;       // (linear only) shared up output
    void* shared_up_packed;    // (full only) packed gate||up after dual GEMV
    void* shared_intermediate; // (linear only) after silu_mul_separate
    void* shared_down_input;   // after shared silu/rotate
    void* shared_out;          // final shared expert output

    // shared_gate_logits = router_logits + num_experts*sizeof(float)
    // computed inside the dispatcher

    // ---- Layer-constant weight pointers ----
    void* router_combined_weight;
    // Selected gate-up (per-expert stacked)
    void* selected_gate_up_pairs;
    void* selected_gate_up_theta;
    void* selected_gate_up_channel_scales;
    void* selected_stacked_gate_qweight;
    void* selected_stacked_gate_qzeros;
    void* selected_stacked_gate_scales;
    void* selected_stacked_up_qweight;
    void* selected_stacked_up_qzeros;
    void* selected_stacked_up_scales;
    // Selected down (per-expert stacked)
    void* selected_down_pairs;
    void* selected_down_theta;
    void* selected_down_channel_scales;
    void* selected_stacked_down_qweight;
    void* selected_stacked_down_qzeros;
    void* selected_stacked_down_scales;
    // Shared gate
    void* shared_gate_pairs;
    void* shared_gate_theta;
    void* shared_gate_channel_scales;
    void* shared_gate_qweight;
    void* shared_gate_qzeros;
    void* shared_gate_scales;
    // Shared up
    void* shared_up_pairs;
    void* shared_up_theta;
    void* shared_up_channel_scales;
    void* shared_up_qweight;
    void* shared_up_qzeros;
    void* shared_up_scales;
    // Shared down
    void* shared_down_pairs;
    void* shared_down_theta;
    void* shared_down_channel_scales;
    void* shared_down_qweight;
    void* shared_down_qzeros;
    void* shared_down_scales;

    // ---- Layer-constant dims (cached at warmup) ----
    int64_t hidden_size;
    int64_t num_experts;
    int64_t num_experts_per_tok;
    int64_t moe_intermediate_size;
    int64_t shared_expert_intermediate_size;
    int64_t group_size;
    int64_t krot_selected_gate_up;
    int64_t krot_selected_down;
    int64_t krot_shared_gate;
    int64_t krot_shared_up;
    int64_t krot_shared_down;
    int64_t out_packed_selected_gate;
    int64_t out_packed_selected_up;
    int64_t out_packed_selected_down;
    int64_t out_packed_shared_gate;
    int64_t out_packed_shared_up;
    int64_t out_packed_shared_down;

    // ---- Threads / tile params ----
    int64_t router_threads;
    int64_t selected_threads;
    int64_t shared_threads;
    int64_t shared_prefill_tile_m;  // linear-attn only
    int64_t shared_prefill_tile_n;  // linear-attn only
    int64_t shared_down_mode;       // linear-attn: 0=prefill, 1=multi-row decode, 2=output-tiled auto, 3=gemv, 4=multi-row
    int64_t combine_threads;
};

// ---------- Helpers -------------------------------------------------------

#define DISPATCH_CHECK(call_) do { \
    hipengine_err_t _err = (call_); \
    if (_err != 0) return _err; \
} while (0)

// Compute shared_gate_logits pointer: router_logits + num_experts * sizeof(float)
static inline void* shared_gate_logits_ptr(const MoeC1Args* args) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uint8_t*>(args->router_logits) +
        static_cast<size_t>(args->num_experts) * sizeof(float));
}

static inline hipengine_err_t dispatch_selected_gate_up(
    const MoeC1Fns* fns,
    const MoeC1Args* args,
    int64_t selected_rows)
{
    if (args->selected_rotate_barrier_target > 0 && args->selected_rotate_barrier_epoch > 0) {
        auto fn = reinterpret_cast<gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16_fn>(
            fns->gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->gate_up_input, args->selected_experts,
            args->selected_gate_up_pairs, args->selected_gate_up_theta,
            args->selected_gate_up_channel_scales,
            args->selected_stacked_gate_qweight,
            args->selected_stacked_gate_qzeros,
            args->selected_stacked_gate_scales,
            args->selected_stacked_up_qweight,
            args->selected_stacked_up_qzeros,
            args->selected_stacked_up_scales,
            args->gate_up, args->selected_rotate_fuse_barrier,
            args->tokens, selected_rows, args->hidden_size,
            args->out_packed_selected_gate, args->out_packed_selected_up,
            args->num_experts, args->group_size, args->krot_selected_gate_up,
            args->selected_threads,
            args->selected_rotate_barrier_target,
            args->selected_rotate_barrier_epoch,
            args->stream));
        return 0;
    }

    {
        auto fn = reinterpret_cast<paro_rotate1_fp16_fn>(fns->paro_rotate1_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->gate_up_input,
            args->selected_gate_up_pairs, args->selected_gate_up_theta,
            args->selected_gate_up_channel_scales,
            args->tokens, args->hidden_size, args->group_size,
            args->krot_selected_gate_up,
            args->stream));
    }
    {
        auto fn = reinterpret_cast<gemv_awq_selected_dual_pack8_transposed_fp16_fn>(
            fns->gemv_awq_selected_dual_pack8_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->gate_up_input, args->selected_experts,
            args->selected_stacked_gate_qweight,
            args->selected_stacked_gate_qzeros,
            args->selected_stacked_gate_scales,
            args->selected_stacked_up_qweight,
            args->selected_stacked_up_qzeros,
            args->selected_stacked_up_scales,
            args->gate_up,
            args->tokens, selected_rows, args->hidden_size,
            args->out_packed_selected_gate, args->out_packed_selected_up,
            args->num_experts, args->group_size, args->selected_threads,
            args->stream));
    }
    return 0;
}

static inline hipengine_err_t dispatch_selected_down(
    const MoeC1Fns* fns,
    const MoeC1Args* args,
    int64_t selected_rows)
{
    if (args->selected_down_barrier_target > 0 && args->selected_down_barrier_epoch > 0) {
        auto fn = reinterpret_cast<gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16_fn>(
            fns->gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16);
        DISPATCH_CHECK(fn(
            args->gate_up, args->down_input, args->selected_experts,
            args->selected_down_pairs, args->selected_down_theta,
            args->selected_down_channel_scales,
            args->selected_stacked_down_qweight,
            args->selected_stacked_down_qzeros,
            args->selected_stacked_down_scales,
            args->down_out, args->selected_rotate_fuse_barrier,
            selected_rows, args->moe_intermediate_size,
            args->out_packed_selected_down,
            args->num_experts, args->group_size, args->krot_selected_down,
            args->selected_threads,
            args->selected_down_barrier_target,
            args->selected_down_barrier_epoch,
            args->stream));
        return 0;
    }

    {
        auto fn = reinterpret_cast<silu_mul_dual_rotate_out_fp16_fn>(
            fns->silu_mul_dual_rotate_out_fp16);
        DISPATCH_CHECK(fn(
            args->gate_up,
            args->selected_down_pairs, args->selected_down_theta,
            args->selected_down_channel_scales,
            args->down_input,
            selected_rows, args->moe_intermediate_size, args->group_size,
            args->krot_selected_down,
            args->stream));
    }
    {
        auto fn = reinterpret_cast<gemv_awq_selected_pack8_transposed_fp16_fn>(
            fns->gemv_awq_selected_pack8_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->down_input, args->selected_experts,
            args->selected_stacked_down_qweight,
            args->selected_stacked_down_qzeros,
            args->selected_stacked_down_scales,
            args->down_out,
            selected_rows, args->moe_intermediate_size,
            args->out_packed_selected_down,
            args->num_experts, args->group_size, args->selected_threads,
            args->stream));
    }
    return 0;
}

// ---------- Linear-attention dispatcher -----------------------------------

hipengine_err_t hipengine_moe_c1_dispatch_linear_fp16(
    const MoeC1Fns* fns,
    const MoeC1Args* args)
{
    if (!fns || !args) return -1;

    const int64_t selected_rows = args->tokens * args->num_experts_per_tok;

    // ---- 1. Router: hidden -> logits, selected, routing ----
    {
        auto fn = reinterpret_cast<router_topk_shared_out_fp16_fn>(
            fns->router_topk_shared_out_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->router_combined_weight,
            args->router_logits, args->selected_experts, args->routing_weights,
            args->tokens, args->hidden_size,
            args->num_experts + 1,                  // num_rows = num_experts + 1
            args->num_experts, args->num_experts_per_tok,
            args->router_threads,
            args->stream));
    }

    // ---- 2-3. Selected gate-up: optional staged rotate + selected dual GEMV ----
    DISPATCH_CHECK(dispatch_selected_gate_up(fns, args, selected_rows));

    // ---- 4-5. Selected down: optional staged SiLU/rotate + selected down GEMV ----
    DISPATCH_CHECK(dispatch_selected_down(fns, args, selected_rows));

    // ---- 6. Shared gate+up rotate: hidden -> shared_gate_input, shared_up_input ----
    // Linear path assumes gate_krot == up_krot (rotate2).  If krots differ, the
    // Python fallback handles two rotate1s; the dispatcher path doesn't.
    {
        auto fn = reinterpret_cast<paro_rotate2_fp16_fn>(fns->paro_rotate2_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->shared_gate_input, args->shared_up_input,
            args->shared_gate_pairs, args->shared_up_pairs,
            args->shared_gate_theta, args->shared_up_theta,
            args->shared_gate_channel_scales, args->shared_up_channel_scales,
            args->tokens, args->hidden_size, args->group_size,
            args->krot_shared_gate,
            args->stream));
    }

    // ---- 7a. Shared gate+up W4 prefill (separate outputs, linear-attn) ----
    {
        const bool output_tiled_rows =
            args->tokens == 2 || args->tokens == 4 || args->tokens == 8;
        if (fns->gemv_awq_dual_pack8_output_tiled_split_transposed_fp16 != nullptr &&
            output_tiled_rows) {
            auto fn = reinterpret_cast<gemv_awq_dual_pack8_output_tiled_split_transposed_fp16_fn>(
                fns->gemv_awq_dual_pack8_output_tiled_split_transposed_fp16);
            DISPATCH_CHECK(fn(
                args->shared_gate_input, args->shared_up_input,
                args->shared_gate_qweight, args->shared_gate_qzeros,
                args->shared_gate_scales,
                args->shared_up_qweight, args->shared_up_qzeros,
                args->shared_up_scales,
                args->shared_gate_out, args->shared_up_out,
                args->tokens, args->hidden_size,
                args->out_packed_shared_gate, args->out_packed_shared_up,
                args->group_size,
                128,
                args->stream));
        } else {
            auto fn = reinterpret_cast<awq_fusedw4_prefill_dual_fp16_fn>(
                fns->awq_fusedw4_prefill_dual_fp16);
            DISPATCH_CHECK(fn(
                args->shared_gate_input, args->shared_up_input,
                args->shared_gate_qweight, args->shared_gate_qzeros,
                args->shared_gate_scales,
                args->shared_up_qweight, args->shared_up_qzeros,
                args->shared_up_scales,
                args->shared_gate_out, args->shared_up_out,
                args->tokens, args->hidden_size,
                args->out_packed_shared_gate, args->out_packed_shared_up,
                args->group_size,
                args->shared_prefill_tile_m, args->shared_prefill_tile_n,
                args->stream));
        }
    }

    // ---- 8a-9. Shared silu/mul + optional down rotate fusion ----
    if (fns->silu_mul_pair_rotate_out_fp16 != nullptr) {
        auto fn = reinterpret_cast<silu_mul_pair_rotate_out_fp16_fn>(
            fns->silu_mul_pair_rotate_out_fp16);
        DISPATCH_CHECK(fn(
            args->shared_gate_out, args->shared_up_out,
            args->shared_down_pairs, args->shared_down_theta,
            args->shared_down_channel_scales,
            args->shared_down_input,
            args->tokens, args->shared_expert_intermediate_size,
            args->group_size, args->krot_shared_down,
            args->stream));
    } else {
        auto fn = reinterpret_cast<silu_mul_separate_out_fp16_fn>(
            fns->silu_mul_separate_out_fp16);
        // silu_mul_separate uses a default `threads=256` in the Python wrapper;
        // expose via shared_threads here for cache-locality.  256 matches the
        // default; pass through args->combine_threads (also 256).
        DISPATCH_CHECK(fn(
            args->shared_gate_out, args->shared_up_out,
            args->shared_intermediate,
            args->tokens, args->shared_expert_intermediate_size,
            args->combine_threads,  // 256
            args->stream));

        // Shared down rotate (linear-attn only): intermediate -> shared_down_input.
        auto rotate_fn = reinterpret_cast<paro_rotate1_fp16_fn>(fns->paro_rotate1_fp16);
        DISPATCH_CHECK(rotate_fn(
            args->shared_intermediate, args->shared_down_input,
            args->shared_down_pairs, args->shared_down_theta,
            args->shared_down_channel_scales,
            args->tokens, args->shared_expert_intermediate_size,
            args->group_size, args->krot_shared_down,
            args->stream));
    }

    // ---- 10a. Shared down W4: shared_down_input -> shared_out ----
    const bool output_tiled_rows =
        args->tokens == 2 || args->tokens == 4 || args->tokens == 8;
    bool combined_in_shared_down = false;
    if (args->shared_down_mode == 2 && output_tiled_rows &&
        fns->gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16) {
        auto fn = reinterpret_cast<gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16_fn>(
            fns->gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->down_out, args->routing_weights,
            shared_gate_logits_ptr(args),
            args->residual, args->out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->num_experts_per_tok, args->num_experts + 1,
            args->shared_threads,
            args->stream));
        combined_in_shared_down = true;
    } else if (args->shared_down_mode == 2 && output_tiled_rows) {
        auto fn = reinterpret_cast<gemv_awq_pack8_transposed_fp16_fn>(
            fns->gemv_awq_pack8_output_tiled_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_threads,
            args->stream));
    } else if (args->shared_down_mode == 1 || args->shared_down_mode == 2) {
        auto fn = reinterpret_cast<gemv_awq_pack8_transposed_fp16_fn>(
            fns->gemv_awq_pack8_multi_row_decode_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_threads,
            args->stream));
    } else if (args->shared_down_mode == 3) {
        auto fn = reinterpret_cast<gemv_awq_pack8_transposed_fp16_fn>(
            fns->gemv_awq_pack8_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_threads,
            args->stream));
    } else if (args->shared_down_mode == 4) {
        auto fn = reinterpret_cast<gemv_awq_pack8_transposed_fp16_fn>(
            fns->gemv_awq_pack8_multi_row_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_threads,
            args->stream));
    } else {
        auto fn = reinterpret_cast<awq_fusedw4_prefill_fp16_fn>(
            fns->awq_fusedw4_prefill_fp16);
        // Linear-attn shared down prefill: tile_m = same as gate-up,
        // tile_n = 32 if rows>=32 else 16 — but for verifier tokens=4
        // both are 16.  Reuse the gate-up tile choice.
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_prefill_tile_m, args->shared_prefill_tile_n,
            args->stream));
    }

    // ---- 11. Combine batch: down_out + routing + shared + residual -> out ----
    if (!combined_in_shared_down) {
        auto fn = reinterpret_cast<combine_batch_fp16_fn>(fns->combine_batch_fp16);
        DISPATCH_CHECK(fn(
            args->down_out, args->routing_weights,
            args->shared_out, shared_gate_logits_ptr(args),
            args->residual, args->out,
            args->tokens, args->num_experts_per_tok, args->hidden_size,
            args->num_experts + 1,                  // gate_stride
            args->combine_threads,
            args->stream));
    }

    return 0;
}

// ---------- Full-attention dispatcher -------------------------------------

hipengine_err_t hipengine_moe_c1_dispatch_full_fp16(
    const MoeC1Fns* fns,
    const MoeC1Args* args)
{
    if (!fns || !args) return -1;

    const int64_t selected_rows = args->tokens * args->num_experts_per_tok;

    // ---- 1. Router (same as linear) ----
    {
        auto fn = reinterpret_cast<router_topk_shared_out_fp16_fn>(
            fns->router_topk_shared_out_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->router_combined_weight,
            args->router_logits, args->selected_experts, args->routing_weights,
            args->tokens, args->hidden_size,
            args->num_experts + 1,
            args->num_experts, args->num_experts_per_tok,
            args->router_threads,
            args->stream));
    }

    // ---- 2-3. Selected gate-up: optional staged rotate + selected dual GEMV ----
    DISPATCH_CHECK(dispatch_selected_gate_up(fns, args, selected_rows));

    // ---- 4-5. Selected down: optional staged SiLU/rotate + selected down GEMV ----
    DISPATCH_CHECK(dispatch_selected_down(fns, args, selected_rows));

    // ---- 6. Shared gate+up rotate (same as linear) ----
    {
        auto fn = reinterpret_cast<paro_rotate2_fp16_fn>(fns->paro_rotate2_fp16);
        DISPATCH_CHECK(fn(
            args->hidden, args->shared_gate_input, args->shared_up_input,
            args->shared_gate_pairs, args->shared_up_pairs,
            args->shared_gate_theta, args->shared_up_theta,
            args->shared_gate_channel_scales, args->shared_up_channel_scales,
            args->tokens, args->hidden_size, args->group_size,
            args->krot_shared_gate,
            args->stream));
    }

    // ---- 7b. Shared gate+up small-batch dual GEMV (packed gate||up) ----
    {
        auto fn = reinterpret_cast<gemv_awq_dual_pack8_transposed_fp16_fn>(
            fns->gemv_awq_dual_pack8_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_gate_input, args->shared_up_input,
            args->shared_gate_qweight, args->shared_gate_qzeros,
            args->shared_gate_scales,
            args->shared_up_qweight, args->shared_up_qzeros,
            args->shared_up_scales,
            args->shared_up_packed,
            args->tokens, args->hidden_size,
            args->out_packed_shared_gate, args->out_packed_shared_up,
            args->group_size, args->shared_threads,
            args->stream));
    }

    // ---- 8b. Shared silu_mul_dual_rotate (also handles down rotate inline) ----
    {
        auto fn = reinterpret_cast<silu_mul_dual_rotate_out_fp16_fn>(
            fns->silu_mul_dual_rotate_out_fp16);
        DISPATCH_CHECK(fn(
            args->shared_up_packed,
            args->shared_down_pairs, args->shared_down_theta,
            args->shared_down_channel_scales,
            args->shared_down_input,
            args->tokens, args->shared_expert_intermediate_size,
            args->group_size, args->krot_shared_down,
            args->stream));
    }

    // ---- 10b. Shared down GEMV (small-batch path) ----
    bool combined_in_shared_down = false;
    const bool output_tiled_rows =
        args->tokens == 2 || args->tokens == 4 || args->tokens == 8;
    if (output_tiled_rows &&
        fns->gemv_awq_pack8_output_tiled_combine_residual_full_transposed_fp16) {
        auto fn = reinterpret_cast<gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16_fn>(
            fns->gemv_awq_pack8_output_tiled_combine_residual_full_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->down_out, args->routing_weights,
            shared_gate_logits_ptr(args),
            args->residual, args->out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->num_experts_per_tok, args->num_experts + 1,
            args->shared_threads,
            args->stream));
        combined_in_shared_down = true;
    } else {
        auto fn = reinterpret_cast<gemv_awq_pack8_transposed_fp16_fn>(
            fns->gemv_awq_pack8_transposed_fp16);
        DISPATCH_CHECK(fn(
            args->shared_down_input,
            args->shared_down_qweight, args->shared_down_qzeros,
            args->shared_down_scales,
            args->shared_out,
            args->tokens, args->shared_expert_intermediate_size,
            args->out_packed_shared_down, args->group_size,
            args->shared_threads,
            args->stream));
    }

    // ---- 11. Combine batch (same as linear) ----
    if (!combined_in_shared_down) {
        auto fn = reinterpret_cast<combine_batch_fp16_fn>(fns->combine_batch_fp16);
        DISPATCH_CHECK(fn(
            args->down_out, args->routing_weights,
            args->shared_out, shared_gate_logits_ptr(args),
            args->residual, args->out,
            args->tokens, args->num_experts_per_tok, args->hidden_size,
            args->num_experts + 1,
            args->combine_threads,
            args->stream));
    }

    return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif
