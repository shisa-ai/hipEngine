// Grouped/selected MoE decode-shaped pack8 GEMV for raw GGUF Q4_K experts.
//
// P9.B1 implementation note: this file mirrors the structure of
// ``paro_awq_gemv.hip::gemv_awq_selected_dual_pack8_strided_kernel`` (PARO-
// style pack8 row layout, ``__launch_bounds__(128, 4)``, register-resident
// dequant, wave-level reduction across 4 wave32 waves) but consumes the
// compact-MoE ABI introduced in P8.4 instead of the AWQ
// ``selected[row] -> expert_id`` mapping:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   qweight_a            [E, out_features_a, row_bytes]   raw block_q4_K bytes
//   qweight_b            [E, out_features_b, row_bytes]   raw block_q4_K bytes
//   out                  [compact_rows, out_features_a + out_features_b]
//
// The block determines its expert via a linear scan over
// ``expert_start_compact`` (``num_experts`` is small in practice, e.g. 32 for
// qwen35moe; no new ABI is introduced for decode dispatch). The inner K-loop
// swaps AWQ pack8 dequant for raw GGUF ``block_q4_K`` decoding (reusing the
// helpers from ``gguf_q4_k_gemv.hip`` / ``gguf_q4_k_selected_prefill.hip``).
// Per-block ``d``/``dmin``/scale/min are hoisted into shared memory once per
// 256-element Q4_K block so the inner k loop stays in registers.
//
// No new compact-MoE ABI and no resident weight sidecar/repack are introduced.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + 12;  // d + dmin + 12 scale bytes
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int PACK8 = 8;

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

__device__ inline uint8_t gguf_q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_quant(
    const uint8_t* qs,
    int subblock,
    int lane32) {
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * Q4_K_SUBBLOCK + lane32];
  return (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

#define GGUF_Q4K_GEMV_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
  for (int offset = 16; offset > 0; offset >>= 1) { \
    acc0 += __shfl_down(acc0, offset); \
    acc1 += __shfl_down(acc1, offset); \
    acc2 += __shfl_down(acc2, offset); \
    acc3 += __shfl_down(acc3, offset); \
    acc4 += __shfl_down(acc4, offset); \
    acc5 += __shfl_down(acc5, offset); \
    acc6 += __shfl_down(acc6, offset); \
    acc7 += __shfl_down(acc7, offset); \
  } \
  __shared__ float xchg[8 * 8]; \
  const int sub_id = tid >> 5; \
  const int sub_lane = tid & 31; \
  if (sub_lane == 0) { \
    xchg[sub_id * 8 + 0] = acc0; \
    xchg[sub_id * 8 + 1] = acc1; \
    xchg[sub_id * 8 + 2] = acc2; \
    xchg[sub_id * 8 + 3] = acc3; \
    xchg[sub_id * 8 + 4] = acc4; \
    xchg[sub_id * 8 + 5] = acc5; \
    xchg[sub_id * 8 + 6] = acc6; \
    xchg[sub_id * 8 + 7] = acc7; \
  } \
  __syncthreads(); \
  float r0 = 0.0f; \
  float r1 = 0.0f; \
  float r2 = 0.0f; \
  float r3 = 0.0f; \
  float r4 = 0.0f; \
  float r5 = 0.0f; \
  float r6 = 0.0f; \
  float r7 = 0.0f; \
  if (tid == 0) { \
    const int sub_groups = blockDim.x >> 5; \
    for (int g = 0; g < sub_groups; ++g) { \
      r0 += xchg[g * 8 + 0]; \
      r1 += xchg[g * 8 + 1]; \
      r2 += xchg[g * 8 + 2]; \
      r3 += xchg[g * 8 + 3]; \
      r4 += xchg[g * 8 + 4]; \
      r5 += xchg[g * 8 + 5]; \
      r6 += xchg[g * 8 + 6]; \
      r7 += xchg[g * 8 + 7]; \
    } \
  }

// Decode-shaped pack8 GEMV for raw GGUF Q4_K selected dual gate+up.
//
// Grid:
//   blockIdx.x = global output pack in [0, out_packed_a + out_packed_b)
//   blockIdx.y = compact_row in [0, compact_rows)
// Block:
//   blockDim.x = 128 (4 wave32 waves; __launch_bounds__(128, 4))
//
// Output layout matches the P8.4 compact WMMA prefill: row-major
// ``[compact_rows, out_features_a + out_features_b]`` with gate occupying the
// first ``out_features_a`` columns and up occupying the next
// ``out_features_b`` columns. The block-y index is the compact row directly;
// expert id is recovered via a linear scan over ``expert_start_compact``.
template <typename scalar_t>
__launch_bounds__(128, 4) __global__ void gguf_q4_k_selected_dual_pack8_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int64_t global_pack = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  // Linear scan to find this row's expert. ``num_experts`` is small (~32 for
  // qwen35moe) and ``expert_start_compact`` is hot in L1 across the grid.
  int64_t expert_id = -1;
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      expert_id = e;
      break;
    }
  }
  if (expert_id < 0) return;

  const int64_t out_packed_a = out_features_a >> 3;
  const int64_t out_packed_b = out_features_b >> 3;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t local_out_col = local_pack << 3;
  const int64_t out_col = global_pack << 3;
  if (out_col >= out_features_total) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;

  // Eight per-output-channel row pointers (one Q4_K row per output column in
  // this pack). At decode shapes the per-block work is small; precomputing
  // these once amortises across the K loop.
  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = expert_base + (local_out_col + ch) * row_bytes;
  }

  const scalar_t* __restrict__ x_row_ptr = x + compact_row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  // Per-block hoist of (d * scales[sb], dmin * mins[sb]) for all 8 output
  // channels x 8 subblocks. 64 floats each, cooperatively loaded by the first
  // 64 threads of the block.
  __shared__ float s_scale[PACK8 * Q4_K_SUBBLOCKS];  // [ch, sb] = d * scales(sb)
  __shared__ float s_min[PACK8 * Q4_K_SUBBLOCKS];    // [ch, sb] = dmin * mins(sb)

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < PACK8 * Q4_K_SUBBLOCKS) {
      const int ch = tid >> 3;
      const int sb = tid & 7;
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block);
      const float dmin_f = fp16_bytes_to_float(block + 2);
      const uint8_t* __restrict__ scales = block + 4;
      s_scale[ch * Q4_K_SUBBLOCKS + sb] =
          d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
      s_min[ch * Q4_K_SUBBLOCKS + sb] =
          dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
    }
    __syncthreads();

    // Inner k loop within this 256-element Q4_K block.
    const int64_t k_block_base = blk * QK_K;
    const uint8_t* __restrict__ qs0 = row_ptr[0] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs1 = row_ptr[1] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs2 = row_ptr[2] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs3 = row_ptr[3] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs4 = row_ptr[4] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs5 = row_ptr[5] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs6 = row_ptr[6] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs7 = row_ptr[7] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row_ptr[k_block_base + k_in_block]);
      const float q0 = static_cast<float>(gguf_q4_k_quant(qs0, sb, lane32));
      const float q1 = static_cast<float>(gguf_q4_k_quant(qs1, sb, lane32));
      const float q2 = static_cast<float>(gguf_q4_k_quant(qs2, sb, lane32));
      const float q3 = static_cast<float>(gguf_q4_k_quant(qs3, sb, lane32));
      const float q4 = static_cast<float>(gguf_q4_k_quant(qs4, sb, lane32));
      const float q5 = static_cast<float>(gguf_q4_k_quant(qs5, sb, lane32));
      const float q6 = static_cast<float>(gguf_q4_k_quant(qs6, sb, lane32));
      const float q7 = static_cast<float>(gguf_q4_k_quant(qs7, sb, lane32));
      acc0 = fmaf(xv, s_scale[0 * Q4_K_SUBBLOCKS + sb] * q0 - s_min[0 * Q4_K_SUBBLOCKS + sb], acc0);
      acc1 = fmaf(xv, s_scale[1 * Q4_K_SUBBLOCKS + sb] * q1 - s_min[1 * Q4_K_SUBBLOCKS + sb], acc1);
      acc2 = fmaf(xv, s_scale[2 * Q4_K_SUBBLOCKS + sb] * q2 - s_min[2 * Q4_K_SUBBLOCKS + sb], acc2);
      acc3 = fmaf(xv, s_scale[3 * Q4_K_SUBBLOCKS + sb] * q3 - s_min[3 * Q4_K_SUBBLOCKS + sb], acc3);
      acc4 = fmaf(xv, s_scale[4 * Q4_K_SUBBLOCKS + sb] * q4 - s_min[4 * Q4_K_SUBBLOCKS + sb], acc4);
      acc5 = fmaf(xv, s_scale[5 * Q4_K_SUBBLOCKS + sb] * q5 - s_min[5 * Q4_K_SUBBLOCKS + sb], acc5);
      acc6 = fmaf(xv, s_scale[6 * Q4_K_SUBBLOCKS + sb] * q6 - s_min[6 * Q4_K_SUBBLOCKS + sb], acc6);
      acc7 = fmaf(xv, s_scale[7 * Q4_K_SUBBLOCKS + sb] * q7 - s_min[7 * Q4_K_SUBBLOCKS + sb], acc7);
    }
    __syncthreads();  // protect s_scale/s_min before the next block reloads them
  }

  GGUF_Q4K_GEMV_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                              r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* __restrict__ out_ptr = out + compact_row * out_features_total + out_col;
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

bool valid_selected_dual_pack8_gemv(
    const void* x,
    const void* expert_start_compact,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  return x != nullptr && expert_start_compact != nullptr && qweight_a != nullptr &&
      qweight_b != nullptr && out != nullptr && compact_rows > 0 && in_features > 0 &&
      out_features_a > 0 && out_features_b > 0 && num_experts > 0 &&
      (out_features_a & 7) == 0 && (out_features_b & 7) == 0 &&
      (in_features % QK_K) == 0;
}

template <typename scalar_t>
int launch_selected_dual_pack8_gemv(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    hipStream_t stream) {
  if (!valid_selected_dual_pack8_gemv(
          x,
          expert_start_compact,
          qweight_a,
          qweight_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed_total = (out_features_a + out_features_b) >> 3;
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_pack8_gemv_kernel<scalar_t>),
      dim3(
          static_cast<unsigned int>(out_packed_total),
          static_cast<unsigned int>(compact_rows)),
      dim3(128),
      0,
      stream,
      x,
      expert_start_compact,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_dual_pack8_gemv<uint16_t>(
      x,
      expert_start_compact,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out(
    const _Float16* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    _Float16* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_dual_pack8_gemv<_Float16>(
      x,
      expert_start_compact,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      stream);
}
