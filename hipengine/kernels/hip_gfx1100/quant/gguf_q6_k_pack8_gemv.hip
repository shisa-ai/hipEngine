// Dense decode-shaped pack8 GEMV for raw GGUF Q6_K weights.
//
// P9.B4b implementation note (kernel added during task #24 to enable the
// Q6_K F32 lm-head correctness fixture): mirrors the structure of
// ``gguf_q4_k_pack8_gemv.hip`` (single-output PARO-style pack8 GEMV) with
// the inner k loop swapped for raw GGUF Q6_K block dequant (int8 per-16-K
// scales + fp16 super-scale + 2 high bits per element).
//
//   x          [rows, in_features]
//   qweight    [out_features, row_bytes]   raw block_q6_K bytes
//   out        [rows, out_features]
//
// Four ``(scalar_in_t, scalar_out_t)`` combinations are supported: BF16/BF16,
// FP16/FP16, BF16/F32, FP16/F32. The F32 output variants are the lm-head
// logits projection case for qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M (the tied
// output weight is stored as Q6_K).
//
// No new ABI and no resident weight sidecar/repack: raw GGUF Q6_K bytes
// stay on device.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q6_K_GROUPS16 = QK_K / 16;  // == 16
constexpr int Q8_1_BLOCK = 32;
constexpr int PACK8 = 8;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_1_BLOCK];
};

static_assert(sizeof(gguf_q8_1_block) == 36, "GGUF q8_1 block layout must match GGML");

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

template <>
__device__ inline float float_to_scalar<float>(float v) {
  return v;
}

#define GGUF_Q6K_PACK8_WAVE_REDUCE(acc)               \
  do {                                                 \
    acc += __shfl_down(acc, 16);                       \
    acc += __shfl_down(acc, 8);                        \
    acc += __shfl_down(acc, 4);                        \
    acc += __shfl_down(acc, 2);                        \
    acc += __shfl_down(acc, 1);                        \
  } while (0)

__device__ inline bool better_pair_i32(float value, int32_t index, float best_value, int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__device__ inline int q6_pack8_dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  const int8_t b0 = static_cast<int8_t>(b & 0xFFu);
  const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFFu);
  const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFFu);
  const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFFu);
  return c + static_cast<int>(a & 0xFFu) * static_cast<int>(b0) +
         static_cast<int>((a >> 8) & 0xFFu) * static_cast<int>(b1) +
         static_cast<int>((a >> 16) & 0xFFu) * static_cast<int>(b2) +
         static_cast<int>((a >> 24) & 0xFFu) * static_cast<int>(b3);
#endif
}

__device__ inline uint32_t q6_k_pack4_unsigned(const uint8_t* block, int group32, int lane4) {
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const int base64 = (group32 >= 4) ? 64 : 0;
  const bool low_nibble = (group32 & 2) == 0;
  const int ql_group = group32 & 1;
  uint32_t low = *reinterpret_cast<const uint32_t*>(ql + base64 + ql_group * 32 + lane4);
  low = low_nibble ? (low & 0x0F0F0F0Fu) : ((low >> 4) & 0x0F0F0F0Fu);
  const int qh_base = (group32 >= 4) ? 32 : 0;
  const uint32_t high_bits = *reinterpret_cast<const uint32_t*>(qh + qh_base + lane4);
  const uint32_t high = ((high_bits >> (2 * (group32 & 3))) & 0x03030303u) << 4;
  return low | high;
}

__device__ inline float q6_k_q8_1_dp4a_term(
    const uint8_t* block,
    int group32,
    int lane4,
    int scale_index,
    uint32_t x_pack,
    int q8_sum,
    float xd) {
  const int dot_u = q6_pack8_dp4a_u8_s8(q6_k_pack4_unsigned(block, group32, lane4), x_pack, 0);
  const int dot = dot_u - 32 * q8_sum;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = fp16_bytes_to_float(block + 208);
  return xd * d * static_cast<float>(scales[scale_index]) * static_cast<float>(dot);
}

// Dense Q6_K pack8 GEMV decode.
//
// Grid: (out_features / 8, rows). One block per (output pack of 8, row).
// Block: 128 threads (4 wave32 waves; __launch_bounds__(128, 4)).
template <typename scalar_in_t, typename scalar_out_t>
__launch_bounds__(128, 4) __global__ void gguf_q6_k_pack8_gemv_kernel(
    const scalar_in_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    scalar_out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }

  const scalar_in_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  // Per-block hoist of (d * scales[g16]) for all 8 output channels x 16
  // groups of 16. 128 floats, cooperatively loaded by the first 128 threads.
  __shared__ float s_scale[PACK8 * Q6_K_GROUPS16];

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < PACK8 * Q6_K_GROUPS16) {
      const int ch = tid >> 4;
      const int g16 = tid & 15;
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q6_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block + 208);
      const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(block + 192);
      s_scale[ch * Q6_K_GROUPS16 + g16] = d_f * static_cast<float>(scales[g16]);
    }
    __syncthreads();

    const int64_t k_block_base = blk * QK_K;
    // Per-channel ql (128 bytes) and qh (64 bytes) pointers.
    const uint8_t* __restrict__ ql0 = row_ptr[0] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql1 = row_ptr[1] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql2 = row_ptr[2] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql3 = row_ptr[3] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql4 = row_ptr[4] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql5 = row_ptr[5] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql6 = row_ptr[6] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql7 = row_ptr[7] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ qhp0 = ql0 + 128;
    const uint8_t* __restrict__ qhp1 = ql1 + 128;
    const uint8_t* __restrict__ qhp2 = ql2 + 128;
    const uint8_t* __restrict__ qhp3 = ql3 + 128;
    const uint8_t* __restrict__ qhp4 = ql4 + 128;
    const uint8_t* __restrict__ qhp5 = ql5 + 128;
    const uint8_t* __restrict__ qhp6 = ql6 + 128;
    const uint8_t* __restrict__ qhp7 = ql7 + 128;

    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int group32 = k_in_block >> 5;
      const int lane = k_in_block & 31;
      const int g16 = k_in_block >> 4;
      const int base64 = (group32 >= 4) ? 64 : 0;
      const bool low_nibble = (group32 & 2) == 0;
      const int ql_group = group32 & 1;
      const int ql_idx = base64 + ql_group * 32 + lane;
      const int qh_base = (group32 >= 4) ? 32 : 0;
      const int qh_idx = qh_base + lane;
      const int qh_shift = 2 * (group32 & 3);

      const float xv = scalar_to_float<scalar_in_t>(x_row[k_block_base + k_in_block]);

      const uint8_t lb0 = ql0[ql_idx];
      const uint8_t lb1 = ql1[ql_idx];
      const uint8_t lb2 = ql2[ql_idx];
      const uint8_t lb3 = ql3[ql_idx];
      const uint8_t lb4 = ql4[ql_idx];
      const uint8_t lb5 = ql5[ql_idx];
      const uint8_t lb6 = ql6[ql_idx];
      const uint8_t lb7 = ql7[ql_idx];
      const uint8_t l0 = low_nibble ? (lb0 & 0x0Fu) : (lb0 >> 4);
      const uint8_t l1 = low_nibble ? (lb1 & 0x0Fu) : (lb1 >> 4);
      const uint8_t l2 = low_nibble ? (lb2 & 0x0Fu) : (lb2 >> 4);
      const uint8_t l3 = low_nibble ? (lb3 & 0x0Fu) : (lb3 >> 4);
      const uint8_t l4 = low_nibble ? (lb4 & 0x0Fu) : (lb4 >> 4);
      const uint8_t l5 = low_nibble ? (lb5 & 0x0Fu) : (lb5 >> 4);
      const uint8_t l6 = low_nibble ? (lb6 & 0x0Fu) : (lb6 >> 4);
      const uint8_t l7 = low_nibble ? (lb7 & 0x0Fu) : (lb7 >> 4);
      const uint8_t h0 = (qhp0[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h1 = (qhp1[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h2 = (qhp2[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h3 = (qhp3[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h4 = (qhp4[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h5 = (qhp5[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h6 = (qhp6[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h7 = (qhp7[qh_idx] >> qh_shift) & 0x03u;
      const float q0 = static_cast<float>(static_cast<int>(l0 | (h0 << 4)) - 32);
      const float q1 = static_cast<float>(static_cast<int>(l1 | (h1 << 4)) - 32);
      const float q2 = static_cast<float>(static_cast<int>(l2 | (h2 << 4)) - 32);
      const float q3 = static_cast<float>(static_cast<int>(l3 | (h3 << 4)) - 32);
      const float q4 = static_cast<float>(static_cast<int>(l4 | (h4 << 4)) - 32);
      const float q5 = static_cast<float>(static_cast<int>(l5 | (h5 << 4)) - 32);
      const float q6 = static_cast<float>(static_cast<int>(l6 | (h6 << 4)) - 32);
      const float q7 = static_cast<float>(static_cast<int>(l7 | (h7 << 4)) - 32);

      acc0 = fmaf(xv, s_scale[0 * Q6_K_GROUPS16 + g16] * q0, acc0);
      acc1 = fmaf(xv, s_scale[1 * Q6_K_GROUPS16 + g16] * q1, acc1);
      acc2 = fmaf(xv, s_scale[2 * Q6_K_GROUPS16 + g16] * q2, acc2);
      acc3 = fmaf(xv, s_scale[3 * Q6_K_GROUPS16 + g16] * q3, acc3);
      acc4 = fmaf(xv, s_scale[4 * Q6_K_GROUPS16 + g16] * q4, acc4);
      acc5 = fmaf(xv, s_scale[5 * Q6_K_GROUPS16 + g16] * q5, acc5);
      acc6 = fmaf(xv, s_scale[6 * Q6_K_GROUPS16 + g16] * q6, acc6);
      acc7 = fmaf(xv, s_scale[7 * Q6_K_GROUPS16 + g16] * q7, acc7);
    }
    __syncthreads();
  }

  GGUF_Q6K_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc7);

  __shared__ float xchg[4 * PACK8];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
    xchg[sub_id * PACK8 + 0] = acc0;
    xchg[sub_id * PACK8 + 1] = acc1;
    xchg[sub_id * PACK8 + 2] = acc2;
    xchg[sub_id * PACK8 + 3] = acc3;
    xchg[sub_id * PACK8 + 4] = acc4;
    xchg[sub_id * PACK8 + 5] = acc5;
    xchg[sub_id * PACK8 + 6] = acc6;
    xchg[sub_id * PACK8 + 7] = acc7;
  }
  __syncthreads();

  if (tid == 0) {
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    float r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f;
    const int sub_groups = blockDim.x >> 5;
    for (int g = 0; g < sub_groups; ++g) {
      r0 += xchg[g * PACK8 + 0];
      r1 += xchg[g * PACK8 + 1];
      r2 += xchg[g * PACK8 + 2];
      r3 += xchg[g * PACK8 + 3];
      r4 += xchg[g * PACK8 + 4];
      r5 += xchg[g * PACK8 + 5];
      r6 += xchg[g * PACK8 + 6];
      r7 += xchg[g * PACK8 + 7];
    }
    scalar_out_t* __restrict__ out_ptr = out + row * out_features + out_col;
    out_ptr[0] = float_to_scalar<scalar_out_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_out_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_out_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_out_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_out_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_out_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_out_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_out_t>(r7);
  }
}

// Top-1 variant for speculative MTP draft sampling.  It preserves the exact
// per-output dot-product reduction order of gguf_q6_k_pack8_gemv_kernel, but
// writes one (max logit, token id) pair per output pack instead of materializing
// the full logits row.  A second tiny kernel reduces the pack winners and can
// gather the next token embedding row for device-chained drafting.
template <typename scalar_in_t>
__launch_bounds__(128, 4) __global__ void gguf_q6_k_pack8_gemv_top1_stage1_kernel(
    const scalar_in_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    float* __restrict__ block_values,
    int32_t* __restrict__ block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }

  const scalar_in_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  __shared__ float s_scale[PACK8 * Q6_K_GROUPS16];

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < PACK8 * Q6_K_GROUPS16) {
      const int ch = tid >> 4;
      const int g16 = tid & 15;
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q6_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block + 208);
      const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(block + 192);
      s_scale[ch * Q6_K_GROUPS16 + g16] = d_f * static_cast<float>(scales[g16]);
    }
    __syncthreads();

    const int64_t k_block_base = blk * QK_K;
    const uint8_t* __restrict__ ql0 = row_ptr[0] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql1 = row_ptr[1] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql2 = row_ptr[2] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql3 = row_ptr[3] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql4 = row_ptr[4] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql5 = row_ptr[5] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql6 = row_ptr[6] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql7 = row_ptr[7] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ qhp0 = ql0 + 128;
    const uint8_t* __restrict__ qhp1 = ql1 + 128;
    const uint8_t* __restrict__ qhp2 = ql2 + 128;
    const uint8_t* __restrict__ qhp3 = ql3 + 128;
    const uint8_t* __restrict__ qhp4 = ql4 + 128;
    const uint8_t* __restrict__ qhp5 = ql5 + 128;
    const uint8_t* __restrict__ qhp6 = ql6 + 128;
    const uint8_t* __restrict__ qhp7 = ql7 + 128;

    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int group32 = k_in_block >> 5;
      const int lane = k_in_block & 31;
      const int g16 = k_in_block >> 4;
      const int base64 = (group32 >= 4) ? 64 : 0;
      const bool low_nibble = (group32 & 2) == 0;
      const int ql_group = group32 & 1;
      const int ql_idx = base64 + ql_group * 32 + lane;
      const int qh_base = (group32 >= 4) ? 32 : 0;
      const int qh_idx = qh_base + lane;
      const int qh_shift = 2 * (group32 & 3);
      const float xv = scalar_to_float<scalar_in_t>(x_row[k_block_base + k_in_block]);

      const uint8_t lb0 = ql0[ql_idx];
      const uint8_t lb1 = ql1[ql_idx];
      const uint8_t lb2 = ql2[ql_idx];
      const uint8_t lb3 = ql3[ql_idx];
      const uint8_t lb4 = ql4[ql_idx];
      const uint8_t lb5 = ql5[ql_idx];
      const uint8_t lb6 = ql6[ql_idx];
      const uint8_t lb7 = ql7[ql_idx];
      const uint8_t l0 = low_nibble ? (lb0 & 0x0Fu) : (lb0 >> 4);
      const uint8_t l1 = low_nibble ? (lb1 & 0x0Fu) : (lb1 >> 4);
      const uint8_t l2 = low_nibble ? (lb2 & 0x0Fu) : (lb2 >> 4);
      const uint8_t l3 = low_nibble ? (lb3 & 0x0Fu) : (lb3 >> 4);
      const uint8_t l4 = low_nibble ? (lb4 & 0x0Fu) : (lb4 >> 4);
      const uint8_t l5 = low_nibble ? (lb5 & 0x0Fu) : (lb5 >> 4);
      const uint8_t l6 = low_nibble ? (lb6 & 0x0Fu) : (lb6 >> 4);
      const uint8_t l7 = low_nibble ? (lb7 & 0x0Fu) : (lb7 >> 4);
      const uint8_t h0 = (qhp0[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h1 = (qhp1[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h2 = (qhp2[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h3 = (qhp3[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h4 = (qhp4[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h5 = (qhp5[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h6 = (qhp6[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h7 = (qhp7[qh_idx] >> qh_shift) & 0x03u;
      const float q0 = static_cast<float>(static_cast<int>(l0 | (h0 << 4)) - 32);
      const float q1 = static_cast<float>(static_cast<int>(l1 | (h1 << 4)) - 32);
      const float q2 = static_cast<float>(static_cast<int>(l2 | (h2 << 4)) - 32);
      const float q3 = static_cast<float>(static_cast<int>(l3 | (h3 << 4)) - 32);
      const float q4 = static_cast<float>(static_cast<int>(l4 | (h4 << 4)) - 32);
      const float q5 = static_cast<float>(static_cast<int>(l5 | (h5 << 4)) - 32);
      const float q6 = static_cast<float>(static_cast<int>(l6 | (h6 << 4)) - 32);
      const float q7 = static_cast<float>(static_cast<int>(l7 | (h7 << 4)) - 32);

      acc0 = fmaf(xv, s_scale[0 * Q6_K_GROUPS16 + g16] * q0, acc0);
      acc1 = fmaf(xv, s_scale[1 * Q6_K_GROUPS16 + g16] * q1, acc1);
      acc2 = fmaf(xv, s_scale[2 * Q6_K_GROUPS16 + g16] * q2, acc2);
      acc3 = fmaf(xv, s_scale[3 * Q6_K_GROUPS16 + g16] * q3, acc3);
      acc4 = fmaf(xv, s_scale[4 * Q6_K_GROUPS16 + g16] * q4, acc4);
      acc5 = fmaf(xv, s_scale[5 * Q6_K_GROUPS16 + g16] * q5, acc5);
      acc6 = fmaf(xv, s_scale[6 * Q6_K_GROUPS16 + g16] * q6, acc6);
      acc7 = fmaf(xv, s_scale[7 * Q6_K_GROUPS16 + g16] * q7, acc7);
    }
    __syncthreads();
  }

  GGUF_Q6K_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc7);

  __shared__ float xchg[4 * PACK8];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
    xchg[sub_id * PACK8 + 0] = acc0;
    xchg[sub_id * PACK8 + 1] = acc1;
    xchg[sub_id * PACK8 + 2] = acc2;
    xchg[sub_id * PACK8 + 3] = acc3;
    xchg[sub_id * PACK8 + 4] = acc4;
    xchg[sub_id * PACK8 + 5] = acc5;
    xchg[sub_id * PACK8 + 6] = acc6;
    xchg[sub_id * PACK8 + 7] = acc7;
  }
  __syncthreads();

  if (tid == 0) {
    float r[PACK8];
    for (int ch = 0; ch < PACK8; ++ch) {
      r[ch] = 0.0f;
    }
    const int sub_groups = blockDim.x >> 5;
    for (int g = 0; g < sub_groups; ++g) {
      for (int ch = 0; ch < PACK8; ++ch) {
        r[ch] += xchg[g * PACK8 + ch];
      }
    }
    float best_value = -3.4028234663852886e+38F;
    int32_t best_index = INT32_MAX;
    for (int ch = 0; ch < PACK8; ++ch) {
      const int64_t index64 = out_col + ch;
      if (index64 < out_features) {
        const int32_t index = static_cast<int32_t>(index64);
        if (better_pair_i32(r[ch], index, best_value, best_index)) {
          best_value = r[ch];
          best_index = index;
        }
      }
    }
    const int64_t off = row * out_packed + out_pack;
    block_values[off] = best_value;
    block_indices[off] = best_index;
  }
}

// Accuracy-traded llama-compat top-1 variant.  The activation row is already
// GGML q8_1-quantized, matching llama.cpp's quantized matvec economy.
__launch_bounds__(128, 4) __global__ void gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    float* __restrict__ block_values,
    int32_t* __restrict__ block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int64_t q8_blocks_per_row) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
  const gguf_q8_1_block* __restrict__ xq_row = xq + row * q8_blocks_per_row;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  const int64_t groups4 = in_features >> 2;
  for (int64_t group = tid; group < groups4; group += blockDim.x) {
    const int64_t k4 = group << 2;
    const int64_t block_idx = k4 / QK_K;
    const int within = static_cast<int>(k4 - block_idx * QK_K);
    const int group32 = within >> 5;
    const int lane4 = within & 31;
    const int scale_index = within >> 4;
    const int64_t q8_index = block_idx * 8 + group32;

    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = q6_pack8_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ b0 = row_ptr[0] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b1 = row_ptr[1] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b2 = row_ptr[2] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b3 = row_ptr[3] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b4 = row_ptr[4] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b5 = row_ptr[5] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b6 = row_ptr[6] + block_idx * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b7 = row_ptr[7] + block_idx * Q6_K_BLOCK_BYTES;

    acc0 += q6_k_q8_1_dp4a_term(b0, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc1 += q6_k_q8_1_dp4a_term(b1, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc2 += q6_k_q8_1_dp4a_term(b2, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc3 += q6_k_q8_1_dp4a_term(b3, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc4 += q6_k_q8_1_dp4a_term(b4, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc5 += q6_k_q8_1_dp4a_term(b5, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc6 += q6_k_q8_1_dp4a_term(b6, group32, lane4, scale_index, x_pack, q8_sum, xd);
    acc7 += q6_k_q8_1_dp4a_term(b7, group32, lane4, scale_index, x_pack, q8_sum, xd);
  }

  GGUF_Q6K_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc7);

  __shared__ float xchg[4 * PACK8];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
    xchg[sub_id * PACK8 + 0] = acc0;
    xchg[sub_id * PACK8 + 1] = acc1;
    xchg[sub_id * PACK8 + 2] = acc2;
    xchg[sub_id * PACK8 + 3] = acc3;
    xchg[sub_id * PACK8 + 4] = acc4;
    xchg[sub_id * PACK8 + 5] = acc5;
    xchg[sub_id * PACK8 + 6] = acc6;
    xchg[sub_id * PACK8 + 7] = acc7;
  }
  __syncthreads();

  if (tid == 0) {
    float r[PACK8];
    for (int ch = 0; ch < PACK8; ++ch) {
      r[ch] = 0.0f;
    }
    const int sub_groups = blockDim.x >> 5;
    for (int g = 0; g < sub_groups; ++g) {
      for (int ch = 0; ch < PACK8; ++ch) {
        r[ch] += xchg[g * PACK8 + ch];
      }
    }
    float best_value = -3.4028234663852886e+38F;
    int32_t best_index = INT32_MAX;
    for (int ch = 0; ch < PACK8; ++ch) {
      const int64_t index64 = out_col + ch;
      if (index64 < out_features) {
        const int32_t index = static_cast<int32_t>(index64);
        if (better_pair_i32(r[ch], index, best_value, best_index)) {
          best_value = r[ch];
          best_index = index;
        }
      }
    }
    const int64_t off = row * out_packed + out_pack;
    block_values[off] = best_value;
    block_indices[off] = best_index;
  }
}

__global__ void top1_stage2_gather_kernel(
    const float* __restrict__ block_values,
    const int32_t* __restrict__ block_indices,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_values,
    const float* __restrict__ embed_table,
    float* __restrict__ next_embed,
    int64_t rows,
    int64_t num_blocks,
    int64_t hidden,
    int64_t vocab) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int32_t* indices = reinterpret_cast<int32_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const int64_t block_base = row * num_blocks;
  float best_value = -3.4028234663852886e+38F;
  int32_t best_index = INT32_MAX;
  for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
    const float value = block_values[block_base + i];
    const int32_t index = block_indices[block_base + i];
    if (better_pair_i32(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair_i32(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  const int32_t id = indices[0];
  if (tid == 0) {
    out_indices[row] = id;
    if (out_values != nullptr) {
      out_values[row] = values[0];
    }
  }
  if (embed_table != nullptr && next_embed != nullptr && hidden > 0) {
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
      next_embed[row * hidden + col] =
          (id < 0 || static_cast<int64_t>(id) >= vocab) ? 0.0f : embed_table[static_cast<int64_t>(id) * hidden + col];
    }
  }
}

__host__ inline bool valid_top1_stage1_threads(int threads) {
  return threads == 64 || threads == 128;
}

template <typename scalar_in_t, typename scalar_out_t>
int launch_q6_k_pack8_gemv(
    const scalar_in_t* x,
    const uint8_t* qweight,
    scalar_out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (out_features & 7) != 0 || (in_features % QK_K) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_q6_k_pack8_gemv_kernel<scalar_in_t, scalar_out_t>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x, qweight, out, rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_in_t>
int launch_q6_k_pack8_gemv_top1_stage1(
    const scalar_in_t* x,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int stage1_threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || block_values == nullptr ||
      block_indices == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      out_features > INT32_MAX || (out_features & 7) != 0 ||
      (in_features % QK_K) != 0 || !valid_top1_stage1_threads(stage1_threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_q6_k_pack8_gemv_top1_stage1_kernel<scalar_in_t>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(stage1_threads)),
      0,
      stream,
      x, qweight, block_values, block_indices, rows, in_features, out_features, out_packed);
  return static_cast<int>(hipGetLastError());
}

int launch_top1_stage2_gather(
    const float* block_values,
    const int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t num_blocks,
    int64_t hidden,
    int64_t vocab,
    hipStream_t stream) {
  if (block_values == nullptr || block_indices == nullptr || out_indices == nullptr ||
      rows <= 0 || num_blocks <= 0 || hidden < 0 || vocab <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if ((embed_table == nullptr) != (next_embed == nullptr)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (embed_table != nullptr && hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int stage2_threads = 256;
  const size_t stage2_smem = static_cast<size_t>(stage2_threads) * (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      top1_stage2_gather_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(stage2_threads),
      stage2_smem,
      stream,
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      num_blocks,
      hidden,
      vocab);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_in_t>
int launch_q6_k_pack8_gemv_top1_gather(
    const scalar_in_t* x,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    int stage1_threads,
    hipStream_t stream) {
  if (hidden < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t err = static_cast<hipError_t>(launch_q6_k_pack8_gemv_top1_stage1<scalar_in_t>(
      x, qweight, block_values, block_indices, rows, in_features, out_features, stage1_threads, stream));
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  return launch_top1_stage2_gather(
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      out_features >> 3,
      hidden,
      out_features,
      stream);
}

int launch_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1(
    const gguf_q8_1_block* xq,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int stage1_threads,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || block_values == nullptr ||
      block_indices == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      out_features > INT32_MAX || (out_features & 7) != 0 ||
      (in_features % QK_K) != 0 || (in_features % Q8_1_BLOCK) != 0 ||
      !valid_top1_stage1_threads(stage1_threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  hipLaunchKernelGGL(
      gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1_kernel,
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(stage1_threads)),
      0,
      stream,
      xq, qweight, block_values, block_indices, rows, in_features, out_features, out_packed, q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}

int launch_q6_k_pack8_gemv_q8_1_dp4a_top1_gather(
    const gguf_q8_1_block* xq,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    int stage1_threads,
    hipStream_t stream) {
  if (hidden < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t err = static_cast<hipError_t>(launch_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1(
      xq, qweight, block_values, block_indices, rows, in_features, out_features, stage1_threads, stream));
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  return launch_top1_stage2_gather(
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      out_features >> 3,
      hidden,
      out_features,
      stream);
}

}  // namespace

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_fp16_fp16_out(
    const _Float16* x,
    const uint8_t* qweight,
    _Float16* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<_Float16, _Float16>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_fp16_f32_out(
    const _Float16* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<_Float16, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(
    const uint16_t* x,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_top1_gather<uint16_t>(
      x,
      qweight,
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      in_features,
      out_features,
      hidden,
      128,
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32_threads(
    const uint16_t* x,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    int32_t stage1_threads,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_top1_gather<uint16_t>(
      x,
      qweight,
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      in_features,
      out_features,
      hidden,
      static_cast<int>(stage1_threads),
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32(
    const uint16_t* x,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int32_t stage1_threads,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_top1_stage1<uint16_t>(
      x,
      qweight,
      block_values,
      block_indices,
      rows,
      in_features,
      out_features,
      static_cast<int>(stage1_threads),
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(
    const void* xq,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_q8_1_dp4a_top1_gather(
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      qweight,
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      in_features,
      out_features,
      hidden,
      128,
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32_threads(
    const void* xq,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t hidden,
    int32_t stage1_threads,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_q8_1_dp4a_top1_gather(
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      qweight,
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      in_features,
      out_features,
      hidden,
      static_cast<int>(stage1_threads),
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32(
    const void* xq,
    const uint8_t* qweight,
    float* block_values,
    int32_t* block_indices,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int32_t stage1_threads,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1(
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      qweight,
      block_values,
      block_indices,
      rows,
      in_features,
      out_features,
      static_cast<int>(stage1_threads),
      stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_top1_stage2_gather_f32(
    const float* block_values,
    const int32_t* block_indices,
    int32_t* out_indices,
    float* out_values,
    const float* embed_table,
    float* next_embed,
    int64_t rows,
    int64_t num_blocks,
    int64_t hidden,
    int64_t vocab,
    hipStream_t stream) {
  return launch_top1_stage2_gather(
      block_values,
      block_indices,
      out_indices,
      out_values,
      embed_table,
      next_embed,
      rows,
      num_blocks,
      hidden,
      vocab,
      stream);
}
