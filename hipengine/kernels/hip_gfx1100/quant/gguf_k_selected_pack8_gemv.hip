// Grouped/selected MoE decode-shaped pack8 GEMV for raw GGUF Q5_K/Q6_K experts.
//
// P9.B2 implementation note: mirrors the structure of
// ``paro_awq_gemv.hip::gemv_awq_selected_pack8_kernel`` (PARO-style pack8 row
// layout, ``__launch_bounds__(128, 4)``, register-resident dequant, wave-level
// reduction across 4 wave32 waves) but consumes the P8.5 compact-MoE ABI:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   qweight              [E, out_features, row_bytes]   raw block_qK bytes
//   out                  [compact_rows, out_features]
//
// One block computes 8 output channels for one compact row, dequantizing
// Q5_K (qtype=5) or Q6_K (qtype=6) blocks in-register. The inner K loop
// hoists each Q5_K/Q6_K block's ``d``/``dmin``/scale/min into shared memory
// once across all 8 output channels of the pack, so the 256-element inner
// loop stays register-resident. Expert id is recovered via a linear scan
// over ``expert_start_compact`` (decode rows=1 per active expert lane; ``num_experts``
// is small in practice for qwen35moe).
//
// No new compact-MoE ABI and no resident weight sidecar/repack are introduced.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q5_K_SUBBLOCKS = 8;             // 8 subblocks of 32 elements each
constexpr int Q6_K_GROUPS16 = QK_K / 16;       // 16 groups of 16 elements each
constexpr int PACK8 = 8;

// Maximum shared-memory scale layout across qtype:
//   Q5_K uses 8 channels x 8 subblocks = 64 scale entries + 64 min entries.
//   Q6_K uses 8 channels x 16 groups = 128 scale entries; no min.
constexpr int S_SCALE_MAX = PACK8 * Q6_K_GROUPS16;  // 128
constexpr int S_MIN_MAX = PACK8 * Q5_K_SUBBLOCKS;   // 64

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

__device__ inline uint8_t gguf_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

template <int qtype>
__device__ inline int qtype_block_bytes() {
  if constexpr (qtype == 5) {
    return Q5_K_BLOCK_BYTES;
  } else {
    return Q6_K_BLOCK_BYTES;
  }
}

template <int qtype>
__device__ inline int64_t qtype_row_bytes(int64_t in_features) {
  if constexpr (qtype == 5) {
    return (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    return (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
}

// Decode-shaped pack8 GEMV for raw GGUF Q5_K (qtype=5) or Q6_K (qtype=6)
// selected single output (down projection).
//
// Grid:
//   blockIdx.x = output pack in [0, out_features / 8)
//   blockIdx.y = compact_row in [0, compact_rows)
// Block:
//   blockDim.x = 128 (4 wave32 waves; __launch_bounds__(128, 4))
template <typename scalar_t, int qtype>
__launch_bounds__(128, 4) __global__ void gguf_k_selected_pack8_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 5 || qtype == 6, "qtype must be 5 or 6");

  const int64_t out_pack = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  int64_t expert_id = -1;
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      expert_id = e;
      break;
    }
  }
  if (expert_id < 0) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t row_bytes = qtype_row_bytes<qtype>(in_features);
  const int64_t expert_bytes = out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = expert_base + (out_col + ch) * row_bytes;
  }

  const scalar_t* __restrict__ x_row = x + compact_row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  __shared__ float s_scale[S_SCALE_MAX];   // Q5 uses 64; Q6 uses 128
  __shared__ float s_min[S_MIN_MAX];        // Q5 only

  const int64_t blocks_per_row = in_features / QK_K;
  const int block_bytes = qtype_block_bytes<qtype>();

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    // Cooperative per-block hoist of scale/min for all 8 output channels.
    if constexpr (qtype == 5) {
      if (tid < PACK8 * Q5_K_SUBBLOCKS) {
        const int ch = tid >> 3;
        const int sb = tid & 7;
        const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q5_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block);
        const float dmin_f = fp16_bytes_to_float(block + 2);
        const uint8_t* __restrict__ scales = block + 4;
        s_scale[ch * Q5_K_SUBBLOCKS + sb] =
            d_f * static_cast<float>(gguf_k_scale(scales, sb));
        s_min[ch * Q5_K_SUBBLOCKS + sb] =
            dmin_f * static_cast<float>(gguf_k_min(scales, sb));
      }
    } else {
      if (tid < PACK8 * Q6_K_GROUPS16) {
        const int ch = tid >> 4;
        const int g16 = tid & 15;
        const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q6_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block + 208);
        const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(block + 192);
        s_scale[ch * Q6_K_GROUPS16 + g16] =
            d_f * static_cast<float>(scales[g16]);
      }
    }
    __syncthreads();

    const int64_t k_block_base = blk * QK_K;

    if constexpr (qtype == 5) {
      // Q5_K per-channel qh (32 bytes) and qs (128 bytes) pointers.
      const uint8_t* __restrict__ qh0 = row_ptr[0] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh1 = row_ptr[1] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh2 = row_ptr[2] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh3 = row_ptr[3] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh4 = row_ptr[4] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh5 = row_ptr[5] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh6 = row_ptr[6] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qh7 = row_ptr[7] + blk * Q5_K_BLOCK_BYTES + 16;
      const uint8_t* __restrict__ qs0 = row_ptr[0] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs1 = row_ptr[1] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs2 = row_ptr[2] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs3 = row_ptr[3] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs4 = row_ptr[4] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs5 = row_ptr[5] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs6 = row_ptr[6] + blk * Q5_K_BLOCK_BYTES + 48;
      const uint8_t* __restrict__ qs7 = row_ptr[7] + blk * Q5_K_BLOCK_BYTES + 48;
      for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
        const int pair_off = (sb >> 1) * 32 + lane32;
        const int shift = (sb & 1) * 4;
        const int hi_shift = sb;
        const float xv = scalar_to_float<scalar_t>(x_row[k_block_base + k_in_block]);

        const uint8_t l0 = (qs0[pair_off] >> shift) & 0x0Fu;
        const uint8_t l1 = (qs1[pair_off] >> shift) & 0x0Fu;
        const uint8_t l2 = (qs2[pair_off] >> shift) & 0x0Fu;
        const uint8_t l3 = (qs3[pair_off] >> shift) & 0x0Fu;
        const uint8_t l4 = (qs4[pair_off] >> shift) & 0x0Fu;
        const uint8_t l5 = (qs5[pair_off] >> shift) & 0x0Fu;
        const uint8_t l6 = (qs6[pair_off] >> shift) & 0x0Fu;
        const uint8_t l7 = (qs7[pair_off] >> shift) & 0x0Fu;
        const uint8_t h0 = (qh0[lane32] >> hi_shift) & 0x01u;
        const uint8_t h1 = (qh1[lane32] >> hi_shift) & 0x01u;
        const uint8_t h2 = (qh2[lane32] >> hi_shift) & 0x01u;
        const uint8_t h3 = (qh3[lane32] >> hi_shift) & 0x01u;
        const uint8_t h4 = (qh4[lane32] >> hi_shift) & 0x01u;
        const uint8_t h5 = (qh5[lane32] >> hi_shift) & 0x01u;
        const uint8_t h6 = (qh6[lane32] >> hi_shift) & 0x01u;
        const uint8_t h7 = (qh7[lane32] >> hi_shift) & 0x01u;
        const float q0 = static_cast<float>(l0 | (h0 << 4));
        const float q1 = static_cast<float>(l1 | (h1 << 4));
        const float q2 = static_cast<float>(l2 | (h2 << 4));
        const float q3 = static_cast<float>(l3 | (h3 << 4));
        const float q4 = static_cast<float>(l4 | (h4 << 4));
        const float q5 = static_cast<float>(l5 | (h5 << 4));
        const float q6 = static_cast<float>(l6 | (h6 << 4));
        const float q7 = static_cast<float>(l7 | (h7 << 4));

        acc0 = fmaf(xv, s_scale[0 * Q5_K_SUBBLOCKS + sb] * q0 - s_min[0 * Q5_K_SUBBLOCKS + sb], acc0);
        acc1 = fmaf(xv, s_scale[1 * Q5_K_SUBBLOCKS + sb] * q1 - s_min[1 * Q5_K_SUBBLOCKS + sb], acc1);
        acc2 = fmaf(xv, s_scale[2 * Q5_K_SUBBLOCKS + sb] * q2 - s_min[2 * Q5_K_SUBBLOCKS + sb], acc2);
        acc3 = fmaf(xv, s_scale[3 * Q5_K_SUBBLOCKS + sb] * q3 - s_min[3 * Q5_K_SUBBLOCKS + sb], acc3);
        acc4 = fmaf(xv, s_scale[4 * Q5_K_SUBBLOCKS + sb] * q4 - s_min[4 * Q5_K_SUBBLOCKS + sb], acc4);
        acc5 = fmaf(xv, s_scale[5 * Q5_K_SUBBLOCKS + sb] * q5 - s_min[5 * Q5_K_SUBBLOCKS + sb], acc5);
        acc6 = fmaf(xv, s_scale[6 * Q5_K_SUBBLOCKS + sb] * q6 - s_min[6 * Q5_K_SUBBLOCKS + sb], acc6);
        acc7 = fmaf(xv, s_scale[7 * Q5_K_SUBBLOCKS + sb] * q7 - s_min[7 * Q5_K_SUBBLOCKS + sb], acc7);
      }
    } else {
      // Q6_K per-channel ql (128 bytes) and qh (64 bytes) pointers.
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

        const float xv = scalar_to_float<scalar_t>(x_row[k_block_base + k_in_block]);

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
    }
    __syncthreads();  // protect s_scale/s_min before the next block reloads them
  }

  // Wave-level reduction across 4 wave32 waves -> cross-wave sum.
#define GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc)                  \
  do {                                                       \
    acc += __shfl_down(acc, 16);                             \
    acc += __shfl_down(acc, 8);                              \
    acc += __shfl_down(acc, 4);                              \
    acc += __shfl_down(acc, 2);                              \
    acc += __shfl_down(acc, 1);                              \
  } while (0)

  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc0);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc1);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc2);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc3);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc4);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc5);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc6);
  GGUF_K_GEMV_PACK8_REDUCE_SHFL(acc7);
#undef GGUF_K_GEMV_PACK8_REDUCE_SHFL

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
    scalar_t* __restrict__ out_ptr = out + compact_row * out_features + out_col;
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

bool valid_selected_pack8_gemv(
    const void* x,
    const void* expert_start_compact,
    const void* qweight,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  return x != nullptr && expert_start_compact != nullptr && qweight != nullptr &&
      out != nullptr && compact_rows > 0 && in_features > 0 && out_features > 0 &&
      num_experts > 0 && (out_features & 7) == 0 && (in_features % QK_K) == 0;
}

template <typename scalar_t, int qtype>
int launch_selected_pack8_gemv(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!valid_selected_pack8_gemv(
          x, expert_start_compact, qweight, out,
          compact_rows, in_features, out_features, num_experts)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_k_selected_pack8_gemv_kernel<scalar_t, qtype>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(compact_rows)),
      dim3(128),
      0,
      stream,
      x,
      expert_start_compact,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_pack8_gemv<uint16_t, 5>(
      x, expert_start_compact, qweight, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out(
    const _Float16* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    _Float16* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_pack8_gemv<_Float16, 5>(
      x, expert_start_compact, qweight, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_pack8_gemv<uint16_t, 6>(
      x, expert_start_compact, qweight, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out(
    const _Float16* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    _Float16* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_selected_pack8_gemv<_Float16, 6>(
      x, expert_start_compact, qweight, out,
      compact_rows, in_features, out_features, num_experts, stream);
}
