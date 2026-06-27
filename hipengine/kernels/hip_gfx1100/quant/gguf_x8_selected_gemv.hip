// Selected MoE GEMV for GGUF X8 replacement layouts.
//
// X8 stores eight adjacent output rows for one K block contiguously while
// preserving the original GGUF Q5_K/Q6_K block bytes. This lets the production
// resident path use the same q8_1+sudot4 recipe as the raw selected-pack8 POC
// without keeping raw expert tensors resident.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q8_1_BLOCK = 32;
constexpr int X8_COLS = 8;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;

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

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
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

__device__ inline uint8_t q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline int x8_dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
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

__device__ inline uint32_t q4_k_pack4_unsigned(const uint8_t* block, int subblock, int lane4) {
  const uint8_t* qs = block + 16;
  const int pair = subblock >> 1;
  uint32_t packed = *reinterpret_cast<const uint32_t*>(qs + pair * 32 + lane4);
  return (subblock & 1) ? ((packed >> 4) & 0x0F0F0F0Fu) : (packed & 0x0F0F0F0Fu);
}

__device__ inline uint32_t q5_k_pack4_unsigned(const uint8_t* block, int subblock, int lane4) {
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;
  uint32_t low = *reinterpret_cast<const uint32_t*>(qs + (subblock >> 1) * 32 + lane4);
  low = (subblock & 1) ? ((low >> 4) & 0x0F0F0F0Fu) : (low & 0x0F0F0F0Fu);
  const uint32_t high_bits = *reinterpret_cast<const uint32_t*>(qh + lane4);
  const uint32_t high = ((high_bits >> subblock) & 0x01010101u) << 4;
  return low | high;
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

__device__ inline float q4_k_q8_1_dp4a_term(
    const uint8_t* block,
    int subblock,
    int lane4,
    uint32_t x_pack,
    int q8_sum,
    float xd) {
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const int dot = x8_dp4a_u8_s8(q4_k_pack4_unsigned(block, subblock, lane4), x_pack, 0);
  return xd * (d * static_cast<float>(q4_k_scale(scales, subblock)) * static_cast<float>(dot) -
               dmin * static_cast<float>(q4_k_min(scales, subblock)) * static_cast<float>(q8_sum));
}

__device__ inline float q5_k_q8_1_dp4a_term(
    const uint8_t* block,
    int subblock,
    int lane4,
    uint32_t x_pack,
    int q8_sum,
    float xd) {
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const int dot = x8_dp4a_u8_s8(q5_k_pack4_unsigned(block, subblock, lane4), x_pack, 0);
  return xd * (d * static_cast<float>(q4_k_scale(scales, subblock)) * static_cast<float>(dot) -
               dmin * static_cast<float>(q4_k_min(scales, subblock)) * static_cast<float>(q8_sum));
}

__device__ inline float q6_k_q8_1_dp4a_term(
    const uint8_t* block,
    int group32,
    int lane4,
    int scale_index,
    uint32_t x_pack,
    int q8_sum,
    float xd) {
  const int dot_u = x8_dp4a_u8_s8(q6_k_pack4_unsigned(block, group32, lane4), x_pack, 0);
  const int dot = dot_u - 32 * q8_sum;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = fp16_bytes_to_float(block + 208);
  return xd * d * static_cast<float>(scales[scale_index]) * static_cast<float>(dot);
}

__device__ inline int64_t find_expert_for_compact_row(
    const int64_t* __restrict__ expert_start_compact,
    int64_t compact_row,
    int64_t num_experts) {
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      return static_cast<int64_t>(e);
    }
  }
  return -1;
}

template <int block_bytes>
__device__ inline const uint8_t* x8_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t expert_id,
    int64_t out_pack,
    int64_t block_idx,
    int64_t out_packed,
    int64_t blocks_per_row) {
  const int64_t tile_index = (expert_id * out_packed + out_pack) * blocks_per_row + block_idx;
  return tiles + tile_index * (X8_COLS * block_bytes);
}

#define GGUF_X8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
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

template <typename out_t>
__launch_bounds__(256, 2)
__global__ void gguf_q4_x8_selected_dual_q8_1_dp4a_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int q8_blocks_per_row) {
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const gguf_q8_1_block* __restrict__ xq_row = xq + static_cast<int64_t>(x_row) * q8_blocks_per_row;

  float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
  float acc_a4 = 0.0f, acc_a5 = 0.0f, acc_a6 = 0.0f, acc_a7 = 0.0f;
  float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
  float acc_b4 = 0.0f, acc_b5 = 0.0f, acc_b6 = 0.0f, acc_b7 = 0.0f;
  const int groups4 = in_features >> 2;
  const int blocks_per_row = in_features / QK_K;
  const int out_packed = out_features / X8_COLS;

  for (int group = tid; group < groups4; group += blockDim.x) {
    const int k4 = group << 2;
    const int block_idx = k4 / QK_K;
    const int within = k4 - block_idx * QK_K;
    const int subblock = within >> 5;
    const int lane4 = within & 31;
    const int q8_index = block_idx * 8 + subblock;

    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = x8_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ tile_a = x8_tile_ptr<Q4_K_BLOCK_BYTES>(
        tiles_a, expert_id, out_pack, block_idx, out_packed, blocks_per_row);
    const uint8_t* __restrict__ tile_b = x8_tile_ptr<Q4_K_BLOCK_BYTES>(
        tiles_b, expert_id, out_pack, block_idx, out_packed, blocks_per_row);
    const uint8_t* __restrict__ a0 = tile_a + 0 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a1 = tile_a + 1 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a2 = tile_a + 2 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a3 = tile_a + 3 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a4 = tile_a + 4 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a5 = tile_a + 5 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a6 = tile_a + 6 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ a7 = tile_a + 7 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b0 = tile_b + 0 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b1 = tile_b + 1 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b2 = tile_b + 2 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b3 = tile_b + 3 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b4 = tile_b + 4 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b5 = tile_b + 5 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b6 = tile_b + 6 * Q4_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b7 = tile_b + 7 * Q4_K_BLOCK_BYTES;

    acc_a0 += q4_k_q8_1_dp4a_term(a0, subblock, lane4, x_pack, q8_sum, xd);
    acc_a1 += q4_k_q8_1_dp4a_term(a1, subblock, lane4, x_pack, q8_sum, xd);
    acc_a2 += q4_k_q8_1_dp4a_term(a2, subblock, lane4, x_pack, q8_sum, xd);
    acc_a3 += q4_k_q8_1_dp4a_term(a3, subblock, lane4, x_pack, q8_sum, xd);
    acc_a4 += q4_k_q8_1_dp4a_term(a4, subblock, lane4, x_pack, q8_sum, xd);
    acc_a5 += q4_k_q8_1_dp4a_term(a5, subblock, lane4, x_pack, q8_sum, xd);
    acc_a6 += q4_k_q8_1_dp4a_term(a6, subblock, lane4, x_pack, q8_sum, xd);
    acc_a7 += q4_k_q8_1_dp4a_term(a7, subblock, lane4, x_pack, q8_sum, xd);
    acc_b0 += q4_k_q8_1_dp4a_term(b0, subblock, lane4, x_pack, q8_sum, xd);
    acc_b1 += q4_k_q8_1_dp4a_term(b1, subblock, lane4, x_pack, q8_sum, xd);
    acc_b2 += q4_k_q8_1_dp4a_term(b2, subblock, lane4, x_pack, q8_sum, xd);
    acc_b3 += q4_k_q8_1_dp4a_term(b3, subblock, lane4, x_pack, q8_sum, xd);
    acc_b4 += q4_k_q8_1_dp4a_term(b4, subblock, lane4, x_pack, q8_sum, xd);
    acc_b5 += q4_k_q8_1_dp4a_term(b5, subblock, lane4, x_pack, q8_sum, xd);
    acc_b6 += q4_k_q8_1_dp4a_term(b6, subblock, lane4, x_pack, q8_sum, xd);
    acc_b7 += q4_k_q8_1_dp4a_term(b7, subblock, lane4, x_pack, q8_sum, xd);
  }

  const int64_t out_base = static_cast<int64_t>(row) * out_features + (out_pack << 3);
  {
    GGUF_X8_REDUCE(acc_a0, acc_a1, acc_a2, acc_a3, acc_a4, acc_a5, acc_a6, acc_a7,
                   ar0, ar1, ar2, ar3, ar4, ar5, ar6, ar7);
    if (tid == 0) {
      store_output<out_t>(out_a, out_base + 0, ar0);
      store_output<out_t>(out_a, out_base + 1, ar1);
      store_output<out_t>(out_a, out_base + 2, ar2);
      store_output<out_t>(out_a, out_base + 3, ar3);
      store_output<out_t>(out_a, out_base + 4, ar4);
      store_output<out_t>(out_a, out_base + 5, ar5);
      store_output<out_t>(out_a, out_base + 6, ar6);
      store_output<out_t>(out_a, out_base + 7, ar7);
    }
  }
  __syncthreads();
  {
    GGUF_X8_REDUCE(acc_b0, acc_b1, acc_b2, acc_b3, acc_b4, acc_b5, acc_b6, acc_b7,
                   br0, br1, br2, br3, br4, br5, br6, br7);
    if (tid == 0) {
      store_output<out_t>(out_b, out_base + 0, br0);
      store_output<out_t>(out_b, out_base + 1, br1);
      store_output<out_t>(out_b, out_base + 2, br2);
      store_output<out_t>(out_b, out_base + 3, br3);
      store_output<out_t>(out_b, out_base + 4, br4);
      store_output<out_t>(out_b, out_base + 5, br5);
      store_output<out_t>(out_b, out_base + 6, br6);
      store_output<out_t>(out_b, out_base + 7, br7);
    }
  }
}

template <typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_x8_selected_q8_1_dp4a_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int q8_blocks_per_row) {
  static_assert(qtype == 5 || qtype == 6, "X8 selected dp4a supports Q5_K/Q6_K");
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const gguf_q8_1_block* __restrict__ xq_row = xq + static_cast<int64_t>(x_row) * q8_blocks_per_row;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  const int groups4 = in_features >> 2;
  const int blocks_per_row = in_features / QK_K;
  const int out_packed = out_features / X8_COLS;
  constexpr int block_bytes = (qtype == 5) ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;

  for (int group = tid; group < groups4; group += blockDim.x) {
    const int k4 = group << 2;
    const int block_idx = k4 / QK_K;
    const int within = k4 - block_idx * QK_K;
    const int subblock = within >> 5;
    const int lane4 = within & 31;
    const int q8_index = block_idx * 8 + subblock;

    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = x8_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ tile = x8_tile_ptr<block_bytes>(
        tiles, expert_id, out_pack, block_idx, out_packed, blocks_per_row);
    const uint8_t* __restrict__ b0 = tile + 0 * block_bytes;
    const uint8_t* __restrict__ b1 = tile + 1 * block_bytes;
    const uint8_t* __restrict__ b2 = tile + 2 * block_bytes;
    const uint8_t* __restrict__ b3 = tile + 3 * block_bytes;
    const uint8_t* __restrict__ b4 = tile + 4 * block_bytes;
    const uint8_t* __restrict__ b5 = tile + 5 * block_bytes;
    const uint8_t* __restrict__ b6 = tile + 6 * block_bytes;
    const uint8_t* __restrict__ b7 = tile + 7 * block_bytes;
    if constexpr (qtype == 5) {
      acc0 += q5_k_q8_1_dp4a_term(b0, subblock, lane4, x_pack, q8_sum, xd);
      acc1 += q5_k_q8_1_dp4a_term(b1, subblock, lane4, x_pack, q8_sum, xd);
      acc2 += q5_k_q8_1_dp4a_term(b2, subblock, lane4, x_pack, q8_sum, xd);
      acc3 += q5_k_q8_1_dp4a_term(b3, subblock, lane4, x_pack, q8_sum, xd);
      acc4 += q5_k_q8_1_dp4a_term(b4, subblock, lane4, x_pack, q8_sum, xd);
      acc5 += q5_k_q8_1_dp4a_term(b5, subblock, lane4, x_pack, q8_sum, xd);
      acc6 += q5_k_q8_1_dp4a_term(b6, subblock, lane4, x_pack, q8_sum, xd);
      acc7 += q5_k_q8_1_dp4a_term(b7, subblock, lane4, x_pack, q8_sum, xd);
    } else {
      const int scale_index = within >> 4;
      acc0 += q6_k_q8_1_dp4a_term(b0, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc1 += q6_k_q8_1_dp4a_term(b1, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc2 += q6_k_q8_1_dp4a_term(b2, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc3 += q6_k_q8_1_dp4a_term(b3, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc4 += q6_k_q8_1_dp4a_term(b4, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc5 += q6_k_q8_1_dp4a_term(b5, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc6 += q6_k_q8_1_dp4a_term(b6, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc7 += q6_k_q8_1_dp4a_term(b7, subblock, lane4, scale_index, x_pack, q8_sum, xd);
    }
  }

  GGUF_X8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + (out_pack << 3);
    store_output<out_t>(out, out_base + 0, r0);
    store_output<out_t>(out, out_base + 1, r1);
    store_output<out_t>(out, out_base + 2, r2);
    store_output<out_t>(out, out_base + 3, r3);
    store_output<out_t>(out, out_base + 4, r4);
    store_output<out_t>(out, out_base + 5, r5);
    store_output<out_t>(out, out_base + 6, r6);
    store_output<out_t>(out, out_base + 7, r7);
  }
}

template <typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_x8_selected_q8_1_dp4a_compact_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts,
    int q8_blocks_per_row) {
  static_assert(qtype == 5 || qtype == 6, "X8 compact dp4a supports Q5_K/Q6_K");
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= compact_rows) return;

  const int64_t expert_id = find_expert_for_compact_row(expert_start_compact, row, num_experts);
  if (expert_id < 0 || expert_id >= num_experts) return;
  const gguf_q8_1_block* __restrict__ xq_row = xq + static_cast<int64_t>(row) * q8_blocks_per_row;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  const int groups4 = in_features >> 2;
  const int blocks_per_row = in_features / QK_K;
  const int out_packed = out_features / X8_COLS;
  constexpr int block_bytes = (qtype == 5) ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;

  for (int group = tid; group < groups4; group += blockDim.x) {
    const int k4 = group << 2;
    const int block_idx = k4 / QK_K;
    const int within = k4 - block_idx * QK_K;
    const int subblock = within >> 5;
    const int lane4 = within & 31;
    const int q8_index = block_idx * 8 + subblock;

    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = x8_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ tile = x8_tile_ptr<block_bytes>(
        tiles, expert_id, out_pack, block_idx, out_packed, blocks_per_row);
    const uint8_t* __restrict__ b0 = tile + 0 * block_bytes;
    const uint8_t* __restrict__ b1 = tile + 1 * block_bytes;
    const uint8_t* __restrict__ b2 = tile + 2 * block_bytes;
    const uint8_t* __restrict__ b3 = tile + 3 * block_bytes;
    const uint8_t* __restrict__ b4 = tile + 4 * block_bytes;
    const uint8_t* __restrict__ b5 = tile + 5 * block_bytes;
    const uint8_t* __restrict__ b6 = tile + 6 * block_bytes;
    const uint8_t* __restrict__ b7 = tile + 7 * block_bytes;
    if constexpr (qtype == 5) {
      acc0 += q5_k_q8_1_dp4a_term(b0, subblock, lane4, x_pack, q8_sum, xd);
      acc1 += q5_k_q8_1_dp4a_term(b1, subblock, lane4, x_pack, q8_sum, xd);
      acc2 += q5_k_q8_1_dp4a_term(b2, subblock, lane4, x_pack, q8_sum, xd);
      acc3 += q5_k_q8_1_dp4a_term(b3, subblock, lane4, x_pack, q8_sum, xd);
      acc4 += q5_k_q8_1_dp4a_term(b4, subblock, lane4, x_pack, q8_sum, xd);
      acc5 += q5_k_q8_1_dp4a_term(b5, subblock, lane4, x_pack, q8_sum, xd);
      acc6 += q5_k_q8_1_dp4a_term(b6, subblock, lane4, x_pack, q8_sum, xd);
      acc7 += q5_k_q8_1_dp4a_term(b7, subblock, lane4, x_pack, q8_sum, xd);
    } else {
      const int scale_index = within >> 4;
      acc0 += q6_k_q8_1_dp4a_term(b0, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc1 += q6_k_q8_1_dp4a_term(b1, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc2 += q6_k_q8_1_dp4a_term(b2, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc3 += q6_k_q8_1_dp4a_term(b3, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc4 += q6_k_q8_1_dp4a_term(b4, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc5 += q6_k_q8_1_dp4a_term(b5, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc6 += q6_k_q8_1_dp4a_term(b6, subblock, lane4, scale_index, x_pack, q8_sum, xd);
      acc7 += q6_k_q8_1_dp4a_term(b7, subblock, lane4, scale_index, x_pack, q8_sum, xd);
    }
  }

  GGUF_X8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + (out_pack << 3);
    store_output<out_t>(out, out_base + 0, r0);
    store_output<out_t>(out, out_base + 1, r1);
    store_output<out_t>(out, out_base + 2, r2);
    store_output<out_t>(out, out_base + 3, r3);
    store_output<out_t>(out, out_base + 4, r4);
    store_output<out_t>(out, out_base + 5, r5);
    store_output<out_t>(out, out_base + 6, r6);
    store_output<out_t>(out, out_base + 7, r7);
  }
}

template <typename out_t, int qtype>
int launch_x8_selected_direct(
    const void* xq,
    const void* selected,
    const void* tiles,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!xq || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 ||
      (rows % x_rows) != 0 || num_experts <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 || (out_features % X8_COLS) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features / X8_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((gguf_x8_selected_q8_1_dp4a_gemv_kernel<out_t, qtype>),
                     grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<out_t*>(out),
                     static_cast<int>(x_rows),
                     static_cast<int>(rows),
                     static_cast<int>(num_experts),
                     static_cast<int>(in_features),
                     static_cast<int>(out_features),
                     static_cast<int>(q8_blocks_per_row));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_q4_x8_selected_dual_direct(
    const void* xq,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out_a,
    void* out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!xq || !selected || !tiles_a || !tiles_b || !out_a || !out_b ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % X8_COLS) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features / X8_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((gguf_q4_x8_selected_dual_q8_1_dp4a_gemv_kernel<out_t>),
                     grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<out_t*>(out_a),
                     reinterpret_cast<out_t*>(out_b),
                     static_cast<int>(x_rows),
                     static_cast<int>(rows),
                     static_cast<int>(num_experts),
                     static_cast<int>(in_features),
                     static_cast<int>(out_features),
                     static_cast<int>(q8_blocks_per_row));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int qtype>
int launch_x8_selected_compact(
    const void* xq,
    const void* expert_start_compact,
    const void* tiles,
    void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t threads,
    hipStream_t stream) {
  if (!xq || !expert_start_compact || !tiles || !out || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % X8_COLS) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features / X8_COLS), static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL((gguf_x8_selected_q8_1_dp4a_compact_gemv_kernel<out_t, qtype>),
                     grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const int64_t*>(expert_start_compact),
                     reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<out_t*>(out),
                     static_cast<int>(compact_rows),
                     static_cast<int>(in_features),
                     static_cast<int>(out_features),
                     static_cast<int>(num_experts),
                     static_cast<int>(q8_blocks_per_row));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_q4_x8_selected_dual_direct<uint16_t>(
      xq, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows, num_experts,
      in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_x8_selected_direct<uint16_t, 5>(
      xq, selected, tiles, out, x_rows, rows, num_experts, in_features,
      out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_x8_selected_direct<uint16_t, 6>(
      xq, selected, tiles, out, x_rows, rows, num_experts, in_features,
      out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out(
    const void* xq, const void* expert_start_compact, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, int64_t threads, hipStream_t stream) {
  return launch_x8_selected_compact<uint16_t, 5>(
      xq, expert_start_compact, tiles, out, compact_rows, in_features,
      out_features, num_experts, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out(
    const void* xq, const void* expert_start_compact, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, int64_t threads, hipStream_t stream) {
  return launch_x8_selected_compact<uint16_t, 6>(
      xq, expert_start_compact, tiles, out, compact_rows, in_features,
      out_features, num_experts, threads, stream);
}
