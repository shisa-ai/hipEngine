// GGUF Q8_0/Q5_K/Q6_K raw-byte GEMV and exact Q8 prefill kernels.
// These preserve GGML quant math and are separate from PARO/AWQ kernels.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int QK_K = 256;
constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_1_BLOCK = 32;
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
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
}

__device__ inline float round_to_bf16_float(float value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
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

__device__ inline float q8_0_weight(const uint8_t* row, int k) {
  const int block_idx = k / Q8_0_BLOCK;
  const int lane = k & (Q8_0_BLOCK - 1);
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q8_0_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
  return d * static_cast<float>(q);
}

__device__ inline uint8_t q5_k_quant(const uint8_t* block, int subblock, int lane) {
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane];
  const uint8_t ql = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane] >> subblock) & 0x01u;
  return ql | (high << 4);
}

__device__ inline float q5_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t q = q5_k_quant(block, subblock, lane);
  const float scale = static_cast<float>(q4_k_scale(scales, subblock));
  const float minv = static_cast<float>(q4_k_min(scales, subblock));
  return d * scale * static_cast<float>(q) - dmin * minv;
}

__device__ inline float q6_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int group32 = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = fp16_bytes_to_float(block + 208);

  const int base64 = (group32 >= 4) ? 64 : 0;
  const bool low_nibble = (group32 & 2) == 0;
  const int ql_group = group32 & 1;
  const uint8_t ql_byte = ql[base64 + ql_group * 32 + lane];
  const uint8_t low = low_nibble ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
  const int qh_base = (group32 >= 4) ? 32 : 0;
  const uint8_t high = (qh[qh_base + lane] >> (2 * (group32 & 3))) & 0x03u;
  const int q = static_cast<int>(low | (high << 4)) - 32;
  const int scale = static_cast<int>(scales[within >> 4]);
  return d * static_cast<float>(scale * q);
}

__device__ __forceinline__ int q6_k_quant_from_block(
    const uint8_t* __restrict__ block,
    int within) {
  const int group32 = within >> 5;
  const int lane = within & 31;
  const uint8_t* __restrict__ qh = block + 128;
  const int base64 = group32 >= 4 ? 64 : 0;
  const bool low_nibble = (group32 & 2) == 0;
  const int ql_index = base64 + (group32 & 1) * 32 + lane;
  const uint8_t ql_byte = block[ql_index];
  const uint8_t low = low_nibble ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
  const int qh_index = (group32 >= 4 ? 32 : 0) + lane;
  const int qh_shift = 2 * (group32 & 3);
  const uint8_t high = (qh[qh_index] >> qh_shift) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

__device__ inline int gguf_k_dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
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
  const int dot = gguf_k_dp4a_u8_s8(q5_k_pack4_unsigned(block, subblock, lane4), x_pack, 0);
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
  const int dot_u = gguf_k_dp4a_u8_s8(q6_k_pack4_unsigned(block, group32, lane4), x_pack, 0);
  const int dot = dot_u - 32 * q8_sum;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = fp16_bytes_to_float(block + 208);
  return xd * d * static_cast<float>(scales[scale_index]) * static_cast<float>(dot);
}

__device__ inline float reduce_block_sum(float acc) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) {
    wave_sums[wave] = acc;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      total += wave_sums[i];
    }
  }
  return total;
}

#define GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
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

__device__ inline int gguf_k_row_bytes(int qtype, int in_features) {
  if (qtype == 8) {
    return (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  }
  if (qtype == 5) {
    return (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  }
  return (in_features / QK_K) * Q6_K_BLOCK_BYTES;
}

template <typename scalar_t, typename out_t, int COL_TILE, int ROW_TILE>
__launch_bounds__(128, (COL_TILE * ROW_TILE <= 32 ? 2 : 1))
__global__ void gguf_q8_0_exact_prefill_tiled_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_TILE == 16 || COL_TILE * ROW_TILE == 32 ||
          COL_TILE * ROW_TILE == 64,
      "exact Q8 prefill supports the bounded 16/32/64-accumulator candidates");
  const int out_col_base = static_cast<int>(blockIdx.x) * COL_TILE;
  const int row_base = static_cast<int>(blockIdx.y) * ROW_TILE;
  const int tid = threadIdx.x;
  if (out_col_base >= out_features || row_base >= rows) return;

  const int row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  const uint8_t* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] = qweight + static_cast<int64_t>(out_col_base + col) * row_bytes;
  }

  const scalar_t* x_rows[ROW_TILE];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
    const int safe_row = min(row_base + row, rows - 1);
    x_rows[row] = x + static_cast<int64_t>(safe_row) * in_features;
  }

  float acc[ROW_TILE][COL_TILE];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] = 0.0f;
    }
  }

  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = q8_0_weight(weight_rows[col], k);
    }
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
      const float xv = scalar_to_float<scalar_t>(x_rows[row][k]);
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        acc[row][col] += xv * weights[col];
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        acc[row][col] += __shfl_down(acc[row][col], offset);
      }
    }
  }

  __shared__ float wave_sums[4][ROW_TILE][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row][col] = acc[row][col];
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
      if (row_base + row >= rows) continue;
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        float total = 0.0f;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
          total += wave_sums[group][row][col];
        }
        store_output<out_t>(
            out,
            static_cast<int64_t>(row_base + row) * out_features + out_col_base + col,
            total);
      }
    }
  }
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_prefill_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ x_row =
      x + static_cast<int64_t>(row) * in_features;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    float wv;
    if constexpr (qtype == 8) {
      wv = q8_0_weight(weight_row, k);
    } else if constexpr (qtype == 5) {
      wv = q5_k_weight(weight_row, k);
    } else {
      wv = q6_k_weight(weight_row, k);
    }
    acc += xv * wv;
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    store_output<out_t>(out, static_cast<int64_t>(row) * out_features + out_col, total);
  }
}

// Small-B weight-load-amortized variant of gguf_k_prefill_out_kernel.  One
// block per output column, ROW_TILE row accumulators, each weight element
// dequantized exactly once and reused across all rows.  See the Q4_K
// gguf_q4_k_prefill_out_rowtile_kernel for the rationale (per-row reload vs
// amortized HBM weight streaming at the verifier's small B).  Bit-identical to
// gguf_k_prefill_out_kernel per row (same k order, same wave/cross-wave sums).
template <typename scalar_t, typename out_t, int qtype, int ROW_TILE>
__launch_bounds__(256, 2)
__global__ void gguf_k_prefill_out_rowtile_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  if (out_col >= out_features) return;

  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(out_col) * row_bytes;

  float acc[ROW_TILE];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r) {
    acc[r] = 0.0f;
  }

  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    float wv;
    if constexpr (qtype == 8) {
      wv = q8_0_weight(weight_row, k);
    } else if constexpr (qtype == 5) {
      wv = q5_k_weight(weight_row, k);
    } else {
      wv = q6_k_weight(weight_row, k);
    }
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(r) * in_features;
      acc[r] = fmaf(scalar_to_float<scalar_t>(x_row[k]), wv, acc[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset);
    }
  }
  __shared__ float red[8][ROW_TILE];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      red[wave][r] = acc[r];
    }
  }
  __syncthreads();
  if (threadIdx.x < ROW_TILE) {
    const int r = threadIdx.x;
    const int waves = (blockDim.x + 31) >> 5;
    float total = 0.0f;
    for (int w = 0; w < waves; ++w) {
      total += red[w][r];
    }
    store_output<out_t>(out, static_cast<int64_t>(r) * out_features + out_col, total);
  }
}

// Large-prefill sibling of the exact small-B rowtile above. Grid Y owns fixed
// ROW_BATCH slabs, so every encoded weight value is reconstructed once and
// reused across four/eight/sixteen/thirty-two prompt rows. Each valid row
// keeps the scalar kernel's thread-local K order and wave/cross-wave reduction
// tree exactly.
template <typename scalar_t, typename out_t, int qtype, int ROW_BATCH>
__launch_bounds__(256, 2)
__global__ void gguf_k_prefill_out_rowbatch_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row_base = blockIdx.y * ROW_BATCH;
  if (out_col >= out_features || row_base >= rows) return;

  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(out_col) * row_bytes;

  float acc[ROW_BATCH];
#pragma unroll
  for (int r = 0; r < ROW_BATCH; ++r) {
    acc[r] = 0.0f;
  }

  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    float wv;
    if constexpr (qtype == 8) {
      wv = q8_0_weight(weight_row, k);
    } else if constexpr (qtype == 5) {
      wv = q5_k_weight(weight_row, k);
    } else {
      wv = q6_k_weight(weight_row, k);
    }
#pragma unroll
    for (int r = 0; r < ROW_BATCH; ++r) {
      const int row = row_base + r;
      if (row < rows) {
        const scalar_t* __restrict__ x_row =
            x + static_cast<int64_t>(row) * in_features;
        acc[r] = fmaf(scalar_to_float<scalar_t>(x_row[k]), wv, acc[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_BATCH; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down(acc[r], offset);
    }
  }
  __shared__ float red[8][ROW_BATCH];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_BATCH; ++r) {
      red[wave][r] = acc[r];
    }
  }
  __syncthreads();
  if (threadIdx.x < ROW_BATCH) {
    const int r = threadIdx.x;
    const int row = row_base + r;
    if (row < rows) {
      const int waves = (blockDim.x + 31) >> 5;
      float total = 0.0f;
      for (int w = 0; w < waves; ++w) {
        total += red[w][r];
      }
      store_output<out_t>(
          out, static_cast<int64_t>(row) * out_features + out_col, total);
    }
  }
}

// WPF-1T exact output-column screen. Keep the RB32 accumulator product fixed
// at 32 while sharing each converted activation across adjacent output columns.
// Every output retains local128 K ownership, ordered FMAs, wave32 shuffles, and
// serial wave-0..3 accumulation from the retained rowbatch kernel.
template <
    typename scalar_t,
    typename out_t,
    int qtype,
    int COL_TILE,
    int ROW_BATCH>
__launch_bounds__(128, 2)
__global__ void gguf_k_prefill_out_coltile_rowbatch_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32 || COL_TILE * ROW_BATCH == 64,
      "exact coltile screens require 32 or 64 accumulators");
  const int out_col_base = static_cast<int>(blockIdx.x) * COL_TILE;
  const int row_base = static_cast<int>(blockIdx.y) * ROW_BATCH;
  const int tid = threadIdx.x;
  if (out_col_base >= out_features || row_base >= rows) return;

  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const uint8_t* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        qweight + static_cast<int64_t>(out_col_base + col) * row_bytes;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      if constexpr (qtype == 8) {
        weights[col] = q8_0_weight(weight_rows[col], k);
      } else if constexpr (qtype == 5) {
        weights[col] = q5_k_weight(weight_rows[col], k);
      } else {
        weights[col] = q6_k_weight(weight_rows[col], k);
      }
    }
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
      const int row = row_base + row_idx;
      if (row < rows) {
        const scalar_t* __restrict__ x_row =
            x + static_cast<int64_t>(row) * in_features;
        const float xv = scalar_to_float<scalar_t>(x_row[k]);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_idx][col] = fmaf(xv, weights[col], acc[row_idx][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_idx][col] += __shfl_down(acc[row_idx][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_idx][col] = acc[row_idx][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_idx = tid / COL_TILE;
    const int col = tid - row_idx * COL_TILE;
    const int row = row_base + row_idx;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_idx = 0; wave_idx < 4; ++wave_idx) {
        total += wave_sums[wave_idx][row_idx][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// WPF-H7C preserves the raw-Q6 coltile kernel's ordered FMAs, four-wave LDS
// publication, serial wave sum, output layout, and ABI. Only the exact
// shuffle16->8->4->2->1 wave tree changes instruction form.
__device__ inline float h7c_permlanex16_f32(float value) {
  union FloatBits {
    float value;
    uint32_t bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_permlanex16(
      data.bits, data.bits, 0x76543210U, 0xFEDCBA98U, false, false);
  return data.value;
}

template <int DPP_CTRL>
__device__ inline float h7c_dpp_move_f32(float value) {
  union FloatBits {
    float value;
    int bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_mov_dpp(
      data.bits, DPP_CTRL, 0xF, 0xF, true);
  return data.value;
}

__device__ inline float h7c_dpp_add_row_shl1_f32(float value) {
  float result;
  asm volatile(
      "v_add_f32_dpp %0, %1, %1 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1"
      : "=v"(result)
      : "v"(value));
  return result;
}

template <int ROW_BATCH, int COL_TILE>
__device__ inline void h7c_reduce_wave_accumulators_dpp(
    float (&acc)[ROW_BATCH][COL_TILE]) {
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h7c_permlanex16_f32(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h7c_dpp_move_f32<0x108>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h7c_dpp_move_f32<0x104>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h7c_dpp_move_f32<0x102>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] = h7c_dpp_add_row_shl1_f32(acc[row][col]);
    }
  }
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
__launch_bounds__(128, 2)
__global__ void gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32,
      "H7C preserves constant-32-accumulator output tiles");
  const int out_col_base = static_cast<int>(blockIdx.x) * COL_TILE;
  const int row_base = static_cast<int>(blockIdx.y) * ROW_BATCH;
  const int tid = threadIdx.x;
  if (out_col_base >= out_features || row_base >= rows) return;

  const int row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  const uint8_t* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        qweight + static_cast<int64_t>(out_col_base + col) * row_bytes;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = q6_k_weight(weight_rows[col], k);
    }
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
      const int row = row_base + row_idx;
      if (row < rows) {
        const uint16_t* __restrict__ x_row =
            x + static_cast<int64_t>(row) * in_features;
        const float xv = scalar_to_float<uint16_t>(x_row[k]);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_idx][col] = fmaf(xv, weights[col], acc[row_idx][col]);
        }
      }
    }
  }

  h7c_reduce_wave_accumulators_dpp<ROW_BATCH, COL_TILE>(acc);
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_idx][col] = acc[row_idx][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_idx = tid / COL_TILE;
    const int col = tid - row_idx * COL_TILE;
    const int row = row_base + row_idx;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_idx = 0; wave_idx < 4; ++wave_idx) {
        total += wave_sums[wave_idx][row_idx][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// WPF-H7I: H7C-equivalent raw-Q6 math for exactly-full natural-M512
// row groups. The outer group guard and final store guard remain so this
// sibling differs only by removing the inner live-row compute predicate.
template <typename out_t, int COL_TILE, int ROW_BATCH>
__launch_bounds__(128, 2)
__global__ void gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_full_group_compute_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32,
      "H7I preserves H7C's constant-32-accumulator output tiles");
  const int out_col_base = static_cast<int>(blockIdx.x) * COL_TILE;
  const int row_base = static_cast<int>(blockIdx.y) * ROW_BATCH;
  const int tid = threadIdx.x;
  if (out_col_base >= out_features || row_base >= rows) return;

  const int row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  const uint8_t* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        qweight + static_cast<int64_t>(out_col_base + col) * row_bytes;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = q6_k_weight(weight_rows[col], k);
    }
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
      const int row = row_base + row_idx;
      const uint16_t* __restrict__ x_row =
          x + static_cast<int64_t>(row) * in_features;
      const float xv = scalar_to_float<uint16_t>(x_row[k]);
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        acc[row_idx][col] = fmaf(xv, weights[col], acc[row_idx][col]);
      }
    }
  }

  h7c_reduce_wave_accumulators_dpp<ROW_BATCH, COL_TILE>(acc);
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_idx][col] = acc[row_idx][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_idx = tid / COL_TILE;
    const int col = tid - row_idx * COL_TILE;
    const int row = row_base + row_idx;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_idx = 0; wave_idx < 4; ++wave_idx) {
        total += wave_sums[wave_idx][row_idx][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_selected_prefill_out_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row =
      qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row_ptr[k]);
    float wv;
    if constexpr (qtype == 8) {
      wv = q8_0_weight(weight_row, k);
    } else if constexpr (qtype == 5) {
      wv = q5_k_weight(weight_row, k);
    } else {
      wv = q6_k_weight(weight_row, k);
    }
    acc += xv * wv;
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    store_output<out_t>(out, static_cast<int64_t>(row) * out_features + out_col, total);
  }
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_selected_silu_prefill_out_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  int row_bytes = 0;
  if constexpr (qtype == 8) {
    row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    row_bytes = (in_features / QK_K) * Q5_K_BLOCK_BYTES;
  } else {
    row_bytes = (in_features / QK_K) * Q6_K_BLOCK_BYTES;
  }
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row =
      qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ gate_row =
      gate + static_cast<int64_t>(x_row) * in_features;
  const scalar_t* __restrict__ up_row =
      up + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float gv = scalar_to_float<scalar_t>(gate_row[k]);
    const float uv = scalar_to_float<scalar_t>(up_row[k]);
    const float silu = gv / (1.0f + expf(-gv));
    const float xv = round_to_bf16_float(silu * uv);
    float wv;
    if constexpr (qtype == 8) {
      wv = q8_0_weight(weight_row, k);
    } else if constexpr (qtype == 5) {
      wv = q5_k_weight(weight_row, k);
    } else {
      wv = q6_k_weight(weight_row, k);
    }
    acc += xv * wv;
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    store_output<out_t>(out, static_cast<int64_t>(row) * out_features + out_col, total);
  }
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_pack8_prefill_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features,
    int out_packed) {
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;

  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(qtype, in_features);
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
  const uint8_t* __restrict__ w0 = qweight + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 = qweight + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 = qweight + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 = qweight + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 = qweight + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 = qweight + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 = qweight + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 = qweight + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    if constexpr (qtype == 8) {
      acc0 += xv * q8_0_weight(w0, k);
      acc1 += xv * q8_0_weight(w1, k);
      acc2 += xv * q8_0_weight(w2, k);
      acc3 += xv * q8_0_weight(w3, k);
      acc4 += xv * q8_0_weight(w4, k);
      acc5 += xv * q8_0_weight(w5, k);
      acc6 += xv * q8_0_weight(w6, k);
      acc7 += xv * q8_0_weight(w7, k);
    } else if constexpr (qtype == 5) {
      acc0 += xv * q5_k_weight(w0, k);
      acc1 += xv * q5_k_weight(w1, k);
      acc2 += xv * q5_k_weight(w2, k);
      acc3 += xv * q5_k_weight(w3, k);
      acc4 += xv * q5_k_weight(w4, k);
      acc5 += xv * q5_k_weight(w5, k);
      acc6 += xv * q5_k_weight(w6, k);
      acc7 += xv * q5_k_weight(w7, k);
    } else {
      acc0 += xv * q6_k_weight(w0, k);
      acc1 += xv * q6_k_weight(w1, k);
      acc2 += xv * q6_k_weight(w2, k);
      acc3 += xv * q6_k_weight(w3, k);
      acc4 += xv * q6_k_weight(w4, k);
      acc5 += xv * q6_k_weight(w5, k);
      acc6 += xv * q6_k_weight(w6, k);
      acc7 += xv * q6_k_weight(w7, k);
    }
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
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

// Decode-only raw-Q6 projection pair. One flattened output-pack range covers
// equal-width K/V and Laguna layer-47's unequal query/gate without idle blocks;
// every workgroup preserves the established pack8 singleton's K traversal,
// reduction tree, and F32 store.
template <typename scalar_t, typename out_t>
__launch_bounds__(256, 2)
__global__ void gguf_q6_k_pair_pack8_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features_a,
    int out_features_b) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  const int packs_a = out_features_a >> 3;
  const int flat_pack = blockIdx.x;
  const bool projection_b = flat_pack >= packs_a;
  const int out_pack = projection_b ? flat_pack - packs_a : flat_pack;
  const int out_features = projection_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = projection_b ? qweight_b : qweight_a;
  out_t* __restrict__ out = projection_b ? out_b : out_a;
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(6, in_features);
  const scalar_t* __restrict__ x_row =
      x + static_cast<int64_t>(row) * in_features;
  const uint8_t* __restrict__ w0 =
      qweight + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 =
      qweight + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 =
      qweight + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 =
      qweight + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 =
      qweight + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 =
      qweight + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 =
      qweight + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 =
      qweight + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    acc0 += xv * q6_k_weight(w0, k);
    acc1 += xv * q6_k_weight(w1, k);
    acc2 += xv * q6_k_weight(w2, k);
    acc3 += xv * q6_k_weight(w3, k);
    acc4 += xv * q6_k_weight(w4, k);
    acc5 += xv * q6_k_weight(w5, k);
    acc6 += xv * q6_k_weight(w6, k);
    acc7 += xv * q6_k_weight(w7, k);
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
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

// Exact Q6_K sibling for mixed projection blocks. Cooperative metadata staging
// replaces each lane's repeated d/scale loads while preserving the retained
// local128 [t, t+128] accumulation and reduction order byte-for-byte.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q6_k_pack8_fixed_meta_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int out_pack,
    int in_features,
    float* __restrict__ block_scales) {
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_rows[8];
#pragma unroll
  for (int channel = 0; channel < 8; ++channel) {
    weight_rows[channel] =
        qweight + static_cast<int64_t>(out_col + channel) * row_bytes;
  }

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const int meta_channel = tid >> 4;
    const int scale_index = tid & 15;
    const uint8_t* __restrict__ meta_block =
        weight_rows[meta_channel] +
        static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const float d = fp16_bytes_to_float(meta_block + 208);
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(meta_block + 192);
    block_scales[tid] = d * static_cast<float>(scales[scale_index]);
    __syncthreads();

    const int k_base = block_idx * QK_K;
    const uint8_t* __restrict__ ql0 =
        weight_rows[0] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql1 =
        weight_rows[1] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql2 =
        weight_rows[2] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql3 =
        weight_rows[3] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql4 =
        weight_rows[4] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql5 =
        weight_rows[5] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql6 =
        weight_rows[6] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql7 =
        weight_rows[7] + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ qh0 = ql0 + 128;
    const uint8_t* __restrict__ qh1 = ql1 + 128;
    const uint8_t* __restrict__ qh2 = ql2 + 128;
    const uint8_t* __restrict__ qh3 = ql3 + 128;
    const uint8_t* __restrict__ qh4 = ql4 + 128;
    const uint8_t* __restrict__ qh5 = ql5 + 128;
    const uint8_t* __restrict__ qh6 = ql6 + 128;
    const uint8_t* __restrict__ qh7 = ql7 + 128;

    for (int within = tid; within < QK_K; within += blockDim.x) {
      const int group32 = within >> 5;
      const int lane = within & 31;
      const int scale = within >> 4;
      const int base64 = group32 >= 4 ? 64 : 0;
      const bool low_nibble = (group32 & 2) == 0;
      const int ql_index = base64 + (group32 & 1) * 32 + lane;
      const int qh_index = (group32 >= 4 ? 32 : 0) + lane;
      const int qh_shift = 2 * (group32 & 3);
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + within]);

      const uint8_t lb0 = ql0[ql_index];
      const uint8_t lb1 = ql1[ql_index];
      const uint8_t lb2 = ql2[ql_index];
      const uint8_t lb3 = ql3[ql_index];
      const uint8_t lb4 = ql4[ql_index];
      const uint8_t lb5 = ql5[ql_index];
      const uint8_t lb6 = ql6[ql_index];
      const uint8_t lb7 = ql7[ql_index];
      const uint8_t l0 = low_nibble ? (lb0 & 0x0Fu) : (lb0 >> 4);
      const uint8_t l1 = low_nibble ? (lb1 & 0x0Fu) : (lb1 >> 4);
      const uint8_t l2 = low_nibble ? (lb2 & 0x0Fu) : (lb2 >> 4);
      const uint8_t l3 = low_nibble ? (lb3 & 0x0Fu) : (lb3 >> 4);
      const uint8_t l4 = low_nibble ? (lb4 & 0x0Fu) : (lb4 >> 4);
      const uint8_t l5 = low_nibble ? (lb5 & 0x0Fu) : (lb5 >> 4);
      const uint8_t l6 = low_nibble ? (lb6 & 0x0Fu) : (lb6 >> 4);
      const uint8_t l7 = low_nibble ? (lb7 & 0x0Fu) : (lb7 >> 4);
      const uint8_t h0 = (qh0[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h1 = (qh1[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h2 = (qh2[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h3 = (qh3[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h4 = (qh4[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h5 = (qh5[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h6 = (qh6[qh_index] >> qh_shift) & 0x03u;
      const uint8_t h7 = (qh7[qh_index] >> qh_shift) & 0x03u;
      const float q0 =
          static_cast<float>(static_cast<int>(l0 | (h0 << 4)) - 32);
      const float q1 =
          static_cast<float>(static_cast<int>(l1 | (h1 << 4)) - 32);
      const float q2 =
          static_cast<float>(static_cast<int>(l2 | (h2 << 4)) - 32);
      const float q3 =
          static_cast<float>(static_cast<int>(l3 | (h3 << 4)) - 32);
      const float q4 =
          static_cast<float>(static_cast<int>(l4 | (h4 << 4)) - 32);
      const float q5 =
          static_cast<float>(static_cast<int>(l5 | (h5 << 4)) - 32);
      const float q6 =
          static_cast<float>(static_cast<int>(l6 | (h6 << 4)) - 32);
      const float q7 =
          static_cast<float>(static_cast<int>(l7 | (h7 << 4)) - 32);

      acc0 = fmaf(xv, block_scales[0 * 16 + scale] * q0, acc0);
      acc1 = fmaf(xv, block_scales[1 * 16 + scale] * q1, acc1);
      acc2 = fmaf(xv, block_scales[2 * 16 + scale] * q2, acc2);
      acc3 = fmaf(xv, block_scales[3 * 16 + scale] * q3, acc3);
      acc4 = fmaf(xv, block_scales[4 * 16 + scale] * q4, acc4);
      acc5 = fmaf(xv, block_scales[5 * 16 + scale] * q5, acc5);
      acc6 = fmaf(xv, block_scales[6 * 16 + scale] * q6, acc6);
      acc7 = fmaf(xv, block_scales[7 * 16 + scale] * q7, acc7);
    }
    __syncthreads();
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    store_output<out_t>(out, out_col + 0, r0);
    store_output<out_t>(out, out_col + 1, r1);
    store_output<out_t>(out, out_col + 2, r2);
    store_output<out_t>(out, out_col + 3, r3);
    store_output<out_t>(out, out_col + 4, r4);
    store_output<out_t>(out, out_col + 5, r5);
    store_output<out_t>(out, out_col + 6, r6);
    store_output<out_t>(out, out_col + 7, r7);
  }
}

// Decode-specialized raw Q5_K pack8. The legacy generic pack8 body decodes
// d/dmin and the packed 6-bit scale/min fields independently for every K
// element and output channel. This schedule hoists those 8x8 coefficients once
// per Q5_K superblock while preserving each thread's K order and the existing
// wave/cross-wave reduction exactly.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q5_k_pack8_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int row,
    int out_pack,
    int in_features,
    int out_features,
    float* __restrict__ block_scales,
    float* __restrict__ block_mins) {
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_rows[8];
#pragma unroll
  for (int channel = 0; channel < 8; ++channel) {
    weight_rows[channel] = qweight + static_cast<int64_t>(out_col + channel) * row_bytes;
  }

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    if (tid < 64) {
      const int channel = tid >> 3;
      const int subblock = tid & 7;
      const uint8_t* __restrict__ block =
          weight_rows[channel] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
      const uint8_t* __restrict__ scales = block + 4;
      block_scales[tid] = fp16_bytes_to_float(block) *
                          static_cast<float>(q4_k_scale(scales, subblock));
      block_mins[tid] = fp16_bytes_to_float(block + 2) *
                        static_cast<float>(q4_k_min(scales, subblock));
    }
    __syncthreads();

    const int k_base = block_idx * QK_K;
    const uint8_t* __restrict__ b0 = weight_rows[0] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b1 = weight_rows[1] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b2 = weight_rows[2] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b3 = weight_rows[3] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b4 = weight_rows[4] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b5 = weight_rows[5] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b6 = weight_rows[6] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ b7 = weight_rows[7] + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    for (int k = tid; k < QK_K; k += blockDim.x) {
      const int subblock = k >> 5;
      const int lane = k & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k]);
      acc0 += xv * (block_scales[0 * 8 + subblock] * static_cast<float>(q5_k_quant(b0, subblock, lane)) - block_mins[0 * 8 + subblock]);
      acc1 += xv * (block_scales[1 * 8 + subblock] * static_cast<float>(q5_k_quant(b1, subblock, lane)) - block_mins[1 * 8 + subblock]);
      acc2 += xv * (block_scales[2 * 8 + subblock] * static_cast<float>(q5_k_quant(b2, subblock, lane)) - block_mins[2 * 8 + subblock]);
      acc3 += xv * (block_scales[3 * 8 + subblock] * static_cast<float>(q5_k_quant(b3, subblock, lane)) - block_mins[3 * 8 + subblock]);
      acc4 += xv * (block_scales[4 * 8 + subblock] * static_cast<float>(q5_k_quant(b4, subblock, lane)) - block_mins[4 * 8 + subblock]);
      acc5 += xv * (block_scales[5 * 8 + subblock] * static_cast<float>(q5_k_quant(b5, subblock, lane)) - block_mins[5 * 8 + subblock]);
      acc6 += xv * (block_scales[6 * 8 + subblock] * static_cast<float>(q5_k_quant(b6, subblock, lane)) - block_mins[6 * 8 + subblock]);
      acc7 += xv * (block_scales[7 * 8 + subblock] * static_cast<float>(q5_k_quant(b7, subblock, lane)) - block_mins[7 * 8 + subblock]);
    }
    __syncthreads();
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
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

template <typename scalar_t, typename out_t>
__launch_bounds__(128, 4)
__global__ void gguf_q5_k_pack8_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  __shared__ float block_scales[8 * 8];
  __shared__ float block_mins[8 * 8];
  gguf_q5_k_pack8_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      blockIdx.x,
      in_features,
      out_features,
      block_scales,
      block_mins);
}

// Decode-specialized raw Q5_K wave32x2. One native wave owns two adjacent
// outputs while each physical lane reconstructs the four logical local128
// threads used by gguf_q5_k_pack8_decode_out_block. The four accumulator
// groups therefore preserve every baseline [t, t+128] K sequence and the
// final logical-wave 0..3 addition order without LDS or block barriers.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q5_k_wave32x2_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int row,
    int out_pair,
    int in_features,
    int out_features) {
  const int lane = threadIdx.x;
  const int out_col = out_pair << 1;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row0 =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_row1 = weight_row0 + row_bytes;

  float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
  float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block0 =
        weight_row0 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ block1 =
        weight_row1 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;

    float lane_scale = 0.0f;
    float lane_min = 0.0f;
    if (lane < 16) {
      const bool second_output = lane >= 8;
      const int subblock = lane & 7;
      const uint8_t* __restrict__ block = second_output ? block1 : block0;
      const uint8_t* __restrict__ scales = block + 4;
      lane_scale = fp16_bytes_to_float(block) *
                   static_cast<float>(q4_k_scale(scales, subblock));
      lane_min = fp16_bytes_to_float(block + 2) *
                 static_cast<float>(q4_k_min(scales, subblock));
    }

    const int k_base = block_idx * QK_K;
#define GGUF_Q5_K_WAVE32X2_GROUP(GROUP, ACC0, ACC1) \
    { \
      constexpr int subblock0 = GROUP; \
      constexpr int subblock1 = GROUP + 4; \
      const float scale0_0 = __shfl(lane_scale, subblock0, 32); \
      const float min0_0 = __shfl(lane_min, subblock0, 32); \
      const float scale0_1 = __shfl(lane_scale, 8 + subblock0, 32); \
      const float min0_1 = __shfl(lane_min, 8 + subblock0, 32); \
      const int k0 = lane + 32 * GROUP; \
      const float x0 = scalar_to_float<scalar_t>(x_row[k_base + k0]); \
      ACC0 += x0 * (scale0_0 * static_cast<float>(q5_k_quant(block0, subblock0, lane)) - min0_0); \
      ACC1 += x0 * (scale0_1 * static_cast<float>(q5_k_quant(block1, subblock0, lane)) - min0_1); \
      const float scale1_0 = __shfl(lane_scale, subblock1, 32); \
      const float min1_0 = __shfl(lane_min, subblock1, 32); \
      const float scale1_1 = __shfl(lane_scale, 8 + subblock1, 32); \
      const float min1_1 = __shfl(lane_min, 8 + subblock1, 32); \
      const int k1 = k0 + 128; \
      const float x1 = scalar_to_float<scalar_t>(x_row[k_base + k1]); \
      ACC0 += x1 * (scale1_0 * static_cast<float>(q5_k_quant(block0, subblock1, lane)) - min1_0); \
      ACC1 += x1 * (scale1_1 * static_cast<float>(q5_k_quant(block1, subblock1, lane)) - min1_1); \
    }
    GGUF_Q5_K_WAVE32X2_GROUP(0, acc00, acc10);
    GGUF_Q5_K_WAVE32X2_GROUP(1, acc01, acc11);
    GGUF_Q5_K_WAVE32X2_GROUP(2, acc02, acc12);
    GGUF_Q5_K_WAVE32X2_GROUP(3, acc03, acc13);
#undef GGUF_Q5_K_WAVE32X2_GROUP
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc00 += __shfl_down(acc00, offset);
    acc01 += __shfl_down(acc01, offset);
    acc02 += __shfl_down(acc02, offset);
    acc03 += __shfl_down(acc03, offset);
    acc10 += __shfl_down(acc10, offset);
    acc11 += __shfl_down(acc11, offset);
    acc12 += __shfl_down(acc12, offset);
    acc13 += __shfl_down(acc13, offset);
  }
  if (lane == 0) {
    float total0 = 0.0f;
    total0 += acc00;
    total0 += acc01;
    total0 += acc02;
    total0 += acc03;
    float total1 = 0.0f;
    total1 += acc10;
    total1 += acc11;
    total1 += acc12;
    total1 += acc13;
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
    store_output<out_t>(out, out_base, total0);
    store_output<out_t>(out, out_base + 1, total1);
  }
}

__device__ __forceinline__ uint32_t gguf_q5_k_uniform_u32(
    const uint8_t* __restrict__ ptr) {
  const uint32_t value = *reinterpret_cast<const uint32_t*>(ptr);
  return static_cast<uint32_t>(
      __builtin_amdgcn_readfirstlane(static_cast<int>(value)));
}

__device__ __forceinline__ float gguf_q5_k_fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

template <int INDEX>
__device__ __forceinline__ uint8_t gguf_q5_k_fixed_meta_byte(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2) {
  static_assert(INDEX >= 0 && INDEX < 12, "Q5_K metadata byte index");
  const uint32_t word = INDEX < 4 ? word0 : (INDEX < 8 ? word1 : word2);
  return static_cast<uint8_t>((word >> (8 * (INDEX & 3))) & 0xFFu);
}

template <int SUBBLOCK>
__device__ __forceinline__ uint8_t gguf_q5_k_fixed_scale(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2) {
  static_assert(SUBBLOCK >= 0 && SUBBLOCK < 8, "Q5_K scale subblock");
  if constexpr (SUBBLOCK < 4) {
    return gguf_q5_k_fixed_meta_byte<SUBBLOCK>(word0, word1, word2) & 0x3Fu;
  } else {
    constexpr int idx = SUBBLOCK - 4;
    return (gguf_q5_k_fixed_meta_byte<8 + idx>(word0, word1, word2) & 0x0Fu) |
           ((gguf_q5_k_fixed_meta_byte<idx>(word0, word1, word2) >> 2) & 0x30u);
  }
}

template <int SUBBLOCK>
__device__ __forceinline__ uint8_t gguf_q5_k_fixed_min(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2) {
  static_assert(SUBBLOCK >= 0 && SUBBLOCK < 8, "Q5_K min subblock");
  if constexpr (SUBBLOCK < 4) {
    return gguf_q5_k_fixed_meta_byte<4 + SUBBLOCK>(word0, word1, word2) & 0x3Fu;
  } else {
    constexpr int idx = SUBBLOCK - 4;
    return (gguf_q5_k_fixed_meta_byte<8 + idx>(word0, word1, word2) >> 4) |
           ((gguf_q5_k_fixed_meta_byte<4 + idx>(word0, word1, word2) >> 2) & 0x30u);
  }
}

// Exact D12 sibling that reads each fixed-address d/dmin/scale/min metadata
// word wave-uniformly. The eight accumulator chains, their K order, the four
// independent shuffle reductions, and the final group additions are unchanged.
// This candidate removes only the retained lane-distributed coefficient
// publication exchanges; raw weights and c=1 output-pair ownership are intact.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int row,
    int out_pair,
    int in_features,
    int out_features) {
  const int lane = threadIdx.x & 31;
  const int out_col = out_pair << 1;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row0 =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_row1 = weight_row0 + row_bytes;

  float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
  float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block0 =
        weight_row0 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ block1 =
        weight_row1 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;

    const uint32_t meta0_head = gguf_q5_k_uniform_u32(block0);
    const uint32_t meta0_word0 = gguf_q5_k_uniform_u32(block0 + 4);
    const uint32_t meta0_word1 = gguf_q5_k_uniform_u32(block0 + 8);
    const uint32_t meta0_word2 = gguf_q5_k_uniform_u32(block0 + 12);
    const uint32_t meta1_head = gguf_q5_k_uniform_u32(block1);
    const uint32_t meta1_word0 = gguf_q5_k_uniform_u32(block1 + 4);
    const uint32_t meta1_word1 = gguf_q5_k_uniform_u32(block1 + 8);
    const uint32_t meta1_word2 = gguf_q5_k_uniform_u32(block1 + 12);
    const float d0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta0_head));
    const float dmin0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta0_head >> 16));
    const float d1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta1_head));
    const float dmin1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta1_head >> 16));

    const int k_base = block_idx * QK_K;
#define GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP(GROUP, ACC0, ACC1) \
    { \
      constexpr int subblock0 = GROUP; \
      constexpr int subblock1 = GROUP + 4; \
      const float scale0_0 = d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(meta0_word0, meta0_word1, meta0_word2)); \
      const float min0_0 = dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(meta0_word0, meta0_word1, meta0_word2)); \
      const float scale0_1 = d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(meta1_word0, meta1_word1, meta1_word2)); \
      const float min0_1 = dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(meta1_word0, meta1_word1, meta1_word2)); \
      const int k0 = lane + 32 * GROUP; \
      const float x0 = scalar_to_float<scalar_t>(x_row[k_base + k0]); \
      ACC0 += x0 * (scale0_0 * static_cast<float>(q5_k_quant(block0, subblock0, lane)) - min0_0); \
      ACC1 += x0 * (scale0_1 * static_cast<float>(q5_k_quant(block1, subblock0, lane)) - min0_1); \
      const float scale1_0 = d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(meta0_word0, meta0_word1, meta0_word2)); \
      const float min1_0 = dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(meta0_word0, meta0_word1, meta0_word2)); \
      const float scale1_1 = d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(meta1_word0, meta1_word1, meta1_word2)); \
      const float min1_1 = dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(meta1_word0, meta1_word1, meta1_word2)); \
      const int k1 = k0 + 128; \
      const float x1 = scalar_to_float<scalar_t>(x_row[k_base + k1]); \
      ACC0 += x1 * (scale1_0 * static_cast<float>(q5_k_quant(block0, subblock1, lane)) - min1_0); \
      ACC1 += x1 * (scale1_1 * static_cast<float>(q5_k_quant(block1, subblock1, lane)) - min1_1); \
    }
    GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP(0, acc00, acc10);
    GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP(1, acc01, acc11);
    GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP(2, acc02, acc12);
    GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP(3, acc03, acc13);
#undef GGUF_Q5_K_WAVE32X2_FIXED_META_GROUP
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc00 += __shfl_down(acc00, offset);
    acc01 += __shfl_down(acc01, offset);
    acc02 += __shfl_down(acc02, offset);
    acc03 += __shfl_down(acc03, offset);
    acc10 += __shfl_down(acc10, offset);
    acc11 += __shfl_down(acc11, offset);
    acc12 += __shfl_down(acc12, offset);
    acc13 += __shfl_down(acc13, offset);
  }
  if (lane == 0) {
    float total0 = 0.0f;
    total0 += acc00;
    total0 += acc01;
    total0 += acc02;
    total0 += acc03;
    float total1 = 0.0f;
    total1 += acc10;
    total1 += acc11;
    total1 += acc12;
    total1 += acc13;
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
    store_output<out_t>(out, out_base, total0);
    store_output<out_t>(out, out_base + 1, total1);
  }
}

// Exact paired-output SWAR sibling. The two output rows keep their original
// independent FP32 arithmetic while Q5 nibble/high-bit reconstruction shares
// integer operations across the low/high bytes of one VGPR.
__device__ __forceinline__ uint32_t gguf_q5_k_swar_pair_u8(
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    int offset) {
  return static_cast<uint32_t>(row0[offset]) |
         (static_cast<uint32_t>(row1[offset]) << 8);
}

template <int SUBBLOCK>
__device__ __forceinline__ uint32_t gguf_q5_k_swar_quant_pair(
    uint32_t qh_pair,
    uint32_t packed_pair) {
  static_assert(SUBBLOCK >= 0 && SUBBLOCK < 8, "Q5_K SWAR subblock");
  const uint32_t low = (SUBBLOCK & 1)
      ? ((packed_pair >> 4) & 0x0F0Fu)
      : (packed_pair & 0x0F0Fu);
  const uint32_t high = ((qh_pair >> SUBBLOCK) & 0x0101u) << 4;
  return low | high;
}

template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int row,
    int out_pair,
    int in_features,
    int out_features) {
  const int lane = threadIdx.x & 31;
  const int out_col = out_pair << 1;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row0 =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_row1 = weight_row0 + row_bytes;

  float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
  float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block0 =
        weight_row0 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ block1 =
        weight_row1 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;

    const uint32_t meta0_head = gguf_q5_k_uniform_u32(block0);
    const uint32_t meta0_word0 = gguf_q5_k_uniform_u32(block0 + 4);
    const uint32_t meta0_word1 = gguf_q5_k_uniform_u32(block0 + 8);
    const uint32_t meta0_word2 = gguf_q5_k_uniform_u32(block0 + 12);
    const uint32_t meta1_head = gguf_q5_k_uniform_u32(block1);
    const uint32_t meta1_word0 = gguf_q5_k_uniform_u32(block1 + 4);
    const uint32_t meta1_word1 = gguf_q5_k_uniform_u32(block1 + 8);
    const uint32_t meta1_word2 = gguf_q5_k_uniform_u32(block1 + 12);
    const float d0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta0_head));
    const float dmin0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta0_head >> 16));
    const float d1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta1_head));
    const float dmin1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(meta1_head >> 16));
    const uint32_t qh_pair = gguf_q5_k_swar_pair_u8(block0, block1, 16 + lane);
    const uint32_t packed_pair01 =
        gguf_q5_k_swar_pair_u8(block0, block1, 48 + lane);
    const uint32_t packed_pair23 =
        gguf_q5_k_swar_pair_u8(block0, block1, 80 + lane);
    const uint32_t packed_pair45 =
        gguf_q5_k_swar_pair_u8(block0, block1, 112 + lane);
    const uint32_t packed_pair67 =
        gguf_q5_k_swar_pair_u8(block0, block1, 144 + lane);

    const int k_base = block_idx * QK_K;
#define GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP(GROUP, ACC0, ACC1) \
    { \
      constexpr int subblock0 = GROUP; \
      constexpr int subblock1 = GROUP + 4; \
      const float scale0_0 = d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(meta0_word0, meta0_word1, meta0_word2)); \
      const float min0_0 = dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(meta0_word0, meta0_word1, meta0_word2)); \
      const float scale0_1 = d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(meta1_word0, meta1_word1, meta1_word2)); \
      const float min0_1 = dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(meta1_word0, meta1_word1, meta1_word2)); \
      const uint32_t quant_pair0 = gguf_q5_k_swar_quant_pair<subblock0>( \
          qh_pair, GROUP < 2 ? packed_pair01 : packed_pair23); \
      const int k0 = lane + 32 * GROUP; \
      const float x0 = scalar_to_float<scalar_t>(x_row[k_base + k0]); \
      ACC0 += x0 * (scale0_0 * static_cast<float>(quant_pair0 & 0xFFu) - min0_0); \
      ACC1 += x0 * (scale0_1 * static_cast<float>((quant_pair0 >> 8) & 0xFFu) - min0_1); \
      const float scale1_0 = d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(meta0_word0, meta0_word1, meta0_word2)); \
      const float min1_0 = dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(meta0_word0, meta0_word1, meta0_word2)); \
      const float scale1_1 = d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(meta1_word0, meta1_word1, meta1_word2)); \
      const float min1_1 = dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(meta1_word0, meta1_word1, meta1_word2)); \
      const uint32_t quant_pair1 = gguf_q5_k_swar_quant_pair<subblock1>( \
          qh_pair, GROUP < 2 ? packed_pair45 : packed_pair67); \
      const int k1 = k0 + 128; \
      const float x1 = scalar_to_float<scalar_t>(x_row[k_base + k1]); \
      ACC0 += x1 * (scale1_0 * static_cast<float>(quant_pair1 & 0xFFu) - min1_0); \
      ACC1 += x1 * (scale1_1 * static_cast<float>((quant_pair1 >> 8) & 0xFFu) - min1_1); \
    }
    GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP(0, acc00, acc10);
    GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP(1, acc01, acc11);
    GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP(2, acc02, acc12);
    GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP(3, acc03, acc13);
#undef GGUF_Q5_K_WAVE32X2_SWAR_PAIR_FIXED_META_GROUP
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc00 += __shfl_down(acc00, offset);
    acc01 += __shfl_down(acc01, offset);
    acc02 += __shfl_down(acc02, offset);
    acc03 += __shfl_down(acc03, offset);
    acc10 += __shfl_down(acc10, offset);
    acc11 += __shfl_down(acc11, offset);
    acc12 += __shfl_down(acc12, offset);
    acc13 += __shfl_down(acc13, offset);
  }
  if (lane == 0) {
    float total0 = 0.0f;
    total0 += acc00;
    total0 += acc01;
    total0 += acc02;
    total0 += acc03;
    float total1 = 0.0f;
    total1 += acc10;
    total1 += acc11;
    total1 += acc12;
    total1 += acc13;
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
    store_output<out_t>(out, out_base, total0);
    store_output<out_t>(out, out_base + 1, total1);
  }
}

// Exact local32 Q6 sibling for the retained local128 fixed-metadata pack. One
// wave owns two adjacent outputs and carries the four original wave partitions
// independently. Each partition replays its original [tid, tid+128] FMA order,
// uses the same wave tree, and lane zero adds wave totals in order 0..3.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q6_k_wave32x2_fixed_meta_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int out_pair,
    int in_features,
    int out_features) {
  const int lane = threadIdx.x & 31;
  const int out_col = out_pair << 1;
  const int blocks_per_row = in_features / QK_K;
  const int row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row0 =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_row1 = weight_row0 + row_bytes;

  float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
  float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block0 =
        weight_row0 + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ block1 =
        weight_row1 + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ meta_block = lane < 16 ? block0 : block1;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(meta_block + 192);
    const float d = fp16_bytes_to_float(meta_block + 208);
    const float lane_scale =
        d * static_cast<float>(scales[lane & 15]);
    const int k_base = block_idx * QK_K;

#define GGUF_Q6_K_WAVE32X2_FIXED_META_PART(PART, ACC0, ACC1) \
    { \
      const int within0 = lane + 32 * PART; \
      const int scale_index0 = within0 >> 4; \
      const float scale0 = __shfl(lane_scale, scale_index0, 32); \
      const float scale1 = __shfl(lane_scale, 16 + scale_index0, 32); \
      const float x0 = scalar_to_float<scalar_t>(x_row[k_base + within0]); \
      ACC0 = fmaf( \
          x0, \
          scale0 * static_cast<float>(q6_k_quant_from_block(block0, within0)), \
          ACC0); \
      ACC1 = fmaf( \
          x0, \
          scale1 * static_cast<float>(q6_k_quant_from_block(block1, within0)), \
          ACC1); \
      const int within1 = within0 + 128; \
      const int scale_index1 = within1 >> 4; \
      const float scale2 = __shfl(lane_scale, scale_index1, 32); \
      const float scale3 = __shfl(lane_scale, 16 + scale_index1, 32); \
      const float x1 = scalar_to_float<scalar_t>(x_row[k_base + within1]); \
      ACC0 = fmaf( \
          x1, \
          scale2 * static_cast<float>(q6_k_quant_from_block(block0, within1)), \
          ACC0); \
      ACC1 = fmaf( \
          x1, \
          scale3 * static_cast<float>(q6_k_quant_from_block(block1, within1)), \
          ACC1); \
    }
    GGUF_Q6_K_WAVE32X2_FIXED_META_PART(0, acc00, acc10);
    GGUF_Q6_K_WAVE32X2_FIXED_META_PART(1, acc01, acc11);
    GGUF_Q6_K_WAVE32X2_FIXED_META_PART(2, acc02, acc12);
    GGUF_Q6_K_WAVE32X2_FIXED_META_PART(3, acc03, acc13);
#undef GGUF_Q6_K_WAVE32X2_FIXED_META_PART
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc00 += __shfl_down(acc00, offset);
    acc01 += __shfl_down(acc01, offset);
    acc02 += __shfl_down(acc02, offset);
    acc03 += __shfl_down(acc03, offset);
    acc10 += __shfl_down(acc10, offset);
    acc11 += __shfl_down(acc11, offset);
    acc12 += __shfl_down(acc12, offset);
    acc13 += __shfl_down(acc13, offset);
  }
  if (lane == 0) {
    float total0 = 0.0f;
    total0 += acc00;
    total0 += acc01;
    total0 += acc02;
    total0 += acc03;
    float total1 = 0.0f;
    total1 += acc10;
    total1 += acc11;
    total1 += acc12;
    total1 += acc13;
    store_output<out_t>(out, out_col, total0);
    store_output<out_t>(out, out_col + 1, total1);
  }
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q6_k_wave32x2_fixed_meta_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int in_features,
    int out_features) {
  gguf_q6_k_wave32x2_fixed_meta_decode_out_block(
      x,
      qweight,
      out,
      blockIdx.x,
      in_features,
      out_features);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_wave32x2_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  gguf_q5_k_wave32x2_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      blockIdx.x,
      in_features,
      out_features);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_unequal_pair_wave32x2_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int in_features,
    int out_features_a,
    int out_features_b) {
  const int pairs_a = out_features_a >> 1;
  const int flat_pair = blockIdx.x;
  const bool projection_b = flat_pair >= pairs_a;
  const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
  gguf_q5_k_wave32x2_decode_out_block(
      x,
      projection_b ? qweight_b : qweight_a,
      projection_b ? out_b : out_a,
      0,
      out_pair,
      in_features,
      projection_b ? out_features_b : out_features_a);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_wave32x2_fixed_meta_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      blockIdx.x,
      in_features,
      out_features);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_unequal_pair_wave32x2_fixed_meta_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int in_features,
    int out_features_a,
    int out_features_b) {
  const int pairs_a = out_features_a >> 1;
  const int flat_pair = blockIdx.x;
  const bool projection_b = flat_pair >= pairs_a;
  const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
  gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
      x,
      projection_b ? qweight_b : qweight_a,
      projection_b ? out_b : out_a,
      0,
      out_pair,
      in_features,
      projection_b ? out_features_b : out_features_a);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      blockIdx.x,
      in_features,
      out_features);
}

template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_k_unequal_pair_wave32x2_swar_pair_fixed_meta_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int in_features,
    int out_features_a,
    int out_features_b) {
  const int pairs_a = out_features_a >> 1;
  const int flat_pair = blockIdx.x;
  const bool projection_b = flat_pair >= pairs_a;
  const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
  gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_block(
      x,
      projection_b ? qweight_b : qweight_a,
      projection_b ? out_b : out_a,
      0,
      out_pair,
      in_features,
      projection_b ? out_features_b : out_features_a);
}

// One c=1 dispatch covers Laguna's mixed raw-Q5/raw-Q6 attention projections.
// Q5 blocks contain four independent wave32 output-pair owners and preserve the
// retained fixed-metadata arithmetic. Q6 blocks retain the generic local128
// pack8 traversal/reduction. The flattened grid is role-agnostic: host-side
// four-axis wrappers provide the production Q5 query/gate ordering.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 2)
__global__ void gguf_q5_q6_mixed_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q5_weight_a,
    const uint8_t* __restrict__ q5_weight_b,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    out_t* __restrict__ q5_out_a,
    out_t* __restrict__ q5_out_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    int in_features,
    int q5_features_a,
    int q5_features_b,
    int q6_features_a,
    int q6_features_b) {
  const int q5_blocks = (q5_features_a + q5_features_b) >> 3;
  const int flat_block = blockIdx.x;
  if (flat_block < q5_blocks) {
    const int wave = threadIdx.x >> 5;
    const int flat_pair = (flat_block << 2) + wave;
    const int pairs_a = q5_features_a >> 1;
    const bool projection_b = flat_pair >= pairs_a;
    const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
    gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
        x,
        projection_b ? q5_weight_b : q5_weight_a,
        projection_b ? q5_out_b : q5_out_a,
        0,
        out_pair,
        in_features,
        projection_b ? q5_features_b : q5_features_a);
    return;
  }

  const int flat_pack = flat_block - q5_blocks;
  const int packs_a = q6_features_a >> 3;
  const bool projection_b = flat_pack >= packs_a;
  const int out_pack = projection_b ? flat_pack - packs_a : flat_pack;
  const int out_features = projection_b ? q6_features_b : q6_features_a;
  const uint8_t* __restrict__ qweight =
      projection_b ? q6_weight_b : q6_weight_a;
  out_t* __restrict__ out = projection_b ? q6_out_b : q6_out_a;
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(6, in_features);
  const uint8_t* __restrict__ w0 =
      qweight + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 =
      qweight + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 =
      qweight + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 =
      qweight + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 =
      qweight + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 =
      qweight + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 =
      qweight + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 =
      qweight + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x[k]);
    acc0 += xv * q6_k_weight(w0, k);
    acc1 += xv * q6_k_weight(w1, k);
    acc2 += xv * q6_k_weight(w2, k);
    acc3 += xv * q6_k_weight(w3, k);
    acc4 += xv * q6_k_weight(w4, k);
    acc5 += xv * q6_k_weight(w5, k);
    acc6 += xv * q6_k_weight(w6, k);
    acc7 += xv * q6_k_weight(w7, k);
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    store_output<out_t>(out, out_col + 0, r0);
    store_output<out_t>(out, out_col + 1, r1);
    store_output<out_t>(out, out_col + 2, r2);
    store_output<out_t>(out, out_col + 3, r3);
    store_output<out_t>(out, out_col + 4, r4);
    store_output<out_t>(out, out_col + 5, r5);
    store_output<out_t>(out, out_col + 6, r6);
    store_output<out_t>(out, out_col + 7, r7);
  }
}

// Default-off sibling of the retained mixed projection. Q5 blocks are
// unchanged; Q6 blocks use the exact cooperative fixed-metadata schedule.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 2)
__global__ void gguf_q5_q6_mixed_q6_fixed_meta_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q5_weight_a,
    const uint8_t* __restrict__ q5_weight_b,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    out_t* __restrict__ q5_out_a,
    out_t* __restrict__ q5_out_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    int in_features,
    int q5_features_a,
    int q5_features_b,
    int q6_features_a,
    int q6_features_b) {
  const int q5_blocks = (q5_features_a + q5_features_b) >> 3;
  const int flat_block = blockIdx.x;
  if (flat_block < q5_blocks) {
    const int wave = threadIdx.x >> 5;
    const int flat_pair = (flat_block << 2) + wave;
    const int pairs_a = q5_features_a >> 1;
    const bool projection_b = flat_pair >= pairs_a;
    const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
    gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
        x,
        projection_b ? q5_weight_b : q5_weight_a,
        projection_b ? q5_out_b : q5_out_a,
        0,
        out_pair,
        in_features,
        projection_b ? q5_features_b : q5_features_a);
    return;
  }

  const int flat_pack = flat_block - q5_blocks;
  const int packs_a = q6_features_a >> 3;
  const bool projection_b = flat_pack >= packs_a;
  const int out_pack = projection_b ? flat_pack - packs_a : flat_pack;
  __shared__ float q6_block_scales[8 * 16];
  gguf_q6_k_pack8_fixed_meta_decode_out_block(
      x,
      projection_b ? q6_weight_b : q6_weight_a,
      projection_b ? q6_out_b : q6_out_a,
      out_pack,
      in_features,
      q6_block_scales);
}

// All-local32 sibling of the retained Q5/Q6 fixed-metadata mixed projection.
// Grid threads and wave count are unchanged: every Q5 pair or Q6 pair is one
// independent workgroup, while the local128 fallback remains separately
// registered for runtime admission and rollback.
template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_q6_mixed_local32_fixed_meta_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q5_weight_a,
    const uint8_t* __restrict__ q5_weight_b,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    out_t* __restrict__ q5_out_a,
    out_t* __restrict__ q5_out_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    int in_features,
    int q5_features_a,
    int q5_features_b,
    int q6_features_a,
    int q6_features_b) {
  const int q5_pairs = (q5_features_a + q5_features_b) >> 1;
  const int flat_pair = blockIdx.x;
  if (flat_pair < q5_pairs) {
    const int pairs_a = q5_features_a >> 1;
    const bool projection_b = flat_pair >= pairs_a;
    const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
    gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
        x,
        projection_b ? q5_weight_b : q5_weight_a,
        projection_b ? q5_out_b : q5_out_a,
        0,
        out_pair,
        in_features,
        projection_b ? q5_features_b : q5_features_a);
    return;
  }

  const int q6_flat_pair = flat_pair - q5_pairs;
  const int q6_pairs_a = q6_features_a >> 1;
  const bool projection_b = q6_flat_pair >= q6_pairs_a;
  const int out_pair = projection_b ? q6_flat_pair - q6_pairs_a : q6_flat_pair;
  gguf_q6_k_wave32x2_fixed_meta_decode_out_block(
      x,
      projection_b ? q6_weight_b : q6_weight_a,
      projection_b ? q6_out_b : q6_out_a,
      out_pair,
      in_features,
      projection_b ? q6_features_b : q6_features_a);
}

// SWAR-Q5 sibling of the retained all-local32 mixed projection. Q6 workgroups,
// flattened pair ownership, grid size, and all output boundaries are unchanged.
template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q5_weight_a,
    const uint8_t* __restrict__ q5_weight_b,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    out_t* __restrict__ q5_out_a,
    out_t* __restrict__ q5_out_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    int in_features,
    int q5_features_a,
    int q5_features_b,
    int q6_features_a,
    int q6_features_b) {
  const int q5_pairs = (q5_features_a + q5_features_b) >> 1;
  const int flat_pair = blockIdx.x;
  if (flat_pair < q5_pairs) {
    const int pairs_a = q5_features_a >> 1;
    const bool projection_b = flat_pair >= pairs_a;
    const int out_pair = projection_b ? flat_pair - pairs_a : flat_pair;
    gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_block(
        x,
        projection_b ? q5_weight_b : q5_weight_a,
        projection_b ? q5_out_b : q5_out_a,
        0,
        out_pair,
        in_features,
        projection_b ? q5_features_b : q5_features_a);
    return;
  }

  const int q6_flat_pair = flat_pair - q5_pairs;
  const int q6_pairs_a = q6_features_a >> 1;
  const bool projection_b = q6_flat_pair >= q6_pairs_a;
  const int out_pair = projection_b ? q6_flat_pair - q6_pairs_a : q6_flat_pair;
  gguf_q6_k_wave32x2_fixed_meta_decode_out_block(
      x,
      projection_b ? q6_weight_b : q6_weight_a,
      projection_b ? q6_out_b : q6_out_a,
      out_pair,
      in_features,
      projection_b ? q6_features_b : q6_features_a);
}

// One local32 wave carries one exact Q5 output pair and one exact Q6 output
// pair, reusing each BF16 activation register. All accumulator chains,
// per-chain FMA order, wave trees, and final 0..3 partition additions match
// the retained independent local32 helpers.
template <typename scalar_t, typename out_t>
__device__ __forceinline__ void gguf_q5_q6_wave32x2_pair_reuse_fixed_meta_decode_out_block(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ q5_weight,
    const uint8_t* __restrict__ q6_weight,
    out_t* __restrict__ q5_out,
    out_t* __restrict__ q6_out,
    int q5_out_pair,
    int q6_out_pair,
    int in_features,
    int q5_out_features,
    int q6_out_features) {
  const int lane = threadIdx.x & 31;
  const int q5_out_col = q5_out_pair << 1;
  const int q6_out_col = q6_out_pair << 1;
  const int blocks_per_row = in_features / QK_K;
  const int q5_row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const int q6_row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
  const uint8_t* __restrict__ q5_row0 =
      q5_weight + static_cast<int64_t>(q5_out_col) * q5_row_bytes;
  const uint8_t* __restrict__ q5_row1 = q5_row0 + q5_row_bytes;
  const uint8_t* __restrict__ q6_row0 =
      q6_weight + static_cast<int64_t>(q6_out_col) * q6_row_bytes;
  const uint8_t* __restrict__ q6_row1 = q6_row0 + q6_row_bytes;

  float q5_acc00 = 0.0f, q5_acc01 = 0.0f, q5_acc02 = 0.0f, q5_acc03 = 0.0f;
  float q5_acc10 = 0.0f, q5_acc11 = 0.0f, q5_acc12 = 0.0f, q5_acc13 = 0.0f;
  float q6_acc00 = 0.0f, q6_acc01 = 0.0f, q6_acc02 = 0.0f, q6_acc03 = 0.0f;
  float q6_acc10 = 0.0f, q6_acc11 = 0.0f, q6_acc12 = 0.0f, q6_acc13 = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ q5_block0 =
        q5_row0 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ q5_block1 =
        q5_row1 + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
    const uint8_t* __restrict__ q6_block0 =
        q6_row0 + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ q6_block1 =
        q6_row1 + static_cast<int64_t>(block_idx) * Q6_K_BLOCK_BYTES;

    const uint32_t q5_meta0_head = gguf_q5_k_uniform_u32(q5_block0);
    const uint32_t q5_meta0_word0 = gguf_q5_k_uniform_u32(q5_block0 + 4);
    const uint32_t q5_meta0_word1 = gguf_q5_k_uniform_u32(q5_block0 + 8);
    const uint32_t q5_meta0_word2 = gguf_q5_k_uniform_u32(q5_block0 + 12);
    const uint32_t q5_meta1_head = gguf_q5_k_uniform_u32(q5_block1);
    const uint32_t q5_meta1_word0 = gguf_q5_k_uniform_u32(q5_block1 + 4);
    const uint32_t q5_meta1_word1 = gguf_q5_k_uniform_u32(q5_block1 + 8);
    const uint32_t q5_meta1_word2 = gguf_q5_k_uniform_u32(q5_block1 + 12);
    const float q5_d0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(q5_meta0_head));
    const float q5_dmin0 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(q5_meta0_head >> 16));
    const float q5_d1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(q5_meta1_head));
    const float q5_dmin1 = gguf_q5_k_fp16_bits_to_float(
        static_cast<uint16_t>(q5_meta1_head >> 16));

    const uint8_t* __restrict__ q6_meta_block = lane < 16 ? q6_block0 : q6_block1;
    const int8_t* __restrict__ q6_scales =
        reinterpret_cast<const int8_t*>(q6_meta_block + 192);
    const float q6_d = fp16_bytes_to_float(q6_meta_block + 208);
    const float q6_lane_scale =
        q6_d * static_cast<float>(q6_scales[lane & 15]);
    const int k_base = block_idx * QK_K;

#define GGUF_Q5_Q6_PAIR_REUSE_GROUP(GROUP, Q5_ACC0, Q5_ACC1, Q6_ACC0, Q6_ACC1) \
    { \
      constexpr int subblock0 = GROUP; \
      constexpr int subblock1 = GROUP + 4; \
      const float q5_scale0_0 = q5_d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(q5_meta0_word0, q5_meta0_word1, q5_meta0_word2)); \
      const float q5_min0_0 = q5_dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(q5_meta0_word0, q5_meta0_word1, q5_meta0_word2)); \
      const float q5_scale0_1 = q5_d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock0>(q5_meta1_word0, q5_meta1_word1, q5_meta1_word2)); \
      const float q5_min0_1 = q5_dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock0>(q5_meta1_word0, q5_meta1_word1, q5_meta1_word2)); \
      const int within0 = lane + 32 * GROUP; \
      const int q6_scale_index0 = within0 >> 4; \
      const float q6_scale0 = __shfl(q6_lane_scale, q6_scale_index0, 32); \
      const float q6_scale1 = __shfl(q6_lane_scale, 16 + q6_scale_index0, 32); \
      const float x0 = scalar_to_float<scalar_t>(x_row[k_base + within0]); \
      Q5_ACC0 += x0 * (q5_scale0_0 * static_cast<float>( \
          q5_k_quant(q5_block0, subblock0, lane)) - q5_min0_0); \
      Q5_ACC1 += x0 * (q5_scale0_1 * static_cast<float>( \
          q5_k_quant(q5_block1, subblock0, lane)) - q5_min0_1); \
      Q6_ACC0 = fmaf( \
          x0, q6_scale0 * static_cast<float>( \
              q6_k_quant_from_block(q6_block0, within0)), Q6_ACC0); \
      Q6_ACC1 = fmaf( \
          x0, q6_scale1 * static_cast<float>( \
              q6_k_quant_from_block(q6_block1, within0)), Q6_ACC1); \
      const float q5_scale1_0 = q5_d0 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(q5_meta0_word0, q5_meta0_word1, q5_meta0_word2)); \
      const float q5_min1_0 = q5_dmin0 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(q5_meta0_word0, q5_meta0_word1, q5_meta0_word2)); \
      const float q5_scale1_1 = q5_d1 * static_cast<float>( \
          gguf_q5_k_fixed_scale<subblock1>(q5_meta1_word0, q5_meta1_word1, q5_meta1_word2)); \
      const float q5_min1_1 = q5_dmin1 * static_cast<float>( \
          gguf_q5_k_fixed_min<subblock1>(q5_meta1_word0, q5_meta1_word1, q5_meta1_word2)); \
      const int within1 = within0 + 128; \
      const int q6_scale_index1 = within1 >> 4; \
      const float q6_scale2 = __shfl(q6_lane_scale, q6_scale_index1, 32); \
      const float q6_scale3 = __shfl(q6_lane_scale, 16 + q6_scale_index1, 32); \
      const float x1 = scalar_to_float<scalar_t>(x_row[k_base + within1]); \
      Q5_ACC0 += x1 * (q5_scale1_0 * static_cast<float>( \
          q5_k_quant(q5_block0, subblock1, lane)) - q5_min1_0); \
      Q5_ACC1 += x1 * (q5_scale1_1 * static_cast<float>( \
          q5_k_quant(q5_block1, subblock1, lane)) - q5_min1_1); \
      Q6_ACC0 = fmaf( \
          x1, q6_scale2 * static_cast<float>( \
              q6_k_quant_from_block(q6_block0, within1)), Q6_ACC0); \
      Q6_ACC1 = fmaf( \
          x1, q6_scale3 * static_cast<float>( \
              q6_k_quant_from_block(q6_block1, within1)), Q6_ACC1); \
    }
    GGUF_Q5_Q6_PAIR_REUSE_GROUP(0, q5_acc00, q5_acc10, q6_acc00, q6_acc10);
    GGUF_Q5_Q6_PAIR_REUSE_GROUP(1, q5_acc01, q5_acc11, q6_acc01, q6_acc11);
    GGUF_Q5_Q6_PAIR_REUSE_GROUP(2, q5_acc02, q5_acc12, q6_acc02, q6_acc12);
    GGUF_Q5_Q6_PAIR_REUSE_GROUP(3, q5_acc03, q5_acc13, q6_acc03, q6_acc13);
#undef GGUF_Q5_Q6_PAIR_REUSE_GROUP
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    q5_acc00 += __shfl_down(q5_acc00, offset);
    q5_acc01 += __shfl_down(q5_acc01, offset);
    q5_acc02 += __shfl_down(q5_acc02, offset);
    q5_acc03 += __shfl_down(q5_acc03, offset);
    q5_acc10 += __shfl_down(q5_acc10, offset);
    q5_acc11 += __shfl_down(q5_acc11, offset);
    q5_acc12 += __shfl_down(q5_acc12, offset);
    q5_acc13 += __shfl_down(q5_acc13, offset);
    q6_acc00 += __shfl_down(q6_acc00, offset);
    q6_acc01 += __shfl_down(q6_acc01, offset);
    q6_acc02 += __shfl_down(q6_acc02, offset);
    q6_acc03 += __shfl_down(q6_acc03, offset);
    q6_acc10 += __shfl_down(q6_acc10, offset);
    q6_acc11 += __shfl_down(q6_acc11, offset);
    q6_acc12 += __shfl_down(q6_acc12, offset);
    q6_acc13 += __shfl_down(q6_acc13, offset);
  }
  if (lane == 0) {
    float q5_total0 = 0.0f;
    q5_total0 += q5_acc00;
    q5_total0 += q5_acc01;
    q5_total0 += q5_acc02;
    q5_total0 += q5_acc03;
    float q5_total1 = 0.0f;
    q5_total1 += q5_acc10;
    q5_total1 += q5_acc11;
    q5_total1 += q5_acc12;
    q5_total1 += q5_acc13;
    float q6_total0 = 0.0f;
    q6_total0 += q6_acc00;
    q6_total0 += q6_acc01;
    q6_total0 += q6_acc02;
    q6_total0 += q6_acc03;
    float q6_total1 = 0.0f;
    q6_total1 += q6_acc10;
    q6_total1 += q6_acc11;
    q6_total1 += q6_acc12;
    q6_total1 += q6_acc13;
    store_output<out_t>(q5_out, q5_out_col, q5_total0);
    store_output<out_t>(q5_out, q5_out_col + 1, q5_total1);
    store_output<out_t>(q6_out, q6_out_col, q6_total0);
    store_output<out_t>(q6_out, q6_out_col + 1, q6_total1);
  }
}

// The first Q6-pair waves consume corresponding Q5 pairs; excess Q5 pairs
// retain their independently measured fixed-metadata helper. Host validation
// guarantees that the Q5 pair count covers the Q6 pair count.
template <typename scalar_t, typename out_t>
__launch_bounds__(32, 8)
__global__ void gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q5_weight_a,
    const uint8_t* __restrict__ q5_weight_b,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    out_t* __restrict__ q5_out_a,
    out_t* __restrict__ q5_out_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    int in_features,
    int q5_features_a,
    int q5_features_b,
    int q6_features_a,
    int q6_features_b) {
  const int q5_pairs_a = q5_features_a >> 1;
  const int q5_pairs = (q5_features_a + q5_features_b) >> 1;
  const int flat_pair = blockIdx.x;
  const bool q5_projection_b = flat_pair >= q5_pairs_a;
  const int q5_out_pair = q5_projection_b ? flat_pair - q5_pairs_a : flat_pair;
  const uint8_t* __restrict__ q5_weight =
      q5_projection_b ? q5_weight_b : q5_weight_a;
  out_t* __restrict__ q5_out = q5_projection_b ? q5_out_b : q5_out_a;
  const int q5_out_features =
      q5_projection_b ? q5_features_b : q5_features_a;

  const int q6_pairs = (q6_features_a + q6_features_b) >> 1;
  if (flat_pair < q6_pairs) {
    const int q6_pairs_a = q6_features_a >> 1;
    const bool q6_projection_b = flat_pair >= q6_pairs_a;
    const int q6_out_pair = q6_projection_b ? flat_pair - q6_pairs_a : flat_pair;
    gguf_q5_q6_wave32x2_pair_reuse_fixed_meta_decode_out_block(
        x,
        q5_weight,
        q6_projection_b ? q6_weight_b : q6_weight_a,
        q5_out,
        q6_projection_b ? q6_out_b : q6_out_a,
        q5_out_pair,
        q6_out_pair,
        in_features,
        q5_out_features,
        q6_projection_b ? q6_features_b : q6_features_a);
    return;
  }

  if (flat_pair < q5_pairs) {
    gguf_q5_k_wave32x2_fixed_meta_decode_out_block(
        x,
        q5_weight,
        q5_out,
        0,
        q5_out_pair,
        in_features,
        q5_out_features);
  }
}

// Laguna layer 47 reverses the raw-Q6 role onto query/gate while K/V are Q8_0.
// One local128 flattened range preserves the retained generic Q6 pair and Q8
// singleton arithmetic while contracting their three launches into one.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 2)
__global__ void gguf_q6_q8_mixed_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    const uint8_t* __restrict__ q8_weight_a,
    const uint8_t* __restrict__ q8_weight_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    out_t* __restrict__ q8_out_a,
    out_t* __restrict__ q8_out_b,
    int in_features,
    int q6_features_a,
    int q6_features_b,
    int q8_features_a,
    int q8_features_b) {
  const int q6_packs = (q6_features_a + q6_features_b) >> 3;
  const int flat_pack = blockIdx.x;
  const bool q8_projection = flat_pack >= q6_packs;
  const int family_pack = q8_projection ? flat_pack - q6_packs : flat_pack;
  const int packs_a = (q8_projection ? q8_features_a : q6_features_a) >> 3;
  const bool projection_b = family_pack >= packs_a;
  const int out_pack = projection_b ? family_pack - packs_a : family_pack;
  const int out_features = q8_projection
      ? (projection_b ? q8_features_b : q8_features_a)
      : (projection_b ? q6_features_b : q6_features_a);
  const uint8_t* __restrict__ qweight = q8_projection
      ? (projection_b ? q8_weight_b : q8_weight_a)
      : (projection_b ? q6_weight_b : q6_weight_a);
  out_t* __restrict__ out = q8_projection
      ? (projection_b ? q8_out_b : q8_out_a)
      : (projection_b ? q6_out_b : q6_out_a);
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(q8_projection ? 8 : 6, in_features);
  const uint8_t* __restrict__ w0 =
      qweight + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 =
      qweight + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 =
      qweight + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 =
      qweight + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 =
      qweight + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 =
      qweight + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 =
      qweight + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 =
      qweight + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x[k]);
    if (q8_projection) {
      acc0 += xv * q8_0_weight(w0, k);
      acc1 += xv * q8_0_weight(w1, k);
      acc2 += xv * q8_0_weight(w2, k);
      acc3 += xv * q8_0_weight(w3, k);
      acc4 += xv * q8_0_weight(w4, k);
      acc5 += xv * q8_0_weight(w5, k);
      acc6 += xv * q8_0_weight(w6, k);
      acc7 += xv * q8_0_weight(w7, k);
    } else {
      acc0 += xv * q6_k_weight(w0, k);
      acc1 += xv * q6_k_weight(w1, k);
      acc2 += xv * q6_k_weight(w2, k);
      acc3 += xv * q6_k_weight(w3, k);
      acc4 += xv * q6_k_weight(w4, k);
      acc5 += xv * q6_k_weight(w5, k);
      acc6 += xv * q6_k_weight(w6, k);
      acc7 += xv * q6_k_weight(w7, k);
    }
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    store_output<out_t>(out, out_col + 0, r0);
    store_output<out_t>(out, out_col + 1, r1);
    store_output<out_t>(out, out_col + 2, r2);
    store_output<out_t>(out, out_col + 3, r3);
    store_output<out_t>(out, out_col + 4, r4);
    store_output<out_t>(out, out_col + 5, r5);
    store_output<out_t>(out, out_col + 6, r6);
    store_output<out_t>(out, out_col + 7, r7);
  }
}

// Layer-47 sibling: Q6 query/gate blocks use cooperative fixed metadata while
// Q8 K/V blocks preserve the retained singleton-equivalent arithmetic.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 2)
__global__ void gguf_q6_q8_mixed_q6_fixed_meta_attention_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ q6_weight_a,
    const uint8_t* __restrict__ q6_weight_b,
    const uint8_t* __restrict__ q8_weight_a,
    const uint8_t* __restrict__ q8_weight_b,
    out_t* __restrict__ q6_out_a,
    out_t* __restrict__ q6_out_b,
    out_t* __restrict__ q8_out_a,
    out_t* __restrict__ q8_out_b,
    int in_features,
    int q6_features_a,
    int q6_features_b,
    int q8_features_a,
    int q8_features_b) {
  const int q6_packs = (q6_features_a + q6_features_b) >> 3;
  const int flat_pack = blockIdx.x;
  const bool q8_projection = flat_pack >= q6_packs;
  const int family_pack = q8_projection ? flat_pack - q6_packs : flat_pack;
  const int packs_a = (q8_projection ? q8_features_a : q6_features_a) >> 3;
  const bool projection_b = family_pack >= packs_a;
  const int out_pack = projection_b ? family_pack - packs_a : family_pack;
  if (!q8_projection) {
    __shared__ float q6_block_scales[8 * 16];
    gguf_q6_k_pack8_fixed_meta_decode_out_block(
        x,
        projection_b ? q6_weight_b : q6_weight_a,
        projection_b ? q6_out_b : q6_out_a,
        out_pack,
        in_features,
        q6_block_scales);
    return;
  }

  const uint8_t* __restrict__ qweight =
      projection_b ? q8_weight_b : q8_weight_a;
  out_t* __restrict__ out = projection_b ? q8_out_b : q8_out_a;
  const int tid = threadIdx.x;
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(8, in_features);
  const uint8_t* __restrict__ w0 =
      qweight + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 =
      qweight + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 =
      qweight + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 =
      qweight + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 =
      qweight + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 =
      qweight + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 =
      qweight + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 =
      qweight + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x[k]);
    acc0 += xv * q8_0_weight(w0, k);
    acc1 += xv * q8_0_weight(w1, k);
    acc2 += xv * q8_0_weight(w2, k);
    acc3 += xv * q8_0_weight(w3, k);
    acc4 += xv * q8_0_weight(w4, k);
    acc5 += xv * q8_0_weight(w5, k);
    acc6 += xv * q8_0_weight(w6, k);
    acc7 += xv * q8_0_weight(w7, k);
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    store_output<out_t>(out, out_col + 0, r0);
    store_output<out_t>(out, out_col + 1, r1);
    store_output<out_t>(out, out_col + 2, r2);
    store_output<out_t>(out, out_col + 3, r3);
    store_output<out_t>(out, out_col + 4, r4);
    store_output<out_t>(out, out_col + 5, r5);
    store_output<out_t>(out, out_col + 6, r6);
    store_output<out_t>(out, out_col + 7, r7);
  }
}

// Decode-only projection pair. blockIdx.y selects gate/up, so each workgroup
// executes the exact singleton body above while one launch provides enough
// independent K3072/N1024 blocks to fill the W7900.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 4)
__global__ void gguf_q5_k_pair_pack8_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features) {
  const int projection = blockIdx.y;
  const int row = blockIdx.z;
  if (projection >= 2 || row >= rows) return;
  const uint8_t* __restrict__ qweight = projection == 0 ? qweight_a : qweight_b;
  out_t* __restrict__ out = projection == 0 ? out_a : out_b;
  __shared__ float block_scales[8 * 8];
  __shared__ float block_mins[8 * 8];
  gguf_q5_k_pack8_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      blockIdx.x,
      in_features,
      out_features,
      block_scales,
      block_mins);
}

// Decode-only unequal-width projection pair. Flattening both output-pack
// ranges avoids the idle workgroups that a rectangular projection grid would
// create for Laguna's wide attention query plus tiny per-head gate.
template <typename scalar_t, typename out_t>
__launch_bounds__(128, 4)
__global__ void gguf_q5_k_unequal_pair_pack8_decode_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features_a,
    int out_features_b) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  const int packs_a = out_features_a >> 3;
  const int flat_pack = blockIdx.x;
  const bool projection_b = flat_pack >= packs_a;
  const int out_pack = projection_b ? flat_pack - packs_a : flat_pack;
  const uint8_t* __restrict__ qweight = projection_b ? qweight_b : qweight_a;
  out_t* __restrict__ out = projection_b ? out_b : out_a;
  const int out_features = projection_b ? out_features_b : out_features_a;
  __shared__ float block_scales[8 * 8];
  __shared__ float block_mins[8 * 8];
  gguf_q5_k_pack8_decode_out_block(
      x + static_cast<int64_t>(row) * in_features,
      qweight,
      out,
      row,
      out_pack,
      in_features,
      out_features,
      block_scales,
      block_mins);
}

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_selected_pack8_prefill_out_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int out_packed) {
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(qtype, in_features);
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const scalar_t* __restrict__ x_row_ptr = x + static_cast<int64_t>(x_row) * in_features;
  const uint8_t* __restrict__ w0 = expert_base + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 = expert_base + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 = expert_base + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 = expert_base + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 = expert_base + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 = expert_base + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 = expert_base + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 = expert_base + static_cast<int64_t>(out_col + 7) * row_bytes;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  for (int k = tid; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row_ptr[k]);
    if constexpr (qtype == 8) {
      acc0 += xv * q8_0_weight(w0, k);
      acc1 += xv * q8_0_weight(w1, k);
      acc2 += xv * q8_0_weight(w2, k);
      acc3 += xv * q8_0_weight(w3, k);
      acc4 += xv * q8_0_weight(w4, k);
      acc5 += xv * q8_0_weight(w5, k);
      acc6 += xv * q8_0_weight(w6, k);
      acc7 += xv * q8_0_weight(w7, k);
    } else if constexpr (qtype == 5) {
      acc0 += xv * q5_k_weight(w0, k);
      acc1 += xv * q5_k_weight(w1, k);
      acc2 += xv * q5_k_weight(w2, k);
      acc3 += xv * q5_k_weight(w3, k);
      acc4 += xv * q5_k_weight(w4, k);
      acc5 += xv * q5_k_weight(w5, k);
      acc6 += xv * q5_k_weight(w6, k);
      acc7 += xv * q5_k_weight(w7, k);
    } else {
      acc0 += xv * q6_k_weight(w0, k);
      acc1 += xv * q6_k_weight(w1, k);
      acc2 += xv * q6_k_weight(w2, k);
      acc3 += xv * q6_k_weight(w3, k);
      acc4 += xv * q6_k_weight(w4, k);
      acc5 += xv * q6_k_weight(w5, k);
      acc6 += xv * q6_k_weight(w6, k);
      acc7 += xv * q6_k_weight(w7, k);
    }
  }

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
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
__global__ void gguf_k_selected_pack8_q8_1_dp4a_prefill_out_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features,
    int out_packed,
    int q8_blocks_per_row) {
  static_assert(qtype == 5 || qtype == 6, "q8_1 selected-pack8 dp4a supports Q5_K/Q6_K");
  const int out_pack = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;

  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int lanes_per_x_row = rows / x_rows;
  const int x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int out_col = out_pack << 3;
  const int row_bytes = gguf_k_row_bytes(qtype, in_features);
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const gguf_q8_1_block* __restrict__ xq_row =
      xq + static_cast<int64_t>(x_row) * q8_blocks_per_row;
  const uint8_t* __restrict__ w0 = expert_base + static_cast<int64_t>(out_col + 0) * row_bytes;
  const uint8_t* __restrict__ w1 = expert_base + static_cast<int64_t>(out_col + 1) * row_bytes;
  const uint8_t* __restrict__ w2 = expert_base + static_cast<int64_t>(out_col + 2) * row_bytes;
  const uint8_t* __restrict__ w3 = expert_base + static_cast<int64_t>(out_col + 3) * row_bytes;
  const uint8_t* __restrict__ w4 = expert_base + static_cast<int64_t>(out_col + 4) * row_bytes;
  const uint8_t* __restrict__ w5 = expert_base + static_cast<int64_t>(out_col + 5) * row_bytes;
  const uint8_t* __restrict__ w6 = expert_base + static_cast<int64_t>(out_col + 6) * row_bytes;
  const uint8_t* __restrict__ w7 = expert_base + static_cast<int64_t>(out_col + 7) * row_bytes;
  constexpr int block_bytes = (qtype == 5) ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
  const int groups4 = in_features >> 2;
  for (int group = tid; group < groups4; group += blockDim.x) {
    const int k4 = group << 2;
    const int block_idx = k4 / QK_K;
    const int within = k4 - block_idx * QK_K;
    const int subblock = within >> 5;
    const int lane4 = within & 31;
    const int q8_index = block_idx * 8 + subblock;

    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = gguf_k_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ b0 = w0 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b1 = w1 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b2 = w2 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b3 = w3 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b4 = w4 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b5 = w5 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b6 = w6 + static_cast<int64_t>(block_idx) * block_bytes;
    const uint8_t* __restrict__ b7 = w7 + static_cast<int64_t>(block_idx) * block_bytes;
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

  GGUF_K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                      r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = static_cast<int64_t>(row) * out_features + out_col;
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

template <typename scalar_t, typename out_t, int qtype>
__launch_bounds__(256, 2)
__global__ void gguf_k_dual_prefill_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int row_bytes = gguf_k_row_bytes(qtype, in_features);
  const uint8_t* __restrict__ weight_a = qweight_a + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ weight_b = qweight_b + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;

  float acc_a = 0.0f;
  float acc_b = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    if constexpr (qtype == 8) {
      acc_a += xv * q8_0_weight(weight_a, k);
      acc_b += xv * q8_0_weight(weight_b, k);
    } else if constexpr (qtype == 5) {
      acc_a += xv * q5_k_weight(weight_a, k);
      acc_b += xv * q5_k_weight(weight_b, k);
    } else {
      acc_a += xv * q6_k_weight(weight_a, k);
      acc_b += xv * q6_k_weight(weight_b, k);
    }
  }
  const float total_a = reduce_block_sum(acc_a);
  __syncthreads();
  const float total_b = reduce_block_sum(acc_b);
  if (threadIdx.x == 0) {
    const int64_t out_idx = static_cast<int64_t>(row) * out_features + out_col;
    store_output<out_t>(out_a, out_idx, total_a);
    store_output<out_t>(out_b, out_idx, total_b);
  }
}

template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_dual_gemv_out(
    const scalar_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr || out_a == nullptr ||
      out_b == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_k_dual_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      qweight_a,
      qweight_b,
      out_a,
      out_b,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_pack8_gemv_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || (out_features % 8) != 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_k_pack8_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(out_packed));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q6_k_pair_pack8_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (out_features_a % 8) != 0 || (out_features_b % 8) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((out_features_a + out_features_b) / 8),
      static_cast<unsigned int>(rows));
  gguf_q6_k_pair_pack8_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(128), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(rows),
          static_cast<int>(in_features),
          static_cast<int>(out_features_a),
          static_cast<int>(out_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_pack8_decode_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % 8) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>(out_features / 8),
      static_cast<unsigned int>(rows));
  gguf_q5_k_pack8_decode_out_kernel<uint16_t, out_t><<<grid, dim3(128), 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_pair_pack8_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features <= 0 || (out_features % 8) != 0 ||
      threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>(out_features / 8),
      2,
      static_cast<unsigned int>(rows));
  gguf_q5_k_pair_pack8_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(128), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(rows),
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_unequal_pair_pack8_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (out_features_a % 8) != 0 || (out_features_b % 8) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((out_features_a + out_features_b) / 8),
      static_cast<unsigned int>(rows));
  gguf_q5_k_unequal_pair_pack8_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(128), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(rows),
          static_cast<int>(in_features),
          static_cast<int>(out_features_a),
          static_cast<int>(out_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_wave32x2_decode_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_features / 2));
  gguf_q5_k_wave32x2_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight,
          out,
          1,
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_unequal_pair_wave32x2_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows != 1 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (out_features_a % 2) != 0 || (out_features_b % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / 2));
  gguf_q5_k_unequal_pair_wave32x2_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(in_features),
          static_cast<int>(out_features_a),
          static_cast<int>(out_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_wave32x2_fixed_meta_decode_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_features / 2));
  gguf_q5_k_wave32x2_fixed_meta_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight,
          out,
          1,
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q6_k_wave32x2_fixed_meta_decode_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_features / 2));
  gguf_q6_k_wave32x2_fixed_meta_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight,
          out,
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_unequal_pair_wave32x2_fixed_meta_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows != 1 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (out_features_a % 2) != 0 || (out_features_b % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / 2));
  gguf_q5_k_unequal_pair_wave32x2_fixed_meta_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(in_features),
          static_cast<int>(out_features_a),
          static_cast<int>(out_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_features / 2));
  gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_kernel<uint16_t, out_t>
      <<<grid, dim3(32), 0, stream>>>(
          x,
          qweight,
          out,
          1,
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_k_unequal_pair_wave32x2_swar_pair_fixed_meta_decode_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows != 1 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (out_features_a % 2) != 0 || (out_features_b % 2) != 0 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / 2));
  gguf_q5_k_unequal_pair_wave32x2_swar_pair_fixed_meta_decode_out_kernel<
      uint16_t, out_t><<<grid, dim3(32), 0, stream>>>(
          x,
          qweight_a,
          qweight_b,
          out_a,
          out_b,
          static_cast<int>(in_features),
          static_cast<int>(out_features_a),
          static_cast<int>(out_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_q6_mixed_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    out_t* q5_out_a,
    out_t* q5_out_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  if (x == nullptr || q5_weight_a == nullptr || q5_weight_b == nullptr ||
      q6_weight_a == nullptr || q6_weight_b == nullptr || q5_out_a == nullptr ||
      q5_out_b == nullptr || q6_out_a == nullptr || q6_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q5_features_a <= 0 || q5_features_b <= 0 || q6_features_a <= 0 ||
      q6_features_b <= 0 || (q5_features_a % 8) != 0 ||
      (q5_features_b % 8) != 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q5_blocks = (q5_features_a + q5_features_b) / 8;
  const int64_t q6_blocks = (q6_features_a + q6_features_b) / 8;
  gguf_q5_q6_mixed_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q5_blocks + q6_blocks)), dim3(128), 0, stream>>>(
          x,
          q5_weight_a,
          q5_weight_b,
          q6_weight_a,
          q6_weight_b,
          q5_out_a,
          q5_out_b,
          q6_out_a,
          q6_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q5_features_a),
          static_cast<int>(q5_features_b),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_q6_mixed_q6_fixed_meta_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    out_t* q5_out_a,
    out_t* q5_out_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  if (x == nullptr || q5_weight_a == nullptr || q5_weight_b == nullptr ||
      q6_weight_a == nullptr || q6_weight_b == nullptr || q5_out_a == nullptr ||
      q5_out_b == nullptr || q6_out_a == nullptr || q6_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q5_features_a <= 0 || q5_features_b <= 0 || q6_features_a <= 0 ||
      q6_features_b <= 0 || (q5_features_a % 8) != 0 ||
      (q5_features_b % 8) != 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q5_blocks = (q5_features_a + q5_features_b) / 8;
  const int64_t q6_blocks = (q6_features_a + q6_features_b) / 8;
  gguf_q5_q6_mixed_q6_fixed_meta_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q5_blocks + q6_blocks)), dim3(128), 0, stream>>>(
          x,
          q5_weight_a,
          q5_weight_b,
          q6_weight_a,
          q6_weight_b,
          q5_out_a,
          q5_out_b,
          q6_out_a,
          q6_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q5_features_a),
          static_cast<int>(q5_features_b),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_q6_mixed_local32_fixed_meta_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    out_t* q5_out_a,
    out_t* q5_out_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  if (x == nullptr || q5_weight_a == nullptr || q5_weight_b == nullptr ||
      q6_weight_a == nullptr || q6_weight_b == nullptr || q5_out_a == nullptr ||
      q5_out_b == nullptr || q6_out_a == nullptr || q6_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q5_features_a <= 0 || q5_features_b <= 0 || q6_features_a <= 0 ||
      q6_features_b <= 0 || (q5_features_a % 8) != 0 ||
      (q5_features_b % 8) != 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q5_pairs = (q5_features_a + q5_features_b) / 2;
  const int64_t q6_pairs = (q6_features_a + q6_features_b) / 2;
  gguf_q5_q6_mixed_local32_fixed_meta_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q5_pairs + q6_pairs)), dim3(32), 0, stream>>>(
          x,
          q5_weight_a,
          q5_weight_b,
          q6_weight_a,
          q6_weight_b,
          q5_out_a,
          q5_out_b,
          q6_out_a,
          q6_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q5_features_a),
          static_cast<int>(q5_features_b),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    out_t* q5_out_a,
    out_t* q5_out_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  if (x == nullptr || q5_weight_a == nullptr || q5_weight_b == nullptr ||
      q6_weight_a == nullptr || q6_weight_b == nullptr || q5_out_a == nullptr ||
      q5_out_b == nullptr || q6_out_a == nullptr || q6_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q5_features_a <= 0 || q5_features_b <= 0 || q6_features_a <= 0 ||
      q6_features_b <= 0 || (q5_features_a % 8) != 0 ||
      (q5_features_b % 8) != 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q5_pairs = (q5_features_a + q5_features_b) / 2;
  const int64_t q6_pairs = (q6_features_a + q6_features_b) / 2;
  gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_decode_out_kernel<
      uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q5_pairs + q6_pairs)), dim3(32), 0, stream>>>(
          x,
          q5_weight_a,
          q5_weight_b,
          q6_weight_a,
          q6_weight_b,
          q5_out_a,
          q5_out_b,
          q6_out_a,
          q6_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q5_features_a),
          static_cast<int>(q5_features_b),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    out_t* q5_out_a,
    out_t* q5_out_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  if (x == nullptr || q5_weight_a == nullptr || q5_weight_b == nullptr ||
      q6_weight_a == nullptr || q6_weight_b == nullptr || q5_out_a == nullptr ||
      q5_out_b == nullptr || q6_out_a == nullptr || q6_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q5_features_a <= 0 || q5_features_b <= 0 || q6_features_a <= 0 ||
      q6_features_b <= 0 || (q5_features_a % 8) != 0 ||
      (q5_features_b % 8) != 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0 ||
      (q5_features_a + q5_features_b) < (q6_features_a + q6_features_b)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q5_pairs = (q5_features_a + q5_features_b) / 2;
  gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q5_pairs)), dim3(32), 0, stream>>>(
          x,
          q5_weight_a,
          q5_weight_b,
          q6_weight_a,
          q6_weight_b,
          q5_out_a,
          q5_out_b,
          q6_out_a,
          q6_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q5_features_a),
          static_cast<int>(q5_features_b),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q6_q8_mixed_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    const uint8_t* q8_weight_a,
    const uint8_t* q8_weight_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    out_t* q8_out_a,
    out_t* q8_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q6_features_a,
    int64_t q6_features_b,
    int64_t q8_features_a,
    int64_t q8_features_b,
    hipStream_t stream) {
  if (x == nullptr || q6_weight_a == nullptr || q6_weight_b == nullptr ||
      q8_weight_a == nullptr || q8_weight_b == nullptr || q6_out_a == nullptr ||
      q6_out_b == nullptr || q8_out_a == nullptr || q8_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q6_features_a <= 0 || q6_features_b <= 0 || q8_features_a <= 0 ||
      q8_features_b <= 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0 || (q8_features_a % 8) != 0 ||
      (q8_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q6_blocks = (q6_features_a + q6_features_b) / 8;
  const int64_t q8_blocks = (q8_features_a + q8_features_b) / 8;
  gguf_q6_q8_mixed_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q6_blocks + q8_blocks)), dim3(128), 0, stream>>>(
          x,
          q6_weight_a,
          q6_weight_b,
          q8_weight_a,
          q8_weight_b,
          q6_out_a,
          q6_out_b,
          q8_out_a,
          q8_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b),
          static_cast<int>(q8_features_a),
          static_cast<int>(q8_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t>
int launch_gguf_q6_q8_mixed_q6_fixed_meta_attention_decode_out(
    const uint16_t* x,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    const uint8_t* q8_weight_a,
    const uint8_t* q8_weight_b,
    out_t* q6_out_a,
    out_t* q6_out_b,
    out_t* q8_out_a,
    out_t* q8_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q6_features_a,
    int64_t q6_features_b,
    int64_t q8_features_a,
    int64_t q8_features_b,
    hipStream_t stream) {
  if (x == nullptr || q6_weight_a == nullptr || q6_weight_b == nullptr ||
      q8_weight_a == nullptr || q8_weight_b == nullptr || q6_out_a == nullptr ||
      q6_out_b == nullptr || q8_out_a == nullptr || q8_out_b == nullptr ||
      rows != 1 || in_features <= 0 || (in_features % QK_K) != 0 ||
      q6_features_a <= 0 || q6_features_b <= 0 || q8_features_a <= 0 ||
      q8_features_b <= 0 || (q6_features_a % 8) != 0 ||
      (q6_features_b % 8) != 0 || (q8_features_a % 8) != 0 ||
      (q8_features_b % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q6_blocks = (q6_features_a + q6_features_b) / 8;
  const int64_t q8_blocks = (q8_features_a + q8_features_b) / 8;
  gguf_q6_q8_mixed_q6_fixed_meta_attention_decode_out_kernel<uint16_t, out_t>
      <<<dim3(static_cast<unsigned int>(q6_blocks + q8_blocks)), dim3(128), 0, stream>>>(
          x,
          q6_weight_a,
          q6_weight_b,
          q8_weight_a,
          q8_weight_b,
          q6_out_a,
          q6_out_b,
          q8_out_a,
          q8_out_b,
          static_cast<int>(in_features),
          static_cast<int>(q6_features_a),
          static_cast<int>(q6_features_b),
          static_cast<int>(q8_features_a),
          static_cast<int>(q8_features_b));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t, int COL_TILE, int ROW_TILE>
int launch_gguf_q8_0_exact_prefill_tiled_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 1 ||
      in_features <= 0 || (in_features % Q8_0_BLOCK) != 0 || out_features <= 0 ||
      (out_features % COL_TILE) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / COL_TILE),
      static_cast<unsigned int>((rows + ROW_TILE - 1) / ROW_TILE));
  gguf_q8_0_exact_prefill_tiled_out_kernel<scalar_t, out_t, COL_TILE, ROW_TILE>
      <<<grid, block, 0, stream>>>(
          x,
          qweight,
          out,
          static_cast<int>(rows),
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_selected_pack8_gemv_out(
    const scalar_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (out_features % 8) != 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_k_selected_pack8_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(out_packed));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int qtype>
int launch_gguf_k_selected_pack8_q8_1_dp4a_gemv_out(
    const gguf_q8_1_block* xq,
    const int64_t* selected,
    const uint8_t* qweight,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  static_assert(qtype == 5 || qtype == 6, "q8_1 selected-pack8 dp4a supports Q5_K/Q6_K");
  if (xq == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % 8) != 0 || (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_k_selected_pack8_q8_1_dp4a_prefill_out_kernel<out_t, qtype><<<grid, block, 0, stream>>>(
      xq,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(out_packed),
      static_cast<int>(q8_blocks_per_row));
  return static_cast<int>(hipGetLastError());
}


template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_gemv_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_k_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

// Dispatch the row-tile kernel by exact row count (small-B verifier path).
// Supports rows in [2, 8]; rows==1 / rows>8 route to per-row / WMMA elsewhere.
template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_gemv_rowtile_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight == nullptr || out == nullptr || rows < 2 || rows > 8 ||
      in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features));
  const int r = static_cast<int>(rows);
  const int infe = static_cast<int>(in_features);
  const int outf = static_cast<int>(out_features);
#define GGUF_K_ROWTILE_CASE(N)                                                \
  case N:                                                                     \
    gguf_k_prefill_out_rowtile_kernel<scalar_t, out_t, qtype, N>              \
        <<<grid, block, 0, stream>>>(x, qweight, out, r, infe, outf);         \
    break;
  switch (r) {
    GGUF_K_ROWTILE_CASE(2)
    GGUF_K_ROWTILE_CASE(3)
    GGUF_K_ROWTILE_CASE(4)
    GGUF_K_ROWTILE_CASE(5)
    GGUF_K_ROWTILE_CASE(6)
    GGUF_K_ROWTILE_CASE(7)
    GGUF_K_ROWTILE_CASE(8)
    default:
      return static_cast<int>(hipErrorInvalidValue);
  }
#undef GGUF_K_ROWTILE_CASE
  return static_cast<int>(hipGetLastError());
}

// Dispatch fixed-grid-Y row batching for arbitrary positive prefill lengths.
template <typename scalar_t, typename out_t, int qtype, int ROW_BATCH>
int launch_gguf_k_gemv_rowbatch_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features),
      static_cast<unsigned int>((rows + ROW_BATCH - 1) / ROW_BATCH));
  gguf_k_prefill_out_rowbatch_kernel<scalar_t, out_t, qtype, ROW_BATCH>
      <<<grid, block, 0, stream>>>(
          x,
          qweight,
          out,
          static_cast<int>(rows),
          static_cast<int>(in_features),
          static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <
    typename scalar_t,
    typename out_t,
    int qtype,
    int COL_TILE,
    int ROW_BATCH>
int launch_gguf_k_gemv_coltile_rowbatch_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                        : ((in_features % QK_K) == 0);
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || !valid_shape || out_features <= 0 ||
      (out_features % COL_TILE) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / COL_TILE),
      static_cast<unsigned int>((rows + ROW_BATCH - 1) / ROW_BATCH));
  gguf_k_prefill_out_coltile_rowbatch_kernel<
      scalar_t,
      out_t,
      qtype,
      COL_TILE,
      ROW_BATCH><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
int launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % COL_TILE) != 0 || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / COL_TILE),
      static_cast<unsigned int>((rows + ROW_BATCH - 1) / ROW_BATCH));
  gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_kernel<
      out_t,
      COL_TILE,
      ROW_BATCH><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
int launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_full_group_compute_out(
    const uint16_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows != 512 ||
      in_features <= 0 || (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % COL_TILE) != 0 || (rows % ROW_BATCH) != 0 ||
      threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / COL_TILE),
      static_cast<unsigned int>(rows / ROW_BATCH));
  gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_full_group_compute_kernel<
      out_t,
      COL_TILE,
      ROW_BATCH><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_selected_gemv_out(
    const scalar_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (x == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_k_selected_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t, int qtype>
int launch_gguf_k_selected_silu_gemv_out(
    const scalar_t* gate,
    const scalar_t* up,
    const int64_t* selected,
    const uint8_t* qweight,
    out_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const bool valid_shape = (qtype == 8) ? ((in_features % Q8_0_BLOCK) == 0)
                                       : ((in_features % QK_K) == 0);
  if (gate == nullptr || up == nullptr || selected == nullptr || qweight == nullptr ||
      out == nullptr || x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 || !valid_shape ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_k_selected_silu_prefill_out_kernel<scalar_t, out_t, qtype><<<grid, block, 0, stream>>>(
      gate,
      up,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

#define HIPENGINE_GGUF_K_EXPORTS(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_gemv_f32_f32_out( \
    const float* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<float, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_f32_fp16_out( \
    const float* x, const uint8_t* qweight, half_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<float, half_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_fp16_f32_out( \
    const half_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<half_t, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_fp16_fp16_out( \
    const half_t* x, const uint8_t* qweight, half_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<half_t, half_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_bf16_f32_out( \
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<uint16_t, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_bf16_fp16_out( \
    const uint16_t* x, const uint8_t* qweight, half_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<uint16_t, half_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_dual_gemv_f32_f32_out( \
    const float* x, const uint8_t* qweight_a, const uint8_t* qweight_b, \
    float* out_a, float* out_b, int64_t rows, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_dual_gemv_out<float, float, QTYPE>( \
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_dual_gemv_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b, \
    uint16_t* out_a, uint16_t* out_b, int64_t rows, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_dual_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_pack8_gemv_f32_f32_out( \
    const float* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_pack8_gemv_out<float, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_pack8_gemv_bf16_f32_out( \
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_pack8_gemv_out<uint16_t, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_pack8_gemv_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_pack8_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_selected_pack8_gemv_bf16_bf16_out( \
    const uint16_t* x, const int64_t* selected, const uint8_t* qweight, uint16_t* out, \
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_selected_pack8_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, selected, qweight, out, x_rows, rows, num_experts, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_selected_gemv_bf16_bf16_out( \
    const uint16_t* x, const int64_t* selected, const uint8_t* qweight, uint16_t* out, \
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_selected_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, selected, qweight, out, x_rows, rows, num_experts, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_selected_silu_gemv_bf16_bf16_out( \
    const uint16_t* gate, const uint16_t* up, const int64_t* selected, const uint8_t* qweight, uint16_t* out, \
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_selected_silu_gemv_out<uint16_t, uint16_t, QTYPE>( \
      gate, up, selected, qweight, out, x_rows, rows, num_experts, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_rowtile_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowtile_out<uint16_t, uint16_t, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_rowtile_bf16_f32_out( \
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowtile_out<uint16_t, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_rowtile_f32_f32_out( \
    const float* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowtile_out<float, float, QTYPE>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_K_EXPORTS(gguf_q8_0, 8)
HIPENGINE_GGUF_K_EXPORTS(gguf_q5_k, 5)
HIPENGINE_GGUF_K_EXPORTS(gguf_q6_k, 6)

#define HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(NAME, QTYPE, ROW_BATCH) \
extern "C" int hipengine_##NAME##_gemv_rowbatch##ROW_BATCH##_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowbatch_out<uint16_t, uint16_t, QTYPE, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowbatch_out<uint16_t, float, QTYPE, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q5_k, 5, 4)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q5_k, 5, 8)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q5_k, 5, 16)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q5_k, 5, 32)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q6_k, 6, 4)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q6_k, 6, 8)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q6_k, 6, 16)
HIPENGINE_GGUF_K_ROWBATCH_EXPORTS(gguf_q6_k, 6, 32)

#undef HIPENGINE_GGUF_K_ROWBATCH_EXPORTS

#define HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT(ROW_BATCH) \
extern "C" int hipengine_gguf_q8_0_gemv_rowbatch##ROW_BATCH##_f32_f32_out( \
    const float* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_rowbatch_out<float, float, 8, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT(4)
HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT(8)
HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT(16)
HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT(32)

#undef HIPENGINE_GGUF_Q8_F32_ROWBATCH_EXPORT

#define HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q8_0_gemv_coltile##COL_TILE##_rowbatch##ROW_BATCH##_f32_f32_out( \
    const float* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_coltile_rowbatch_out< \
      float, float, 8, COL_TILE, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(4, 8)
HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(8, 4)
HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(8, 8)
HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(16, 2)
HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(16, 4)
HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT(32, 1)

#undef HIPENGINE_GGUF_Q8_F32_COLTILE_EXPORT

#define HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS( \
    NAME, QTYPE, COL_TILE, ROW_BATCH) \
extern "C" int hipengine_##NAME##_gemv_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_coltile_rowbatch_out< \
      uint16_t, uint16_t, QTYPE, COL_TILE, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
} \
extern "C" int hipengine_##NAME##_gemv_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_gemv_coltile_rowbatch_out< \
      uint16_t, float, QTYPE, COL_TILE, ROW_BATCH>( \
      x, qweight, out, rows, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS(gguf_q5_k, 5, 2, 16)
HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS(gguf_q5_k, 5, 4, 8)
HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS(gguf_q6_k, 6, 2, 16)
HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS(gguf_q6_k, 6, 4, 8)

#undef HIPENGINE_GGUF_K_COLTILE_ROWBATCH_EXPORTS

extern "C" int hipengine_gguf_q6_k_gemv_dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  if ((in_features != 9216 && in_features != 12288) || out_features != 3072) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_out<
      uint16_t, 4, 8>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_gemv_dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  if (in_features != 3072 || out_features != 9216) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_out<
      float, 2, 16>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile4_rowbatch8_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  if ((in_features != 9216 && in_features != 12288) || out_features != 3072) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_full_group_compute_out<
      uint16_t, 4, 8>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  if (in_features != 3072 || out_features != 9216) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_gguf_q6_k_gemv_coltile_rowbatch_dpp_wave_reduction_full_group_compute_out<
      float, 2, 16>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    float* out_a, float* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q6_k_pair_pack8_decode_out<float>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features_a,
      out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q6_k_wave32x2_fixed_meta_decode_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_q6_mixed_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    float* q5_out_a,
    float* q5_out_b,
    float* q6_out_a,
    float* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  return launch_gguf_q5_q6_mixed_attention_decode_out<float>(
      x, q5_weight_a, q5_weight_b, q6_weight_a, q6_weight_b, q5_out_a,
      q5_out_b, q6_out_a, q6_out_b, rows, in_features, q5_features_a,
      q5_features_b, q6_features_a, q6_features_b, stream);
}

extern "C" int hipengine_gguf_q5_q6_mixed_q6_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    float* q5_out_a,
    float* q5_out_b,
    float* q6_out_a,
    float* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  return launch_gguf_q5_q6_mixed_q6_fixed_meta_attention_decode_out<float>(
      x, q5_weight_a, q5_weight_b, q6_weight_a, q6_weight_b, q5_out_a,
      q5_out_b, q6_out_a, q6_out_b, rows, in_features, q5_features_a,
      q5_features_b, q6_features_a, q6_features_b, stream);
}

extern "C" int hipengine_gguf_q5_q6_mixed_local32_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    float* q5_out_a,
    float* q5_out_b,
    float* q6_out_a,
    float* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  return launch_gguf_q5_q6_mixed_local32_fixed_meta_attention_decode_out<float>(
      x, q5_weight_a, q5_weight_b, q6_weight_a, q6_weight_b, q5_out_a,
      q5_out_b, q6_out_a, q6_out_b, rows, in_features, q5_features_a,
      q5_features_b, q6_features_a, q6_features_b, stream);
}

extern "C" int hipengine_gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    float* q5_out_a,
    float* q5_out_b,
    float* q6_out_a,
    float* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  return launch_gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_decode_out<float>(
      x, q5_weight_a, q5_weight_b, q6_weight_a, q6_weight_b, q5_out_a,
      q5_out_b, q6_out_a, q6_out_b, rows, in_features, q5_features_a,
      q5_features_b, q6_features_a, q6_features_b, stream);
}

extern "C" int hipengine_gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q5_weight_a,
    const uint8_t* q5_weight_b,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    float* q5_out_a,
    float* q5_out_b,
    float* q6_out_a,
    float* q6_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q5_features_a,
    int64_t q5_features_b,
    int64_t q6_features_a,
    int64_t q6_features_b,
    hipStream_t stream) {
  return launch_gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_decode_out<float>(
      x, q5_weight_a, q5_weight_b, q6_weight_a, q6_weight_b, q5_out_a,
      q5_out_b, q6_out_a, q6_out_b, rows, in_features, q5_features_a,
      q5_features_b, q6_features_a, q6_features_b, stream);
}

extern "C" int hipengine_gguf_q6_q8_mixed_q6_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    const uint8_t* q8_weight_a,
    const uint8_t* q8_weight_b,
    float* q6_out_a,
    float* q6_out_b,
    float* q8_out_a,
    float* q8_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q6_features_a,
    int64_t q6_features_b,
    int64_t q8_features_a,
    int64_t q8_features_b,
    hipStream_t stream) {
  return launch_gguf_q6_q8_mixed_q6_fixed_meta_attention_decode_out<float>(
      x, q6_weight_a, q6_weight_b, q8_weight_a, q8_weight_b, q6_out_a,
      q6_out_b, q8_out_a, q8_out_b, rows, in_features, q6_features_a,
      q6_features_b, q8_features_a, q8_features_b, stream);
}

extern "C" int hipengine_gguf_q6_q8_mixed_attention_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* q6_weight_a,
    const uint8_t* q6_weight_b,
    const uint8_t* q8_weight_a,
    const uint8_t* q8_weight_b,
    float* q6_out_a,
    float* q6_out_b,
    float* q8_out_a,
    float* q8_out_b,
    int64_t rows,
    int64_t in_features,
    int64_t q6_features_a,
    int64_t q6_features_b,
    int64_t q8_features_a,
    int64_t q8_features_b,
    hipStream_t stream) {
  return launch_gguf_q6_q8_mixed_attention_decode_out<float>(
      x, q6_weight_a, q6_weight_b, q8_weight_a, q8_weight_b, q6_out_a,
      q6_out_b, q8_out_a, q8_out_b, rows, in_features, q6_features_a,
      q6_features_b, q8_features_a, q8_features_b, stream);
}

extern "C" int hipengine_gguf_q5_k_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_pack8_decode_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_pack8_decode_out<float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    uint16_t* out_a, uint16_t* out_b, int64_t rows, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_pair_pack8_decode_out<uint16_t>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features,
      threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    float* out_a, float* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_pack8_decode_out<float>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features_a,
      out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_decode_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_decode_out<float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    float* out_a, float* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_wave32x2_decode_out<float>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features_a,
      out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_fixed_meta_decode_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_fixed_meta_decode_out<float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    uint16_t* out_a, uint16_t* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_wave32x2_fixed_meta_decode_out<uint16_t>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features_a,
      out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    float* out_a, float* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_wave32x2_fixed_meta_decode_out<float>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features_a,
      out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out<float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    uint16_t* out_a, uint16_t* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_wave32x2_swar_pair_fixed_meta_decode_out<
      uint16_t>(x, qweight_a, qweight_b, out_a, out_b, rows, in_features,
                out_features_a, out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out(
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b,
    float* out_a, float* out_b, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q5_k_unequal_pair_wave32x2_swar_pair_fixed_meta_decode_out<
      float>(x, qweight_a, qweight_b, out_a, out_b, rows, in_features,
             out_features_a, out_features_b, threads, stream);
}

#define HIPENGINE_GGUF_K_SELECTED_PACK8_DP4A_EXPORT(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_selected_pack8_gemv_q8_1_dp4a_bf16_bf16_out( \
    const void* xq, const int64_t* selected, const uint8_t* qweight, uint16_t* out, \
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_selected_pack8_q8_1_dp4a_gemv_out<uint16_t, QTYPE>( \
      reinterpret_cast<const gguf_q8_1_block*>(xq), selected, qweight, out, x_rows, \
      rows, num_experts, in_features, out_features, threads, stream); \
}

HIPENGINE_GGUF_K_SELECTED_PACK8_DP4A_EXPORT(gguf_q5_k, 5)
HIPENGINE_GGUF_K_SELECTED_PACK8_DP4A_EXPORT(gguf_q6_k, 6)

extern "C" int hipengine_gguf_q8_0_exact_prefill_tile8x2_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q8_0_exact_prefill_tiled_out<uint16_t, uint16_t, 8, 2>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_exact_prefill_tile8x4_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q8_0_exact_prefill_tiled_out<uint16_t, uint16_t, 8, 4>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q8_0_exact_prefill_tiled_out<uint16_t, uint16_t, 16, 4>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}
