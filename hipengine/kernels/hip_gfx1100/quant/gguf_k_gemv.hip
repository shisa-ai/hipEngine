// GGUF Q8_0/Q5_K/Q6_K raw-byte GEMV bring-up kernels.
// These preserve GGML quant math and are separate from PARO/AWQ kernels.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int QK_K = 256;
constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;

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

__device__ inline float q5_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q5_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane];
  const uint8_t ql = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane] >> subblock) & 0x01u;
  const uint8_t q = ql | (high << 4);
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
extern "C" int hipengine_##NAME##_dual_gemv_bf16_bf16_out( \
    const uint16_t* x, const uint8_t* qweight_a, const uint8_t* qweight_b, \
    uint16_t* out_a, uint16_t* out_b, int64_t rows, int64_t in_features, \
    int64_t out_features, int64_t threads, hipStream_t stream) { \
  return launch_gguf_k_dual_gemv_out<uint16_t, uint16_t, QTYPE>( \
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features, threads, stream); \
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
}

HIPENGINE_GGUF_K_EXPORTS(gguf_q8_0, 8)
HIPENGINE_GGUF_K_EXPORTS(gguf_q5_k, 5)
HIPENGINE_GGUF_K_EXPORTS(gguf_q6_k, 6)
