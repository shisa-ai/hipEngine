// GGUF Q4_K/Q5_K/Q6_K/Q8_0 token embedding lookup kernels.

#include <hip/hip_runtime.h>

#include <cstdint>

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  uint16_t bits = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
  uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
  uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x03FFu;
  uint32_t out_bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      out_bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03FFu;
      out_bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    out_bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    out_bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = out_bits;
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

__device__ inline float q4_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane];
  const uint8_t q = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  return d * static_cast<float>(q4_k_scale(scales, subblock)) * static_cast<float>(q) -
         dmin * static_cast<float>(q4_k_min(scales, subblock));
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

__device__ inline float q8_0_weight(const uint8_t* row, int k) {
  const int block_idx = k / Q8_0_BLOCK;
  const int lane = k & (Q8_0_BLOCK - 1);
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q8_0_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
  return d * static_cast<float>(q);
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

__launch_bounds__(256, 2)
__global__ void gguf_q4_k_embedding_bf16_out_kernel(
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int rows,
    int hidden_size,
    int vocab_size) {
  const int row = static_cast<int>(blockIdx.y);
  const int k0 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t token_id64 = token_ids[row];
  if (token_id64 < 0 || token_id64 >= static_cast<int64_t>(vocab_size)) {
    return;
  }
  const int token_id = static_cast<int>(token_id64);
  const int64_t row_bytes = (static_cast<int64_t>(hidden_size) / QK_K) * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row = qweight + static_cast<int64_t>(token_id) * row_bytes;
  for (int k = k0; k < hidden_size; k += static_cast<int>(gridDim.x) * blockDim.x) {
    out[static_cast<int64_t>(row) * hidden_size + k] = float_to_bf16_bits(q4_k_weight(weight_row, k));
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_q5_k_embedding_bf16_out_kernel(
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int rows,
    int hidden_size,
    int vocab_size) {
  const int row = static_cast<int>(blockIdx.y);
  const int k0 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t token_id64 = token_ids[row];
  if (token_id64 < 0 || token_id64 >= static_cast<int64_t>(vocab_size)) {
    return;
  }
  const int token_id = static_cast<int>(token_id64);
  const int64_t row_bytes = (static_cast<int64_t>(hidden_size) / QK_K) * Q5_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row = qweight + static_cast<int64_t>(token_id) * row_bytes;
  for (int k = k0; k < hidden_size; k += static_cast<int>(gridDim.x) * blockDim.x) {
    out[static_cast<int64_t>(row) * hidden_size + k] = float_to_bf16_bits(q5_k_weight(weight_row, k));
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_q6_k_embedding_bf16_out_kernel(
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int rows,
    int hidden_size,
    int vocab_size) {
  const int row = static_cast<int>(blockIdx.y);
  const int k0 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t token_id64 = token_ids[row];
  if (token_id64 < 0 || token_id64 >= static_cast<int64_t>(vocab_size)) {
    return;
  }
  const int token_id = static_cast<int>(token_id64);
  const int64_t row_bytes = (static_cast<int64_t>(hidden_size) / QK_K) * Q6_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row = qweight + static_cast<int64_t>(token_id) * row_bytes;
  for (int k = k0; k < hidden_size; k += static_cast<int>(gridDim.x) * blockDim.x) {
    out[static_cast<int64_t>(row) * hidden_size + k] = float_to_bf16_bits(q6_k_weight(weight_row, k));
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_q8_0_embedding_bf16_out_kernel(
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int rows,
    int hidden_size,
    int vocab_size) {
  const int row = static_cast<int>(blockIdx.y);
  const int k0 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t token_id64 = token_ids[row];
  if (token_id64 < 0 || token_id64 >= static_cast<int64_t>(vocab_size)) {
    return;
  }
  const int token_id = static_cast<int>(token_id64);
  const int64_t row_bytes = (static_cast<int64_t>(hidden_size) / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row = qweight + static_cast<int64_t>(token_id) * row_bytes;
  for (int k = k0; k < hidden_size; k += static_cast<int>(gridDim.x) * blockDim.x) {
    out[static_cast<int64_t>(row) * hidden_size + k] = float_to_bf16_bits(q8_0_weight(weight_row, k));
  }
}

extern "C" int hipengine_gguf_q4_k_embedding_bf16_out(
    const int64_t* token_ids,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || hidden_size % QK_K != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const int64_t blocks_x = (hidden_size + threads - 1) / threads;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(rows));
  gguf_q4_k_embedding_bf16_out_kernel<<<grid, block, 0, stream>>>(
      token_ids,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden_size),
      static_cast<int>(vocab_size));
  return hipGetLastError();
}

extern "C" int hipengine_gguf_q5_k_embedding_bf16_out(
    const int64_t* token_ids,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || hidden_size % QK_K != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const int64_t blocks_x = (hidden_size + threads - 1) / threads;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(rows));
  gguf_q5_k_embedding_bf16_out_kernel<<<grid, block, 0, stream>>>(
      token_ids,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden_size),
      static_cast<int>(vocab_size));
  return hipGetLastError();
}

extern "C" int hipengine_gguf_q6_k_embedding_bf16_out(
    const int64_t* token_ids,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || hidden_size % QK_K != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const int64_t blocks_x = (hidden_size + threads - 1) / threads;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(rows));
  gguf_q6_k_embedding_bf16_out_kernel<<<grid, block, 0, stream>>>(
      token_ids,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden_size),
      static_cast<int>(vocab_size));
  return hipGetLastError();
}

extern "C" int hipengine_gguf_q8_0_embedding_bf16_out(
    const int64_t* token_ids,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || vocab_size <= 0 || hidden_size % Q8_0_BLOCK != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const int64_t blocks_x = (hidden_size + threads - 1) / threads;
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(rows));
  gguf_q8_0_embedding_bf16_out_kernel<<<grid, block, 0, stream>>>(
      token_ids,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden_size),
      static_cast<int>(vocab_size));
  return hipGetLastError();
}
