// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip.
// Baseline decode/prefill convolution kernels track @ 59195ed; DFlash tree
// t-loop variants track @ b95eaa5. Wrappers are hipEngine raw-pointer C ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;

namespace {

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
    uint32_t u32;
    float f32;
  } bits;
  bits.f32 = value;
  const uint32_t lsb = (bits.u32 >> 16) & 1U;
  bits.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits.u32 >> 16);
}

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

// Force the production kernel's separately rounded multiply without using a
// volatile local, which would spill every product to scratch under hipcc.
__device__ __forceinline__ float mul_f32_round(float lhs, float rhs) {
  float product;
  asm("v_mul_f32_e32 %0, %1, %2" : "=v"(product) : "v"(lhs), "v"(rhs));
  return product;
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

__device__ __forceinline__ float mul_add_decode_order_f32(float acc, float lhs, float rhs) {
  float product;
  float result;
  asm volatile(
      "v_mul_f32_e32 %[product], %[lhs], %[rhs]\n\t"
      "v_add_f32_e32 %[result], %[acc], %[product]"
      : [product] "=&v"(product), [result] "=v"(result)
      : [lhs] "v"(lhs), [rhs] "v"(rhs), [acc] "v"(acc));
  return result;
}

__global__ void qwen35_linear_attn_conv_decode_kernel(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }

  const int64_t offset = channel * kernel_size;
  float acc = 0.0f;
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    acc += value * conv_weight[offset + idx];
    conv_state[offset + idx] = value;
  }
  const float newest = hidden_states[channel];
  acc += newest * conv_weight[offset + kernel_size - 1];
  conv_state[offset + kernel_size - 1] = newest;
  out[channel] = silu_f32(acc);
}

template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_decode_lowp_kernel(
    const scalar_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }

  const int64_t offset = channel * kernel_size;
  float acc = 0.0f;
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    acc += value * conv_weight[offset + idx];
    conv_state[offset + idx] = value;
  }
  const float newest = scalar_to_float_qwen35(hidden_states[channel]);
  acc += newest * conv_weight[offset + kernel_size - 1];
  conv_state[offset + kernel_size - 1] = newest;
  out[channel] = silu_f32(acc);
}

// Batch-shaped sibling of the scalar decode kernel. Every grid-Y row owns one
// independent token and selects its persistent Conv state through
// state_indices. The arithmetic statements intentionally mirror the scalar
// kernel exactly; only row/state pointer offsets differ.
template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_decode_indexed_lowp_kernel(
    const scalar_t* __restrict__ hidden_states,
    float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    const int64_t* __restrict__ state_indices,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || channel >= channels) {
    return;
  }

  const int64_t state_slot = state_indices[row];
  const int64_t state_stride = channels * kernel_size;
  const int64_t offset = state_slot * state_stride + channel * kernel_size;
  const int64_t weight_offset = channel * kernel_size;
  float acc = 0.0f;
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    acc += value * conv_weight[weight_offset + idx];
    conv_state[offset + idx] = value;
  }
  const float newest = scalar_to_float_qwen35(hidden_states[row * channels + channel]);
  acc += newest * conv_weight[weight_offset + kernel_size - 1];
  conv_state[offset + kernel_size - 1] = newest;
  out[row * channels + channel] = silu_f32(acc);
}


__global__ void qwen35_linear_attn_conv_prefill_no_state_rows_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = tokens * channels;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = idx / channels;
    const int64_t channel = idx - token * channels;
    const int64_t offset = channel * kernel_size;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[offset + padded0 + 1] : hidden_states[(padded0 - 3) * channels + channel]);
      const float value1 = (
          padded1 < 3 ? conv_state[offset + padded1 + 1] : hidden_states[(padded1 - 3) * channels + channel]);
      const float value2 = (
          padded2 < 3 ? conv_state[offset + padded2 + 1] : hidden_states[(padded2 - 3) * channels + channel]);
      const float value3 = hidden_states[(padded3 - 3) * channels + channel];
      acc = mul_add_decode_order_f32(acc, value0, conv_weight[offset]);
      acc = mul_add_decode_order_f32(acc, value1, conv_weight[offset + 1]);
      acc = mul_add_decode_order_f32(acc, value2, conv_weight[offset + 2]);
      acc = mul_add_decode_order_f32(acc, value3, conv_weight[offset + 3]);
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[offset + padded + 1]
                : hidden_states[(padded - (kernel_size - 1)) * channels + channel]);
        acc = mul_add_decode_order_f32(acc, value, conv_weight[offset + k]);
      }
    }
    out[idx] = silu_f32(acc);
  }
}

__global__ void qwen35_linear_attn_conv_prefill_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    float* __restrict__ conv_state_rows,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = tokens * channels;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = idx / channels;
    const int64_t channel = idx - token * channels;
    const int64_t offset = channel * kernel_size;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[offset + padded0 + 1] : hidden_states[(padded0 - 3) * channels + channel]);
      const float value1 = (
          padded1 < 3 ? conv_state[offset + padded1 + 1] : hidden_states[(padded1 - 3) * channels + channel]);
      const float value2 = (
          padded2 < 3 ? conv_state[offset + padded2 + 1] : hidden_states[(padded2 - 3) * channels + channel]);
      const float value3 = hidden_states[(padded3 - 3) * channels + channel];
      if (conv_state_rows != nullptr) {
        const int64_t row_state = token * channels * kernel_size + offset;
        conv_state_rows[row_state + 0] = value0;
        conv_state_rows[row_state + 1] = value1;
        conv_state_rows[row_state + 2] = value2;
        conv_state_rows[row_state + 3] = value3;
      }
      volatile float product0 = value0 * conv_weight[offset];
      volatile float product1 = value1 * conv_weight[offset + 1];
      volatile float product2 = value2 * conv_weight[offset + 2];
      volatile float product3 = value3 * conv_weight[offset + 3];
      acc += product0;
      acc += product1;
      acc += product2;
      acc += product3;
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[offset + padded + 1]
                : hidden_states[(padded - (kernel_size - 1)) * channels + channel]);
        if (conv_state_rows != nullptr) {
          conv_state_rows[token * channels * kernel_size + offset + k] = value;
        }
        volatile float product = value * conv_weight[offset + k];
        acc += product;
      }
    }
    out[idx] = silu_f32(acc);
  }
}

constexpr int64_t QWEN35_CONV_TOKEN_TILE = 32;
constexpr int64_t QWEN35_CONV_CHANNEL_TILE = 128;
constexpr int64_t QWEN35_CONV_KERNEL_SIZE = 4;
constexpr int64_t QWEN35_CONV_WINDOW = QWEN35_CONV_TOKEN_TILE + QWEN35_CONV_KERNEL_SIZE - 1;

// Long-token exact convolution schedule inspired by llama.cpp@6e9007ae6
// ssm-conv.cu. One 128-thread block owns a channel tile and computes 32
// consecutive output rows. The 35x128 input window is loaded once into LDS;
// each output preserves the production kernel's four products, serial adds,
// and SiLU order.
__global__ __launch_bounds__(128) void qwen35_linear_attn_conv_prefill_tile32x128_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t channels) {
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * QWEN35_CONV_CHANNEL_TILE + lane;
  const int64_t token_base = static_cast<int64_t>(blockIdx.y) * QWEN35_CONV_TOKEN_TILE;
  const int64_t tile_tokens = min(QWEN35_CONV_TOKEN_TILE, tokens - token_base);
  __shared__ float tile[QWEN35_CONV_WINDOW][QWEN35_CONV_CHANNEL_TILE];

  const int64_t state_offset = channel * QWEN35_CONV_KERNEL_SIZE;
  const int64_t load_rows = tile_tokens + QWEN35_CONV_KERNEL_SIZE - 1;
  for (int64_t row = 0; row < load_rows; ++row) {
    float value = 0.0f;
    if (channel < channels) {
      const int64_t padded = token_base + row;
      value = padded < QWEN35_CONV_KERNEL_SIZE - 1
          ? conv_state[state_offset + padded + 1]
          : hidden_states[(padded - (QWEN35_CONV_KERNEL_SIZE - 1)) * channels + channel];
    }
    tile[row][lane] = value;
  }
  __syncthreads();

  if (channel >= channels) {
    return;
  }
  const float weight0 = conv_weight[state_offset + 0];
  const float weight1 = conv_weight[state_offset + 1];
  const float weight2 = conv_weight[state_offset + 2];
  const float weight3 = conv_weight[state_offset + 3];
#pragma unroll 1
  for (int64_t token = 0; token < tile_tokens; ++token) {
    const float product0 = mul_f32_round(tile[token + 0][lane], weight0);
    const float product1 = mul_f32_round(tile[token + 1][lane], weight1);
    const float product2 = mul_f32_round(tile[token + 2][lane], weight2);
    const float product3 = mul_f32_round(tile[token + 3][lane], weight3);
    float acc = 0.0f;
    acc += product0;
    acc += product1;
    acc += product2;
    acc += product3;
    out[(token_base + token) * channels + channel] = silu_f32(acc);
  }
}

template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_prefill_lowp_kernel(
    const scalar_t* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = tokens * channels;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = idx / channels;
    const int64_t channel = idx - token * channels;
    const int64_t offset = channel * kernel_size;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[offset + padded0 + 1] : scalar_to_float_qwen35(hidden_states[(padded0 - 3) * channels + channel]));
      const float value1 = (
          padded1 < 3 ? conv_state[offset + padded1 + 1] : scalar_to_float_qwen35(hidden_states[(padded1 - 3) * channels + channel]));
      const float value2 = (
          padded2 < 3 ? conv_state[offset + padded2 + 1] : scalar_to_float_qwen35(hidden_states[(padded2 - 3) * channels + channel]));
      const float value3 = scalar_to_float_qwen35(hidden_states[(padded3 - 3) * channels + channel]);
      volatile float product0 = value0 * conv_weight[offset];
      volatile float product1 = value1 * conv_weight[offset + 1];
      volatile float product2 = value2 * conv_weight[offset + 2];
      volatile float product3 = value3 * conv_weight[offset + 3];
      acc += product0;
      acc += product1;
      acc += product2;
      acc += product3;
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[offset + padded + 1]
                : scalar_to_float_qwen35(hidden_states[(padded - (kernel_size - 1)) * channels + channel]));
        volatile float product = value * conv_weight[offset + k];
        acc += product;
      }
    }
    out[idx] = silu_f32(acc);
  }
}

template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_prefill_state_lowp_kernel(
    const scalar_t* __restrict__ hidden_states,
    float* __restrict__ conv_state,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = channels * kernel_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t channel = idx / kernel_size;
    const int64_t k = idx - channel * kernel_size;
    const int64_t offset = channel * kernel_size;
    if (tokens >= kernel_size) {
      conv_state[offset + k] = scalar_to_float_qwen35(hidden_states[(tokens - kernel_size + k) * channels + channel]);
    } else {
      const int64_t kept = kernel_size - tokens;
      conv_state[offset + k] = (
          k < kept
              ? conv_state[offset + k + tokens]
              : scalar_to_float_qwen35(hidden_states[(k - kept) * channels + channel]));
    }
  }
}

__global__ void qwen35_linear_attn_conv_prefill_state_kernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ conv_state,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = channels * kernel_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t channel = idx / kernel_size;
    const int64_t k = idx - channel * kernel_size;
    const int64_t offset = channel * kernel_size;
    if (tokens >= kernel_size) {
      conv_state[offset + k] = hidden_states[(tokens - kernel_size + k) * channels + channel];
    } else {
      const int64_t kept = kernel_size - tokens;
      conv_state[offset + k] = (
          k < kept
              ? conv_state[offset + k + tokens]
              : hidden_states[(k - kept) * channels + channel]);
    }
  }
}

__global__ void qwen35_linear_attn_conv_prefill_segments_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t segment = static_cast<int64_t>(blockIdx.y);
  if (segment >= segments) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  const int64_t tokens = end - start;
  if (tokens <= 0) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = channels * kernel_size;
  const int64_t segment_size = tokens * channels;
  for (int64_t local_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       local_idx < segment_size;
       local_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = local_idx / channels;
    const int64_t channel = local_idx - token * channels;
    const int64_t state_offset = state_slot * state_stride + channel * kernel_size;
    const int64_t global_token = start + token;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[state_offset + padded0 + 1] : hidden_states[(start + padded0 - 3) * channels + channel]);
      const float value1 = (
          padded1 < 3 ? conv_state[state_offset + padded1 + 1] : hidden_states[(start + padded1 - 3) * channels + channel]);
      const float value2 = (
          padded2 < 3 ? conv_state[state_offset + padded2 + 1] : hidden_states[(start + padded2 - 3) * channels + channel]);
      const float value3 = hidden_states[(start + padded3 - 3) * channels + channel];
      volatile float product0 = value0 * conv_weight[channel * kernel_size];
      volatile float product1 = value1 * conv_weight[channel * kernel_size + 1];
      volatile float product2 = value2 * conv_weight[channel * kernel_size + 2];
      volatile float product3 = value3 * conv_weight[channel * kernel_size + 3];
      acc += product0;
      acc += product1;
      acc += product2;
      acc += product3;
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[state_offset + padded + 1]
                : hidden_states[(start + padded - (kernel_size - 1)) * channels + channel]);
        volatile float product = value * conv_weight[channel * kernel_size + k];
        acc += product;
      }
    }
    out[global_token * channels + channel] = silu_f32(acc);
  }
}

__global__ void qwen35_linear_attn_conv_prefill_segments_state_rows_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    float* __restrict__ conv_state_rows,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t segment = static_cast<int64_t>(blockIdx.y);
  if (segment >= segments) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  const int64_t tokens = end - start;
  if (tokens <= 0) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = channels * kernel_size;
  const int64_t segment_size = tokens * channels;
  for (int64_t local_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       local_idx < segment_size;
       local_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = local_idx / channels;
    const int64_t channel = local_idx - token * channels;
    const int64_t state_offset = state_slot * state_stride + channel * kernel_size;
    const int64_t row_state = (start + token) * state_stride + channel * kernel_size;
    const int64_t global_token = start + token;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[state_offset + padded0 + 1] : hidden_states[(start + padded0 - 3) * channels + channel]);
      const float value1 = (
          padded1 < 3 ? conv_state[state_offset + padded1 + 1] : hidden_states[(start + padded1 - 3) * channels + channel]);
      const float value2 = (
          padded2 < 3 ? conv_state[state_offset + padded2 + 1] : hidden_states[(start + padded2 - 3) * channels + channel]);
      const float value3 = hidden_states[(start + padded3 - 3) * channels + channel];
      conv_state_rows[row_state + 0] = value0;
      conv_state_rows[row_state + 1] = value1;
      conv_state_rows[row_state + 2] = value2;
      conv_state_rows[row_state + 3] = value3;
      volatile float product0 = value0 * conv_weight[channel * kernel_size];
      volatile float product1 = value1 * conv_weight[channel * kernel_size + 1];
      volatile float product2 = value2 * conv_weight[channel * kernel_size + 2];
      volatile float product3 = value3 * conv_weight[channel * kernel_size + 3];
      acc += product0;
      acc += product1;
      acc += product2;
      acc += product3;
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[state_offset + padded + 1]
                : hidden_states[(start + padded - (kernel_size - 1)) * channels + channel]);
        conv_state_rows[row_state + k] = value;
        volatile float product = value * conv_weight[channel * kernel_size + k];
        acc += product;
      }
    }
    out[global_token * channels + channel] = silu_f32(acc);
  }
}

__global__ void qwen35_linear_attn_conv_prefill_segments_state_kernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ conv_state,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t segment = static_cast<int64_t>(blockIdx.y);
  if (segment >= segments) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  const int64_t tokens = end - start;
  if (tokens <= 0) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = channels * kernel_size;
  for (int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < channels;
       channel += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t state_offset = state_slot * state_stride + channel * kernel_size;
    if (tokens >= kernel_size) {
      for (int64_t k = 0; k < kernel_size; ++k) {
        conv_state[state_offset + k] = hidden_states[(end - kernel_size + k) * channels + channel];
      }
    } else {
      const int64_t kept = kernel_size - tokens;
      for (int64_t k = 0; k < kept; ++k) {
        conv_state[state_offset + k] = conv_state[state_offset + k + tokens];
      }
      for (int64_t k = kept; k < kernel_size; ++k) {
        conv_state[state_offset + k] = hidden_states[(start + k - kept) * channels + channel];
      }
    }
  }
}

// One-launch parent-indexed linear-attention conv update for an N-node tree.
//
// Ported from nano-vllm-amd@b95eaa5. Each thread owns one channel and walks
// t=0..N in topological order, so parent_ids[t] < t references a state row
// written earlier by the same thread. No host depth loop or cross-thread sync
// is required.
template <typename scalar_t>
__global__ void qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel(
    const scalar_t* __restrict__ hidden_states,
    const float* __restrict__ base_conv_state,
    float* __restrict__ tree_conv_state,
    const float* __restrict__ conv_weight,
    const int64_t* __restrict__ parent_ids,
    float* __restrict__ out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int64_t state_stride = channels * kernel_size;
  const int64_t channel_base = channel * kernel_size;
  const float* weight = conv_weight + channel_base;
  for (int64_t t = 0; t < max_nodes; ++t) {
    const int64_t parent = parent_ids[t];
    const float* parent_state = parent < 0
        ? base_conv_state + channel_base
        : tree_conv_state + parent * state_stride + channel_base;
    float* node_state = tree_conv_state + t * state_stride + channel_base;
    float acc = 0.0f;
    for (int64_t k = 0; k + 1 < kernel_size; ++k) {
      const float value = parent_state[k + 1];
      node_state[k] = value;
      acc += value * weight[k];
    }
    const float token_value = scalar_to_float_qwen35(hidden_states[t * channels + channel]);
    node_state[kernel_size - 1] = token_value;
    acc += token_value * weight[kernel_size - 1];
    out[t * channels + channel] = silu_f32(acc);
  }
}

// Single-chain verifier variant.  For parent_ids == [-1,0,1,...], keep the
// rolling convolution state in registers and still materialize every row so
// partial accepts can commit the exact selected prefix without a replay.
template <typename scalar_t>
__global__ void qwen35_linear_attn_chain_conv_decode_lowp_tloop_kernel(
    const scalar_t* __restrict__ hidden_states,
    const float* __restrict__ base_conv_state,
    float* __restrict__ chain_conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size) {
  constexpr int K_MAX = 8;
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels || kernel_size <= 0 || kernel_size > K_MAX) {
    return;
  }
  const int64_t state_stride = channels * kernel_size;
  const int64_t channel_base = channel * kernel_size;
  const float* weight = conv_weight + channel_base;
  float state[K_MAX];
  #pragma unroll
  for (int k = 0; k < K_MAX; ++k) {
    state[k] = k < kernel_size ? base_conv_state[channel_base + k] : 0.0f;
  }
  for (int64_t t = 0; t < max_nodes; ++t) {
    float* node_state = chain_conv_state + t * state_stride + channel_base;
    float acc = 0.0f;
    for (int64_t k = 0; k + 1 < kernel_size; ++k) {
      const float value = state[k + 1];
      state[k] = value;
      node_state[k] = value;
      acc += value * weight[k];
    }
    const float token_value = scalar_to_float_qwen35(hidden_states[t * channels + channel]);
    state[kernel_size - 1] = token_value;
    node_state[kernel_size - 1] = token_value;
    acc += token_value * weight[kernel_size - 1];
    out[t * channels + channel] = silu_f32(acc);
  }
}

// Rollback-safe native verifier sibling.  The producer already loads every
// immutable base-state element; retain those exact bits while materializing the
// ordinary row journal, avoiding a separate pre-verify state-copy pass.
template <typename scalar_t>
__global__ void qwen35_linear_attn_chain_conv_decode_lowp_snapshot_tloop_kernel(
    const scalar_t* __restrict__ hidden_states,
    const float* __restrict__ base_conv_state,
    float* __restrict__ chain_conv_state,
    float* __restrict__ initial_conv_state_snapshot,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size) {
  constexpr int K_MAX = 8;
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels || kernel_size <= 0 || kernel_size > K_MAX) {
    return;
  }
  const int64_t state_stride = channels * kernel_size;
  const int64_t channel_base = channel * kernel_size;
  const float* weight = conv_weight + channel_base;
  float state[K_MAX];
  #pragma unroll
  for (int k = 0; k < K_MAX; ++k) {
    const float value = k < kernel_size ? base_conv_state[channel_base + k] : 0.0f;
    state[k] = value;
    if (k < kernel_size) {
      initial_conv_state_snapshot[channel_base + k] = value;
    }
  }
  for (int64_t t = 0; t < max_nodes; ++t) {
    float* node_state = chain_conv_state + t * state_stride + channel_base;
    float acc = 0.0f;
    for (int64_t k = 0; k + 1 < kernel_size; ++k) {
      const float value = state[k + 1];
      state[k] = value;
      node_state[k] = value;
      acc += value * weight[k];
    }
    const float token_value = scalar_to_float_qwen35(hidden_states[t * channels + channel]);
    state[kernel_size - 1] = token_value;
    node_state[kernel_size - 1] = token_value;
    acc += token_value * weight[kernel_size - 1];
    out[t * channels + channel] = silu_f32(acc);
  }
}

// One-grid Qwen3.6 verifier owner for two independent dense-F32 alpha/beta
// projections plus the rollback-safe chain-Conv producer. Pair blocks preserve
// the scalar local256 FMA/reduction order; ROW_TILE=2 is selected only for four
// rows. Final blocks preserve the existing snapshot-Conv channel body. GDN is
// a later launch, so kernel completion is the only synchronization required
// between these independent block groups.
template <int ROW_TILE>
__global__ void qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_kernel(
    const uint16_t* __restrict__ norm,
    const float* __restrict__ weight_a,
    const float* __restrict__ weight_b,
    uint16_t* __restrict__ out_a,
    uint16_t* __restrict__ out_b,
    const uint16_t* __restrict__ hidden_states,
    const float* __restrict__ base_conv_state,
    float* __restrict__ chain_conv_state,
    float* __restrict__ initial_conv_state_snapshot,
    const float* __restrict__ conv_weight,
    float* __restrict__ conv_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t channels,
    int64_t kernel_size) {
  constexpr int kThreads = 256;
  constexpr int kValuesPerThread = 8;
  constexpr int kStride = kThreads * kValuesPerThread;
  const int tid = threadIdx.x;
  const int64_t pair_cols = 2 * out_features;
  const int64_t row_groups = (rows + ROW_TILE - 1) / ROW_TILE;
  const int64_t pair_blocks = pair_cols * row_groups;
  const int64_t linear_block = blockIdx.x;

  if (linear_block < pair_blocks) {
    const int64_t pair_col = linear_block % pair_cols;
    const int64_t row_base = (linear_block / pair_cols) * ROW_TILE;
    const bool use_b = pair_col >= out_features;
    const int64_t out_col = use_b ? pair_col - out_features : pair_col;
    const float* __restrict__ weight = use_b ? weight_b : weight_a;
    uint16_t* __restrict__ out = use_b ? out_b : out_a;
    const int64_t w_base = out_col * in_features;

    float acc[ROW_TILE];
#pragma unroll
    for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
      acc[local_row] = 0.0f;
    }
    const int64_t main_end = (in_features / kStride) * kStride;
    for (int64_t k_base = static_cast<int64_t>(tid) * kValuesPerThread;
         k_base < main_end;
         k_base += kStride) {
      const float* __restrict__ w_ptr = weight + w_base + k_base;
#pragma unroll
      for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
        const int64_t row = row_base + local_row;
        if (row < rows) {
          const uint16_t* __restrict__ x_ptr = norm + row * in_features + k_base;
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[0]), w_ptr[0], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[1]), w_ptr[1], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[2]), w_ptr[2], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[3]), w_ptr[3], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[4]), w_ptr[4], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[5]), w_ptr[5], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[6]), w_ptr[6], acc[local_row]);
          acc[local_row] = fmaf(bf16_bits_to_float(x_ptr[7]), w_ptr[7], acc[local_row]);
        }
      }
    }
    for (int64_t in_col = main_end + tid;
         in_col < in_features;
         in_col += kThreads) {
      const float w = weight[w_base + in_col];
#pragma unroll
      for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
        const int64_t row = row_base + local_row;
        if (row < rows) {
          acc[local_row] = fmaf(
              bf16_bits_to_float(norm[row * in_features + in_col]),
              w,
              acc[local_row]);
        }
      }
    }

    __shared__ float partial[ROW_TILE][kThreads];
#pragma unroll
    for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
      partial[local_row][tid] = acc[local_row];
    }
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
#pragma unroll
        for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
          partial[local_row][tid] += partial[local_row][tid + stride];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
#pragma unroll
      for (int local_row = 0; local_row < ROW_TILE; ++local_row) {
        const int64_t row = row_base + local_row;
        if (row < rows) {
          out[row * out_features + out_col] =
              float_to_bf16_bits(partial[local_row][0]);
        }
      }
    }
    return;
  }

  constexpr int K_MAX = 8;
  const int64_t conv_block = linear_block - pair_blocks;
  const int64_t channel = conv_block * blockDim.x + tid;
  if (channel >= channels || kernel_size <= 0 || kernel_size > K_MAX) {
    return;
  }
  const int64_t state_stride = channels * kernel_size;
  const int64_t channel_base = channel * kernel_size;
  const float* weight = conv_weight + channel_base;
  float state[K_MAX];
#pragma unroll
  for (int k = 0; k < K_MAX; ++k) {
    const float value = k < kernel_size ? base_conv_state[channel_base + k] : 0.0f;
    state[k] = value;
    if (k < kernel_size) {
      initial_conv_state_snapshot[channel_base + k] = value;
    }
  }
  for (int64_t row = 0; row < rows; ++row) {
    float* node_state = chain_conv_state + row * state_stride + channel_base;
    float acc = 0.0f;
    for (int64_t k = 0; k + 1 < kernel_size; ++k) {
      const float value = state[k + 1];
      state[k] = value;
      node_state[k] = value;
      acc += value * weight[k];
    }
    const float token_value =
        scalar_to_float_qwen35(hidden_states[row * channels + channel]);
    state[kernel_size - 1] = token_value;
    node_state[kernel_size - 1] = token_value;
    acc += token_value * weight[kernel_size - 1];
    conv_out[row * channels + channel] = silu_f32(acc);
  }
}

bool valid_conv_shape(int64_t channels, int64_t kernel_size) {
  return channels > 0 && kernel_size > 0;
}

}  // namespace

extern "C" int hipengine_qwen35_linear_attn_conv_decode_f32(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_decode_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_decode_bf16(
    const uint16_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_decode_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_decode_indexed_bf16(
    const uint16_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    const int64_t* state_indices,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (rows <= 0 || !valid_conv_shape(channels, kernel_size) || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_decode_indexed_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks), static_cast<unsigned int>(rows)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      state_indices,
      rows,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_decode_fp16(
    const half_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_decode_lowp_kernel<half_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_tree_conv_decode_bf16_tloop(
    const uint16_t* hidden_states,
    const float* base_conv_state,
    float* tree_conv_state,
    const float* conv_weight,
    const int64_t* parent_ids,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_conv_shape(channels, kernel_size) || kernel_size < 2) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      tree_conv_state,
      conv_weight,
      parent_ids,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_tree_conv_decode_fp16_tloop(
    const half_t* hidden_states,
    const float* base_conv_state,
    float* tree_conv_state,
    const float* conv_weight,
    const int64_t* parent_ids,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_conv_shape(channels, kernel_size) || kernel_size < 2) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel<half_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      tree_conv_state,
      conv_weight,
      parent_ids,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_chain_conv_decode_bf16_tloop(
    const uint16_t* hidden_states,
    const float* base_conv_state,
    float* leaf_conv_state,
    const float* conv_weight,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_conv_shape(channels, kernel_size) || kernel_size < 2 || kernel_size > 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_chain_conv_decode_lowp_tloop_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      leaf_conv_state,
      conv_weight,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_chain_conv_decode_bf16_snapshot_tloop(
    const uint16_t* hidden_states,
    const float* base_conv_state,
    float* leaf_conv_state,
    float* initial_conv_state_snapshot,
    const float* conv_weight,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!initial_conv_state_snapshot || max_nodes <= 0 ||
      !valid_conv_shape(channels, kernel_size) || kernel_size < 2 ||
      kernel_size > 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_chain_conv_decode_lowp_snapshot_tloop_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      leaf_conv_state,
      initial_conv_state_snapshot,
      conv_weight,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w(
    const uint16_t* norm,
    const float* weight_a,
    const float* weight_b,
    uint16_t* out_a,
    uint16_t* out_b,
    const uint16_t* hidden_states,
    const float* base_conv_state,
    float* chain_conv_state,
    float* initial_conv_state_snapshot,
    const float* conv_weight,
    float* conv_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  constexpr int64_t kInFeatures = 5120;
  constexpr int64_t kOutFeatures = 48;
  constexpr int64_t kChannels = 10240;
  constexpr int64_t kKernelSize = 4;
  if (!norm || !weight_a || !weight_b || !out_a || !out_b ||
      !hidden_states || !base_conv_state || !chain_conv_state ||
      !initial_conv_state_snapshot || !conv_weight || !conv_out ||
      rows < 1 || rows > 4 || in_features != kInFeatures ||
      out_features != kOutFeatures || channels != kChannels ||
      kernel_size != kKernelSize) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int row_tile = rows == 4 ? 2 : 1;
  const int64_t row_groups = (rows + row_tile - 1) / row_tile;
  const int64_t pair_blocks = 2 * out_features * row_groups;
  const int64_t conv_blocks = (channels + 255) / 256;
  const dim3 grid(static_cast<unsigned int>(pair_blocks + conv_blocks));
  if (rows == 4) {
    hipLaunchKernelGGL(
        (qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_kernel<2>),
        grid,
        dim3(256),
        0,
        stream,
        norm,
        weight_a,
        weight_b,
        out_a,
        out_b,
        hidden_states,
        base_conv_state,
        chain_conv_state,
        initial_conv_state_snapshot,
        conv_weight,
        conv_out,
        rows,
        in_features,
        out_features,
        channels,
        kernel_size);
  } else {
    hipLaunchKernelGGL(
        (qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_kernel<1>),
        grid,
        dim3(256),
        0,
        stream,
        norm,
        weight_a,
        weight_b,
        out_a,
        out_b,
        hidden_states,
        base_conv_state,
        chain_conv_state,
        initial_conv_state_snapshot,
        conv_weight,
        conv_out,
        rows,
        in_features,
        out_features,
        channels,
        kernel_size);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_chain_conv_decode_f32_tloop(
    const float* hidden_states,
    const float* base_conv_state,
    float* leaf_conv_state,
    const float* conv_weight,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_conv_shape(channels, kernel_size) || kernel_size < 2 || kernel_size > 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_chain_conv_decode_lowp_tloop_kernel<float>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      leaf_conv_state,
      conv_weight,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_chain_conv_decode_fp16_tloop(
    const half_t* hidden_states,
    const float* base_conv_state,
    float* leaf_conv_state,
    const float* conv_weight,
    float* out,
    int64_t max_nodes,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_conv_shape(channels, kernel_size) || kernel_size < 2 || kernel_size > 8) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_chain_conv_decode_lowp_tloop_kernel<half_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      base_conv_state,
      leaf_conv_state,
      conv_weight,
      out,
      max_nodes,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_f32(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_no_state_rows_kernel,
      dim3(static_cast<unsigned int>(output_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      tokens,
      channels,
      kernel_size);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int64_t state_size = channels * kernel_size;
  const int64_t state_blocks64 = (state_size + threads - 1) / threads;
  const int state_blocks = static_cast<int>(state_blocks64 < 65535 ? state_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_state_kernel,
      dim3(static_cast<unsigned int>(state_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      tokens,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_f32_tile32x128(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size) ||
      kernel_size != QWEN35_CONV_KERNEL_SIZE) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t channel_blocks64 =
      (channels + QWEN35_CONV_CHANNEL_TILE - 1) / QWEN35_CONV_CHANNEL_TILE;
  const int64_t token_blocks64 =
      (tokens + QWEN35_CONV_TOKEN_TILE - 1) / QWEN35_CONV_TOKEN_TILE;
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_tile32x128_kernel,
      dim3(static_cast<unsigned int>(channel_blocks64), static_cast<unsigned int>(token_blocks64)),
      dim3(static_cast<unsigned int>(QWEN35_CONV_CHANNEL_TILE)),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      tokens,
      channels);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int threads = 256;
  const int64_t state_size = channels * kernel_size;
  const int64_t state_blocks64 = (state_size + threads - 1) / threads;
  const int state_blocks = static_cast<int>(state_blocks64 < 65535 ? state_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_state_kernel,
      dim3(static_cast<unsigned int>(state_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      tokens,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_f32_state_rows(
    const float* hidden_states,
    const float* conv_state,
    float* conv_state_rows,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (conv_state_rows == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_kernel,
      dim3(static_cast<unsigned int>(output_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_state_rows,
      conv_weight,
      out,
      tokens,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_fp16(
    const half_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_prefill_lowp_kernel<half_t>),
      dim3(static_cast<unsigned int>(output_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      tokens,
      channels,
      kernel_size);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int64_t state_size = channels * kernel_size;
  const int64_t state_blocks64 = (state_size + threads - 1) / threads;
  const int state_blocks = static_cast<int>(state_blocks64 < 65535 ? state_blocks64 : 65535);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_prefill_state_lowp_kernel<half_t>),
      dim3(static_cast<unsigned int>(state_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      tokens,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_segments_f32(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || !valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = total_tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_segments_kernel,
      dim3(static_cast<unsigned int>(output_blocks), static_cast<unsigned int>(segments)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      cu_seqlens,
      state_indices,
      segments,
      channels,
      kernel_size);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int64_t state_blocks64 = (channels + threads - 1) / threads;
  const int state_blocks = static_cast<int>(state_blocks64 < 65535 ? state_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_segments_state_kernel,
      dim3(static_cast<unsigned int>(state_blocks), static_cast<unsigned int>(segments)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      cu_seqlens,
      state_indices,
      segments,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_segments_f32_state_rows(
    const float* hidden_states,
    const float* conv_state,
    float* conv_state_rows,
    const float* conv_weight,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || !valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (conv_state == nullptr || conv_state_rows == nullptr || cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = total_tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_segments_state_rows_kernel,
      dim3(static_cast<unsigned int>(output_blocks), static_cast<unsigned int>(segments)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_state_rows,
      conv_weight,
      out,
      cu_seqlens,
      state_indices,
      segments,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}
