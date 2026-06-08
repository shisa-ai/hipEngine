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

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
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


__global__ void qwen35_linear_attn_conv_prefill_kernel(
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
        volatile float product = value * conv_weight[offset + k];
        acc += product;
      }
    }
    out[idx] = silu_f32(acc);
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
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size) || tokens < kernel_size) {
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

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_fp16(
    const half_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size) || tokens < kernel_size) {
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
