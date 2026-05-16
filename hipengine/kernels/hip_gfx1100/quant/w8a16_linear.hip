// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// W8A16 linear kernel bodies are preserved; wrappers are hipENGINE C ABI.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

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

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

__global__ void w8a16_linear_kernel(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += bf16_bits_to_float(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += bf16_bits_to_float(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += bf16_bits_to_float(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += bf16_bits_to_float(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += bf16_bits_to_float(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += bf16_bits_to_float(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += bf16_bits_to_float(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += bf16_bits_to_float(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += bf16_bits_to_float(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


template <typename scalar_t>
__global__ void w8a16_linear_lowp_out_kernel(
    const scalar_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    scalar_t* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += scalar_to_float_qwen35(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += scalar_to_float_qwen35(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = static_cast<scalar_t>(partial[0] * weight_scale[row]);
  }
}


__global__ void w8a16_shared_gate_up_silu_fp16_kernel(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t hidden_size,
    int64_t intermediate_size) {
  extern __shared__ float partial[];
  float* gate_partials[4] = {partial, partial + blockDim.x, partial + 2 * blockDim.x, partial + 3 * blockDim.x};
  float* up_partials[4] = {
      partial + 4 * blockDim.x,
      partial + 5 * blockDim.x,
      partial + 6 * blockDim.x,
      partial + 7 * blockDim.x,
  };
  const int64_t j_base = static_cast<int64_t>(blockIdx.x) * 4;
  const int64_t token = blockIdx.y;
  if (j_base >= intermediate_size) {
    return;
  }

  bool active[4];
  int64_t gate_rows[4];
  int64_t up_rows[4];
  float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int64_t j = j_base + slot;
    active[slot] = j < intermediate_size;
    gate_rows[slot] = j * hidden_size;
    up_rows[slot] = (j + intermediate_size) * hidden_size;
  }

  const int64_t hidden_row = token * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const float x0 = static_cast<float>(hidden_states[h_off + 0]);
    const float x1 = static_cast<float>(hidden_states[h_off + 1]);
    const float x2 = static_cast<float>(hidden_states[h_off + 2]);
    const float x3 = static_cast<float>(hidden_states[h_off + 3]);
    const float x4 = static_cast<float>(hidden_states[h_off + 4]);
    const float x5 = static_cast<float>(hidden_states[h_off + 5]);
    const float x6 = static_cast<float>(hidden_states[h_off + 6]);
    const float x7 = static_cast<float>(hidden_states[h_off + 7]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t g_off = gate_rows[slot] + k;
        const int64_t u_off = up_rows[slot] + k;
        gate_acc[slot] += x0 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 0]));
        gate_acc[slot] += x1 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 1]));
        gate_acc[slot] += x2 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 2]));
        gate_acc[slot] += x3 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 3]));
        gate_acc[slot] += x4 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 4]));
        gate_acc[slot] += x5 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 5]));
        gate_acc[slot] += x6 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 6]));
        gate_acc[slot] += x7 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 7]));
        up_acc[slot] += x0 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 0]));
        up_acc[slot] += x1 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 1]));
        up_acc[slot] += x2 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 2]));
        up_acc[slot] += x3 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 3]));
        up_acc[slot] += x4 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 4]));
        up_acc[slot] += x5 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 5]));
        up_acc[slot] += x6 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 6]));
        up_acc[slot] += x7 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 7]));
      }
    }
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    const float x = static_cast<float>(hidden_states[hidden_row + k]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        gate_acc[slot] += x * static_cast<float>(static_cast<int>(gate_up_weight[gate_rows[slot] + k]));
        up_acc[slot] += x * static_cast<float>(static_cast<int>(gate_up_weight[up_rows[slot] + k]));
      }
    }
  }
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    gate_partials[slot][threadIdx.x] = gate_acc[slot];
    up_partials[slot][threadIdx.x] = up_acc[slot];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        gate_partials[slot][threadIdx.x] += gate_partials[slot][threadIdx.x + stride];
        up_partials[slot][threadIdx.x] += up_partials[slot][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int64_t out_offset = token * intermediate_size;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t j = j_base + slot;
        const half_t gate_lowp = static_cast<half_t>(gate_partials[slot][0] * gate_up_scale[j]);
        const half_t up_lowp = static_cast<half_t>(up_partials[slot][0] * gate_up_scale[intermediate_size + j]);
        const float gate = static_cast<float>(gate_lowp);
        const float up = static_cast<float>(up_lowp);
        out[out_offset + j] = static_cast<half_t>(silu_f32(gate) * up);
      }
    }
  }
}


__global__ void w8a16_shared_down_combine_residual_fp16_kernel(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride) {
  extern __shared__ float partial[];
  float* partials[4] = {partial, partial + blockDim.x, partial + 2 * blockDim.x, partial + 3 * blockDim.x};
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * 4;
  const int64_t token = blockIdx.y;
  if (row_base >= hidden_size) {
    return;
  }

  bool active[4];
  int64_t down_rows[4];
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int64_t row = row_base + slot;
    active[slot] = row < hidden_size;
    down_rows[slot] = row * intermediate_size;
  }

  const int64_t mid_offset = token * intermediate_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t j = threadIdx.x * 8; j + 7 < intermediate_size; j += vec_stride) {
    const int64_t m_off = mid_offset + j;
    const float m0 = static_cast<float>(shared_intermediate[m_off + 0]);
    const float m1 = static_cast<float>(shared_intermediate[m_off + 1]);
    const float m2 = static_cast<float>(shared_intermediate[m_off + 2]);
    const float m3 = static_cast<float>(shared_intermediate[m_off + 3]);
    const float m4 = static_cast<float>(shared_intermediate[m_off + 4]);
    const float m5 = static_cast<float>(shared_intermediate[m_off + 5]);
    const float m6 = static_cast<float>(shared_intermediate[m_off + 6]);
    const float m7 = static_cast<float>(shared_intermediate[m_off + 7]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t w_off = down_rows[slot] + j;
        acc[slot] += m0 * static_cast<float>(static_cast<int>(down_weight[w_off + 0]));
        acc[slot] += m1 * static_cast<float>(static_cast<int>(down_weight[w_off + 1]));
        acc[slot] += m2 * static_cast<float>(static_cast<int>(down_weight[w_off + 2]));
        acc[slot] += m3 * static_cast<float>(static_cast<int>(down_weight[w_off + 3]));
        acc[slot] += m4 * static_cast<float>(static_cast<int>(down_weight[w_off + 4]));
        acc[slot] += m5 * static_cast<float>(static_cast<int>(down_weight[w_off + 5]));
        acc[slot] += m6 * static_cast<float>(static_cast<int>(down_weight[w_off + 6]));
        acc[slot] += m7 * static_cast<float>(static_cast<int>(down_weight[w_off + 7]));
      }
    }
  }
  for (int64_t j = (intermediate_size & ~7) + threadIdx.x; j < intermediate_size; j += blockDim.x) {
    const float mid = static_cast<float>(shared_intermediate[mid_offset + j]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        acc[slot] += mid * static_cast<float>(static_cast<int>(down_weight[down_rows[slot] + j]));
      }
    }
  }
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    partials[slot][threadIdx.x] = acc[slot];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        partials[slot][threadIdx.x] += partials[slot][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
    const int64_t out_offset = token * hidden_size;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t row = row_base + slot;
        const int64_t idx = out_offset + row;
        const half_t shared_lowp = static_cast<half_t>(partials[slot][0] * down_scale[row]);
        const float value = static_cast<float>(residual[idx]) + static_cast<float>(selected[idx]) + gate * static_cast<float>(shared_lowp);
        out[idx] = static_cast<half_t>(value);
      }
    }
  }
}


__global__ void w8a16_linear_f32_kernel(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += hidden_states[h_off + 0] * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += hidden_states[h_off + 1] * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += hidden_states[h_off + 2] * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += hidden_states[h_off + 3] * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += hidden_states[h_off + 4] * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += hidden_states[h_off + 5] * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += hidden_states[h_off + 6] * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += hidden_states[h_off + 7] * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += hidden_states[hidden_row + k] * static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_w8a16_linear_bf16_f32_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_bf16_lowp_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<hip_bfloat16>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      reinterpret_cast<const hip_bfloat16*>(hidden_states),
      weight,
      weight_scale,
      reinterpret_cast<hip_bfloat16*>(out),
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_fp16_lowp_out(
    const half_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_gate_up_silu_fp16(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_shared_gate_up_silu_fp16_kernel,
      dim3(static_cast<unsigned int>((intermediate_size + 3) / 4), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(8 * threads) * sizeof(float),
      stream,
      hidden_states,
      gate_up_weight,
      gate_up_scale,
      out,
      hidden_size,
      intermediate_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_down_combine_residual_fp16(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_shared_down_combine_residual_fp16_kernel,
      dim3(static_cast<unsigned int>((hidden_size + 3) / 4), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(4 * threads) * sizeof(float),
      stream,
      shared_intermediate,
      down_weight,
      down_scale,
      selected,
      gate_logits,
      residual,
      out,
      hidden_size,
      intermediate_size,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_f32_f32_out(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_f32_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}
