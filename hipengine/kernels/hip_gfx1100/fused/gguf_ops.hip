// Small BF16 elementwise helpers for GGUF Qwen3.5 bring-up.

#include <hip/hip_runtime.h>

#include <cstdint>

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

__device__ inline float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__global__ void gguf_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ weight,
    uint16_t* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) * inv_rms * weight[idx];
    out[row_base + idx] = float_to_bf16_bits(value);
  }
}

__global__ void
gguf_rmsnorm_bf16_f32_weight_out_fp16_via_bf16_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ weight,
    half_t* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms =
      rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value =
        bf16_bits_to_float(x[row_base + idx]) *
        inv_rms *
        weight[idx];
    out[row_base + idx] = static_cast<half_t>(
        bf16_bits_to_float(float_to_bf16_bits(value)));
  }
}

__global__ void gguf_rmsnorm_bf16_f32_weight_out_f32_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    out[row_base + idx] = bf16_bits_to_float(x[row_base + idx]) * inv_rms * weight[idx];
  }
}

__global__ void gguf_rmsnorm_f32_f32_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    uint16_t* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    out[row_base + idx] = float_to_bf16_bits(x[row_base + idx] * inv_rms * weight[idx]);
  }
}

__global__ void gguf_rmsnorm_f32_f32_weight_out_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    out[row_base + idx] = x[row_base + idx] * inv_rms * weight[idx];
  }
}

template <bool kAddResidual>
__launch_bounds__(256, 4)
__global__ void gguf_norm_fixed1024_wave256_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps) {
  constexpr int kHidden = 1024;
  constexpr int kThreads = 256;
  constexpr int kWave = 32;
  constexpr int kWaves = kThreads / kWave;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWave - 1);
  const int wave = tid / kWave;
  const int i0 = tid;
  const int i1 = tid + kThreads;
  const int i2 = tid + 2 * kThreads;
  const int i3 = tid + 3 * kThreads;
  const float v0 = bf16_bits_to_float(x[i0]) +
      (kAddResidual ? bf16_bits_to_float(add[i0]) : 0.0f);
  const float v1 = bf16_bits_to_float(x[i1]) +
      (kAddResidual ? bf16_bits_to_float(add[i1]) : 0.0f);
  const float v2 = bf16_bits_to_float(x[i2]) +
      (kAddResidual ? bf16_bits_to_float(add[i2]) : 0.0f);
  const float v3 = bf16_bits_to_float(x[i3]) +
      (kAddResidual ? bf16_bits_to_float(add[i3]) : 0.0f);
  float sum = 0.0f;
  sum += v0 * v0;
  sum += v1 * v1;
  sum += v2 * v2;
  sum += v3 * v3;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset, kWave);
  }
  __shared__ float wave_sums[kWaves];
  __shared__ float shared_inv_rms;
  if (lane == 0) {
    wave_sums[wave] = sum;
  }
  __syncthreads();
  if (wave == 0) {
    float total = lane < kWaves ? wave_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down(total, offset, kWave);
    }
    if (lane == 0) {
      shared_inv_rms = rsqrtf(total / static_cast<float>(kHidden) + eps);
    }
  }
  __syncthreads();
  const float inv_rms = shared_inv_rms;
  norm_out[i0] = float_to_bf16_bits(v0 * inv_rms * weight[i0]);
  norm_out[i1] = float_to_bf16_bits(v1 * inv_rms * weight[i1]);
  norm_out[i2] = float_to_bf16_bits(v2 * inv_rms * weight[i2]);
  norm_out[i3] = float_to_bf16_bits(v3 * inv_rms * weight[i3]);
  if constexpr (kAddResidual) {
    residual_out[i0] = float_to_bf16_bits(v0);
    residual_out[i1] = float_to_bf16_bits(v1);
    residual_out[i2] = float_to_bf16_bits(v2);
    residual_out[i3] = float_to_bf16_bits(v3);
  }
}

template <bool kAddResidual>
__launch_bounds__(256, 1)
__global__ void gguf_norm_fixed5120_wave256_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps) {
  constexpr int kHidden = 5120;
  constexpr int kThreads = 256;
  constexpr int kValuesPerThread = kHidden / kThreads;
  constexpr int kWave = 32;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWave - 1);
  float values[kValuesPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const int idx = tid + i * kThreads;
    const float value = bf16_bits_to_float(x[idx]) +
        (kAddResidual ? bf16_bits_to_float(add[idx]) : 0.0f);
    values[i] = value;
    sum += value * value;
  }

  // Preserve the generic local256 reduction tree exactly while replacing its
  // nine tree barriers with two block barriers and five wave exchanges. The
  // first three tree levels join threads 128, 64, and 32 apart from the
  // immutable per-thread partials; the remaining levels are wholly wave-local.
  __shared__ float partial[kThreads];
  __shared__ float shared_inv_rms;
  partial[tid] = sum;
  __syncthreads();
  if (tid < kWave) {
    float total = partial[tid] + partial[tid + 128];
    total += partial[tid + 64] + partial[tid + 192];
    const float upper =
        (partial[tid + 32] + partial[tid + 160]) +
        (partial[tid + 96] + partial[tid + 224]);
    total += upper;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down(total, offset, kWave);
    }
    if (lane == 0) {
      shared_inv_rms =
          rsqrtf(total / static_cast<float>(kHidden) + eps);
    }
  }
  __syncthreads();

  const float inv_rms = shared_inv_rms;
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const int idx = tid + i * kThreads;
    const float value = values[i];
    norm_out[idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
    if constexpr (kAddResidual) {
      residual_out[idx] = float_to_bf16_bits(value);
    }
  }
}

__global__ void gguf_add_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
    residual_out[row_base + idx] = float_to_bf16_bits(value);
    norm_out[row_base + idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

// Exact replacement for a standalone BF16 add followed by F32-weight
// RMSNorm. Both the residual output and the norm reduction consume the same
// explicitly rounded BF16 sum.
__launch_bounds__(256, 1)
__global__ void gguf_rounded_add_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ residual,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = static_cast<int64_t>(threadIdx.x);
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += 256) {
    const float raw = bf16_bits_to_float(residual[row_base + idx]) +
        bf16_bits_to_float(add[row_base + idx]);
    const uint16_t rounded_bits = float_to_bf16_bits(raw);
    const float value = bf16_bits_to_float(rounded_bits);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms =
      rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += 256) {
    const float raw = bf16_bits_to_float(residual[row_base + idx]) +
        bf16_bits_to_float(add[row_base + idx]);
    const uint16_t rounded_bits = float_to_bf16_bits(raw);
    const float value = bf16_bits_to_float(rounded_bits);
    residual_out[row_base + idx] = rounded_bits;
    norm_out[row_base + idx] =
        float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

__launch_bounds__(256, 1)
__global__ void gguf_add_rmsnorm_bf16_f32_weight_staged_f32_local256_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float scratch[];
  float* value_cache = scratch;
  float* partial = scratch + hidden_size;
  const int64_t tid = static_cast<int64_t>(threadIdx.x);
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += 256) {
    const float value =
        bf16_bits_to_float(x[idx]) + bf16_bits_to_float(add[idx]);
    value_cache[idx] = value;
    residual_out[idx] = float_to_bf16_bits(value);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms =
      rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += 256) {
    norm_out[idx] = float_to_bf16_bits(
        value_cache[idx] * inv_rms * weight[idx]);
  }
}

__global__ void gguf_add_rmsnorm_f32_bf16_f32_weight_kernel(
    const float* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    float* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx] + bf16_bits_to_float(add[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx] + bf16_bits_to_float(add[row_base + idx]);
    residual_out[row_base + idx] = value;
    norm_out[row_base + idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

__global__ void gguf_add_rmsnorm_f32_f32_f32_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    float* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx] + add[row_base + idx];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = x[row_base + idx] + add[row_base + idx];
    residual_out[row_base + idx] = value;
    norm_out[row_base + idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

__global__ void gguf_bf16_add_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    out[idx] = float_to_bf16_bits(bf16_bits_to_float(a[idx]) + bf16_bits_to_float(b[idx]));
  }
}

__global__ void gguf_f32_bf16_add_out_f32_kernel(
    const float* __restrict__ a,
    const uint16_t* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    out[idx] = a[idx] + bf16_bits_to_float(b[idx]);
  }
}

__global__ void gguf_gate_repeat_value_bf16_kernel(
    const uint16_t* __restrict__ q_gate,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim) {
  const int64_t n = head_count * head_dim;
  const int64_t ratio = head_count / head_count_kv;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const int64_t head = idx / head_dim;
    const int64_t dim = idx - head * head_dim;
    const int64_t kv_head = head / ratio;
    const float gate = sigmoidf_stable(bf16_bits_to_float(q_gate[idx]));
    const float v = bf16_bits_to_float(value[kv_head * head_dim + dim]);
    out[idx] = float_to_bf16_bits(gate * v);
  }
}

__global__ void gguf_gate_mul_bf16_kernel(
    const uint16_t* __restrict__ attn,
    const uint16_t* __restrict__ gate,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const float a = bf16_bits_to_float(attn[idx]);
    const float g = sigmoidf_stable(bf16_bits_to_float(gate[idx]));
    out[idx] = float_to_bf16_bits(a * g);
  }
}

__device__ __forceinline__ void gguf_head_rmsnorm_partial_rotary_f32_weight_row(
    const float* __restrict__ src,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim] * inv_rms * weight[dim];
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * weight[pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__device__ __forceinline__ void gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
    const uint16_t* __restrict__ src,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = bf16_bits_to_float(src[dim]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float src_value = bf16_bits_to_float(src[dim]);
    const float value = src_value * inv_rms * weight[dim];
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired_src = bf16_bits_to_float(src[pair_dim]);
      const float paired = paired_src * inv_rms * weight[pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__global__ void gguf_head_rmsnorm_partial_rotary_position_f32_weight_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ position_ptr,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const int64_t head = blockIdx.x;
  if (head >= num_q_heads + num_kv_heads) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t offset = local_head * head_dim;
  const float* src = (is_query ? query : key) + offset;
  const float* weight = is_query ? q_weight : k_weight;
  float* dst = (is_query ? query_out : key_out) + offset;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  gguf_head_rmsnorm_partial_rotary_f32_weight_row(
      src,
      weight,
      cos,
      sin,
      dst,
      eps,
      head_dim,
      rotary_dim,
      partial);
}

__global__ void gguf_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ position_ptr,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const int64_t head = blockIdx.x;
  if (head >= num_q_heads + num_kv_heads) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t offset = local_head * head_dim;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  if (is_query) {
    gguf_head_rmsnorm_partial_rotary_f32_weight_row(
        query + offset,
        q_weight,
        cos,
        sin,
        query_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  } else {
    gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
        key + offset,
        k_weight,
        cos,
        sin,
        key_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  }
}

// Exact serial full-attention postprocess contraction. Query blocks read the
// packed BF16 Q/gate projection directly, preserve the established FP32
// RMSNorm/RoPE tree, and copy gate BF16 bits. K blocks consume the projected
// BF16 key directly through the same conversion and reduction sequence as the
// primitive split/cast/head chain.
__global__ void gguf_split_qgate_head_rmsnorm_partial_rotary_position_qk_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ q_projected,
    const uint16_t* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ position_ptr,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    uint16_t* __restrict__ gate_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const int64_t head = blockIdx.x;
  if (head >= num_q_heads + num_kv_heads) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  if (is_query) {
    const int64_t src_offset = local_head * 2 * head_dim;
    const int64_t out_offset = local_head * head_dim;
    const uint16_t* __restrict__ src = q_projected + src_offset;
    gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
        src,
        q_weight,
        cos,
        sin,
        query_out + out_offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      gate_out[out_offset + dim] = src[head_dim + dim];
    }
  } else {
    const int64_t offset = local_head * head_dim;
    gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
        key + offset,
        k_weight,
        cos,
        sin,
        key_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  }
}

template <bool PACK_QUERY_TILES>
__global__ void gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ positions,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t packed_query_begin,
    int64_t packed_query_end) {
  const int64_t head = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (head >= num_q_heads + num_kv_heads || token >= tokens) {
    return;
  }
  const int64_t position = positions[token];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t heads = is_query ? num_q_heads : num_kv_heads;
  const int64_t src_offset = (token * heads + local_head) * head_dim;
  int64_t dst_offset = src_offset;
  if constexpr (PACK_QUERY_TILES) {
    if (is_query && token >= packed_query_begin && token < packed_query_end) {
      constexpr int64_t kTileRows = 128;
      const int64_t tile_begin = (token / kTileRows) * kTileRows;
      const int64_t tile_row = token - tile_begin;
      dst_offset =
          ((tile_begin * num_q_heads + local_head * kTileRows + tile_row) *
           head_dim);
    }
  }
  const float* src = (is_query ? query : key) + src_offset;
  const float* weight = is_query ? q_weight : k_weight;
  float* dst = (is_query ? query_out : key_out) + dst_offset;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  gguf_head_rmsnorm_partial_rotary_f32_weight_row(
      src,
      weight,
      cos,
      sin,
      dst,
      eps,
      head_dim,
      rotary_dim,
      partial);
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int
hipengine_gguf_rmsnorm_bf16_f32_weight_out_fp16_via_bf16(
    const uint16_t* x,
    const float* weight,
    half_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_rmsnorm_bf16_f32_weight_out_fp16_via_bf16_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight_out_f32(
    const uint16_t* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || threads <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads > 1024) {
    threads = 1024;
  }
  gguf_rmsnorm_bf16_f32_weight_out_f32_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_rmsnorm_f32_f32_weight(
    const float* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_rmsnorm_f32_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_rmsnorm_f32_f32_weight_out_f32(
    const float* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || threads <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads > 1024) {
    threads = 1024;
  }
  gguf_rmsnorm_f32_f32_weight_out_f32_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight_fixed1024_wave256(
    const uint16_t* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr || rows != 1 ||
      hidden_size != 1024 || threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_norm_fixed1024_wave256_kernel<false><<<dim3(1), dim3(256), 0, stream>>>(
      x, nullptr, weight, out, nullptr, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight_fixed1024_wave256(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || add == nullptr || weight == nullptr || norm_out == nullptr ||
      residual_out == nullptr || rows != 1 || hidden_size != 1024 ||
      threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_norm_fixed1024_wave256_kernel<true><<<dim3(1), dim3(256), 0, stream>>>(
      x, add, weight, norm_out, residual_out, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight_fixed5120_wave256(
    const uint16_t* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr || rows != 1 ||
      hidden_size != 5120 || threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_norm_fixed5120_wave256_kernel<false><<<dim3(1), dim3(256), 0, stream>>>(
      x, nullptr, weight, out, nullptr, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight_fixed5120_wave256(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || add == nullptr || weight == nullptr || norm_out == nullptr ||
      residual_out == nullptr || rows != 1 || hidden_size != 5120 ||
      threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_norm_fixed5120_wave256_kernel<true><<<dim3(1), dim3(256), 0, stream>>>(
      x, add, weight, norm_out, residual_out, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_add_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, add, weight, norm_out, residual_out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_rounded_add_rmsnorm_bf16_f32_weight(
    const uint16_t* residual,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (residual == nullptr || add == nullptr || weight == nullptr ||
      norm_out == nullptr || residual_out == nullptr || rows < 2 || rows > 4 ||
      hidden_size <= 0 || threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_rounded_add_rmsnorm_bf16_f32_weight_kernel<<<
      dim3(static_cast<unsigned int>(rows)), dim3(256),
      256 * sizeof(float), stream>>>(
      residual, add, weight, norm_out, residual_out, eps, hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight_staged_f32_local256(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || add == nullptr || weight == nullptr ||
      norm_out == nullptr || residual_out == nullptr || rows != 1 ||
      hidden_size <= 0 || hidden_size > 4096 ||
      (hidden_size % 256) != 0 || threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shared_bytes =
      static_cast<size_t>(hidden_size + threads) * sizeof(float);
  gguf_add_rmsnorm_bf16_f32_weight_staged_f32_local256_kernel<<<
      dim3(1), dim3(256), shared_bytes, stream>>>(
      x, add, weight, norm_out, residual_out, eps, hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_add_rmsnorm_f32_bf16_f32_weight(
    const float* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    float* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_add_rmsnorm_f32_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, add, weight, norm_out, residual_out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_add_rmsnorm_f32_f32_f32_weight(
    const float* x,
    const float* add,
    const float* weight,
    uint16_t* norm_out,
    float* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_add_rmsnorm_f32_f32_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, add, weight, norm_out, residual_out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_bf16_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_bf16_add_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      a, b, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_f32_bf16_add_out_f32(
    const float* a,
    const uint16_t* b,
    float* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_f32_bf16_add_out_f32_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      a, b, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_gate_repeat_value_bf16(
    const uint16_t* q_gate,
    const uint16_t* value,
    uint16_t* out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  if (head_count <= 0 || head_count_kv <= 0 || head_dim <= 0 || head_count % head_count_kv != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t n = head_count * head_dim;
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_gate_repeat_value_bf16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      q_gate, value, out, head_count, head_count_kv, head_dim);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_gate_mul_bf16(
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_gate_mul_bf16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      attn, gate, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
    const float* query,
    const float* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_position_f32_weight_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight(
    const float* query,
    const uint16_t* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_split_qgate_head_rmsnorm_partial_rotary_position_qk_bf16_f32_weight(
    const uint16_t* q_projected,
    const uint16_t* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    uint16_t* gate_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !cos_table ||
      !sin_table || !position || !query_out || !key_out || !gate_out ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
      max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_split_qgate_head_rmsnorm_partial_rotary_position_qk_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      q_projected,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      gate_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
    const float* query,
    const float* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* positions,
    float* query_out,
    float* key_out,
    float eps,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel<false><<<
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads), static_cast<unsigned int>(tokens)),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      positions,
      query_out,
      key_out,
      eps,
      tokens,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions,
      0,
      0);
  return hipGetLastError();
}

extern "C" int
hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_positions_packed_query_f32_weight(
    const float* query,
    const float* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* positions,
    float* query_out,
    float* key_out,
    float eps,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t packed_query_begin,
    int64_t packed_query_end,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || max_positions <= 0 ||
      packed_query_begin < 0 || packed_query_begin >= packed_query_end ||
      packed_query_end > tokens || packed_query_begin % 128 != 0 ||
      packed_query_end % 128 != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel<true><<<
      dim3(
          static_cast<unsigned int>(num_q_heads + num_kv_heads),
          static_cast<unsigned int>(tokens)),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      positions,
      query_out,
      key_out,
      eps,
      tokens,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions,
      packed_query_begin,
      packed_query_end);
  return hipGetLastError();
}

// B2 P1: stage-owned BF16 -> IEEE-half activation cast for the F16-input
// dense T16 siblings. Round-to-nearest via the F32 intermediate, matching
// the host-side staging used by the sibling numerics contract.
__global__ void gguf_cast_bf16_to_f16_kernel(
    const uint16_t* __restrict__ x,
    half_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    out[idx] = static_cast<half_t>(bf16_bits_to_float(x[idx]));
  }
}

extern "C" int hipengine_gguf_cast_bf16_to_f16(
    const uint16_t* x,
    half_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_cast_bf16_to_f16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      x, out, n);
  return hipGetLastError();
}
