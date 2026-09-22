// Gemma 4 norm, router pre-scale, and layer-output family.
//
// Gemma 4's RMSNorm is *not* the Gemma-2 form. HuggingFace's Gemma4RMSNorm
// computes `normed * weight` where the existing qwen35 kernels compute
// `normed * (1 + weight)`. Reusing those would add one to every weight and
// corrupt every norm in the model, so these kernels are separate.
//
// Gemma 4 also builds two norms with `with_scale=False`: the value norm on
// global (full attention) layers, where V = K and no v_proj tensor exists, and
// the router's input norm. Those have no weight tensor at all.
//
// Every weight tensor these kernels read is F32 in the artifact. The norm
// weights, the router scale, the router projection, and the layer scalar are
// all F32; only the projections and the expert stacks are quantized.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGemma4NormThreads = 256;

__device__ inline float gemma4_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_float_to_bf16_bits(float value) {
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
__device__ inline scalar_t gemma4_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_to_scalar<uint16_t>(float value) {
  return gemma4_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_to_float<uint16_t>(uint16_t value) {
  return gemma4_bf16_bits_to_float(value);
}

__device__ inline void gemma4_block_sum(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
}

// Plain-weight RMSNorm over a whole row. `weight` is F32 and applied as-is.
template <typename scalar_t>
__global__ void gemma4_rmsnorm_kernel(
    const scalar_t* hidden_states,
    const float* weight,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] =
        gemma4_to_scalar<scalar_t>(value * inv_rms * weight[idx]);
  }
}

// Weightless RMSNorm. Used for the value norm on global layers and for the
// router's input norm, neither of which has a weight tensor.
template <typename scalar_t>
__global__ void gemma4_rmsnorm_weightless_kernel(
    const scalar_t* hidden_states,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>(value * inv_rms);
  }
}

// Per-head plain-weight RMSNorm over `head_dim`, one block per head.
// Gemma 4 applies this to queries and keys, and to keys again for the value on
// global layers. `weight` may be null, which is the weightless case.
template <typename scalar_t>
__global__ void gemma4_head_rmsnorm_kernel(
    const scalar_t* hidden_states,
    const float* weight,
    scalar_t* out,
    float eps,
    int64_t head_dim) {
  extern __shared__ float partial[];
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_offset = row * head_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(head_dim)) + eps);
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    const float scale = weight == nullptr ? 1.0f : weight[idx];
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>(value * inv_rms * scale);
  }
}

// The router's input transform: weightless RMSNorm followed by
// `scale[i] * hidden_size**-0.5`.
//
// Gemma4TextRouter does exactly this before its projection. Folding the two
// scalars into one multiply keeps the logits kernel reading an already-scaled
// row, so the router needs no kernel of its own.
template <typename scalar_t>
__global__ void gemma4_router_prescale_kernel(
    const scalar_t* hidden_states,
    const float* scale,
    scalar_t* out,
    float eps,
    float root_size,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    out[row_offset + idx] =
        gemma4_to_scalar<scalar_t>(value * inv_rms * scale[idx] * root_size);
  }
}

// `out = (residual + rmsnorm(x, weight)) * layer_scalar`.
//
// This is the tail of a Gemma 4 decoder layer: the post-feedforward norm, the
// residual add, and the trained per-layer output scale, which is a real scalar
// (0.0703125 on layer 0, 0.1953125 on the last) rather than 1.
template <typename scalar_t>
__global__ void gemma4_add_rmsnorm_scale_kernel(
    const scalar_t* hidden_states,
    const scalar_t* residual,
    const float* weight,
    const float* layer_scalar,
    scalar_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const float scalar = layer_scalar == nullptr ? 1.0f : layer_scalar[0];
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    sumsq += value * value;
  }
  gemma4_block_sum(partial, sumsq);
  const float inv_rms =
      rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = gemma4_to_float(hidden_states[row_offset + idx]);
    const float normed = value * inv_rms * weight[idx];
    const float base = gemma4_to_float(residual[row_offset + idx]);
    out[row_offset + idx] = gemma4_to_scalar<scalar_t>((base + normed) * scalar);
  }
}

// `weights[token, slot] *= per_expert_scale[expert[token, slot]]`.
//
// Gemma4TextRouter applies the per-expert scale after renormalizing the top-k
// weights, so it lands on the selected slots only.
__global__ void gemma4_expert_weight_scale_kernel(
    float* weights,
    const int64_t* selected,
    const float* per_expert_scale,
    int64_t rows,
    int64_t top_k) {
  const int64_t total = rows * top_k;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    weights[idx] *= per_expert_scale[selected[idx]];
  }
}

// `out = a + b` over a flat buffer, used to combine the dense and expert
// branches before their shared post-norm.
template <typename scalar_t>
__global__ void gemma4_branch_add_kernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = gemma4_to_scalar<scalar_t>(
        gemma4_to_float(a[idx]) + gemma4_to_float(b[idx]));
  }
}

}  // namespace

extern "C" int hipengine_gemma4_rmsnorm_f32w_bf16(
    const uint16_t* hidden_states,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_rmsnorm_f32w_f32(
    const float* hidden_states,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_kernel<float>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_rmsnorm_weightless_bf16(
    const uint16_t* hidden_states,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_rmsnorm_weightless_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_head_rmsnorm_f32w_bf16(
    const uint16_t* hidden_states,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_head_rmsnorm_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_router_prescale_bf16(
    const uint16_t* hidden_states,
    const float* scale,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    float root_size,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_router_prescale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      scale,
      out,
      eps,
      root_size,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_add_rmsnorm_scale_bf16(
    const uint16_t* hidden_states,
    const uint16_t* residual,
    const float* weight,
    const float* layer_scalar,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_add_rmsnorm_scale_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kGemma4NormThreads),
      kGemma4NormThreads * sizeof(float),
      stream,
      hidden_states,
      residual,
      weight,
      layer_scalar,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_expert_weight_scale_f32(
    float* weights,
    const int64_t* selected,
    const float* per_expert_scale,
    int64_t rows,
    int64_t top_k,
    hipStream_t stream) {
  if (rows <= 0 || top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * top_k;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(
      gemma4_expert_weight_scale_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      weights,
      selected,
      per_expert_scale,
      rows,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemma4_branch_add_bf16(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int64_t capped = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      gemma4_branch_add_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(capped)),
      dim3(threads),
      0,
      stream,
      a,
      b,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}
