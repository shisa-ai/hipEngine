// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// PARO weighted/shared-gate combine kernel bodies are preserved; wrappers are HIPENGINE C ABI.

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
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t, typename weight_t>
__global__ void weighted_sum_out_kernel(
    const scalar_t* __restrict__ values,
    const weight_t* __restrict__ weights,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float acc = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    acc = fmaf(scalar_to_float(values[row * features + col]), scalar_to_float(weights[row]), acc);
  }
  out[col] = float_to_scalar<scalar_t>(acc);
}

template <typename scalar_t, typename weight_t>
__global__ void weighted_sum_shared_gate_combine_residual_out_kernel(
    const scalar_t* __restrict__ values,
    const weight_t* __restrict__ weights,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float acc = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    acc = fmaf(scalar_to_float(values[row * features + col]), scalar_to_float(weights[row]), acc);
  }
  // Match the old two-kernel path: weighted_sum_out rounded the selected
  // branch to scalar_t before shared-gate/residual combine consumed it.
  const scalar_t selected = float_to_scalar<scalar_t>(acc);
  const float gate = 1.0f / (1.0f + expf(-gate_logits[0]));
  const float value = scalar_to_float(residual[col]) + scalar_to_float(selected) + gate * scalar_to_float(shared[col]);
  out[col] = float_to_scalar<scalar_t>(value);
}

template <typename scalar_t, typename weight_t>
__global__ void weighted_sum_shared_gate_combine_residual_batch_out_kernel(
    const scalar_t* __restrict__ values,
    const weight_t* __restrict__ weights,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = blockIdx.y;
  if (col >= features || token >= tokens) return;
  const int64_t lane_base = token * rows_per_token;
  float acc = 0.0f;
  for (int64_t row = 0; row < rows_per_token; ++row) {
    const int64_t lane = lane_base + row;
    acc = fmaf(scalar_to_float(values[lane * features + col]), scalar_to_float(weights[lane]), acc);
  }
  const scalar_t selected = float_to_scalar<scalar_t>(acc);
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  const int64_t offset = token * features + col;
  const float value = scalar_to_float(residual[offset]) + scalar_to_float(selected) + gate * scalar_to_float(shared[offset]);
  out[offset] = float_to_scalar<scalar_t>(value);
}

template <typename scalar_t>
__global__ void shared_gate_combine_out_kernel(
    const scalar_t* __restrict__ expert,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    scalar_t* __restrict__ out,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  const float gate_logit = scalar_to_float(gate_logits[0]);
  const float gate = 1.0f / (1.0f + expf(-gate_logit));
  const float value = scalar_to_float(expert[col]) + gate * scalar_to_float(shared[col]);
  out[col] = float_to_scalar<scalar_t>(value);
}

template <typename scalar_t>
__global__ void shared_gate_combine_residual_out_kernel(
    const scalar_t* __restrict__ expert,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  const float gate_logit = scalar_to_float(gate_logits[0]);
  const float gate = 1.0f / (1.0f + expf(-gate_logit));
  const float value = scalar_to_float(residual[col]) + scalar_to_float(expert[col]) + gate * scalar_to_float(shared[col]);
  out[col] = float_to_scalar<scalar_t>(value);
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

dim3 feature_grid(int64_t features, int64_t threads) {
  return dim3(static_cast<unsigned int>((features + threads - 1) / threads));
}

}  // namespace

extern "C" int hipengine_weighted_sum_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_out_kernel<uint16_t, float>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_out_fp16_f32w(
    const half_t* values,
    const float* weights,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_out_kernel<half_t, float>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_shared_gate_combine_residual_out_kernel<uint16_t, float>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      shared,
      gate_logits,
      residual,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_out_fp16_f32w(
    const half_t* values,
    const float* weights,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_shared_gate_combine_residual_out_kernel<half_t, float>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      shared,
      gate_logits,
      residual,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    uint16_t* out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || rows_per_token <= 0 || features <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_shared_gate_combine_residual_batch_out_kernel<uint16_t, float>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      shared,
      gate_logits,
      residual,
      out,
      tokens,
      rows_per_token,
      features,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(
    const half_t* values,
    const float* weights,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || rows_per_token <= 0 || features <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_shared_gate_combine_residual_batch_out_kernel<half_t, float>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      shared,
      gate_logits,
      residual,
      out,
      tokens,
      rows_per_token,
      features,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_out_bf16(
    const uint16_t* expert,
    const uint16_t* shared,
    const float* gate_logits,
    uint16_t* out,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_out_kernel<uint16_t>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      out,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_out_fp16(
    const half_t* expert,
    const half_t* shared,
    const float* gate_logits,
    half_t* out,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_out_kernel<half_t>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      out,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_out_bf16(
    const uint16_t* expert,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    uint16_t* out,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_residual_out_kernel<uint16_t>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      residual,
      out,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_out_fp16(
    const half_t* expert,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_residual_out_kernel<half_t>),
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      residual,
      out,
      features);
  return static_cast<int>(hipGetLastError());
}
