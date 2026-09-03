// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// PARO weighted/shared-gate combine kernel bodies are preserved; wrappers are hipEngine C ABI.
// The MoE-tail + next-RMS schedule follows qwen-kernel shaders/add_rms3.comp @ 52e240f,
// while preserving hipEngine's activation-dtype selected-sum and residual rounding boundaries.

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

__global__ void weighted_lanes_inverse_kernel(
    const int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ lane_to_row,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int64_t lane = sorted_lanes[row];
  if (lane >= 0 && lane < rows) {
    lane_to_row[lane] = row;
  }
}

template <typename scalar_t>
__global__ void weighted_lanes_sum_out_kernel(
    const scalar_t* __restrict__ values,
    const float* __restrict__ weights,
    const int64_t* __restrict__ lane_to_row,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t top_k,
    int64_t features) {
  const int64_t token = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || col >= features) return;
  float acc = 0.0f;
  const int64_t lane_base = token * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    const int64_t row = lane_to_row[lane_base + k];
    acc = fmaf(scalar_to_float(values[row * features + col]), weights[row], acc);
  }
  out[token * features + col] = float_to_scalar<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void weighted_lanes_sum_shared_add_out_kernel(
    const scalar_t* __restrict__ values,
    const float* __restrict__ weights,
    const int64_t* __restrict__ lane_to_row,
    const scalar_t* __restrict__ shared,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t top_k,
    int64_t features) {
  const int64_t token = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || col >= features) return;
  float acc = 0.0f;
  const int64_t lane_base = token * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    const int64_t row = lane_to_row[lane_base + k];
    acc = fmaf(scalar_to_float(values[row * features + col]), weights[row], acc);
  }
  // Preserve the unfused weighted-sum -> shared-add BF16 boundary exactly.
  const scalar_t selected = float_to_scalar<scalar_t>(acc);
  const int64_t offset = token * features + col;
  out[offset] = float_to_scalar<scalar_t>(
      scalar_to_float(selected) + scalar_to_float(shared[offset]));
}

template <typename scalar_t>
__global__ void weighted_lanes_sum_shared_gate_combine_batch_out_kernel(
    const scalar_t* __restrict__ values,
    const float* __restrict__ weights,
    const int64_t* __restrict__ lane_to_row,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t top_k,
    int64_t features) {
  const int64_t token = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || col >= features) return;
  float acc = 0.0f;
  const int64_t lane_base = token * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    const int64_t row = lane_to_row[lane_base + k];
    acc = fmaf(scalar_to_float(values[row * features + col]), weights[row], acc);
  }
  // Preserve the unfused weighted-sum -> shared-gate-combine BF16 boundary
  // exactly (T0): round the routed sum to BF16 before the gated shared add,
  // matching the production chain's routed memory round-trip.
  const scalar_t selected = float_to_scalar<scalar_t>(acc);
  const int64_t offset = token * features + col;
  const float gate_logit = gate_logits[token];
  const float gate = 1.0f / (1.0f + expf(-gate_logit));
  out[offset] = float_to_scalar<scalar_t>(
      scalar_to_float(selected) + gate * scalar_to_float(shared[offset]));
}

template <typename scalar_t, typename weight_t>
__global__ void weighted_sum_batch_out_kernel(
    const scalar_t* __restrict__ values,
    const weight_t* __restrict__ weights,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t top_k,
    int64_t features) {
  const int64_t token = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || col >= features) return;
  float acc = 0.0f;
  const int64_t row_base = token * top_k;
  for (int64_t lane = 0; lane < top_k; ++lane) {
    const int64_t row = row_base + lane;
    acc = fmaf(
        scalar_to_float(values[row * features + col]),
        scalar_to_float(weights[row]),
        acc);
  }
  out[token * features + col] = float_to_scalar<scalar_t>(acc);
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

__global__ void laguna_weighted_top10_routed_hidden_out_kernel(
    const uint16_t* __restrict__ expert_down,
    const float* __restrict__ routing_weights,
    const uint16_t* __restrict__ shared,
    const uint16_t* __restrict__ post_attention,
    uint16_t* __restrict__ routed_out,
    uint16_t* __restrict__ hidden_out,
    int64_t top_k,
    int64_t features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= features) return;
  float selected_acc = 0.0f;
#pragma unroll
  for (int64_t row = 0; row < 10; ++row) {
    selected_acc = fmaf(
        bf16_bits_to_float(expert_down[row * features + idx]),
        routing_weights[row],
        selected_acc);
  }
  const uint16_t routed_bits = float_to_bf16_bits(selected_acc);
  routed_out[idx] = routed_bits;
  const uint16_t moe_bits = float_to_bf16_bits(
      bf16_bits_to_float(routed_bits) + bf16_bits_to_float(shared[idx]));
  hidden_out[idx] = float_to_bf16_bits(
      bf16_bits_to_float(post_attention[idx]) + bf16_bits_to_float(moe_bits));
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

__global__ void weighted_sum_shared_gate_combine_residual_out_f32_kernel(
    const uint16_t* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float acc = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    acc = fmaf(bf16_bits_to_float(values[row * features + col]), weights[row], acc);
  }
  // Preserve the existing hipEngine selected-branch contraction while keeping
  // the residual stream and final layer output in fp32 for llama-compat probes.
  const float selected = bf16_bits_to_float(float_to_bf16_bits(acc));
  const float gate = 1.0f / (1.0f + expf(-gate_logits[0]));
  out[col] = residual[col] + selected + gate * bf16_bits_to_float(shared[col]);
}

__global__ void weighted_sum_shared_gate_combine_residual_out_f32_accum_kernel(
    const uint16_t* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    selected = fmaf(bf16_bits_to_float(values[row * features + col]), weights[row], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[0]));
  out[col] = residual[col] + selected + gate * bf16_bits_to_float(shared[col]);
}

__global__ void weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_kernel(
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    selected = fmaf(values[row * features + col], weights[row], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[0]));
  out[col] = residual[col] + selected + gate * bf16_bits_to_float(shared[col]);
}

__global__ void weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_kernel(
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const float* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= features) return;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    selected = fmaf(values[row * features + col], weights[row], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[0]));
  out[col] = residual[col] + selected + gate * shared[col];
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

__global__ void weighted_sum_shared_gate_combine_residual_batch_out_f32_kernel(
    const uint16_t* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
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
    acc = fmaf(bf16_bits_to_float(values[lane * features + col]), weights[lane], acc);
  }
  const float selected = bf16_bits_to_float(float_to_bf16_bits(acc));
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  const int64_t offset = token * features + col;
  out[offset] = residual[offset] + selected + gate * bf16_bits_to_float(shared[offset]);
}

__global__ void weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_kernel(
    const uint16_t* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = blockIdx.y;
  if (col >= features || token >= tokens) return;
  const int64_t lane_base = token * rows_per_token;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows_per_token; ++row) {
    const int64_t lane = lane_base + row;
    selected = fmaf(bf16_bits_to_float(values[lane * features + col]), weights[lane], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  const int64_t offset = token * features + col;
  out[offset] = residual[offset] + selected + gate * bf16_bits_to_float(shared[offset]);
}

__global__ void weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_kernel(
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const uint16_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = blockIdx.y;
  if (col >= features || token >= tokens) return;
  const int64_t lane_base = token * rows_per_token;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows_per_token; ++row) {
    const int64_t lane = lane_base + row;
    selected = fmaf(values[lane * features + col], weights[lane], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  const int64_t offset = token * features + col;
  out[offset] = residual[offset] + selected + gate * bf16_bits_to_float(shared[offset]);
}

__global__ void weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_kernel(
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const float* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = blockIdx.y;
  if (col >= features || token >= tokens) return;
  const int64_t lane_base = token * rows_per_token;
  float selected = 0.0f;
  for (int64_t row = 0; row < rows_per_token; ++row) {
    const int64_t lane = lane_base + row;
    selected = fmaf(values[lane * features + col], weights[lane], selected);
  }
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  const int64_t offset = token * features + col;
  out[offset] = residual[offset] + selected + gate * shared[offset];
}

template <typename scalar_t, bool weighted_selected>
__device__ inline float rounded_moe_tail_value(
    const scalar_t* __restrict__ selected_values,
    const float* __restrict__ routing_weights,
    const scalar_t* __restrict__ shared,
    const scalar_t* __restrict__ residual,
    int64_t token,
    int64_t col,
    int64_t rows_per_token,
    int64_t features,
    float gate) {
  const int64_t offset = token * features + col;
  float selected;
  if constexpr (weighted_selected) {
    float acc = 0.0f;
    const int64_t lane_base = token * rows_per_token;
    for (int64_t row = 0; row < rows_per_token; ++row) {
      const int64_t lane = lane_base + row;
      acc = fmaf(
          scalar_to_float(selected_values[lane * features + col]),
          routing_weights[lane],
          acc);
    }
    selected = scalar_to_float(float_to_scalar<scalar_t>(acc));
  } else {
    selected = scalar_to_float(selected_values[offset]);
  }
  const float value =
      scalar_to_float(residual[offset]) + selected + gate * scalar_to_float(shared[offset]);
  return scalar_to_float(float_to_scalar<scalar_t>(value));
}

__global__ void laguna_aggregate_moe_tail_next_rmsnorm_out_kernel(
    const uint16_t* __restrict__ routed,
    const uint16_t* __restrict__ shared,
    const uint16_t* __restrict__ post_attention,
    const float* __restrict__ norm_weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ hidden_out,
    int64_t features,
    float eps) {
  const int tid = threadIdx.x;
  float sumsq = 0.0f;
  for (int64_t idx = tid; idx < features; idx += blockDim.x) {
    const uint16_t moe_bits = float_to_bf16_bits(
        bf16_bits_to_float(routed[idx]) + bf16_bits_to_float(shared[idx]));
    const uint16_t hidden_bits = float_to_bf16_bits(
        bf16_bits_to_float(post_attention[idx]) + bf16_bits_to_float(moe_bits));
    hidden_out[idx] = hidden_bits;
    const float hidden = bf16_bits_to_float(hidden_bits);
    sumsq += hidden * hidden;
  }

  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(features) + eps);

  for (int64_t idx = tid; idx < features; idx += blockDim.x) {
    norm_out[idx] = float_to_bf16_bits(
        bf16_bits_to_float(hidden_out[idx]) * inv_rms * norm_weight[idx]);
  }
}

__global__ void laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_out_kernel(
    const uint16_t* __restrict__ routed,
    const uint16_t* __restrict__ shared,
    const uint16_t* __restrict__ post_attention,
    const float* __restrict__ norm_weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ hidden_out,
    int64_t features,
    float eps) {
  const int tid = threadIdx.x;
  float sumsq = 0.0f;
  for (int64_t idx = tid; idx < features; idx += blockDim.x) {
    const uint16_t moe_bits = float_to_bf16_bits(
        bf16_bits_to_float(routed[idx]) + bf16_bits_to_float(shared[idx]));
    const uint16_t hidden_bits = float_to_bf16_bits(
        bf16_bits_to_float(post_attention[idx]) + bf16_bits_to_float(moe_bits));
    hidden_out[idx] = hidden_bits;
    const float hidden = bf16_bits_to_float(hidden_bits);
    sumsq += hidden * hidden;
  }

  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  if (tid < 32) {
    float wave0 = partial[tid];
    wave0 += partial[tid + 128];
    float wave1 = partial[tid + 32];
    wave1 += partial[tid + 160];
    float wave2 = partial[tid + 64];
    wave2 += partial[tid + 192];
    float wave3 = partial[tid + 96];
    wave3 += partial[tid + 224];
    wave0 += wave2;
    wave1 += wave3;
    wave0 += wave1;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      wave0 += __shfl_down(wave0, stride, 32);
    }
    if (tid == 0) {
      partial[0] = wave0;
    }
  }
  __syncthreads();
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(features) + eps);

  for (int64_t idx = tid; idx < features; idx += blockDim.x) {
    norm_out[idx] = float_to_bf16_bits(
        bf16_bits_to_float(hidden_out[idx]) * inv_rms * norm_weight[idx]);
  }
}

template <typename scalar_t, typename norm_weight_t, bool weighted_selected, bool paro_reduction>
__global__ void moe_tail_next_rmsnorm_out_kernel(
    const scalar_t* __restrict__ selected_values,
    const float* __restrict__ routing_weights,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const scalar_t* __restrict__ residual,
    const norm_weight_t* __restrict__ norm_weight,
    scalar_t* __restrict__ norm_out,
    scalar_t* __restrict__ residual_out,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    float eps) {
  const int64_t token = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_base = token * features;
  const float gate = 1.0f / (1.0f + expf(-gate_logits[token * gate_stride]));
  float sumsq = 0.0f;

  if constexpr (paro_reduction) {
    const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
    const int64_t main_end = (features / vec_stride) * vec_stride;
    for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
      const int64_t off = row_base + idx;
      const float v0 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 0, rows_per_token, features, gate);
      const float v1 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 1, rows_per_token, features, gate);
      const float v2 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 2, rows_per_token, features, gate);
      const float v3 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 3, rows_per_token, features, gate);
      const float v4 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 4, rows_per_token, features, gate);
      const float v5 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 5, rows_per_token, features, gate);
      const float v6 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 6, rows_per_token, features, gate);
      const float v7 = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx + 7, rows_per_token, features, gate);
      residual_out[off + 0] = float_to_scalar<scalar_t>(v0);
      residual_out[off + 1] = float_to_scalar<scalar_t>(v1);
      residual_out[off + 2] = float_to_scalar<scalar_t>(v2);
      residual_out[off + 3] = float_to_scalar<scalar_t>(v3);
      residual_out[off + 4] = float_to_scalar<scalar_t>(v4);
      residual_out[off + 5] = float_to_scalar<scalar_t>(v5);
      residual_out[off + 6] = float_to_scalar<scalar_t>(v6);
      residual_out[off + 7] = float_to_scalar<scalar_t>(v7);
      sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
    }
    for (int64_t idx = main_end + tid; idx < features; idx += blockDim.x) {
      const float value = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx, rows_per_token, features, gate);
      residual_out[row_base + idx] = float_to_scalar<scalar_t>(value);
      sumsq += value * value;
    }
  } else {
    for (int64_t idx = tid; idx < features; idx += blockDim.x) {
      const float value = rounded_moe_tail_value<scalar_t, weighted_selected>(
          selected_values, routing_weights, shared, residual, token, idx, rows_per_token, features, gate);
      residual_out[row_base + idx] = float_to_scalar<scalar_t>(value);
      sumsq += value * value;
    }
  }

  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(features) + eps);

  if constexpr (paro_reduction) {
    const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
    const int64_t main_end = (features / vec_stride) * vec_stride;
    for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
      const int64_t off = row_base + idx;
      norm_out[off + 0] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 0]) * inv_rms * scalar_to_float(norm_weight[idx + 0]));
      norm_out[off + 1] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 1]) * inv_rms * scalar_to_float(norm_weight[idx + 1]));
      norm_out[off + 2] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 2]) * inv_rms * scalar_to_float(norm_weight[idx + 2]));
      norm_out[off + 3] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 3]) * inv_rms * scalar_to_float(norm_weight[idx + 3]));
      norm_out[off + 4] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 4]) * inv_rms * scalar_to_float(norm_weight[idx + 4]));
      norm_out[off + 5] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 5]) * inv_rms * scalar_to_float(norm_weight[idx + 5]));
      norm_out[off + 6] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 6]) * inv_rms * scalar_to_float(norm_weight[idx + 6]));
      norm_out[off + 7] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[off + 7]) * inv_rms * scalar_to_float(norm_weight[idx + 7]));
    }
    for (int64_t idx = main_end + tid; idx < features; idx += blockDim.x) {
      norm_out[row_base + idx] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[row_base + idx]) * inv_rms * scalar_to_float(norm_weight[idx]));
    }
  } else {
    for (int64_t idx = tid; idx < features; idx += blockDim.x) {
      norm_out[row_base + idx] = float_to_scalar<scalar_t>(
          scalar_to_float(residual_out[row_base + idx]) * inv_rms * scalar_to_float(norm_weight[idx]));
    }
  }
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
__global__ void shared_gate_combine_batch_out_kernel(
    const scalar_t* __restrict__ expert,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t features) {
  const int64_t token = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens || col >= features) return;
  const int64_t offset = token * features + col;
  const float gate_logit = gate_logits[token];
  const float gate = 1.0f / (1.0f + expf(-gate_logit));
  const float value = scalar_to_float(expert[offset])
      + gate * scalar_to_float(shared[offset]);
  out[offset] = float_to_scalar<scalar_t>(value);
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

template <typename scalar_t>
__global__ void shared_gate_combine_residual_batch_out_kernel(
    const scalar_t* __restrict__ expert,
    const scalar_t* __restrict__ shared,
    const float* __restrict__ gate_logits,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token = blockIdx.y;
  if (col >= features || token >= tokens) return;
  const int64_t offset = token * features + col;
  const float gate_logit = scalar_to_float(gate_logits[token * gate_stride]);
  const float gate = 1.0f / (1.0f + expf(-gate_logit));
  const float value = scalar_to_float(residual[offset]) + scalar_to_float(expert[offset]) + gate * scalar_to_float(shared[offset]);
  out[offset] = float_to_scalar<scalar_t>(value);
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

dim3 feature_grid(int64_t features, int64_t threads) {
  return dim3(static_cast<unsigned int>((features + threads - 1) / threads));
}

}  // namespace

extern "C" int hipengine_weighted_lanes_sum_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    const int64_t* sorted_lanes,
    int64_t* lane_to_row,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * top_k;
  const int inverse_threads = 256;
  hipLaunchKernelGGL(
      weighted_lanes_inverse_kernel,
      dim3(static_cast<unsigned int>((rows + inverse_threads - 1) / inverse_threads)),
      dim3(inverse_threads),
      0,
      stream,
      sorted_lanes,
      lane_to_row,
      rows);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  hipLaunchKernelGGL(
      (weighted_lanes_sum_out_kernel<uint16_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      lane_to_row,
      out,
      tokens,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_lanes_sum_shared_add_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    const int64_t* sorted_lanes,
    int64_t* lane_to_row,
    const uint16_t* shared,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * top_k;
  const int inverse_threads = 256;
  hipLaunchKernelGGL(
      weighted_lanes_inverse_kernel,
      dim3(static_cast<unsigned int>((rows + inverse_threads - 1) / inverse_threads)),
      dim3(inverse_threads),
      0,
      stream,
      sorted_lanes,
      lane_to_row,
      rows);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  hipLaunchKernelGGL(
      (weighted_lanes_sum_shared_add_out_kernel<uint16_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      lane_to_row,
      shared,
      out,
      tokens,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    const int64_t* sorted_lanes,
    int64_t* lane_to_row,
    const uint16_t* shared,
    const float* gate_logits,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * top_k;
  const int inverse_threads = 256;
  hipLaunchKernelGGL(
      weighted_lanes_inverse_kernel,
      dim3(static_cast<unsigned int>((rows + inverse_threads - 1) / inverse_threads)),
      dim3(inverse_threads),
      0,
      stream,
      sorted_lanes,
      lane_to_row,
      rows);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  hipLaunchKernelGGL(
      (weighted_lanes_sum_shared_gate_combine_batch_out_kernel<uint16_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      lane_to_row,
      shared,
      gate_logits,
      out,
      tokens,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_lanes_sum_out_fp16_f32w(
    const half_t* values,
    const float* weights,
    const int64_t* sorted_lanes,
    int64_t* lane_to_row,
    half_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * top_k;
  const int inverse_threads = 256;
  hipLaunchKernelGGL(
      weighted_lanes_inverse_kernel,
      dim3(static_cast<unsigned int>((rows + inverse_threads - 1) / inverse_threads)),
      dim3(inverse_threads),
      0,
      stream,
      sorted_lanes,
      lane_to_row,
      rows);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  hipLaunchKernelGGL(
      (weighted_lanes_sum_out_kernel<half_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      lane_to_row,
      out,
      tokens,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_batch_out_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (weighted_sum_batch_out_kernel<uint16_t, float>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      out,
      tokens,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

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

extern "C" int hipengine_laguna_weighted_top10_routed_hidden_bf16_out(
    const uint16_t* expert_down,
    const float* routing_weights,
    const uint16_t* shared,
    const uint16_t* post_attention,
    uint16_t* routed_out,
    uint16_t* hidden_out,
    int64_t rows,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (!expert_down || !routing_weights || !shared || !post_attention ||
      !routed_out || !hidden_out || rows != 1 || top_k != 10 ||
      features != 3072 || threads != 32) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_weighted_top10_routed_hidden_out_kernel,
      feature_grid(features, threads),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert_down,
      routing_weights,
      shared,
      post_attention,
      routed_out,
      hidden_out,
      top_k,
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

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_out_f32_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      weighted_sum_shared_gate_combine_residual_out_f32_kernel,
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

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      weighted_sum_shared_gate_combine_residual_out_f32_accum_kernel,
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

extern "C" int hipengine_weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w(
    const float* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_kernel,
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

extern "C" int hipengine_weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w(
    const float* values,
    const float* weights,
    const float* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_kernel,
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

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
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
      weighted_sum_shared_gate_combine_residual_batch_out_f32_kernel,
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

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w(
    const uint16_t* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
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
      weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_kernel,
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

extern "C" int hipengine_weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w(
    const float* values,
    const float* weights,
    const uint16_t* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
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
      weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_kernel,
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

extern "C" int hipengine_weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w(
    const float* values,
    const float* weights,
    const float* shared,
    const float* gate_logits,
    const float* residual,
    float* out,
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
      weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_kernel,
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

extern "C" int hipengine_laguna_aggregate_moe_tail_next_rmsnorm_gguf_bf16_out(
    const uint16_t* routed,
    const uint16_t* shared,
    const uint16_t* post_attention,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* hidden_out,
    int64_t features,
    float eps,
    hipStream_t stream) {
  if (features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_aggregate_moe_tail_next_rmsnorm_out_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      routed,
      shared,
      post_attention,
      norm_weight,
      norm_out,
      hidden_out,
      features,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_gguf_bf16_out(
    const uint16_t* routed,
    const uint16_t* shared,
    const uint16_t* post_attention,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* hidden_out,
    int64_t features,
    float eps,
    hipStream_t stream) {
  if (features <= 0 || routed == nullptr || shared == nullptr ||
      post_attention == nullptr || norm_weight == nullptr ||
      norm_out == nullptr || hidden_out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_aggregate_moe_tail_next_rmsnorm_wave0_tree_out_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      routed,
      shared,
      post_attention,
      norm_weight,
      norm_out,
      hidden_out,
      features,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
    const uint16_t* selected,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<uint16_t, float, false, false>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected,
      static_cast<const float*>(nullptr),
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      1,
      features,
      gate_stride,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
    const uint16_t* selected_values,
    const float* routing_weights,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || rows_per_token <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<uint16_t, float, true, false>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected_values,
      routing_weights,
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      rows_per_token,
      features,
      gate_stride,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_rmsnorm_paro_bf16_out(
    const uint16_t* selected,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    const uint16_t* norm_weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<uint16_t, uint16_t, false, true>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected,
      static_cast<const float*>(nullptr),
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      1,
      features,
      gate_stride,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_paro_bf16_out(
    const uint16_t* selected_values,
    const float* routing_weights,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    const uint16_t* norm_weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || rows_per_token <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<uint16_t, uint16_t, true, true>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected_values,
      routing_weights,
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      rows_per_token,
      features,
      gate_stride,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_rmsnorm_paro_fp16_out(
    const half_t* selected,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    const half_t* norm_weight,
    half_t* norm_out,
    half_t* residual_out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<half_t, half_t, false, true>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected,
      static_cast<const float*>(nullptr),
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      1,
      features,
      gate_stride,
      eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_weighted_sum_shared_gate_combine_residual_rmsnorm_paro_fp16_out(
    const half_t* selected_values,
    const float* routing_weights,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    const half_t* norm_weight,
    half_t* norm_out,
    half_t* residual_out,
    int64_t tokens,
    int64_t rows_per_token,
    int64_t features,
    int64_t gate_stride,
    float eps,
    hipStream_t stream) {
  if (tokens <= 0 || rows_per_token <= 0 || features <= 0 || gate_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (moe_tail_next_rmsnorm_out_kernel<half_t, half_t, true, true>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      selected_values,
      routing_weights,
      shared,
      gate_logits,
      residual,
      norm_weight,
      norm_out,
      residual_out,
      rows_per_token,
      features,
      gate_stride,
      eps);
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

extern "C" int hipengine_shared_gate_combine_batch_out_bf16(
    const uint16_t* expert,
    const uint16_t* shared,
    const float* gate_logits,
    uint16_t* out,
    int64_t tokens,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_batch_out_kernel<uint16_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      out,
      tokens,
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

extern "C" int hipengine_shared_gate_combine_residual_batch_out_bf16(
    const uint16_t* expert,
    const uint16_t* shared,
    const float* gate_logits,
    const uint16_t* residual,
    uint16_t* out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_residual_batch_out_kernel<uint16_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      residual,
      out,
      tokens,
      features,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_shared_gate_combine_residual_batch_out_fp16(
    const half_t* expert,
    const half_t* shared,
    const float* gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t features,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || features <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (shared_gate_combine_residual_batch_out_kernel<half_t>),
      dim3(feature_grid(features, threads).x, static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      expert,
      shared,
      gate_logits,
      residual,
      out,
      tokens,
      features,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}
