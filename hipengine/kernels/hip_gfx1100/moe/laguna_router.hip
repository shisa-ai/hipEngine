// Native Laguna selection-only-correction sigmoid router.
// Net-new hipEngine kernel; the unfused BF16-hidden/F32-weight projection is
// the registered gfx11 router_logits primitive.

#include <float.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float laguna_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t laguna_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float stable_sigmoid_f32(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float numerator = expf(value);
  return numerator / (1.0f + numerator);
}

__device__ inline bool better_route(
    float candidate_value,
    int32_t candidate_index,
    float current_value,
    int32_t current_index) {
  return candidate_index >= 0 &&
      (current_index < 0 || candidate_value > current_value ||
       (candidate_value == current_value && candidate_index < current_index));
}

__global__ void laguna_sigmoid_correction_topk_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    float* __restrict__ routing_scores,
    float* __restrict__ selection_scores,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    float* __restrict__ scaled_routing,
    int64_t num_experts,
    int64_t top_k,
    float routed_scaling_factor) {
  extern __shared__ float work_scores[];
  __shared__ float warp_values[8];
  __shared__ int32_t warp_indices[8];
  __shared__ float selected_probabilities[16];

  const int64_t token = blockIdx.x;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (warpSize - 1);
  const int32_t wave = static_cast<int32_t>(threadIdx.x) / warpSize;
  const int32_t wave_count = static_cast<int32_t>(blockDim.x) / warpSize;
  const int64_t token_offset = token * num_experts;

  for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    const float probability = stable_sigmoid_f32(logits[token_offset + expert]);
    const float selection = probability + correction_bias[expert];
    routing_scores[token_offset + expert] = probability;
    selection_scores[token_offset + expert] = selection;
    work_scores[expert] = selection;
  }
  __syncthreads();

  for (int64_t position = 0; position < top_k; ++position) {
    float local_value = -FLT_MAX;
    int32_t local_index = -1;
    for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
      const float candidate = work_scores[expert];
      const int32_t candidate_index = static_cast<int32_t>(expert);
      if (better_route(candidate, candidate_index, local_value, local_index)) {
        local_value = candidate;
        local_index = candidate_index;
      }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down(local_value, offset);
      const int32_t other_index = __shfl_down(local_index, offset);
      if (better_route(other_value, other_index, local_value, local_index)) {
        local_value = other_value;
        local_index = other_index;
      }
    }
    if (lane == 0) {
      warp_values[wave] = local_value;
      warp_indices[wave] = local_index;
    }
    __syncthreads();

    if (wave == 0) {
      float block_value = lane < wave_count ? warp_values[lane] : -FLT_MAX;
      int32_t block_index = lane < wave_count ? warp_indices[lane] : -1;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down(block_value, offset);
        const int32_t other_index = __shfl_down(block_index, offset);
        if (better_route(other_value, other_index, block_value, block_index)) {
          block_value = other_value;
          block_index = other_index;
        }
      }
      if (lane == 0) {
        selected[token * top_k + position] = static_cast<int64_t>(block_index);
        selected_probabilities[position] = routing_scores[token_offset + block_index];
        work_scores[block_index] = -FLT_MAX;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float denominator = 0.0f;
    for (int64_t position = 0; position < top_k; ++position) {
      denominator += selected_probabilities[position];
    }
    denominator = fmaxf(denominator, FLT_MIN);
    for (int64_t position = 0; position < top_k; ++position) {
      const float normalized = selected_probabilities[position] / denominator;
      routing[token * top_k + position] = normalized;
      scaled_routing[token * top_k + position] = normalized * routed_scaling_factor;
    }
  }
}

__global__ void laguna_weighted_sum_rows_bf16_f32w_kernel(
    const uint16_t* __restrict__ values,
    const float* __restrict__ weights,
    uint16_t* __restrict__ out,
    int64_t top_k,
    int64_t features) {
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  const int64_t feature = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (feature >= features) return;
  const int64_t lane_base = token * top_k;
  float acc = 0.0f;
  for (int64_t route = 0; route < top_k; ++route) {
    const int64_t lane = lane_base + route;
    acc = fmaf(
        laguna_bf16_bits_to_float(values[lane * features + feature]),
        weights[lane],
        acc);
  }
  out[token * features + feature] = laguna_float_to_bf16_bits(acc);
}

__global__ void laguna_prune_tail_routes_f32_kernel(
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    float* __restrict__ scaled_routing,
    int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ lane_to_row,
    int64_t tokens,
    int64_t top_k,
    int64_t drop_count,
    float max_tail_mass,
    float routed_scaling_factor) {
  const int64_t token =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const int64_t base = token * top_k;
  for (int64_t route = 0; route < top_k; ++route) {
    sorted_lanes[base + route] = -1;
    lane_to_row[base + route] = -1;
  }
  const int64_t keep_count = top_k - drop_count;
  float tail_mass = 0.0f;
  for (int64_t route = keep_count; route < top_k; ++route) {
    tail_mass += routing[base + route];
  }
  if (tail_mass > max_tail_mass) return;

  float denominator = 0.0f;
  for (int64_t route = 0; route < keep_count; ++route) {
    denominator += routing[base + route];
  }
  denominator = fmaxf(denominator, FLT_MIN);
  const float inverse = 1.0f / denominator;
  for (int64_t route = 0; route < keep_count; ++route) {
    const float normalized = routing[base + route] * inverse;
    routing[base + route] = normalized;
    scaled_routing[base + route] = normalized * routed_scaling_factor;
  }
  for (int64_t route = keep_count; route < top_k; ++route) {
    selected[base + route] = -1;
    routing[base + route] = 0.0f;
    scaled_routing[base + route] = 0.0f;
  }
}

}  // namespace

extern "C" int hipengine_laguna_sigmoid_correction_topk_f32(
    const float* logits,
    const float* correction_bias,
    float* routing_scores,
    float* selection_scores,
    int64_t* selected,
    float* routing,
    float* scaled_routing,
    int64_t tokens,
    int64_t num_experts,
    int64_t top_k,
    float routed_scaling_factor,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || num_experts <= 0 || num_experts > 256 ||
      top_k <= 0 || top_k > 16 || top_k > num_experts ||
      routed_scaling_factor <= 0.0f || threads != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_sigmoid_correction_topk_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      correction_bias,
      routing_scores,
      selection_scores,
      selected,
      routing,
      scaled_routing,
      num_experts,
      top_k,
      routed_scaling_factor);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_weighted_sum_rows_bf16_f32w(
    const uint16_t* values,
    const float* weights,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 0 || top_k > 16 || features <= 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((features + threads - 1) / threads),
      static_cast<unsigned int>(tokens));
  hipLaunchKernelGGL(
      laguna_weighted_sum_rows_bf16_f32w_kernel,
      grid,
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      values,
      weights,
      out,
      top_k,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_prune_tail_routes_f32(
    int64_t* selected,
    float* routing,
    float* scaled_routing,
    int64_t* sorted_lanes,
    int64_t* lane_to_row,
    int64_t tokens,
    int64_t top_k,
    int64_t drop_count,
    float max_tail_mass,
    float routed_scaling_factor,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || top_k <= 1 || top_k > 16 ||
      drop_count <= 0 || drop_count >= top_k ||
      max_tail_mass <= 0.0f || max_tail_mass >= 1.0f ||
      routed_scaling_factor <= 0.0f || threads != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_prune_tail_routes_f32_kernel,
      dim3(static_cast<unsigned int>((tokens + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      selected,
      routing,
      scaled_routing,
      sorted_lanes,
      lane_to_row,
      tokens,
      top_k,
      drop_count,
      max_tail_mass,
      routed_scaling_factor);
  return static_cast<int>(hipGetLastError());
}
