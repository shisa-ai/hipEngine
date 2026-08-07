// Net-new Maple kernels derived from the public DeepGrove format/math contract.
// Format reference: deepgrove-ai/mlx-lm-deepgrove/mlx_lm/ternary.py (LSB-first).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float reduce_sum(float value, float* scratch) {
  const int tid = static_cast<int>(threadIdx.x);
  scratch[tid] = value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ inline float ternary_dot(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    int64_t in_features) {
  const int64_t words = in_features / 16;
  float sum = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = weight[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
      sum += bf16_to_float(x[base + lane]) * static_cast<float>(code);
    }
  }
  return sum;
}

__global__ void maple_ternary_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.x;
  if (row >= out_features) {
    return;
  }
  extern __shared__ float scratch[];
  const int64_t words = in_features / 16;
  const float partial = ternary_dot(x, weight + row * words, in_features);
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row] = float_to_bf16(total * bf16_to_float(row_alpha[row]));
  }
}

__global__ void maple_ternary_qkv_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ q_weight,
    const uint16_t* __restrict__ q_alpha,
    const uint32_t* __restrict__ k_weight,
    const uint16_t* __restrict__ k_alpha,
    const uint32_t* __restrict__ v_weight,
    const uint16_t* __restrict__ v_alpha,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t row = blockIdx.x;
  const int64_t total_rows = q_features + 2 * kv_features;
  if (row >= total_rows) {
    return;
  }
  const int64_t words = in_features / 16;
  const uint32_t* selected_weight;
  const uint16_t* selected_alpha;
  int64_t selected_row;
  if (row < q_features) {
    selected_weight = q_weight;
    selected_alpha = q_alpha;
    selected_row = row;
  } else if (row < q_features + kv_features) {
    selected_weight = k_weight;
    selected_alpha = k_alpha;
    selected_row = row - q_features;
  } else {
    selected_weight = v_weight;
    selected_alpha = v_alpha;
    selected_row = row - q_features - kv_features;
  }
  extern __shared__ float scratch[];
  const float partial = ternary_dot(
      x,
      selected_weight + selected_row * words,
      in_features);
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row] = float_to_bf16(total * bf16_to_float(selected_alpha[selected_row]));
  }
}

__global__ void maple_selected_ternary_dual_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ a_weight,
    const uint16_t* __restrict__ a_alpha,
    const uint32_t* __restrict__ b_weight,
    const uint16_t* __restrict__ b_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ a_out,
    uint16_t* __restrict__ b_out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t block = blockIdx.x;
  const int64_t route = block / out_features;
  const int64_t row = block - route * out_features;
  if (route >= top_k) {
    return;
  }
  const int32_t expert = selected_experts[route];
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      a_out[route * out_features + row] = float_to_bf16(0.0f);
      b_out[route * out_features + row] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + row;
  float a_partial = 0.0f;
  float b_partial = 0.0f;
  const uint32_t* a_row = a_weight + matrix_row * words;
  const uint32_t* b_row = b_weight + matrix_row * words;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t a_word = a_row[word_idx];
    const uint32_t b_word = b_row[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float xv = bf16_to_float(x[base + lane]);
      const int a_code = static_cast<int>((a_word >> (2 * lane)) & 0x3U) - 1;
      const int b_code = static_cast<int>((b_word >> (2 * lane)) & 0x3U) - 1;
      a_partial += xv * static_cast<float>(a_code);
      b_partial += xv * static_cast<float>(b_code);
    }
  }
  extern __shared__ float scratch[];
  float* a_scratch = scratch;
  float* b_scratch = scratch + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  a_scratch[tid] = a_partial;
  b_scratch[tid] = b_partial;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      a_scratch[tid] += a_scratch[tid + stride];
      b_scratch[tid] += b_scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int64_t out_idx = route * out_features + row;
    a_out[out_idx] = float_to_bf16(
        a_scratch[0] * bf16_to_float(a_alpha[matrix_row]));
    b_out[out_idx] = float_to_bf16(
        b_scratch[0] * bf16_to_float(b_alpha[matrix_row]));
  }
}

// Batched selected-expert dual GEMV (P3 bring-up, row/route gather):
// computes gate/up for all (row, slot) entries over T rows. This is not
// expert-major grouped MoE: duplicate expert weights are reread per entry. Grid is
// (out_features, top_k, T); block (outrow, slot, row) reads
// expert = selected_ids[row, slot] and projects x[row] with that expert's
// gate/up weight rows, writing to a_out/b_out as [T, top_k, out_features].
__global__ void maple_selected_ternary_dual_gemv_batched_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ a_weight,
    const uint16_t* __restrict__ a_alpha,
    const uint32_t* __restrict__ b_weight,
    const uint16_t* __restrict__ b_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ a_out,
    uint16_t* __restrict__ b_out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.z;
  const int64_t route = blockIdx.y;
  const int64_t outrow = blockIdx.x;
  if (row >= rows || route >= top_k) {
    return;
  }
  const int64_t entry = row * top_k + route;
  const int32_t expert = selected_experts[entry];
  const int64_t entry_out = entry * out_features + outrow;
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      a_out[entry_out] = float_to_bf16(0.0f);
      b_out[entry_out] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + outrow;
  float a_partial = 0.0f;
  float b_partial = 0.0f;
  const uint32_t* a_row = a_weight + matrix_row * words;
  const uint32_t* b_row = b_weight + matrix_row * words;
  const uint16_t* xrow = x + row * in_features;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t a_word = a_row[word_idx];
    const uint32_t b_word = b_row[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float xv = bf16_to_float(xrow[base + lane]);
      const int a_code = static_cast<int>((a_word >> (2 * lane)) & 0x3U) - 1;
      const int b_code = static_cast<int>((b_word >> (2 * lane)) & 0x3U) - 1;
      a_partial += xv * static_cast<float>(a_code);
      b_partial += xv * static_cast<float>(b_code);
    }
  }
  extern __shared__ float scratch[];
  float* a_scratch = scratch;
  float* b_scratch = scratch + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  a_scratch[tid] = a_partial;
  b_scratch[tid] = b_partial;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      a_scratch[tid] += a_scratch[tid + stride];
      b_scratch[tid] += b_scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    a_out[entry_out] = float_to_bf16(
        a_scratch[0] * bf16_to_float(a_alpha[matrix_row]));
    b_out[entry_out] = float_to_bf16(
        b_scratch[0] * bf16_to_float(b_alpha[matrix_row]));
  }
}

// Batched selected-expert down GEMV (P3 bring-up): same row/route gather
// pattern as the dual kernel, but single projection. x is the SwiGLU output
// [T, top_k, in];
// block (outrow, slot, row) projects x[row, slot] with expert down weight.
__global__ void maple_selected_ternary_gemv_batched_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.z;
  const int64_t route = blockIdx.y;
  const int64_t outrow = blockIdx.x;
  if (row >= rows || route >= top_k) {
    return;
  }
  const int64_t entry = row * top_k + route;
  const int32_t expert = selected_experts[entry];
  const int64_t entry_out = entry * out_features + outrow;
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      out[entry_out] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + outrow;
  const uint32_t* wrow = weight + matrix_row * words;
  const uint16_t* xrow = x + entry * in_features;
  float partial = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = wrow[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float xv = bf16_to_float(xrow[base + lane]);
      const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
      partial += xv * static_cast<float>(code);
    }
  }
  extern __shared__ float scratch[];
  const int tid = static_cast<int>(threadIdx.x);
  scratch[tid] = partial;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[entry_out] = float_to_bf16(
        scratch[0] * bf16_to_float(row_alpha[matrix_row]));
  }
}

// Expert-major prefill gate/up. Generic stable compaction supplies
// expert_start[expert:expert+2] and sorted_lanes in ascending original-lane
// order. One block owns (expert, output row), stages that packed weight row
// once, and computes every routed lane for the expert while writing results
// directly back to original [row, route, output] order.
template <int DOT_THREADS, int LANES_PER_BLOCK>
__global__ void maple_selected_ternary_dual_grouped_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ a_weight,
    const uint16_t* __restrict__ a_alpha,
    const uint32_t* __restrict__ b_weight,
    const uint16_t* __restrict__ b_alpha,
    const int64_t* __restrict__ expert_start,
    const int64_t* __restrict__ sorted_lanes,
    uint16_t* __restrict__ a_out,
    uint16_t* __restrict__ b_out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t outrow = blockIdx.x;
  const int64_t expert = blockIdx.y;
  if (expert >= num_experts || outrow >= out_features) {
    return;
  }
  const int64_t begin = expert_start[expert];
  const int64_t end = expert_start[expert + 1];
  if (begin >= end) {
    return;
  }

  const int64_t words = in_features / 16;
  const int64_t matrix_row = expert * out_features + outrow;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane_slot = thread / DOT_THREADS;
  const int dot_tid = thread - lane_slot * DOT_THREADS;
  __shared__ uint32_t a_wrow[128];
  __shared__ uint32_t b_wrow[128];
  const uint32_t* a_base = a_weight + matrix_row * words;
  const uint32_t* b_base = b_weight + matrix_row * words;
  for (int64_t word = thread; word < words; word += blockDim.x) {
    a_wrow[word] = a_base[word];
    b_wrow[word] = b_base[word];
  }
  __syncthreads();

  extern __shared__ float scratch[];
  float* a_scratch = scratch;
  float* b_scratch = scratch + blockDim.x;
  const int scratch_index = lane_slot * DOT_THREADS + dot_tid;
  const float a_scale = bf16_to_float(a_alpha[matrix_row]);
  const float b_scale = bf16_to_float(b_alpha[matrix_row]);
  const int64_t total_lanes = rows * top_k;
  for (int64_t sorted_base = begin; sorted_base < end;
       sorted_base += LANES_PER_BLOCK) {
    const int64_t sorted = sorted_base + lane_slot;
    const bool valid_sorted = sorted < end;
    const int64_t lane_id = valid_sorted ? sorted_lanes[sorted] : -1;
    const bool valid_lane = lane_id >= 0 && lane_id < total_lanes;
    const uint16_t* xrow =
        x + (valid_lane ? (lane_id / top_k) * in_features : 0);
    float a_partial = 0.0f;
    float b_partial = 0.0f;
    if (valid_lane) {
      for (int64_t word = dot_tid; word < words; word += DOT_THREADS) {
        const uint32_t a_word = a_wrow[word];
        const uint32_t b_word = b_wrow[word];
        const int64_t base = word * 16;
#pragma unroll
        for (int packed_lane = 0; packed_lane < 16; ++packed_lane) {
          const float xv = bf16_to_float(xrow[base + packed_lane]);
          const int a_code =
              static_cast<int>((a_word >> (2 * packed_lane)) & 0x3U) - 1;
          const int b_code =
              static_cast<int>((b_word >> (2 * packed_lane)) & 0x3U) - 1;
          a_partial += xv * static_cast<float>(a_code);
          b_partial += xv * static_cast<float>(b_code);
        }
      }
    }
    a_scratch[scratch_index] = a_partial;
    b_scratch[scratch_index] = b_partial;
    __syncthreads();
    for (int stride = DOT_THREADS / 2; stride > 0; stride >>= 1) {
      if (dot_tid < stride) {
        a_scratch[scratch_index] += a_scratch[scratch_index + stride];
        b_scratch[scratch_index] += b_scratch[scratch_index + stride];
      }
      __syncthreads();
    }
    const int group_base = lane_slot * DOT_THREADS;
    const float a_total = a_scratch[group_base];
    const float b_total = b_scratch[group_base];
    if (dot_tid == 0 && valid_lane) {
      const int64_t output = lane_id * out_features + outrow;
      a_out[output] = float_to_bf16(a_total * a_scale);
      b_out[output] = float_to_bf16(b_total * b_scale);
    }
  }
}

// Expert-major prefill down projection. Intermediate rows already use original
// flattened [row, route] order, so the same stable lane map restores output
// order without a separate inverse-scatter kernel.
template <int DOT_THREADS, int LANES_PER_BLOCK>
__global__ void maple_selected_ternary_grouped_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    const int64_t* __restrict__ expert_start,
    const int64_t* __restrict__ sorted_lanes,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t outrow = blockIdx.x;
  const int64_t expert = blockIdx.y;
  if (expert >= num_experts || outrow >= out_features) {
    return;
  }
  const int64_t begin = expert_start[expert];
  const int64_t end = expert_start[expert + 1];
  if (begin >= end) {
    return;
  }

  const int64_t words = in_features / 16;
  const int64_t matrix_row = expert * out_features + outrow;
  const int thread = static_cast<int>(threadIdx.x);
  const int lane_slot = thread / DOT_THREADS;
  const int dot_tid = thread - lane_slot * DOT_THREADS;
  __shared__ uint32_t wrow[128];
  const uint32_t* wbase = weight + matrix_row * words;
  for (int64_t word = thread; word < words; word += blockDim.x) {
    wrow[word] = wbase[word];
  }
  __syncthreads();

  extern __shared__ float scratch[];
  const int scratch_index = lane_slot * DOT_THREADS + dot_tid;
  const float scale = bf16_to_float(row_alpha[matrix_row]);
  const int64_t total_lanes = rows * top_k;
  for (int64_t sorted_base = begin; sorted_base < end;
       sorted_base += LANES_PER_BLOCK) {
    const int64_t sorted = sorted_base + lane_slot;
    const bool valid_sorted = sorted < end;
    const int64_t lane_id = valid_sorted ? sorted_lanes[sorted] : -1;
    const bool valid_lane = lane_id >= 0 && lane_id < total_lanes;
    const uint16_t* xrow = x + (valid_lane ? lane_id * in_features : 0);
    float partial = 0.0f;
    if (valid_lane) {
      for (int64_t word = dot_tid; word < words; word += DOT_THREADS) {
        const uint32_t packed = wrow[word];
        const int64_t base = word * 16;
#pragma unroll
        for (int packed_lane = 0; packed_lane < 16; ++packed_lane) {
          const int code =
              static_cast<int>((packed >> (2 * packed_lane)) & 0x3U) - 1;
          partial += bf16_to_float(xrow[base + packed_lane]) *
              static_cast<float>(code);
        }
      }
    }
    scratch[scratch_index] = partial;
    __syncthreads();
    for (int stride = DOT_THREADS / 2; stride > 0; stride >>= 1) {
      if (dot_tid < stride) {
        scratch[scratch_index] += scratch[scratch_index + stride];
      }
      __syncthreads();
    }
    const float total = scratch[lane_slot * DOT_THREADS];
    if (dot_tid == 0 && valid_lane) {
      out[lane_id * out_features + outrow] = float_to_bf16(total * scale);
    }
  }
}

__global__ void maple_selected_ternary_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t block = blockIdx.x;
  const int64_t route = block / out_features;
  const int64_t row = block - route * out_features;
  if (route >= top_k) {
    return;
  }
  const int32_t expert = selected_experts[route];
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      out[route * out_features + row] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + row;
  const float partial = ternary_dot(
      x + route * in_features,
      weight + matrix_row * words,
      in_features);
  extern __shared__ float scratch[];
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[route * out_features + row] = float_to_bf16(
        total * bf16_to_float(row_alpha[matrix_row]));
  }
}

// Fused MoE gate/up dual GEMV + clamped SiLU (D2 fusion).
// Replaces maple_selected_ternary_dual_gemv_kernel + maple_clamped_swiglu_kernel:
// computes gate/up exactly as the dual gemv (FP32 block reduce, bf16 scaling),
// then applies the same clamped swiglu as the standalone kernel and writes a
// single [top_k, out_features] intermediate buffer. Bit-exact with the unfused
// chain. Grid is (out_features, top_k); block (row, route).
__global__ void maple_moe_dual_swiglu_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ a_weight,
    const uint16_t* __restrict__ a_alpha,
    const uint32_t* __restrict__ b_weight,
    const uint16_t* __restrict__ b_alpha,
    const int32_t* __restrict__ selected_experts,
    uint16_t* __restrict__ out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features) {
  const int64_t block = blockIdx.x;
  const int64_t route = block / out_features;
  const int64_t row = block - route * out_features;
  if (route >= top_k) {
    return;
  }
  const int64_t out_idx = route * out_features + row;
  const int32_t expert = selected_experts[route];
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      out[out_idx] = float_to_bf16(0.0f);
    }
    return;
  }
  const int64_t words = in_features / 16;
  const int64_t matrix_row = static_cast<int64_t>(expert) * out_features + row;
  float a_partial = 0.0f;
  float b_partial = 0.0f;
  const uint32_t* a_row = a_weight + matrix_row * words;
  const uint32_t* b_row = b_weight + matrix_row * words;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t a_word = a_row[word_idx];
    const uint32_t b_word = b_row[word_idx];
    const int64_t base = word_idx * 16;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
      const float xv = bf16_to_float(x[base + lane]);
      const int a_code = static_cast<int>((a_word >> (2 * lane)) & 0x3U) - 1;
      const int b_code = static_cast<int>((b_word >> (2 * lane)) & 0x3U) - 1;
      a_partial += xv * static_cast<float>(a_code);
      b_partial += xv * static_cast<float>(b_code);
    }
  }
  extern __shared__ float scratch[];
  float* a_scratch = scratch;
  float* b_scratch = scratch + blockDim.x;
  const int tid = static_cast<int>(threadIdx.x);
  a_scratch[tid] = a_partial;
  b_scratch[tid] = b_partial;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      a_scratch[tid] += a_scratch[tid + stride];
      b_scratch[tid] += b_scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const uint16_t gate = float_to_bf16(
        a_scratch[0] * bf16_to_float(a_alpha[matrix_row]));
    const uint16_t up = float_to_bf16(
        b_scratch[0] * bf16_to_float(b_alpha[matrix_row]));
    const float gate_value = fminf(bf16_to_float(gate), 7.0f);
    const float up_value = fminf(fmaxf(bf16_to_float(up), -7.0f), 7.0f);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    out[out_idx] = float_to_bf16(silu * up_value);
  }
}

// Fused MoE down GEMV + weighted residual (D2 fusion) is intentionally not
// promoted: serializing the top-k selected experts per output row (grid=
// out_features) regresses decode ~3.5% vs the parallel-expert standalone down
// gemv + weighted residual. The bit-exact composite was verified and reverted;
// see docs/REFACTOR.md.

__global__ void maple_affine4_embed_kernel(
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    uint16_t* __restrict__ out,
    int64_t token_id,
    int64_t hidden_size) {
  const int64_t words_per_row = hidden_size / 8;
  const int64_t groups_per_row = hidden_size / 64;
  const uint32_t* row = weight + token_id * words_per_row;
  const uint16_t* row_scales = scales + token_id * groups_per_row;
  const uint16_t* row_biases = biases + token_id * groups_per_row;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const uint32_t word = row[idx / 8];
    const int code = static_cast<int>((word >> (4 * (idx & 7))) & 0xFU);
    const int64_t group = idx / 64;
    const float value = static_cast<float>(code) * bf16_to_float(row_scales[group])
        + bf16_to_float(row_biases[group]);
    out[idx] = float_to_bf16(value);
  }
}

// Batched affine4 embedding (P4 prefill): embeds T token IDs into a
// [T, hidden_size] buffer.  Grid is T blocks, one block per token row.
__global__ void maple_affine4_embed_batched_kernel(
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    const int64_t* __restrict__ token_ids,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t hidden_size) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t token_id = token_ids[row];
  const int64_t words_per_row = hidden_size / 8;
  const int64_t groups_per_row = hidden_size / 64;
  const uint32_t* wrow = weight + token_id * words_per_row;
  const uint16_t* row_scales = scales + token_id * groups_per_row;
  const uint16_t* row_biases = biases + token_id * groups_per_row;
  uint16_t* orow = out + row * hidden_size;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const uint32_t word = wrow[idx / 8];
    const int code = static_cast<int>((word >> (4 * (idx & 7))) & 0xFU);
    const int64_t group = idx / 64;
    const float value = static_cast<float>(code) * bf16_to_float(row_scales[group])
        + bf16_to_float(row_biases[group]);
    orow[idx] = float_to_bf16(value);
  }
}

__device__ inline float maple_affine4_accumulate_word_exact(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ row,
    const uint16_t* __restrict__ row_scales,
    const uint16_t* __restrict__ row_biases,
    int64_t word_idx,
    float partial) {
  const uint32_t word = row[word_idx];
  const int64_t base = word_idx * 8;
  const int64_t group = word_idx / 8;
  const float scale = bf16_to_float(row_scales[group]);
  const float bias = bf16_to_float(row_biases[group]);
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) {
    const int code = static_cast<int>((word >> (4 * lane)) & 0xFU);
    const float w = static_cast<float>(code) * scale + bias;
    partial += bf16_to_float(x[base + lane]) * w;
  }
  return partial;
}

__global__ void maple_affine4_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    float* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row_idx = blockIdx.x;
  if (row_idx >= out_features) {
    return;
  }
  const int64_t words = in_features / 8;
  const int64_t groups = in_features / 64;
  const uint32_t* row = weight + row_idx * words;
  const uint16_t* row_scales = scales + row_idx * groups;
  const uint16_t* row_biases = biases + row_idx * groups;
  float partial = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = row[word_idx];
    const int64_t base = word_idx * 8;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const int64_t idx = base + lane;
      const int code = static_cast<int>((word >> (4 * lane)) & 0xFU);
      const int64_t group = idx / 64;
      const float w = static_cast<float>(code) * bf16_to_float(row_scales[group])
          + bf16_to_float(row_biases[group]);
      partial += bf16_to_float(x[idx]) * w;
    }
  }
  extern __shared__ float scratch[];
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[row_idx] = total;
  }
}

// Exact D0 c1 sibling for the fixed Maple K=2048 head. One wave emulates the
// production kernel's four 32-lane waves: each lane computes virtual tids
// lane/{+32,+64,+96}, including each tid's two word-strided K chunks, then
// reconstructs the original 128-thread LDS tree (stride 64, 32, 16..1) with
// wave shuffles. This changes ownership/barriers, not any per-logit FP32
// association.
__global__ __launch_bounds__(32) void maple_affine4_gemv_wave32_exact_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    float* __restrict__ out,
    int64_t out_features) {
  const int64_t row_idx = blockIdx.x;
  if (row_idx >= out_features) {
    return;
  }
  constexpr int64_t words = 2048 / 8;
  constexpr int64_t groups = 2048 / 64;
  const uint32_t* row = weight + row_idx * words;
  const uint16_t* row_scales = scales + row_idx * groups;
  const uint16_t* row_biases = biases + row_idx * groups;
  const int lane = static_cast<int>(threadIdx.x);
  float virtual_partial[4];
#pragma unroll
  for (int virtual_wave = 0; virtual_wave < 4; ++virtual_wave) {
    const int64_t virtual_tid = lane + virtual_wave * 32;
    float partial = 0.0f;
    partial = maple_affine4_accumulate_word_exact(
        x, row, row_scales, row_biases, virtual_tid, partial);
    partial = maple_affine4_accumulate_word_exact(
        x, row, row_scales, row_biases, virtual_tid + 128, partial);
    virtual_partial[virtual_wave] = partial;
  }

  const float stride64_lo = virtual_partial[0] + virtual_partial[2];
  const float stride64_hi = virtual_partial[1] + virtual_partial[3];
  float total = stride64_lo + stride64_hi;
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    const float peer = __shfl_down(total, stride, 32);
    if (lane < stride) {
      total += peer;
    }
  }
  if (lane == 0) {
    out[row_idx] = total;
  }
}

// Batched affine4 lm_head GEMM (P4 prefill): out[T, out] = x[T, in] x affine4 W.
// Grid (out_features, T); block (row_idx, t) projects input row t with output
// row row_idx's affine4 weights, writing out[t*out + row_idx].
__global__ void maple_affine4_gemv_batched_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ biases,
    float* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row_idx = blockIdx.x;
  const int64_t t = blockIdx.y;
  if (row_idx >= out_features || t >= rows) {
    return;
  }
  const int64_t words = in_features / 8;
  const int64_t groups = in_features / 64;
  const uint32_t* wrow = weight + row_idx * words;
  const uint16_t* row_scales = scales + row_idx * groups;
  const uint16_t* row_biases = biases + row_idx * groups;
  const uint16_t* xrow = x + t * in_features;
  float partial = 0.0f;
  for (int64_t word_idx = threadIdx.x; word_idx < words; word_idx += blockDim.x) {
    const uint32_t word = wrow[word_idx];
    const int64_t base = word_idx * 8;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const int64_t idx = base + lane;
      const int code = static_cast<int>((word >> (4 * lane)) & 0xFU);
      const int64_t group = idx / 64;
      const float w = static_cast<float>(code) * bf16_to_float(row_scales[group])
          + bf16_to_float(row_biases[group]);
      partial += bf16_to_float(xrow[idx]) * w;
    }
  }
  extern __shared__ float scratch[];
  const float total = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    out[t * out_features + row_idx] = total;
  }
}

inline int launch_threads_for_words(int64_t words) {
  if (words >= 128) {
    return 128;
  }
  if (words >= 64) {
    return 64;
  }
  return 32;
}

template <int DOT_THREADS, int LANES_PER_BLOCK>
inline void launch_maple_selected_ternary_dual_grouped(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int64_t* expert_start,
    const int64_t* sorted_lanes,
    uint16_t* a_out,
    uint16_t* b_out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  constexpr int block_threads = DOT_THREADS * LANES_PER_BLOCK;
  hipLaunchKernelGGL(
      (maple_selected_ternary_dual_grouped_kernel<DOT_THREADS, LANES_PER_BLOCK>),
      dim3(static_cast<unsigned int>(out_features),
           static_cast<unsigned int>(num_experts)),
      dim3(block_threads),
      2 * block_threads * sizeof(float),
      stream,
      x,
      a_weight,
      a_alpha,
      b_weight,
      b_alpha,
      expert_start,
      sorted_lanes,
      a_out,
      b_out,
      rows,
      num_experts,
      top_k,
      in_features,
      out_features);
}

template <int DOT_THREADS, int LANES_PER_BLOCK>
inline void launch_maple_selected_ternary_grouped(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    const int64_t* expert_start,
    const int64_t* sorted_lanes,
    uint16_t* out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  constexpr int block_threads = DOT_THREADS * LANES_PER_BLOCK;
  hipLaunchKernelGGL(
      (maple_selected_ternary_grouped_kernel<DOT_THREADS, LANES_PER_BLOCK>),
      dim3(static_cast<unsigned int>(out_features),
           static_cast<unsigned int>(num_experts)),
      dim3(block_threads),
      block_threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      expert_start,
      sorted_lanes,
      out,
      rows,
      num_experts,
      top_k,
      in_features,
      out_features);
}

}  // namespace

extern "C" int hipengine_maple_ternary_gemv_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    uint16_t* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (in_features <= 0 || out_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  hipLaunchKernelGGL(
      maple_ternary_gemv_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      out,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

// Batched ternary GEMM (prefill P1): computes
//   out[t, r] = sum_k x[t, k] * ternary(W[r, k]) * alpha[r]
// for t in [0, rows), r in [0, out_features).  One block owns one output row
// and a tile of TOKENS_PER_BLOCK prompt rows, loads that weight row into LDS
// ONCE, and reuses it across the token tile — the core prefill weight-reuse
// win (the c1 GEMV re-reads the full weight row per token).  Accumulation
// order per (token,row) is word-strided threads + lane loop + tree reduce,
// identical to maple_ternary_gemv_kernel, so bf16 outputs match within the
// gate.
template <int TOKENS_PER_BLOCK>
__global__ void maple_ternary_gemm_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ weight,
    const uint16_t* __restrict__ row_alpha,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t r = blockIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t t0 = tile * TOKENS_PER_BLOCK;
  if (r >= out_features) {
    return;
  }
  const int64_t words = in_features / 16;
  const int tid = static_cast<int>(threadIdx.x);
  extern __shared__ float xs[];
  // Load this output row's weights into LDS once (packed ternary codes).
  __shared__ uint32_t wrow[128];
  const uint32_t* wbase = weight + r * words;
  for (int64_t w = tid; w < words; w += blockDim.x) {
    wrow[w] = wbase[w];
  }
  __syncthreads();  // publish wrow before the token loop reads it
  // Scratch for tree reductions: [blockDim.x] floats per token slice.
  float* scratch = xs;
  const float alpha = bf16_to_float(row_alpha[r]);
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int64_t tok = t0 + t;
    if (tok >= rows) {
      break;
    }
    const uint16_t* xrow = x + tok * in_features;
    float partial = 0.0f;
    for (int64_t w = tid; w < words; w += blockDim.x) {
      const uint32_t word = wrow[w];
      const int64_t base = w * 16;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
        partial += bf16_to_float(xrow[base + lane]) * static_cast<float>(code);
      }
    }
    float* slice = scratch + t * blockDim.x;
    slice[tid] = partial;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        slice[tid] += slice[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[tok * out_features + r] = float_to_bf16(slice[0] * alpha);
    }
    __syncthreads();  // protect scratch reuse across tokens in the tile
  }
}

extern "C" int hipengine_maple_ternary_gemm_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || in_features % 16 != 0
      || in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t words = in_features / 16;
  const int toks_per_block = 8;
  const int64_t blocks_x = out_features;
  const int64_t blocks_y = (rows + toks_per_block - 1) / toks_per_block;
  const int64_t shared = toks_per_block * threads * sizeof(float);
  hipLaunchKernelGGL(
      (maple_ternary_gemm_bf16_kernel<8>),
      dim3(static_cast<unsigned int>(blocks_x),
           static_cast<unsigned int>(blocks_y)),
      dim3(threads),
      static_cast<unsigned int>(shared),
      stream,
      x,
      weight,
      row_alpha,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

// Batched QKV ternary GEMM (P1 prefill): computes the full [rows, q_features +
// 2*kv_features] qkv buffer from the q/k/v ternary projections of a [rows, in]
// input. Grid is (total_output_rows x token-tiles); each block owns one output
// row and an 8-token tile, loading that output row's packed weight codes into
// LDS once and reusing them across the tile (the prefill weight-reuse win).
// Output column layout matches the single-row qkv GEMV: [q | k | v].
template <int TOKENS_PER_BLOCK>
__global__ void maple_ternary_qkv_gemm_kernel(
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ q_weight,
    const uint16_t* __restrict__ q_alpha,
    const uint32_t* __restrict__ k_weight,
    const uint16_t* __restrict__ k_alpha,
    const uint32_t* __restrict__ v_weight,
    const uint16_t* __restrict__ v_alpha,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features) {
  const int64_t total_rows = q_features + 2 * kv_features;
  const int64_t r = blockIdx.x;
  const int64_t tile = blockIdx.y;
  const int64_t t0 = tile * TOKENS_PER_BLOCK;
  if (r >= total_rows) {
    return;
  }
  const uint32_t* selected_weight;
  const uint16_t* selected_alpha;
  int64_t matrix_row;
  int64_t col;
  if (r < q_features) {
    selected_weight = q_weight;
    selected_alpha = q_alpha;
    matrix_row = r;
    col = r;
  } else if (r < q_features + kv_features) {
    selected_weight = k_weight;
    selected_alpha = k_alpha;
    matrix_row = r - q_features;
    col = q_features + matrix_row;
  } else {
    selected_weight = v_weight;
    selected_alpha = v_alpha;
    matrix_row = r - q_features - kv_features;
    col = q_features + kv_features + matrix_row;
  }
  const int64_t words = in_features / 16;
  const int tid = static_cast<int>(threadIdx.x);
  extern __shared__ float xs[];
  __shared__ uint32_t wrow[128];
  const uint32_t* wbase = selected_weight + matrix_row * words;
  for (int64_t w = tid; w < words; w += blockDim.x) {
    wrow[w] = wbase[w];
  }
  __syncthreads();
  float* scratch = xs;
  const float alpha = bf16_to_float(selected_alpha[matrix_row]);
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    const int64_t tok = t0 + t;
    if (tok >= rows) {
      break;
    }
    const uint16_t* xrow = x + tok * in_features;
    float partial = 0.0f;
    for (int64_t w = tid; w < words; w += blockDim.x) {
      const uint32_t word = wrow[w];
      const int64_t base = w * 16;
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        const int code = static_cast<int>((word >> (2 * lane)) & 0x3U) - 1;
        partial += bf16_to_float(xrow[base + lane]) * static_cast<float>(code);
      }
    }
    float* slice = scratch + t * blockDim.x;
    slice[tid] = partial;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        slice[tid] += slice[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[tok * total_rows + col] = float_to_bf16(slice[0] * alpha);
    }
    __syncthreads();
  }
}

extern "C" int hipengine_maple_ternary_qkv_gemm_bf16(
    const uint16_t* x,
    const uint32_t* q_weight,
    const uint16_t* q_alpha,
    const uint32_t* k_weight,
    const uint16_t* k_alpha,
    const uint32_t* v_weight,
    const uint16_t* v_alpha,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || q_features <= 0 || kv_features <= 0
      || in_features % 16 != 0 || in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int toks_per_block = 8;
  const int64_t total_rows = q_features + 2 * kv_features;
  const int64_t blocks_x = total_rows;
  const int64_t blocks_y = (rows + toks_per_block - 1) / toks_per_block;
  const int64_t shared = toks_per_block * threads * sizeof(float);
  hipLaunchKernelGGL(
      (maple_ternary_qkv_gemm_kernel<8>),
      dim3(static_cast<unsigned int>(blocks_x),
           static_cast<unsigned int>(blocks_y)),
      dim3(threads),
      static_cast<unsigned int>(shared),
      stream,
      x,
      q_weight,
      q_alpha,
      k_weight,
      k_alpha,
      v_weight,
      v_alpha,
      out,
      rows,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_ternary_qkv_gemv_bf16(
    const uint16_t* x,
    const uint32_t* q_weight,
    const uint16_t* q_alpha,
    const uint32_t* k_weight,
    const uint16_t* k_alpha,
    const uint32_t* v_weight,
    const uint16_t* v_alpha,
    uint16_t* out,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    hipStream_t stream) {
  if (in_features <= 0 || q_features <= 0 || kv_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = q_features + 2 * kv_features;
  hipLaunchKernelGGL(
      maple_ternary_qkv_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      q_weight,
      q_alpha,
      k_weight,
      k_alpha,
      v_weight,
      v_alpha,
      out,
      in_features,
      q_features,
      kv_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_dual_gemv_bf16(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int32_t* selected_experts,
    uint16_t* a_out,
    uint16_t* b_out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (num_experts <= 0 || top_k <= 0 || in_features <= 0 || out_features <= 0
      || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = top_k * out_features;
  hipLaunchKernelGGL(
      maple_selected_ternary_dual_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      2 * threads * sizeof(float),
      stream,
      x,
      a_weight,
      a_alpha,
      b_weight,
      b_alpha,
      selected_experts,
      a_out,
      b_out,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_dual_gemv_batched_bf16(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int32_t* selected_experts,
    uint16_t* a_out,
    uint16_t* b_out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0 || in_features <= 0
      || out_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  hipLaunchKernelGGL(
      maple_selected_ternary_dual_gemv_batched_kernel,
      dim3(static_cast<unsigned int>(out_features),
           static_cast<unsigned int>(top_k),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      2 * threads * sizeof(float),
      stream,
      x,
      a_weight,
      a_alpha,
      b_weight,
      b_alpha,
      selected_experts,
      a_out,
      b_out,
      rows,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_gemv_batched_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    const int32_t* selected_experts,
    uint16_t* out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0 || in_features <= 0
      || out_features <= 0 || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  hipLaunchKernelGGL(
      maple_selected_ternary_gemv_batched_kernel,
      dim3(static_cast<unsigned int>(out_features),
           static_cast<unsigned int>(top_k),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      selected_experts,
      out,
      rows,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_dual_grouped_bf16(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int64_t* expert_start,
    const int64_t* sorted_lanes,
    uint16_t* a_out,
    uint16_t* b_out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0 || in_features <= 0 ||
      out_features <= 0 || in_features % 16 != 0 ||
      in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int dot_threads = launch_threads_for_words(in_features / 16);
  if (dot_threads == 128) {
    launch_maple_selected_ternary_dual_grouped<128, 1>(
        x, a_weight, a_alpha, b_weight, b_alpha, expert_start, sorted_lanes,
        a_out, b_out, rows, num_experts, top_k, in_features, out_features,
        stream);
  } else if (dot_threads == 64) {
    launch_maple_selected_ternary_dual_grouped<64, 1>(
        x, a_weight, a_alpha, b_weight, b_alpha, expert_start, sorted_lanes,
        a_out, b_out, rows, num_experts, top_k, in_features, out_features,
        stream);
  } else {
    launch_maple_selected_ternary_dual_grouped<32, 1>(
        x, a_weight, a_alpha, b_weight, b_alpha, expert_start, sorted_lanes,
        a_out, b_out, rows, num_experts, top_k, in_features, out_features,
        stream);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_grouped_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    const int64_t* expert_start,
    const int64_t* sorted_lanes,
    uint16_t* out,
    int64_t rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || num_experts <= 0 || top_k <= 0 || in_features <= 0 ||
      out_features <= 0 || in_features % 16 != 0 ||
      in_features / 16 > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int dot_threads = launch_threads_for_words(in_features / 16);
  if (dot_threads == 128) {
    launch_maple_selected_ternary_grouped<128, 1>(
        x, weight, row_alpha, expert_start, sorted_lanes, out, rows,
        num_experts, top_k, in_features, out_features, stream);
  } else if (dot_threads == 64) {
    launch_maple_selected_ternary_grouped<64, 1>(
        x, weight, row_alpha, expert_start, sorted_lanes, out, rows,
        num_experts, top_k, in_features, out_features, stream);
  } else {
    launch_maple_selected_ternary_grouped<32, 1>(
        x, weight, row_alpha, expert_start, sorted_lanes, out, rows,
        num_experts, top_k, in_features, out_features, stream);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_selected_ternary_gemv_bf16(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* row_alpha,
    const int32_t* selected_experts,
    uint16_t* out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (num_experts <= 0 || top_k <= 0 || in_features <= 0 || out_features <= 0
      || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = top_k * out_features;
  hipLaunchKernelGGL(
      maple_selected_ternary_gemv_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      row_alpha,
      selected_experts,
      out,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_moe_dual_swiglu_bf16(
    const uint16_t* x,
    const uint32_t* a_weight,
    const uint16_t* a_alpha,
    const uint32_t* b_weight,
    const uint16_t* b_alpha,
    const int32_t* selected_experts,
    uint16_t* out,
    int64_t num_experts,
    int64_t top_k,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (num_experts <= 0 || top_k <= 0 || in_features <= 0 || out_features <= 0
      || in_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 16);
  const int64_t blocks = top_k * out_features;
  hipLaunchKernelGGL(
      maple_moe_dual_swiglu_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      2 * threads * sizeof(float),
      stream,
      x,
      a_weight,
      a_alpha,
      b_weight,
      b_alpha,
      selected_experts,
      out,
      num_experts,
      top_k,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_embed_bf16(
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    uint16_t* out,
    int64_t token_id,
    int64_t hidden_size,
    hipStream_t stream) {
  if (token_id < 0 || hidden_size <= 0 || hidden_size % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_affine4_embed_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      weight,
      scales,
      biases,
      out,
      token_id,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_embed_batched_bf16(
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    const int64_t* token_ids,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0 || hidden_size % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_affine4_embed_batched_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      weight,
      scales,
      biases,
      token_ids,
      out,
      rows,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_gemv_f32(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (in_features <= 0 || out_features <= 0 || in_features % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 8);
  hipLaunchKernelGGL(
      maple_affine4_gemv_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      scales,
      biases,
      out,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_gemv_wave32_exact_f32(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (in_features != 2048 || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_affine4_gemv_wave32_exact_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(32),
      0,
      stream,
      x,
      weight,
      scales,
      biases,
      out,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_affine4_gemv_batched_f32(
    const uint16_t* x,
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || in_features % 64 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_words(in_features / 8);
  hipLaunchKernelGGL(
      maple_affine4_gemv_batched_kernel,
      dim3(static_cast<unsigned int>(out_features),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      x,
      weight,
      scales,
      biases,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}
