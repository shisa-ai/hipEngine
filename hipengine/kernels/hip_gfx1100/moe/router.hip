// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Kernel bodies are preserved from the Qwen3.5 native router family; wrappers are hipEngine C ABI.

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

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename scalar_t>
__global__ void qwen35_router_logits_kernel(
    const scalar_t* hidden,
    const uint16_t* weight,
    float* logits,
    int64_t hidden_size,
    int64_t num_experts) {
  extern __shared__ float partial[];
  const int64_t expert = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (expert >= num_experts) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = expert * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += scalar_to_float_qwen35(hidden[h_off + 0]) * bf16_bits_to_float(weight[w_off + 0]);
    acc += scalar_to_float_qwen35(hidden[h_off + 1]) * bf16_bits_to_float(weight[w_off + 1]);
    acc += scalar_to_float_qwen35(hidden[h_off + 2]) * bf16_bits_to_float(weight[w_off + 2]);
    acc += scalar_to_float_qwen35(hidden[h_off + 3]) * bf16_bits_to_float(weight[w_off + 3]);
    acc += scalar_to_float_qwen35(hidden[h_off + 4]) * bf16_bits_to_float(weight[w_off + 4]);
    acc += scalar_to_float_qwen35(hidden[h_off + 5]) * bf16_bits_to_float(weight[w_off + 5]);
    acc += scalar_to_float_qwen35(hidden[h_off + 6]) * bf16_bits_to_float(weight[w_off + 6]);
    acc += scalar_to_float_qwen35(hidden[h_off + 7]) * bf16_bits_to_float(weight[w_off + 7]);
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += scalar_to_float_qwen35(hidden[hidden_row + k]) * bf16_bits_to_float(weight[weight_row + k]);
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
    logits[token * num_experts + expert] = partial[0];
  }
}

template <typename scalar_t, int TOKENS_PER_BLOCK>
__global__ void qwen35_router_logits_token_tile_kernel(
    const scalar_t* hidden,
    const uint16_t* weight,
    float* logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_experts) {
  extern __shared__ float partial[];
  const int64_t expert = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(blockIdx.y) * TOKENS_PER_BLOCK;
  if (expert >= num_experts) {
    return;
  }

  float acc[TOKENS_PER_BLOCK];
#pragma unroll
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    acc[t] = 0.0f;
  }

  const int64_t weight_row = expert * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t w_off = weight_row + k;
    const float w0 = bf16_bits_to_float(weight[w_off + 0]);
    const float w1 = bf16_bits_to_float(weight[w_off + 1]);
    const float w2 = bf16_bits_to_float(weight[w_off + 2]);
    const float w3 = bf16_bits_to_float(weight[w_off + 3]);
    const float w4 = bf16_bits_to_float(weight[w_off + 4]);
    const float w5 = bf16_bits_to_float(weight[w_off + 5]);
    const float w6 = bf16_bits_to_float(weight[w_off + 6]);
    const float w7 = bf16_bits_to_float(weight[w_off + 7]);
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) {
        const int64_t h_off = token * hidden_size + k;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 0]) * w0;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 1]) * w1;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 2]) * w2;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 3]) * w3;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 4]) * w4;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 5]) * w5;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 6]) * w6;
        acc[t] += scalar_to_float_qwen35(hidden[h_off + 7]) * w7;
      }
    }
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    const float w = bf16_bits_to_float(weight[weight_row + k]);
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) {
        acc[t] += scalar_to_float_qwen35(hidden[token * hidden_size + k]) * w;
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
    partial[t * blockDim.x + threadIdx.x] = acc[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
        partial[t * blockDim.x + threadIdx.x] += partial[t * blockDim.x + threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
      const int64_t token = token_base + t;
      if (token < tokens) {
        logits[token * num_experts + expert] = partial[t * blockDim.x];
      }
    }
  }
}

template <typename scalar_t>
hipError_t launch_qwen35_router_logits(
    const scalar_t* hidden,
    const uint16_t* weight,
    float* logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t threads,
    hipStream_t stream) {
  if (tokens >= 4) {
    constexpr int kTokensPerBlock = 4;
    hipLaunchKernelGGL(
        (qwen35_router_logits_token_tile_kernel<scalar_t, kTokensPerBlock>),
        dim3(
            static_cast<unsigned int>(num_rows),
            static_cast<unsigned int>((tokens + kTokensPerBlock - 1) / kTokensPerBlock)),
        dim3(static_cast<unsigned int>(threads)),
        static_cast<size_t>(threads) * kTokensPerBlock * sizeof(float),
        stream,
        hidden,
        weight,
        logits,
        tokens,
        hidden_size,
        num_rows);
  } else {
    hipLaunchKernelGGL(
        qwen35_router_logits_kernel<scalar_t>,
        dim3(static_cast<unsigned int>(num_rows), static_cast<unsigned int>(tokens)),
        dim3(static_cast<unsigned int>(threads)),
        static_cast<size_t>(threads) * sizeof(float),
        stream,
        hidden,
        weight,
        logits,
        hidden_size,
        num_rows);
  }
  return hipGetLastError();
}

// Block-parallel top-k + softmax.
//
// Original design: thread 0 did top_k * num_experts iterations serially
// while 511 threads sat idle. With grid=1 (c=1 decode), that starved the
// entire GPU. rocprof at 4K/D4 showed this kernel at 9.6% of total GPU
// time with grid=1 AND a 144-byte scratch spill from per-thread arrays.
//
// Fix (2026-05-05):
//   * All threads cooperate on every top-k iteration via a warp-shuffle
//     argmax reduction.
//   * Per-thread arrays replaced with a small static __shared__ block of
//     top_k <= 16 floats, eliminating the scratch spill.
//   * Softmax over the small top_k values remains in thread 0; its cost
//     is trivial vs the top-k search.
//
// Shared memory (dynamic): num_experts * sizeof(float) -- unchanged at
// launch time so wrappers keep working with the same allocation.
// Shared memory (static): ~144 bytes for warp-reduction scratch plus
// a top_vals[16] buffer.
__device__ void qwen35_router_select_from_logits(
    const float* __restrict__ token_logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    float* __restrict__ shared_logits,
    int64_t token,
    int64_t logits_stride,
    int64_t num_experts,
    int64_t top_k) {
  // Warp-reduction scratch: blockDim.x / warpSize <= 16 for allowed router workgroups.
  __shared__ float warp_max_vals[16];
  __shared__ int32_t warp_max_idxs[16];
  // Top-k buffer for softmax. top_k is asserted <= 16 at the wrapper.
  __shared__ float top_vals[16];

  const int lane_id = static_cast<int>(threadIdx.x) & (warpSize - 1);
  const int warp_id = static_cast<int>(threadIdx.x) / warpSize;
  const int num_warps = static_cast<int>(blockDim.x) / warpSize;

  // Parallel load of this token's logits into shared memory.
  for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    shared_logits[expert] = token_logits[expert];
  }
  __syncthreads();

  // Iterative top-k: each iteration runs a block-parallel argmax, records
  // the winner, and marks that expert's score as -inf so the next
  // iteration finds the next-best.
  for (int64_t pos = 0; pos < top_k; ++pos) {
    // 1) Thread-local argmax across strided experts.
    float my_val = -3.4028234663852886e38f;
    int32_t my_idx = -1;
    for (int64_t expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
      const float v = shared_logits[expert];
      if (v > my_val) {
        my_val = v;
        my_idx = static_cast<int32_t>(expert);
      }
    }
    // 2) Warp-level argmax via shuffle. On gfx1100 CU mode, warpSize==64.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      const float other_val = __shfl_down(my_val, offset);
      const int32_t other_idx = __shfl_down(my_idx, offset);
      if (other_val > my_val) {
        my_val = other_val;
        my_idx = other_idx;
      }
    }
    if (lane_id == 0) {
      warp_max_vals[warp_id] = my_val;
      warp_max_idxs[warp_id] = my_idx;
    }
    __syncthreads();
    // 3) First warp reduces the per-warp maxes.
    if (warp_id == 0) {
      float v = (lane_id < num_warps) ? warp_max_vals[lane_id] : -3.4028234663852886e38f;
      int32_t i = (lane_id < num_warps) ? warp_max_idxs[lane_id] : -1;
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        const float other_v = __shfl_down(v, offset);
        const int32_t other_i = __shfl_down(i, offset);
        if (other_v > v) {
          v = other_v;
          i = other_i;
        }
      }
      if (lane_id == 0) {
        top_vals[pos] = v;
        selected[token * top_k + pos] = static_cast<int64_t>(i);
        shared_logits[i] = -3.4028234663852886e38f;
      }
    }
    __syncthreads();
  }

  // Softmax over top_k values. top_k <= 16 so this is trivial; leave it
  // in thread 0 to avoid extra syncs for a cost that no longer matters.
  if (threadIdx.x == 0) {
    const float max_value = top_vals[0];
    float denom = 0.0f;
    for (int64_t pos = 0; pos < top_k; ++pos) {
      denom += expf(top_vals[pos] - max_value);
    }
    denom = fmaxf(denom, 1.0e-20f);
    for (int64_t pos = 0; pos < top_k; ++pos) {
      routing[token * top_k + pos] = expf(top_vals[pos] - max_value) / denom;
    }
  }
}

__global__ void qwen35_router_select_kernel(
    const float* __restrict__ logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    int64_t logits_stride,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ float shared_logits[];
  const int64_t token = blockIdx.x;
  qwen35_router_select_from_logits(
      logits + token * logits_stride,
      selected,
      routing,
      shared_logits,
      token,
      logits_stride,
      num_experts,
      top_k);
}

__global__ void qwen35_router_select_sigmoid_shared_kernel(
    float* __restrict__ logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    int64_t logits_stride,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ float shared_logits[];
  const int64_t token = blockIdx.x;
  qwen35_router_select_from_logits(
      logits + token * logits_stride,
      selected,
      routing,
      shared_logits,
      token,
      logits_stride,
      num_experts,
      top_k);
  if (threadIdx.x == 0) {
    float* shared_gate = logits + token * logits_stride + num_experts;
    *shared_gate = sigmoid_f32(*shared_gate);
  }
}

template <typename scalar_t>
__device__ inline void qwen35_router_reduce_row(
    const scalar_t* __restrict__ hidden,
    const uint16_t* __restrict__ weight_row,
    float* __restrict__ shared_scratch,
    int64_t hidden_size) {
  const int tid = threadIdx.x;
  float acc = 0.0f;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = tid * 8; k + 7 < hidden_size; k += vec_stride) {
    acc += scalar_to_float_qwen35(hidden[k + 0]) * bf16_bits_to_float(weight_row[k + 0]);
    acc += scalar_to_float_qwen35(hidden[k + 1]) * bf16_bits_to_float(weight_row[k + 1]);
    acc += scalar_to_float_qwen35(hidden[k + 2]) * bf16_bits_to_float(weight_row[k + 2]);
    acc += scalar_to_float_qwen35(hidden[k + 3]) * bf16_bits_to_float(weight_row[k + 3]);
    acc += scalar_to_float_qwen35(hidden[k + 4]) * bf16_bits_to_float(weight_row[k + 4]);
    acc += scalar_to_float_qwen35(hidden[k + 5]) * bf16_bits_to_float(weight_row[k + 5]);
    acc += scalar_to_float_qwen35(hidden[k + 6]) * bf16_bits_to_float(weight_row[k + 6]);
    acc += scalar_to_float_qwen35(hidden[k + 7]) * bf16_bits_to_float(weight_row[k + 7]);
  }
  for (int64_t k = (hidden_size & ~7) + tid; k < hidden_size; k += blockDim.x) {
    acc += scalar_to_float_qwen35(hidden[k]) * bf16_bits_to_float(weight_row[k]);
  }
  shared_scratch[tid] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_scratch[tid] += shared_scratch[tid + stride];
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void qwen35_router_topk_shared_coop_out_kernel(
    const scalar_t* __restrict__ hidden,
    const uint16_t* __restrict__ combined_weight,
    float* __restrict__ logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ float shared_scratch[];
  __shared__ int is_last_block;

  const int64_t row = blockIdx.x;
  const int64_t token = 0;
  const int tid = threadIdx.x;
  const uint16_t* weight_row = combined_weight + row * hidden_size;
  qwen35_router_reduce_row(hidden, weight_row, shared_scratch, hidden_size);

  if (tid == 0) {
    logits[row] = shared_scratch[0];
    __threadfence();
    int32_t* counter = reinterpret_cast<int32_t*>(selected);
    const int prior = atomicAdd(counter, 1);
    is_last_block = (prior + 1 == static_cast<int>(num_rows)) ? 1 : 0;
  }
  __syncthreads();

  if (is_last_block == 0) {
    return;
  }

  qwen35_router_select_from_logits(
      logits,
      selected,
      routing,
      shared_scratch,
      token,
      num_rows,
      num_experts,
      top_k);
}

template <typename scalar_t>
__global__ void qwen35_router_topk_split_shared_coop_out_kernel(
    const scalar_t* __restrict__ hidden,
    const uint16_t* __restrict__ expert_weight,
    const uint16_t* __restrict__ shared_weight,
    float* __restrict__ logits,
    int64_t* __restrict__ selected,
    float* __restrict__ routing,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ float shared_scratch[];
  __shared__ int is_last_block;

  const int64_t row = blockIdx.x;
  const int64_t num_rows = num_experts + 1;
  const int64_t token = 0;
  const int tid = threadIdx.x;
  const uint16_t* weight_row = (row < num_experts) ? (expert_weight + row * hidden_size) : shared_weight;
  qwen35_router_reduce_row(hidden, weight_row, shared_scratch, hidden_size);

  if (tid == 0) {
    logits[row] = shared_scratch[0];
    __threadfence();
    int32_t* counter = reinterpret_cast<int32_t*>(selected);
    const int prior = atomicAdd(counter, 1);
    is_last_block = (prior + 1 == static_cast<int>(num_rows)) ? 1 : 0;
  }
  __syncthreads();

  if (is_last_block == 0) {
    return;
  }

  qwen35_router_select_from_logits(
      logits,
      selected,
      routing,
      shared_scratch,
      token,
      num_rows,
      num_experts,
      top_k);
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_qwen35_router_logits_bf16(
    const uint16_t* hidden,
    const uint16_t* weight,
    float* logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || num_rows <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(launch_qwen35_router_logits(
      hidden,
      weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream));
}

extern "C" int hipengine_qwen35_router_logits_fp16(
    const half_t* hidden,
    const uint16_t* weight,
    float* logits,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || num_rows <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return static_cast<int>(launch_qwen35_router_logits(
      hidden,
      weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream));
}

extern "C" int hipengine_qwen35_router_select(
    const float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t logits_stride,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || logits_stride <= 0 || num_experts <= 0 || top_k <= 0 || top_k > 16 ||
      top_k > num_experts || num_experts > logits_stride || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_router_select_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      selected,
      routing,
      logits_stride,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_coop_out_bf16(
    const uint16_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens != 1 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t memset_err = hipMemsetAsync(selected, 0, sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  const int64_t shared_floats = (threads > num_experts) ? threads : num_experts;
  hipLaunchKernelGGL(
      (qwen35_router_topk_shared_coop_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(num_rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(shared_floats) * sizeof(float),
      stream,
      hidden,
      combined_weight,
      logits,
      selected,
      routing,
      hidden_size,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_coop_out_fp16(
    const half_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens != 1 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t memset_err = hipMemsetAsync(selected, 0, sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  const int64_t shared_floats = (threads > num_experts) ? threads : num_experts;
  hipLaunchKernelGGL(
      (qwen35_router_topk_shared_coop_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(num_rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(shared_floats) * sizeof(float),
      stream,
      hidden,
      combined_weight,
      logits,
      selected,
      routing,
      hidden_size,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_split_shared_coop_out_bf16(
    const uint16_t* hidden,
    const uint16_t* expert_weight,
    const uint16_t* shared_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens != 1 || hidden_size <= 0 || num_experts <= 0 || top_k <= 0 || top_k > 16 ||
      top_k > num_experts || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t memset_err = hipMemsetAsync(selected, 0, sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  const int64_t shared_floats = (threads > num_experts) ? threads : num_experts;
  hipLaunchKernelGGL(
      (qwen35_router_topk_split_shared_coop_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(num_experts + 1)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(shared_floats) * sizeof(float),
      stream,
      hidden,
      expert_weight,
      shared_weight,
      logits,
      selected,
      routing,
      hidden_size,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_split_shared_coop_out_fp16(
    const half_t* hidden,
    const uint16_t* expert_weight,
    const uint16_t* shared_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens != 1 || hidden_size <= 0 || num_experts <= 0 || top_k <= 0 || top_k > 16 ||
      top_k > num_experts || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t memset_err = hipMemsetAsync(selected, 0, sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  const int64_t shared_floats = (threads > num_experts) ? threads : num_experts;
  hipLaunchKernelGGL(
      (qwen35_router_topk_split_shared_coop_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(num_experts + 1)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(shared_floats) * sizeof(float),
      stream,
      hidden,
      expert_weight,
      shared_weight,
      logits,
      selected,
      routing,
      hidden_size,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_sigmoid_out_bf16(
    const uint16_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 1 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const hipError_t logits_err = launch_qwen35_router_logits(
      hidden,
      combined_weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream);
  if (logits_err != hipSuccess) {
    return static_cast<int>(logits_err);
  }
  hipLaunchKernelGGL(
      qwen35_router_select_sigmoid_shared_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      selected,
      routing,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_sigmoid_out_fp16(
    const half_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 1 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const hipError_t logits_err = launch_qwen35_router_logits(
      hidden,
      combined_weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream);
  if (logits_err != hipSuccess) {
    return static_cast<int>(logits_err);
  }
  hipLaunchKernelGGL(
      qwen35_router_select_sigmoid_shared_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      selected,
      routing,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_out_bf16(
    const uint16_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const hipError_t logits_err = launch_qwen35_router_logits(
      hidden,
      combined_weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream);
  if (logits_err != hipSuccess) {
    return static_cast<int>(logits_err);
  }
  hipLaunchKernelGGL(
      qwen35_router_select_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      selected,
      routing,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_router_topk_shared_out_fp16(
    const half_t* hidden,
    const uint16_t* combined_weight,
    float* logits,
    int64_t* selected,
    float* routing,
    int64_t tokens,
    int64_t hidden_size,
    int64_t num_rows,
    int64_t num_experts,
    int64_t top_k,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || num_rows <= 0 || num_experts <= 0 ||
      num_experts >= num_rows || top_k <= 0 || top_k > 16 || top_k > num_experts ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const hipError_t logits_err = launch_qwen35_router_logits(
      hidden,
      combined_weight,
      logits,
      tokens,
      hidden_size,
      num_rows,
      threads,
      stream);
  if (logits_err != hipSuccess) {
    return static_cast<int>(logits_err);
  }
  hipLaunchKernelGGL(
      qwen35_router_select_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(num_experts) * sizeof(float),
      stream,
      logits,
      selected,
      routing,
      num_rows,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}
