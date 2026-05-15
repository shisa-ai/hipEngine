// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 5d8f496.
// Qwen3.5 grouped-MoE metadata and packed-hidden gather kernels for native prefill.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void qwen35_moe_group_count_kernel(
    const int64_t* selected_experts,
    int32_t* counts,
    int64_t total_lanes,
    int64_t num_experts) {
  const int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= total_lanes) {
    return;
  }
  const int64_t expert = selected_experts[lane];
  if (expert >= 0 && expert < num_experts) {
    atomicAdd(&counts[expert], 1);
  }
}

__global__ void qwen35_moe_group_prefix_kernel(
    const int32_t* counts,
    int32_t* padded_counts,
    int64_t* expert_start,
    int64_t* total_padded,
    int64_t num_experts,
    int64_t pad_multiple) {
  extern __shared__ int64_t prefix[];
  const int64_t tid = threadIdx.x;
  int64_t padded = 0;
  if (tid < num_experts) {
    const int64_t count = static_cast<int64_t>(counts[tid]);
    padded = pad_multiple > 1 ? ((count + pad_multiple - 1) / pad_multiple) * pad_multiple : count;
    padded_counts[tid] = static_cast<int32_t>(padded);
  }
  prefix[tid] = padded;
  __syncthreads();

  for (int64_t offset = 1; offset < blockDim.x; offset <<= 1) {
    int64_t value = 0;
    if (tid >= offset) {
      value = prefix[tid - offset];
    }
    __syncthreads();
    prefix[tid] += value;
    __syncthreads();
  }

  if (tid < num_experts) {
    expert_start[tid] = prefix[tid] - padded;
  }
  if (tid == 0) {
    const int64_t total = num_experts > 0 ? prefix[num_experts - 1] : 0;
    expert_start[num_experts] = total;
    total_padded[0] = total;
  }
}

__global__ void qwen35_moe_wmma_tile_map_kernel(
    const int64_t* __restrict__ expert_start_compact,
    int64_t* __restrict__ wmma_expert_start,
    int64_t* __restrict__ tile_expert,
    int64_t* __restrict__ wmma_total_out,
    int64_t num_experts) {
  extern __shared__ int64_t prefix[];
  const int64_t tid = threadIdx.x;

  int64_t count = 0;
  int64_t padded = 0;
  if (tid < num_experts) {
    count = expert_start_compact[tid + 1] - expert_start_compact[tid];
    padded = ((count + 15) / 16) * 16;
  }
  prefix[tid] = padded;
  __syncthreads();

  for (int64_t offset = 1; offset < blockDim.x; offset <<= 1) {
    int64_t value = 0;
    if (tid >= offset) {
      value = prefix[tid - offset];
    }
    __syncthreads();
    prefix[tid] += value;
    __syncthreads();
  }

  if (tid < num_experts) {
    wmma_expert_start[tid] = prefix[tid] - padded;
  }

  int64_t total = 0;
  if (tid == 0) {
    total = num_experts > 0 ? prefix[num_experts - 1] : 0;
    wmma_expert_start[num_experts] = total;
    wmma_total_out[0] = total;
  }
  __syncthreads();

  total = num_experts > 0 ? prefix[num_experts - 1] : 0;
  (void)total;
  if (tid < num_experts && padded > 0) {
    const int64_t my_start_tile = (prefix[tid] - padded) / 16;
    const int64_t my_num_tiles = padded / 16;
    for (int64_t t = 0; t < my_num_tiles; t++) {
      tile_expert[my_start_tile + t] = tid;
    }
  }
}

__global__ void qwen35_moe_group_scatter_kernel(
    const int64_t* selected_experts,
    const float* routing_weights,
    const int64_t* expert_start,
    int32_t* scatter_offsets,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    bool include_routing_weights) {
  const int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= total_lanes) {
    return;
  }
  const int64_t expert = selected_experts[lane];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const int32_t offset = atomicAdd(&scatter_offsets[expert], 1);
  const int64_t out_idx = expert_start[expert] + static_cast<int64_t>(offset);
  sorted_lanes[out_idx] = lane;
  sorted_experts[out_idx] = expert;
  if (include_routing_weights) {
    sorted_weights[out_idx] = routing_weights[lane];
  }
}

__global__ void qwen35_moe_group_scatter_gather_kernel(
    const uint16_t* __restrict__ hidden_states,
    const int64_t* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    const int64_t* __restrict__ expert_start,
    int32_t* __restrict__ scatter_offsets,
    int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ sorted_experts,
    float* __restrict__ sorted_weights,
    uint16_t* __restrict__ packed_hidden,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    int64_t hidden_size) {
  const int64_t lane = static_cast<int64_t>(blockIdx.x);
  if (lane >= total_lanes) {
    return;
  }
  const int64_t expert = selected_experts[lane];
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  __shared__ int64_t out_idx_s;
  if (threadIdx.x == 0) {
    const int32_t offset = atomicAdd(&scatter_offsets[expert], 1);
    const int64_t out_idx = expert_start[expert] + static_cast<int64_t>(offset);
    out_idx_s = out_idx;
    sorted_lanes[out_idx] = lane;
    sorted_experts[out_idx] = expert;
    sorted_weights[out_idx] = routing_weights[lane];
  }
  __syncthreads();

  const int64_t out_idx = out_idx_s;
  const int64_t token_idx = lane / top_k;
  const int64_t in_base = token_idx * hidden_size;
  const int64_t out_base = out_idx * hidden_size;
  for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    packed_hidden[out_base + col] = hidden_states[in_base + col];
  }
}

__global__ void qwen35_moe_gather_packed_hidden_kernel(
    const uint16_t* hidden_states,
    const int64_t* sorted_lanes,
    uint16_t* packed_hidden,
    int64_t total_elements,
    int64_t tokens,
    int64_t top_k,
    int64_t hidden_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements) {
    return;
  }
  const int64_t row = idx / hidden_size;
  const int64_t col = idx - row * hidden_size;
  uint16_t value = 0;
  const int64_t lane = sorted_lanes[row];
  if (lane >= 0) {
    const int64_t token_idx = lane / top_k;
    if (token_idx >= 0 && token_idx < tokens) {
      value = hidden_states[token_idx * hidden_size + col];
    }
  }
  packed_hidden[idx] = value;
}

int64_t next_power_of_two_i64(int64_t value) {
  int64_t out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

}  // namespace

extern "C" int hipengine_qwen35_moe_group_count(
    const int64_t* selected_experts,
    int32_t* counts,
    int64_t total_lanes,
    int64_t num_experts,
    hipStream_t stream) {
  if (total_lanes <= 0 || num_experts <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total_lanes + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_moe_group_count_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      selected_experts,
      counts,
      total_lanes,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_group_prefix(
    const int32_t* counts,
    int32_t* padded_counts,
    int64_t* expert_start,
    int64_t* total_padded,
    int64_t num_experts,
    int64_t pad_multiple,
    hipStream_t stream) {
  if (num_experts <= 0 || pad_multiple <= 0 || num_experts > 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t threads64 = next_power_of_two_i64(num_experts);
  const int threads = static_cast<int>(threads64 < 32 ? 32 : threads64);
  hipLaunchKernelGGL(
      qwen35_moe_group_prefix_kernel,
      dim3(1),
      dim3(threads),
      static_cast<size_t>(threads) * sizeof(int64_t),
      stream,
      counts,
      padded_counts,
      expert_start,
      total_padded,
      num_experts,
      pad_multiple);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_wmma_tile_map(
    const int64_t* expert_start_compact,
    int64_t* wmma_expert_start,
    int64_t* tile_expert,
    int64_t* wmma_total_out,
    int64_t num_experts,
    hipStream_t stream) {
  if (num_experts <= 0 || num_experts > 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t threads64 = next_power_of_two_i64(num_experts);
  const int threads = static_cast<int>(threads64 < 32 ? 32 : threads64);
  hipLaunchKernelGGL(
      qwen35_moe_wmma_tile_map_kernel,
      dim3(1),
      dim3(threads),
      static_cast<size_t>(threads) * sizeof(int64_t),
      stream,
      expert_start_compact,
      wmma_expert_start,
      tile_expert,
      wmma_total_out,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_group_scatter(
    const int64_t* selected_experts,
    const float* routing_weights,
    const int64_t* expert_start,
    int32_t* scatter_offsets,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    bool include_routing_weights,
    hipStream_t stream) {
  if (total_lanes <= 0 || num_experts <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total_lanes + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_moe_group_scatter_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      selected_experts,
      routing_weights,
      expert_start,
      scatter_offsets,
      sorted_lanes,
      sorted_experts,
      sorted_weights,
      total_lanes,
      num_experts,
      include_routing_weights);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_group_scatter_gather_lowp(
    const uint16_t* hidden_states,
    const int64_t* selected_experts,
    const float* routing_weights,
    const int64_t* expert_start,
    int32_t* scatter_offsets,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    uint16_t* packed_hidden,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    int64_t hidden_size,
    hipStream_t stream) {
  if (total_lanes <= 0 || num_experts <= 0 || top_k <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_moe_group_scatter_gather_kernel,
      dim3(static_cast<unsigned int>(total_lanes)),
      dim3(256),
      0,
      stream,
      hidden_states,
      selected_experts,
      routing_weights,
      expert_start,
      scatter_offsets,
      sorted_lanes,
      sorted_experts,
      sorted_weights,
      packed_hidden,
      total_lanes,
      num_experts,
      top_k,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_gather_packed_hidden_lowp(
    const uint16_t* hidden_states,
    const int64_t* sorted_lanes,
    uint16_t* packed_hidden,
    int64_t total_elements,
    int64_t tokens,
    int64_t top_k,
    int64_t hidden_size,
    hipStream_t stream) {
  if (total_elements <= 0 || tokens <= 0 || top_k <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_moe_gather_packed_hidden_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      sorted_lanes,
      packed_hidden,
      total_elements,
      tokens,
      top_k,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}
