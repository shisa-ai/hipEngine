// Stable expert-major Maple prefill metadata for CUDA sm_120a.
// Ported from hipengine/kernels/hip_gfx1100/moe/group_scatter.hip
// (hipEngine@12106b8a4; original lineage
// nano-vllm-amd/csrc/amd/qwen35_expert.hip@5d8f496).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

template <typename ExpertId>
__launch_bounds__(256, 1)
__global__ void qwen35_moe_group_count_active_parallel_kernel(
    const ExpertId* __restrict__ selected_experts,
    int64_t* __restrict__ expert_start,
    int64_t total_lanes,
    int64_t num_experts) {
  const int64_t expert = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (expert >= num_experts) {
    return;
  }

  int64_t count = 0;
  for (int64_t lane = tid; lane < total_lanes; lane += blockDim.x) {
    count += selected_experts[lane] == expert;
  }

  __shared__ int64_t partial[256];
  partial[tid] = count;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    expert_start[expert] = partial[0];
  }
}

__launch_bounds__(256, 1)
__global__ void qwen35_moe_group_prefix_active_parallel_kernel(
    int64_t* __restrict__ expert_start,
    int64_t* __restrict__ active_experts,
    int64_t* __restrict__ active_count,
    int64_t num_experts) {
  if (blockIdx.x != 0) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int64_t count = tid < num_experts ? expert_start[tid] : 0;
  __shared__ int64_t scan[256];
  __shared__ int warp_counts[8];
  __shared__ int warp_starts[8];
  __shared__ int64_t total;
  scan[tid] = count;
  __syncthreads();

  // Blelloch exclusive scan over the fixed 256-expert upper bound.
  for (int offset = 1; offset < 256; offset <<= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < 256) {
      scan[index] += scan[index - offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    total = scan[255];
    scan[255] = 0;
  }
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    const int index = (tid + 1) * offset * 2 - 1;
    if (index < 256) {
      const int64_t left = scan[index - offset];
      scan[index - offset] = scan[index];
      scan[index] += left;
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    expert_start[tid] = scan[tid];
  }
  if (tid == 0) {
    expert_start[num_experts] = total;
  }

  const int lane32 = tid & 31;
  const int warp = tid >> 5;
  const unsigned int mask = __ballot_sync(
      0xffffffffU, tid < num_experts && count > 0);
  if (lane32 == 0) {
    warp_counts[warp] = __popc(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int active = 0;
#pragma unroll
    for (int warp_idx = 0; warp_idx < 8; ++warp_idx) {
      warp_starts[warp_idx] = active;
      active += warp_counts[warp_idx];
    }
    active_count[0] = active;
  }
  __syncthreads();
  if (tid < num_experts && count > 0) {
    const unsigned int lower_mask =
        lane32 == 0 ? 0U : ((1U << lane32) - 1U);
    const int rank = __popc(mask & lower_mask);
    active_experts[warp_starts[warp] + rank] = tid;
  }
}

template <typename ExpertId>
__launch_bounds__(256, 1)
__global__ void qwen35_moe_group_scatter_active_parallel_kernel(
    const ExpertId* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    const int64_t* __restrict__ expert_start,
    int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ sorted_experts,
    float* __restrict__ sorted_weights,
    int64_t total_lanes,
    int64_t num_experts) {
  const int64_t expert = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (expert >= num_experts ||
      expert_start[expert] == expert_start[expert + 1]) {
    return;
  }

  const int warp = tid >> 5;
  const int lane32 = tid & 31;
  __shared__ int warp_counts[8];
  __shared__ int warp_starts[8];
  __shared__ int64_t running;
  __shared__ int64_t chunk_base;
  if (tid == 0) {
    running = expert_start[expert];
  }
  __syncthreads();

  for (int64_t chunk = 0; chunk < total_lanes; chunk += blockDim.x) {
    const int64_t lane = chunk + tid;
    const bool match =
        lane < total_lanes && selected_experts[lane] == expert;
    const unsigned int mask = __ballot_sync(0xffffffffU, match);
    if (lane32 == 0) {
      warp_counts[warp] = __popc(mask);
    }
    __syncthreads();

    if (tid == 0) {
      int prefix = 0;
#pragma unroll
      for (int warp_idx = 0; warp_idx < 8; ++warp_idx) {
        warp_starts[warp_idx] = prefix;
        prefix += warp_counts[warp_idx];
      }
      chunk_base = running;
      running += prefix;
    }
    __syncthreads();

    if (match) {
      const unsigned int lower_mask =
          lane32 == 0 ? 0U : ((1U << lane32) - 1U);
      const int warp_rank = __popc(mask & lower_mask);
      const int64_t out_idx =
          chunk_base + warp_starts[warp] + warp_rank;
      sorted_lanes[out_idx] = lane;
      sorted_experts[out_idx] = expert;
      sorted_weights[out_idx] = routing_weights[lane];
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" int hipengine_qwen35_moe_group_compact_active_i32_parallel(
    const int32_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    cudaStream_t stream) {
  if (!selected_experts || !routing_weights || !expert_start ||
      !active_experts || !active_count || !sorted_lanes || !sorted_experts ||
      !sorted_weights || total_lanes <= 0 || num_experts <= 0 ||
      num_experts > 256) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  qwen35_moe_group_count_active_parallel_kernel<int32_t>
      <<<dim3(static_cast<unsigned int>(num_experts)), dim3(256), 0, stream>>>(
          selected_experts, expert_start, total_lanes, num_experts);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  qwen35_moe_group_prefix_active_parallel_kernel
      <<<dim3(1), dim3(256), 0, stream>>>(
          expert_start, active_experts, active_count, num_experts);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }
  qwen35_moe_group_scatter_active_parallel_kernel<int32_t>
      <<<dim3(static_cast<unsigned int>(num_experts)), dim3(256), 0, stream>>>(
          selected_experts,
          routing_weights,
          expert_start,
          sorted_lanes,
          sorted_experts,
          sorted_weights,
          total_lanes,
          num_experts);
  return static_cast<int>(cudaGetLastError());
}
