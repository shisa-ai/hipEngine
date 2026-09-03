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

__global__ void qwen35_moe_group_prefix_active_kernel(
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ expert_start,
    int64_t* __restrict__ active_experts,
    int64_t* __restrict__ active_count,
    int64_t num_experts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t offset = 0;
  int64_t active = 0;
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    expert_start[expert] = offset;
    const int64_t count = static_cast<int64_t>(counts[expert]);
    if (count > 0) {
      active_experts[active++] = expert;
      offset += count;
    }
  }
  expert_start[num_experts] = offset;
  active_count[0] = active;
}

__global__ void qwen35_moe_group_compact_active_kernel(
    const int64_t* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    int64_t* __restrict__ expert_start,
    int64_t* __restrict__ active_experts,
    int64_t* __restrict__ active_count,
    int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ sorted_experts,
    int64_t* __restrict__ compact_to_source,
    float* __restrict__ sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k) {
  extern __shared__ int64_t counts[];
  const int64_t expert = threadIdx.x;
  if (expert < num_experts) {
    int64_t count = 0;
    for (int64_t lane = 0; lane < total_lanes; ++lane) {
      count += selected_experts[lane] == expert;
    }
    counts[expert] = count;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int64_t offset = 0;
    int64_t active = 0;
    for (int64_t idx = 0; idx < num_experts; ++idx) {
      expert_start[idx] = offset;
      const int64_t count = counts[idx];
      if (count > 0) {
        active_experts[active++] = idx;
        offset += count;
      }
    }
    expert_start[num_experts] = offset;
    active_count[0] = active;
  }
  __syncthreads();

  if (expert < num_experts && counts[expert] > 0) {
    int64_t out_idx = expert_start[expert];
    for (int64_t lane = 0; lane < total_lanes; ++lane) {
      if (selected_experts[lane] == expert) {
        sorted_lanes[out_idx] = lane;
        if (sorted_experts) {
          sorted_experts[out_idx] = expert;
        }
        if (compact_to_source) {
          compact_to_source[out_idx] = lane / top_k;
        }
        sorted_weights[out_idx] = routing_weights[lane];
        ++out_idx;
      }
    }
  }
}

// Stable expert-major compaction without the serial 256 x total_lanes scan.
// Stage 1 gives every expert its own workgroup and stores exact counts in the
// caller-owned expert_start array. Stage 2 converts those counts in place to
// starts. Stage 3 gives every active expert a workgroup and uses wave ballots
// to preserve ascending lane order inside each expert.
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
  const int64_t count =
      tid < num_experts ? expert_start[tid] : 0;
  __shared__ int64_t scan[256];
  __shared__ int wave_counts[8];
  __shared__ int wave_starts[8];
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
  const int wave = tid >> 5;
  const unsigned long long mask =
      __ballot(tid < num_experts && count > 0);
  if (lane32 == 0) {
    wave_counts[wave] = __popcll(mask);
  }
  __syncthreads();
  if (tid == 0) {
    int active = 0;
#pragma unroll
    for (int wave_idx = 0; wave_idx < 8; ++wave_idx) {
      wave_starts[wave_idx] = active;
      active += wave_counts[wave_idx];
    }
    active_count[0] = active;
  }
  __syncthreads();
  if (tid < num_experts && count > 0) {
    const unsigned long long lower_mask =
        lane32 == 0 ? 0ull : ((1ull << lane32) - 1ull);
    const int rank = __popcll(mask & lower_mask);
    active_experts[wave_starts[wave] + rank] = tid;
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
    int64_t* __restrict__ compact_to_source,
    float* __restrict__ sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k) {
  const int64_t expert = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (expert >= num_experts ||
      expert_start[expert] == expert_start[expert + 1]) {
    return;
  }

  const int wave = tid >> 5;
  const int lane32 = tid & 31;
  __shared__ int wave_counts[8];
  __shared__ int wave_starts[8];
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
    const unsigned long long mask = __ballot(match);
    if (lane32 == 0) {
      wave_counts[wave] = __popcll(mask);
    }
    __syncthreads();

    if (tid == 0) {
      int prefix = 0;
#pragma unroll
      for (int wave_idx = 0; wave_idx < 8; ++wave_idx) {
        wave_starts[wave_idx] = prefix;
        prefix += wave_counts[wave_idx];
      }
      chunk_base = running;
      running += prefix;
    }
    __syncthreads();

    if (match) {
      const unsigned long long lower_mask =
          lane32 == 0 ? 0ull : ((1ull << lane32) - 1ull);
      const int wave_rank = __popcll(mask & lower_mask);
      const int64_t out_idx =
          chunk_base + wave_starts[wave] + wave_rank;
      sorted_lanes[out_idx] = lane;
      if (sorted_experts) {
        sorted_experts[out_idx] = expert;
      }
      if (compact_to_source) {
        compact_to_source[out_idx] = lane / top_k;
      }
      sorted_weights[out_idx] = routing_weights[lane];
    }
    __syncthreads();
  }
}

__global__ void qwen35_moe_wmma_tile_map_kernel(
    const int64_t* __restrict__ expert_start_compact,
    int64_t* __restrict__ wmma_expert_start,
    int64_t* __restrict__ tile_expert,
    int64_t* __restrict__ wmma_total_out,
    int64_t num_experts,
    int64_t tile_capacity,
    int64_t tile_rows) {
  extern __shared__ int64_t prefix[];
  const int64_t tid = threadIdx.x;

  if (tile_capacity > 0) {
    for (int64_t idx = tid; idx < tile_capacity; idx += blockDim.x) {
      tile_expert[idx] = -1;
    }
  }
  __syncthreads();

  int64_t count = 0;
  int64_t padded = 0;
  if (tid < num_experts) {
    count = expert_start_compact[tid + 1] - expert_start_compact[tid];
    padded = ((count + tile_rows - 1) / tile_rows) * tile_rows;
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
    const int64_t my_start_tile = (prefix[tid] - padded) / tile_rows;
    const int64_t my_num_tiles = padded / tile_rows;
    for (int64_t t = 0; t < my_num_tiles; t++) {
      const int64_t tile_idx = my_start_tile + t;
      if (tile_capacity <= 0 || tile_idx < tile_capacity) {
        tile_expert[tile_idx] = tid;
      }
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

// Packed (token, top-k slot) pair: 22 bits token + 10 bits slot, matching the
// halo-box a7ad7b7f mmid.cu mm_ids_helper_store layout.
struct moe_map_store {
  uint32_t data;
  __device__ moe_map_store(uint32_t it, uint32_t iex_used) {
    data = (it & 0x003FFFFFu) | (iex_used << 22);
  }
  __device__ uint32_t it() const { return data & 0x003FFFFFu; }
  __device__ uint32_t iex_used() const { return data >> 22; }
};

__device__ inline int moe_map_warp_any16(int x) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    x = __shfl_xor(x, offset, 16) || x;
  }
  return x;
}

__device__ inline int moe_map_warp_sum32(int x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, 32);
  }
  return x;
}

// PF-4 M2 candidate: fused per-expert routing-map compaction (port of the
// halo-box a7ad7b7f mm_ids_helper_top10_parallel mechanism). One 256-thread
// block (8 warps x 32 lanes) per expert; each warp walks its token slice two
// tokens at a time (32 lanes / 16 padded top-k slots), builds the per-expert
// compact list with a warp any-scan + shfl_up exclusive scan, and combines
// the 8 warps through shared per-warp counts/prefixes. Replaces the
// incumbent memset + group_count + group_prefix + memset + map-scatter chain;
// all outputs are integer maps plus copied float weights (T0, no float
// arithmetic). Requires top_k == 10 and at most one slot per token matching
// any single expert (production top-k guarantees distinct experts per token).
// expert_start semantics match the incumbent count+prefix exactly: only
// lanes with 0 <= expert < num_experts are counted; expert_start[e] is the
// exclusive prefix and expert_start[num_experts] the valid-lane total.
__global__ void qwen35_moe_group_map_top10_parallel_kernel(
    const int64_t* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    int64_t* __restrict__ expert_start,
    int64_t* __restrict__ sorted_lanes,
    int64_t* __restrict__ sorted_experts,
    float* __restrict__ sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k) {
  constexpr int warp_size = 32;
  constexpr int nwarps = 8;
  constexpr int n_expert_used = 10;
  constexpr int neu_padded = 16;

  const int64_t expert = blockIdx.x;
  const int warp = threadIdx.x / warp_size;
  const int lane = threadIdx.x % warp_size;
  const int64_t n_tokens = total_lanes / top_k;
  const int64_t tokens_per_warp = (n_tokens + nwarps - 1) / nwarps;
  const int64_t token_begin = min(warp * tokens_per_warp, n_tokens);
  const int64_t token_end = min(token_begin + tokens_per_warp, n_tokens);

  extern __shared__ char data_moe_map[];
  moe_map_store* store = reinterpret_cast<moe_map_store*>(data_moe_map);
  __shared__ int64_t warp_counts[nwarps];
  __shared__ int64_t warp_nex_prev[nwarps];

  int64_t nex_prev = 0;
  int64_t it_compact = 0;
  for (int64_t it0 = token_begin; it0 < token_end; it0 += warp_size / neu_padded) {
    const int64_t it = it0 + lane / neu_padded;
    const int iex = lane % neu_padded;
    const bool slot_valid = (iex < n_expert_used) && (it < token_end);
    const int64_t expert_used = slot_valid ? selected_experts[it * top_k + iex] : INT64_MAX;
    const int iex_used = (expert_used == expert) ? iex : -1;
    if (slot_valid && expert_used >= 0 && expert_used < num_experts &&
        expert_used < expert) {
      nex_prev += 1;
    }

    const int it_compact_add_self = moe_map_warp_any16(iex_used != -1);
    int it_compact_add_lower = 0;
#pragma unroll
    for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
      const int tmp = __shfl_up(it_compact_add_self, offset, warp_size);
      if (lane >= offset) {
        it_compact_add_lower += tmp;
      }
    }
    if (iex_used != -1) {
      store[token_begin + it_compact + it_compact_add_lower] =
          moe_map_store(static_cast<uint32_t>(it), static_cast<uint32_t>(iex_used));
    }
    it_compact += __shfl(it_compact_add_lower + it_compact_add_self,
                              warp_size - 1, warp_size);
  }

  nex_prev = moe_map_warp_sum32(static_cast<int>(nex_prev));
  if (lane == 0) {
    warp_counts[warp] = it_compact;
    warp_nex_prev[warp] = nex_prev;
  }
  __syncthreads();

  int64_t compact_prefix = 0;
  int64_t nex_prev_total = 0;
#pragma unroll
  for (int i = 0; i < nwarps; ++i) {
    nex_prev_total += warp_nex_prev[i];
    if (i < warp) {
      compact_prefix += warp_counts[i];
    }
  }

  for (int64_t itc = lane; itc < it_compact; itc += warp_size) {
    const moe_map_store store_it = store[token_begin + itc];
    const int64_t it = static_cast<int64_t>(store_it.it());
    const int64_t iex_used = static_cast<int64_t>(store_it.iex_used());
    const int64_t compact = nex_prev_total + compact_prefix + itc;
    const int64_t lane_idx = it * top_k + iex_used;
    sorted_lanes[compact] = lane_idx;
    sorted_experts[compact] = expert;
    sorted_weights[compact] = routing_weights[lane_idx];
  }

  if (threadIdx.x == 0) {
    expert_start[expert] = nex_prev_total;
    if (expert == static_cast<int64_t>(gridDim.x) - 1) {
      int64_t total_count = 0;
#pragma unroll
      for (int i = 0; i < nwarps; ++i) {
        total_count += warp_counts[i];
      }
      expert_start[num_experts] = nex_prev_total + total_count;
    }
  }
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

extern "C" int hipengine_qwen35_moe_group_prefix_active(
    const int32_t* counts,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t num_experts,
    hipStream_t stream) {
  if (!counts || !expert_start || !active_experts || !active_count ||
      num_experts <= 0 || num_experts > 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_moe_group_prefix_active_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      counts,
      expert_start,
      active_experts,
      active_count,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_group_compact_active(
    const int64_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    hipStream_t stream) {
  if (!selected_experts || !routing_weights || !expert_start ||
      !active_experts || !active_count || !sorted_lanes || !sorted_experts ||
      !sorted_weights || total_lanes <= 0 || num_experts <= 0 ||
      num_experts > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_moe_group_compact_active_kernel,
      dim3(1),
      dim3(256),
      static_cast<size_t>(num_experts) * sizeof(int64_t),
      stream,
      selected_experts,
      routing_weights,
      expert_start,
      active_experts,
      active_count,
      sorted_lanes,
      sorted_experts,
      nullptr,
      sorted_weights,
      total_lanes,
      num_experts,
      1);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_group_compact_active_source_rows(
    const int64_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* compact_to_source,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (!selected_experts || !routing_weights || !expert_start ||
      !active_experts || !active_count || !sorted_lanes ||
      !compact_to_source || !sorted_weights || total_lanes <= 0 ||
      num_experts <= 0 || num_experts > 256 || top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_moe_group_compact_active_kernel,
      dim3(1),
      dim3(256),
      static_cast<size_t>(num_experts) * sizeof(int64_t),
      stream,
      selected_experts,
      routing_weights,
      expert_start,
      active_experts,
      active_count,
      sorted_lanes,
      nullptr,
      compact_to_source,
      sorted_weights,
      total_lanes,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}

template <typename ExpertId>
int launch_qwen35_moe_group_compact_active_parallel(
    const ExpertId* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    int64_t* compact_to_source,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (!selected_experts || !routing_weights || !expert_start ||
      !active_experts || !active_count || !sorted_lanes ||
      (!sorted_experts && !compact_to_source) || !sorted_weights ||
      total_lanes <= 0 || num_experts <= 0 || num_experts > 256 ||
      top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (qwen35_moe_group_count_active_parallel_kernel<ExpertId>),
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(256),
      0,
      stream,
      selected_experts,
      expert_start,
      total_lanes,
      num_experts);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  hipLaunchKernelGGL(
      qwen35_moe_group_prefix_active_parallel_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      expert_start,
      active_experts,
      active_count,
      num_experts);
  error = hipGetLastError();
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  hipLaunchKernelGGL(
      (qwen35_moe_group_scatter_active_parallel_kernel<ExpertId>),
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(256),
      0,
      stream,
      selected_experts,
      routing_weights,
      expert_start,
      sorted_lanes,
      sorted_experts,
      compact_to_source,
      sorted_weights,
      total_lanes,
      num_experts,
      top_k);
  error = hipGetLastError();
  return static_cast<int>(error);
}

extern "C" int hipengine_qwen35_moe_group_compact_active_parallel(
    const int64_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_qwen35_moe_group_compact_active_parallel<int64_t>(
      selected_experts,
      routing_weights,
      expert_start,
      active_experts,
      active_count,
      sorted_lanes,
      sorted_experts,
      nullptr,
      sorted_weights,
      total_lanes,
      num_experts,
      1,
      stream);
}

extern "C" int hipengine_qwen35_moe_group_compact_active_source_rows_parallel(
    const int64_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* active_experts,
    int64_t* active_count,
    int64_t* sorted_lanes,
    int64_t* compact_to_source,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  return launch_qwen35_moe_group_compact_active_parallel<int64_t>(
      selected_experts,
      routing_weights,
      expert_start,
      active_experts,
      active_count,
      sorted_lanes,
      nullptr,
      compact_to_source,
      sorted_weights,
      total_lanes,
      num_experts,
      top_k,
      stream);
}

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
    hipStream_t stream) {
  return launch_qwen35_moe_group_compact_active_parallel<int32_t>(
      selected_experts,
      routing_weights,
      expert_start,
      active_experts,
      active_count,
      sorted_lanes,
      sorted_experts,
      nullptr,
      sorted_weights,
      total_lanes,
      num_experts,
      1,
      stream);
}

extern "C" int hipengine_qwen35_moe_wmma_tile_map(
    const int64_t* expert_start_compact,
    int64_t* wmma_expert_start,
    int64_t* tile_expert,
    int64_t* wmma_total_out,
    int64_t num_experts,
    int64_t tile_capacity,
    hipStream_t stream) {
  if (num_experts <= 0 || num_experts > 1024 || tile_capacity < 0) {
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
      num_experts,
      tile_capacity,
      16);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_mmq32_tile_map(
    const int64_t* expert_start_compact,
    int64_t* mmq_expert_start,
    int64_t* tile_expert,
    int64_t* mmq_total_out,
    int64_t num_experts,
    int64_t tile_capacity,
    hipStream_t stream) {
  if (!expert_start_compact || !mmq_expert_start || !tile_expert ||
      !mmq_total_out || num_experts <= 0 || num_experts > 1024 ||
      tile_capacity < 0) {
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
      mmq_expert_start,
      tile_expert,
      mmq_total_out,
      num_experts,
      tile_capacity,
      32);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_moe_mmq64_tile_map(
    const int64_t* expert_start_compact,
    int64_t* mmq_expert_start,
    int64_t* tile_expert,
    int64_t* mmq_total_out,
    int64_t num_experts,
    int64_t tile_capacity,
    hipStream_t stream) {
  if (!expert_start_compact || !mmq_expert_start || !tile_expert ||
      !mmq_total_out || num_experts <= 0 || num_experts > 1024 ||
      tile_capacity < 0) {
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
      mmq_expert_start,
      tile_expert,
      mmq_total_out,
      num_experts,
      tile_capacity,
      64);
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

extern "C" int hipengine_qwen35_moe_group_map_top10_parallel(
    const int64_t* selected_experts,
    const float* routing_weights,
    int64_t* expert_start,
    int64_t* sorted_lanes,
    int64_t* sorted_experts,
    float* sorted_weights,
    int64_t total_lanes,
    int64_t num_experts,
    int64_t top_k,
    hipStream_t stream) {
  if (!selected_experts || !routing_weights || !expert_start || !sorted_lanes ||
      !sorted_experts || !sorted_weights || total_lanes <= 0 ||
      num_experts <= 0 || top_k <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (top_k != 10 || total_lanes % top_k != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n_tokens = total_lanes / top_k;
  if (n_tokens >= (1 << 22)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      qwen35_moe_group_map_top10_parallel_kernel,
      dim3(static_cast<unsigned int>(num_experts)),
      dim3(threads),
      n_tokens * sizeof(moe_map_store),
      stream,
      selected_experts,
      routing_weights,
      expert_start,
      sorted_lanes,
      sorted_experts,
      sorted_weights,
      total_lanes,
      num_experts,
      top_k);
  return static_cast<int>(hipGetLastError());
}
