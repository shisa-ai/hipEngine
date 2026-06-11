// Device-side DFlash verified-state commit helpers for hipEngine.
//
// The verifier writes root+candidates into transaction scratch.  This kernel
// installs only the accepted prefix into canonical state/KV/output buffers:
// final selected linear-state/hidden-tap rows are copied by commit_rows, and
// full-attention KV rows are compacted by walking parent_rows from each
// commit_row back to the request root.  Rejected suffix rows are never copied;
// output-ring suffix entries are cleared to -1 by the commit kernel and KV
// destination capacity is memset by the host wrapper before launch.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ int64_t accepted_prefix_offset(const int32_t* __restrict__ accepted_counts, int64_t request_index) {
  int64_t offset = 0;
  for (int64_t i = 0; i < request_index; ++i) {
    offset += static_cast<int64_t>(accepted_counts[i]);
  }
  return offset;
}

__device__ int64_t accepted_path_row(
    const int32_t* __restrict__ parent_rows,
    int64_t commit_row,
    int32_t accepted_count,
    int32_t path_index,
    int64_t target_rows) {
  int64_t row = commit_row;
  const int32_t climbs = accepted_count - 1 - path_index;
  for (int32_t step = 0; step < climbs; ++step) {
    if (row < 0 || row >= target_rows) {
      return -1;
    }
    row = static_cast<int64_t>(parent_rows[row]);
  }
  return (row >= 0 && row < target_rows) ? row : -1;
}

__device__ void copy_bytes_strided(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t bytes,
    int64_t thread_id,
    int64_t thread_count) {
  for (int64_t byte = thread_id; byte < bytes; byte += thread_count) {
    dst[byte] = src[byte];
  }
}

__global__ void dflash_commit_chain_i32_kernel(
    const int32_t* __restrict__ accepted_counts,
    const int32_t* __restrict__ commit_rows,
    const int32_t* __restrict__ commit_positions,
    const int32_t* __restrict__ parent_rows,
    const uint8_t* __restrict__ linear_state_src,
    uint8_t* __restrict__ linear_state_dst,
    int64_t linear_row_bytes,
    const uint8_t* __restrict__ kv_rows_src,
    uint8_t* __restrict__ kv_rows_dst,
    int64_t kv_row_bytes,
    int64_t kv_dst_rows,
    const uint8_t* __restrict__ hidden_taps_src,
    uint8_t* __restrict__ hidden_taps_dst,
    int64_t hidden_tap_count,
    int64_t hidden_tap_src_row_stride_bytes,
    int64_t hidden_tap_dst_row_stride_bytes,
    int64_t hidden_tap_src_plane_stride_bytes,
    int64_t hidden_tap_dst_plane_stride_bytes,
    const int32_t* __restrict__ next_tokens_src,
    const int32_t* __restrict__ committed_output_ids_src,
    const int32_t* __restrict__ committed_output_lengths_src,
    int32_t* __restrict__ output_ids_dst,
    int32_t* __restrict__ output_lengths_dst,
    int64_t output_src_cols,
    int64_t output_src_row_stride,
    int64_t output_dst_cols,
    int64_t output_dst_row_stride,
    int32_t* __restrict__ last_positions_dst,
    int32_t* __restrict__ context_lengths_dst,
    int64_t target_rows,
    int64_t request_count) {
  const int64_t request_index = static_cast<int64_t>(blockIdx.x);
  if (request_index >= request_count) {
    return;
  }
  const int64_t tid = static_cast<int64_t>(threadIdx.x);
  const int64_t threads = static_cast<int64_t>(blockDim.x);
  const int32_t accepted = accepted_counts[request_index];
  const int64_t commit_row = static_cast<int64_t>(commit_rows[request_index]);
  const int32_t commit_position = commit_positions[request_index];

  if (linear_state_src != nullptr && linear_state_dst != nullptr && linear_row_bytes > 0) {
    const uint8_t* src = linear_state_src + commit_row * linear_row_bytes;
    uint8_t* dst = linear_state_dst + request_index * linear_row_bytes;
    copy_bytes_strided(src, dst, linear_row_bytes, tid, threads);
  }

  if (kv_rows_src != nullptr && kv_rows_dst != nullptr && parent_rows != nullptr && kv_row_bytes > 0) {
    const int64_t dst_base = accepted_prefix_offset(accepted_counts, request_index);
    for (int32_t path_index = 0; path_index < accepted; ++path_index) {
      const int64_t dst_row = dst_base + static_cast<int64_t>(path_index);
      if (dst_row < 0 || dst_row >= kv_dst_rows) {
        continue;
      }
      const int64_t src_row = accepted_path_row(parent_rows, commit_row, accepted, path_index, target_rows);
      if (src_row < 0) {
        continue;
      }
      const uint8_t* src = kv_rows_src + src_row * kv_row_bytes;
      uint8_t* dst = kv_rows_dst + dst_row * kv_row_bytes;
      copy_bytes_strided(src, dst, kv_row_bytes, tid, threads);
    }
  }

  if (hidden_taps_src != nullptr && hidden_taps_dst != nullptr && hidden_tap_count > 0 && hidden_tap_src_row_stride_bytes > 0) {
    for (int64_t tap = 0; tap < hidden_tap_count; ++tap) {
      const uint8_t* src = hidden_taps_src + tap * hidden_tap_src_plane_stride_bytes + commit_row * hidden_tap_src_row_stride_bytes;
      uint8_t* dst = hidden_taps_dst + tap * hidden_tap_dst_plane_stride_bytes + request_index * hidden_tap_dst_row_stride_bytes;
      copy_bytes_strided(src, dst, hidden_tap_src_row_stride_bytes, tid, threads);
    }
  }

  if (committed_output_ids_src != nullptr && committed_output_lengths_src != nullptr && output_ids_dst != nullptr && output_lengths_dst != nullptr) {
    const int32_t raw_length = committed_output_lengths_src[request_index];
    const int64_t copy_length = raw_length < output_src_cols ? static_cast<int64_t>(raw_length) : output_src_cols;
    const int32_t next_token = next_tokens_src == nullptr ? -1 : next_tokens_src[request_index];
    const bool append_next = next_token >= 0;
    const int64_t produced_length = copy_length + (append_next ? 1 : 0);
    for (int64_t slot = tid; slot < output_dst_cols; slot += threads) {
      int32_t value = -1;
      if (slot < copy_length) {
        value = committed_output_ids_src[request_index * output_src_row_stride + slot];
      } else if (append_next && slot == copy_length) {
        value = next_token;
      }
      output_ids_dst[request_index * output_dst_row_stride + slot] = value;
    }
    if (tid == 0) {
      output_lengths_dst[request_index] = produced_length < output_dst_cols ? static_cast<int32_t>(produced_length) : static_cast<int32_t>(output_dst_cols);
    }
  }

  if (tid == 0) {
    if (last_positions_dst != nullptr) {
      last_positions_dst[request_index] = commit_position;
    }
    if (context_lengths_dst != nullptr) {
      context_lengths_dst[request_index] = commit_position + 1;
    }
  }
}

bool valid_commit_shape(int64_t target_rows, int64_t request_count) {
  return target_rows > 0 && request_count > 0 && request_count <= target_rows;
}

}  // namespace

extern "C" int hipengine_dflash_commit_chain_i32(
    const int32_t* accepted_counts,
    const int32_t* commit_rows,
    const int32_t* commit_positions,
    const int32_t* parent_rows,
    const void* linear_state_src,
    void* linear_state_dst,
    int64_t linear_row_bytes,
    const void* kv_rows_src,
    void* kv_rows_dst,
    int64_t kv_row_bytes,
    int64_t kv_dst_rows,
    const void* hidden_taps_src,
    void* hidden_taps_dst,
    int64_t hidden_tap_count,
    int64_t hidden_tap_src_row_stride_bytes,
    int64_t hidden_tap_dst_row_stride_bytes,
    int64_t hidden_tap_src_plane_stride_bytes,
    int64_t hidden_tap_dst_plane_stride_bytes,
    const int32_t* next_tokens_src,
    const int32_t* committed_output_ids_src,
    const int32_t* committed_output_lengths_src,
    int32_t* output_ids_dst,
    int32_t* output_lengths_dst,
    int64_t output_src_cols,
    int64_t output_src_row_stride,
    int64_t output_dst_cols,
    int64_t output_dst_row_stride,
    int32_t* last_positions_dst,
    int32_t* context_lengths_dst,
    int64_t target_rows,
    int64_t request_count,
    hipStream_t stream) {
  if (!valid_commit_shape(target_rows, request_count)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  hipLaunchKernelGGL(
      dflash_commit_chain_i32_kernel,
      dim3(static_cast<unsigned int>(request_count)),
      dim3(threads),
      0,
      stream,
      accepted_counts,
      commit_rows,
      commit_positions,
      parent_rows,
      static_cast<const uint8_t*>(linear_state_src),
      static_cast<uint8_t*>(linear_state_dst),
      linear_row_bytes,
      static_cast<const uint8_t*>(kv_rows_src),
      static_cast<uint8_t*>(kv_rows_dst),
      kv_row_bytes,
      kv_dst_rows,
      static_cast<const uint8_t*>(hidden_taps_src),
      static_cast<uint8_t*>(hidden_taps_dst),
      hidden_tap_count,
      hidden_tap_src_row_stride_bytes,
      hidden_tap_dst_row_stride_bytes,
      hidden_tap_src_plane_stride_bytes,
      hidden_tap_dst_plane_stride_bytes,
      next_tokens_src,
      committed_output_ids_src,
      committed_output_lengths_src,
      output_ids_dst,
      output_lengths_dst,
      output_src_cols,
      output_src_row_stride,
      output_dst_cols,
      output_dst_row_stride,
      last_positions_dst,
      context_lengths_dst,
      target_rows,
      request_count);
  return static_cast<int>(hipGetLastError());
}

// M12.4: multi-layer linear-attention state commit.
//
// One launch replaces the per-layer ``for layer_id in linear_layers:
// hipMemcpyAsync(conv) + hipMemcpyAsync(recurrent)`` sequence in
// ``Qwen35ParoResidentSession._commit_bulk_linear_states``.  All linear-
// attention layers share the same workspace-named scratch buffers, so the
// source row base is uniform across layers per buffer family; only the
// per-layer canonical destination differs.  The commit row index comes from
// ``verify_commit_rows[0]`` (device-resident, produced by
// ``dflash_accept_chain_i32`` in the same cycle).  Two families (conv,
// recurrent) are handled by one launch with ``blockIdx.y in {0, 1}``.
//
// Layout:
//   grid  : (n_layers, 2)
//   block : (threads,) one block copies one (layer, family) row
//   smem  : 0
//
// dst tables are device-resident arrays of n_layers pointers each, built once
// by the host at session init (canonical layer-state buffers do not move).
namespace {

__device__ inline void copy_bytes_strided_f4(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t bytes,
    int64_t tid,
    int64_t threads) {
  // float4-aligned fast path when bytes and pointers are 16-byte aligned.
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  if ((bytes & 15) == 0 && (src_addr & 15) == 0 && (dst_addr & 15) == 0) {
    const int64_t n_f4 = bytes >> 4;  // bytes / 16
    const float4* sp = reinterpret_cast<const float4*>(src);
    float4* dp = reinterpret_cast<float4*>(dst);
    for (int64_t i = tid; i < n_f4; i += threads) {
      dp[i] = sp[i];
    }
    return;
  }
  for (int64_t i = tid; i < bytes; i += threads) {
    dst[i] = src[i];
  }
}

__global__ void linear_state_pair_commit_i32_kernel(
    const uint64_t* __restrict__ src_conv_table,
    const uint64_t* __restrict__ dst_conv_table,
    int64_t conv_row_nbytes,
    const uint64_t* __restrict__ src_recurrent_table,
    const uint64_t* __restrict__ dst_recurrent_table,
    int64_t recurrent_row_nbytes,
    const int32_t* __restrict__ commit_row,
    int64_t n_layers) {
  const int64_t layer = static_cast<int64_t>(blockIdx.x);
  const int family = static_cast<int>(blockIdx.y);
  if (layer >= n_layers) {
    return;
  }
  const int32_t row = commit_row[0];
  const uint8_t* src_base;
  uint8_t* dst;
  int64_t row_nbytes;
  if (family == 0) {
    src_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_conv_table[layer]));
    dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_conv_table[layer]));
    row_nbytes = conv_row_nbytes;
  } else {
    src_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_recurrent_table[layer]));
    dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_recurrent_table[layer]));
    row_nbytes = recurrent_row_nbytes;
  }
  if (src_base == nullptr || dst == nullptr || row_nbytes <= 0) {
    return;
  }
  const uint8_t* src = src_base + static_cast<int64_t>(row) * row_nbytes;
  const int64_t tid = static_cast<int64_t>(threadIdx.x);
  const int64_t threads = static_cast<int64_t>(blockDim.x);
  copy_bytes_strided_f4(src, dst, row_nbytes, tid, threads);
}

__global__ void linear_state_pair_commit_chunked_i32_kernel(
    const uint64_t* __restrict__ src_conv_table,
    const uint64_t* __restrict__ dst_conv_table,
    int64_t conv_row_nbytes,
    const uint64_t* __restrict__ src_recurrent_table,
    const uint64_t* __restrict__ dst_recurrent_table,
    int64_t recurrent_row_nbytes,
    const int32_t* __restrict__ commit_row,
    int64_t n_layers,
    int64_t chunk_bytes) {
  const int64_t layer = static_cast<int64_t>(blockIdx.x);
  const int family = static_cast<int>(blockIdx.y);
  const int64_t chunk = static_cast<int64_t>(blockIdx.z);
  if (layer >= n_layers || chunk_bytes <= 0) {
    return;
  }
  const int32_t row = commit_row[0];
  const uint8_t* src_base;
  uint8_t* dst;
  int64_t row_nbytes;
  if (family == 0) {
    src_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_conv_table[layer]));
    dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_conv_table[layer]));
    row_nbytes = conv_row_nbytes;
  } else {
    src_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_recurrent_table[layer]));
    dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_recurrent_table[layer]));
    row_nbytes = recurrent_row_nbytes;
  }
  const int64_t start = chunk * chunk_bytes;
  if (src_base == nullptr || dst == nullptr || row_nbytes <= 0 || start >= row_nbytes) {
    return;
  }
  const int64_t chunk_end = start + chunk_bytes;
  const int64_t end = chunk_end < row_nbytes ? chunk_end : row_nbytes;
  const uint8_t* src = src_base + static_cast<int64_t>(row) * row_nbytes;
  const int64_t tid = static_cast<int64_t>(threadIdx.x);
  const int64_t threads = static_cast<int64_t>(blockDim.x);

  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src + start);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst + start);
  const int64_t bytes = end - start;
  if ((start & 15) == 0 && (bytes & 15) == 0 && (src_addr & 15) == 0 && (dst_addr & 15) == 0) {
    const int64_t f4_start = start >> 4;
    const int64_t n_f4 = bytes >> 4;
    const float4* sp = reinterpret_cast<const float4*>(src);
    float4* dp = reinterpret_cast<float4*>(dst);
    for (int64_t i = tid; i < n_f4; i += threads) {
      dp[f4_start + i] = sp[f4_start + i];
    }
    return;
  }
  for (int64_t i = start + tid; i < end; i += threads) {
    dst[i] = src[i];
  }
}

}  // namespace

extern "C" int hipengine_linear_state_pair_commit_i32(
    const void* src_conv_table,
    const void* dst_conv_table,
    int64_t conv_row_nbytes,
    const void* src_recurrent_table,
    const void* dst_recurrent_table,
    int64_t recurrent_row_nbytes,
    const void* commit_row,
    int64_t n_layers,
    hipStream_t stream) {
  if (n_layers <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (conv_row_nbytes <= 0 && recurrent_row_nbytes <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  hipLaunchKernelGGL(
      linear_state_pair_commit_i32_kernel,
      dim3(static_cast<unsigned int>(n_layers), 2u),
      dim3(threads),
      0,
      stream,
      static_cast<const uint64_t*>(src_conv_table),
      static_cast<const uint64_t*>(dst_conv_table),
      conv_row_nbytes,
      static_cast<const uint64_t*>(src_recurrent_table),
      static_cast<const uint64_t*>(dst_recurrent_table),
      recurrent_row_nbytes,
      static_cast<const int32_t*>(commit_row),
      n_layers);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_linear_state_pair_commit_chunked_i32(
    const void* src_conv_table,
    const void* dst_conv_table,
    int64_t conv_row_nbytes,
    const void* src_recurrent_table,
    const void* dst_recurrent_table,
    int64_t recurrent_row_nbytes,
    const void* commit_row,
    int64_t n_layers,
    hipStream_t stream) {
  if (n_layers <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (conv_row_nbytes <= 0 && recurrent_row_nbytes <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int64_t kChunkBytes = 65536;
  const int64_t max_row_nbytes = conv_row_nbytes > recurrent_row_nbytes ? conv_row_nbytes : recurrent_row_nbytes;
  const int64_t chunks = (max_row_nbytes + kChunkBytes - 1) / kChunkBytes;
  if (chunks <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  hipLaunchKernelGGL(
      linear_state_pair_commit_chunked_i32_kernel,
      dim3(static_cast<unsigned int>(n_layers), 2u, static_cast<unsigned int>(chunks)),
      dim3(threads),
      0,
      stream,
      static_cast<const uint64_t*>(src_conv_table),
      static_cast<const uint64_t*>(dst_conv_table),
      conv_row_nbytes,
      static_cast<const uint64_t*>(src_recurrent_table),
      static_cast<const uint64_t*>(dst_recurrent_table),
      recurrent_row_nbytes,
      static_cast<const int32_t*>(commit_row),
      n_layers,
      kChunkBytes);
  return static_cast<int>(hipGetLastError());
}
