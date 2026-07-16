// Small graph-friendly runtime state helpers for hipEngine Qwen3.5/PARO.
// These kernels keep decode token/position state device-resident so a future
// HIP graph replay path does not depend on host-side memcpy between tokens.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void embedding_lookup_bf16_i64_kernel(
    const uint16_t* embedding,
    const int64_t* token_id,
    uint16_t* out,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t token = token_id[0];
  if (token < 0 || token >= vocab_size) {
    return;
  }
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= hidden_size) {
    return;
  }
  out[idx] = embedding[token * hidden_size + idx];
}

__global__ void embedding_lookup_batch_bf16_i64_kernel(
    const uint16_t* embedding,
    const int64_t* token_ids,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = tokens * hidden_size;
  if (idx >= size) {
    return;
  }
  const int64_t row = idx / hidden_size;
  const int64_t col = idx - row * hidden_size;
  const int64_t token = token_ids[row];
  if (token < 0 || token >= vocab_size) {
    return;
  }
  out[idx] = embedding[token * hidden_size + col];
}

__global__ void embedding_lookup_batch_mapped_bf16_i64_kernel(
    const uint16_t* embedding,
    const int64_t* token_ids,
    const int32_t* row_map,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t token_slots) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = rows * hidden_size;
  if (idx >= size) {
    return;
  }
  const int64_t row = idx / hidden_size;
  const int64_t col = idx - row * hidden_size;
  const int64_t source_row = row_map == nullptr ? row : static_cast<int64_t>(row_map[row]);
  if (source_row < 0 || source_row >= token_slots) {
    return;
  }
  const int64_t token = token_ids[source_row];
  if (token < 0 || token >= vocab_size) {
    return;
  }
  out[idx] = embedding[token * hidden_size + col];
}

__global__ void set_i64_scalar_kernel(int64_t* out, int64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
  }
}

__global__ void prefill_flight_recorder_mark_i64_kernel(int64_t* out, int64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
    // The destination is HIP-registered mapped host memory.  A system fence
    // makes the completion cursor visible to a CPU-side/postmortem reader only
    // after all preceding work on this stream has retired.
    __threadfence_system();
  }
}

__global__ void set_decode_position_i64_kernel(
    int64_t* position,
    int64_t* context,
    int64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    position[0] = value;
    context[0] = value + 1;
  }
}

__global__ void advance_decode_position_i64_kernel(int64_t* position, int64_t* context) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int64_t next = position[0] + 1;
    position[0] = next;
    context[0] = next + 1;
  }
}

__global__ void record_i64_scalar_indexed_kernel(
    const int64_t* value,
    int64_t* out,
    int64_t* index,
    int64_t capacity) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int64_t row = index[0];
    if (row >= 0 && row < capacity) {
      out[row] = value[0];
    }
    index[0] = row + 1;
  }
}

__global__ void record_f32_row_indexed_kernel(
    const float* value,
    float* out,
    const int64_t* index,
    int64_t cols,
    int64_t capacity) {
  const int64_t row = index[0];
  if (row < 0 || row >= capacity) {
    return;
  }
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col < cols) {
    out[row * cols + col] = value[col];
  }
}

__global__ void set_i64_vector_kernel(int64_t* out, const int64_t* values, int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) {
    out[row] = values[row];
  }
}

__global__ void set_decode_positions_i64_kernel(
    int64_t* positions,
    int64_t* contexts,
    const int64_t* values,
    const uint8_t* active_mask,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || (active_mask != nullptr && active_mask[row] == 0)) {
    return;
  }
  const int64_t value = values[row];
  positions[row] = value;
  contexts[row] = value + 1;
}

__global__ void prepare_prefill_chunk_metadata_kernel(
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* positions,
    int64_t* contexts,
    int64_t start,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row == 0) {
    cu_q[0] = 0;
    cu_q[1] = static_cast<int32_t>(rows);
    cu_k[0] = 0;
    cu_k[1] = static_cast<int32_t>(start + rows);
    atomic[0] = 0;
    gdn_cu_seqlens[0] = 0;
    gdn_cu_seqlens[1] = static_cast<int32_t>(rows);
  }
  if (row < rows) {
    const int64_t position = start + row;
    positions[row] = position;
    contexts[row] = position + 1;
  }
}

__device__ __forceinline__ int64_t packed_decode_position(
    int64_t row,
    int64_t position0,
    int64_t position1,
    int64_t position2,
    int64_t position3) {
  if (row == 0) {
    return position0;
  }
  if (row == 1) {
    return position1;
  }
  if (row == 2) {
    return position2;
  }
  return position3;
}

__global__ void prepare_packed_decode_metadata_kernel(
    int32_t* block_table,
    int64_t* positions,
    int64_t* contexts,
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* state_indices,
    int64_t position0,
    int64_t position1,
    int64_t position2,
    int64_t position3,
    int64_t rows,
    int64_t blocks_per_slot) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index == 0) {
    int64_t max_position = position0;
    if (rows > 1 && position1 > max_position) {
      max_position = position1;
    }
    if (rows > 2 && position2 > max_position) {
      max_position = position2;
    }
    if (rows > 3 && position3 > max_position) {
      max_position = position3;
    }
    cu_q[0] = 0;
    cu_q[1] = static_cast<int32_t>(rows);
    cu_k[0] = 0;
    cu_k[1] = static_cast<int32_t>(max_position + 1);
    atomic[0] = 0;
  }
  if (index <= rows) {
    gdn_cu_seqlens[index] = static_cast<int32_t>(index);
  }
  if (index < rows) {
    const int64_t position = packed_decode_position(
        index, position0, position1, position2, position3);
    positions[index] = position;
    contexts[index] = position + 1;
    state_indices[index] = index;
  }
  const int64_t block_table_size = rows * blocks_per_slot;
  if (index < block_table_size) {
    block_table[index] = static_cast<int32_t>(index);
  }
}

__global__ void prepare_packed_decode_metadata_from_positions_kernel(
    int32_t* block_table,
    const int64_t* positions,
    int64_t* contexts,
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* state_indices,
    int64_t rows,
    int64_t blocks_per_slot) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index == 0) {
    int64_t max_position = positions[0];
    for (int64_t row = 1; row < rows; ++row) {
      if (positions[row] > max_position) {
        max_position = positions[row];
      }
    }
    cu_q[0] = 0;
    cu_q[1] = static_cast<int32_t>(rows);
    cu_k[0] = 0;
    cu_k[1] = static_cast<int32_t>(max_position + 1);
    atomic[0] = 0;
  }
  if (index <= rows) {
    gdn_cu_seqlens[index] = static_cast<int32_t>(index);
  }
  if (index < rows) {
    contexts[index] = positions[index] + 1;
    state_indices[index] = index;
  }
  const int64_t block_table_size = rows * blocks_per_slot;
  if (index < block_table_size) {
    block_table[index] = static_cast<int32_t>(index);
  }
}

__global__ void record_u16_rows_indexed_kernel(
    const uint16_t* value,
    uint16_t* out,
    const int64_t* index,
    int64_t elements,
    int64_t step_stride,
    int64_t capacity) {
  const int64_t step = index[0];
  if (step < 0 || step >= capacity) {
    return;
  }
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element < elements) {
    out[step * step_stride + element] = value[element];
  }
}

__global__ void commit_packed_decode_graph_step_kernel(
    const int32_t* token_ids_i32,
    int64_t* token_ids_i64,
    int64_t* positions,
    int64_t* contexts,
    int32_t* recorded_token_ids_i32,
    int64_t* record_index,
    int64_t record_capacity,
    int64_t rows) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int64_t step = record_index == nullptr ? 0 : record_index[0];
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t token = token_ids_i32[row];
    token_ids_i64[row] = static_cast<int64_t>(token);
    if (recorded_token_ids_i32 != nullptr && step >= 0 && step < record_capacity) {
      recorded_token_ids_i32[step * rows + row] = token;
    }
    const int64_t next = positions[row] + 1;
    positions[row] = next;
    contexts[row] = next + 1;
  }
  if (record_index != nullptr) {
    record_index[0] = step + 1;
  }
}

__global__ void advance_decode_positions_i64_kernel(
    int64_t* positions,
    int64_t* contexts,
    const uint8_t* active_mask,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || (active_mask != nullptr && active_mask[row] == 0)) {
    return;
  }
  const int64_t next = positions[row] + 1;
  positions[row] = next;
  contexts[row] = next + 1;
}

__global__ void unpack_verify_chain_dynamic_metadata_i64_kernel(
    const int64_t* packed,
    int64_t* token_ids_i64,
    int32_t* token_ids_i32,
    int64_t* positions_i64,
    int32_t* positions_i32,
    int64_t* contexts_i64,
    int64_t rows) {
  constexpr int64_t kFields = 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * kFields;
  token_ids_i64[row] = packed[base + 0];
  token_ids_i32[row] = static_cast<int32_t>(packed[base + 1]);
  positions_i64[row] = packed[base + 2];
  positions_i32[row] = static_cast<int32_t>(packed[base + 3]);
  contexts_i64[row] = packed[base + 4];
}

bool valid_embedding_shape(int64_t hidden_size, int64_t vocab_size) {
  return hidden_size > 0 && vocab_size > 0;
}

bool valid_embedding_batch_shape(int64_t tokens, int64_t hidden_size, int64_t vocab_size) {
  return tokens > 0 && hidden_size > 0 && vocab_size > 0;
}

bool valid_embedding_mapped_shape(int64_t rows, int64_t hidden_size, int64_t vocab_size, int64_t token_slots) {
  return rows > 0 && hidden_size > 0 && vocab_size > 0 && token_slots > 0;
}

bool valid_vector_shape(int64_t rows) {
  return rows > 0;
}

}  // namespace

extern "C" int hipengine_embedding_lookup_bf16_i64(
    const uint16_t* embedding,
    const int64_t* token_id,
    uint16_t* out,
    int64_t hidden_size,
    int64_t vocab_size,
    hipStream_t stream) {
  if (!valid_embedding_shape(hidden_size, vocab_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((hidden_size + threads - 1) / threads);
  hipLaunchKernelGGL(
      embedding_lookup_bf16_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      embedding,
      token_id,
      out,
      hidden_size,
      vocab_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_embedding_lookup_batch_bf16_i64(
    const uint16_t* embedding,
    const int64_t* token_ids,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t vocab_size,
    hipStream_t stream) {
  if (!valid_embedding_batch_shape(tokens, hidden_size, vocab_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = tokens * hidden_size;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      embedding_lookup_batch_bf16_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      embedding,
      token_ids,
      out,
      tokens,
      hidden_size,
      vocab_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_embedding_lookup_batch_mapped_bf16_i64(
    const uint16_t* embedding,
    const int64_t* token_ids,
    const int32_t* row_map,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t token_slots,
    hipStream_t stream) {
  if (!valid_embedding_mapped_shape(rows, hidden_size, vocab_size, token_slots)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = rows * hidden_size;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      embedding_lookup_batch_mapped_bf16_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      embedding,
      token_ids,
      row_map,
      out,
      rows,
      hidden_size,
      vocab_size,
      token_slots);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_embedding_lookup_fp16_i64(
    const uint16_t* embedding,
    const int64_t* token_id,
    uint16_t* out,
    int64_t hidden_size,
    int64_t vocab_size,
    hipStream_t stream) {
  return hipengine_embedding_lookup_bf16_i64(embedding, token_id, out, hidden_size, vocab_size, stream);
}

extern "C" int hipengine_embedding_lookup_batch_fp16_i64(
    const uint16_t* embedding,
    const int64_t* token_ids,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t vocab_size,
    hipStream_t stream) {
  return hipengine_embedding_lookup_batch_bf16_i64(embedding, token_ids, out, tokens, hidden_size, vocab_size, stream);
}

extern "C" int hipengine_embedding_lookup_batch_mapped_fp16_i64(
    const uint16_t* embedding,
    const int64_t* token_ids,
    const int32_t* row_map,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t token_slots,
    hipStream_t stream) {
  return hipengine_embedding_lookup_batch_mapped_bf16_i64(
      embedding,
      token_ids,
      row_map,
      out,
      rows,
      hidden_size,
      vocab_size,
      token_slots,
      stream);
}

extern "C" int hipengine_set_i64_scalar(int64_t* out, int64_t value, hipStream_t stream) {
  hipLaunchKernelGGL(
      set_i64_scalar_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      out,
      value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_prefill_flight_recorder_mark_i64(
    int64_t* out,
    int64_t value,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      prefill_flight_recorder_mark_i64_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      out,
      value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_set_i64_vector(
    int64_t* out,
    const int64_t* values,
    int64_t rows,
    hipStream_t stream) {
  if (!valid_vector_shape(rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      set_i64_vector_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      out,
      values,
      rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_set_decode_position_i64(
    int64_t* position,
    int64_t* context,
    int64_t value,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      set_decode_position_i64_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      position,
      context,
      value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_advance_decode_position_i64(
    int64_t* position,
    int64_t* context,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      advance_decode_position_i64_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      position,
      context);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_record_i64_scalar_indexed(
    const int64_t* value,
    int64_t* out,
    int64_t* index,
    int64_t capacity,
    hipStream_t stream) {
  if (capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      record_i64_scalar_indexed_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      value,
      out,
      index,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_record_f32_row_indexed(
    const float* value,
    float* out,
    const int64_t* index,
    int64_t cols,
    int64_t capacity,
    hipStream_t stream) {
  if (cols <= 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((cols + threads - 1) / threads);
  hipLaunchKernelGGL(
      record_f32_row_indexed_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      value,
      out,
      index,
      cols,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_set_decode_positions_i64(
    int64_t* positions,
    int64_t* contexts,
    const int64_t* values,
    const uint8_t* active_mask,
    int64_t rows,
    hipStream_t stream) {
  if (!valid_vector_shape(rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      set_decode_positions_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      positions,
      contexts,
      values,
      active_mask,
      rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_prepare_prefill_chunk_metadata(
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* positions,
    int64_t* contexts,
    int64_t start,
    int64_t rows,
    hipStream_t stream) {
  if (start < 0 || !valid_vector_shape(rows) || start + rows > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      prepare_prefill_chunk_metadata_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      cu_q,
      cu_k,
      atomic,
      gdn_cu_seqlens,
      positions,
      contexts,
      start,
      rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_prepare_packed_decode_metadata(
    int32_t* block_table,
    int64_t* positions,
    int64_t* contexts,
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* state_indices,
    int64_t position0,
    int64_t position1,
    int64_t position2,
    int64_t position3,
    int64_t rows,
    int64_t blocks_per_slot,
    hipStream_t stream) {
  if (rows <= 0 || rows > 4 || blocks_per_slot <= 0 ||
      position0 < 0 || (rows > 1 && position1 < 0) ||
      (rows > 2 && position2 < 0) || (rows > 3 && position3 < 0)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int64_t max_position = position0;
  if (rows > 1 && position1 > max_position) {
    max_position = position1;
  }
  if (rows > 2 && position2 > max_position) {
    max_position = position2;
  }
  if (rows > 3 && position3 > max_position) {
    max_position = position3;
  }
  if (max_position >= INT32_MAX || rows * blocks_per_slot > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 64;
  const int64_t elements = rows * blocks_per_slot > rows + 1
      ? rows * blocks_per_slot
      : rows + 1;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      prepare_packed_decode_metadata_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      block_table,
      positions,
      contexts,
      cu_q,
      cu_k,
      atomic,
      gdn_cu_seqlens,
      state_indices,
      position0,
      position1,
      position2,
      position3,
      rows,
      blocks_per_slot);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_prepare_packed_decode_metadata_from_positions(
    int32_t* block_table,
    const int64_t* positions,
    int64_t* contexts,
    int32_t* cu_q,
    int32_t* cu_k,
    int32_t* atomic,
    int32_t* gdn_cu_seqlens,
    int64_t* state_indices,
    int64_t rows,
    int64_t blocks_per_slot,
    hipStream_t stream) {
  if (rows <= 0 || rows > 4 || blocks_per_slot <= 0 ||
      rows * blocks_per_slot > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 64;
  const int64_t elements = rows * blocks_per_slot > rows + 1
      ? rows * blocks_per_slot
      : rows + 1;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      prepare_packed_decode_metadata_from_positions_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      block_table,
      positions,
      contexts,
      cu_q,
      cu_k,
      atomic,
      gdn_cu_seqlens,
      state_indices,
      rows,
      blocks_per_slot);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_record_u16_rows_indexed(
    const uint16_t* value,
    uint16_t* out,
    const int64_t* index,
    int64_t elements,
    int64_t step_stride,
    int64_t capacity,
    hipStream_t stream) {
  if (elements <= 0 || step_stride < elements || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      record_u16_rows_indexed_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      value,
      out,
      index,
      elements,
      step_stride,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_commit_packed_decode_graph_step(
    const int32_t* token_ids_i32,
    int64_t* token_ids_i64,
    int64_t* positions,
    int64_t* contexts,
    int32_t* recorded_token_ids_i32,
    int64_t* record_index,
    int64_t record_capacity,
    int64_t rows,
    hipStream_t stream) {
  if (rows <= 0 || rows > 4 ||
      ((recorded_token_ids_i32 == nullptr) != (record_index == nullptr)) ||
      (recorded_token_ids_i32 != nullptr && record_capacity <= 0)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      commit_packed_decode_graph_step_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      token_ids_i32,
      token_ids_i64,
      positions,
      contexts,
      recorded_token_ids_i32,
      record_index,
      record_capacity,
      rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_advance_decode_positions_i64(
    int64_t* positions,
    int64_t* contexts,
    const uint8_t* active_mask,
    int64_t rows,
    hipStream_t stream) {
  if (!valid_vector_shape(rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      advance_decode_positions_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      positions,
      contexts,
      active_mask,
      rows);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_unpack_verify_chain_dynamic_metadata_i64(
    const int64_t* packed,
    int64_t* token_ids_i64,
    int32_t* token_ids_i32,
    int64_t* positions_i64,
    int32_t* positions_i32,
    int64_t* contexts_i64,
    int64_t rows,
    hipStream_t stream) {
  if (!valid_vector_shape(rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      unpack_verify_chain_dynamic_metadata_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      packed,
      token_ids_i64,
      token_ids_i32,
      positions_i64,
      positions_i32,
      contexts_i64,
      rows);
  return static_cast<int>(hipGetLastError());
}
