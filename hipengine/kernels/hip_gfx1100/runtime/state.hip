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
