// Moonshine FP16 embedding, residual, partial-RoPE, and fixed-cache glue.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;

__device__ inline half_t moonshine_rotate_value(
    const half_t* source,
    int64_t head_offset,
    int64_t dimension,
    int64_t rotary_dim,
    const half_t* cos,
    const half_t* sin,
    int64_t position) {
  if (dimension >= rotary_dim) return source[head_offset + dimension];
  const int64_t pair = dimension >> 1;
  const int64_t pair_offset = head_offset + (pair << 1);
  const float first = static_cast<float>(source[pair_offset]);
  const float second = static_cast<float>(source[pair_offset + 1]);
  const int64_t table_offset = position * (rotary_dim >> 1) + pair;
  const float cosine = static_cast<float>(cos[table_offset]);
  const float sine = static_cast<float>(sin[table_offset]);
  const float value = (dimension & 1)
      ? second * cosine + first * sine
      : first * cosine - second * sine;
  return static_cast<half_t>(value);
}

__global__ void moonshine_argmax_fp16_kernel(
    const half_t* __restrict__ logits,
    int64_t* __restrict__ output,
    int64_t vocab_size) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];
  float best_value = -3.4028234663852886e38f;
  int64_t best_index = vocab_size;
  for (int64_t index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = static_cast<float>(logits[index]);
    if (value > best_value || (value == best_value && index < best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best_value;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = values[threadIdx.x + stride];
      const int64_t other_index = indices[threadIdx.x + stride];
      if (other_value > values[threadIdx.x] ||
          (other_value == values[threadIdx.x] &&
           other_index < indices[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[0] = indices[0];
}

__global__ void moonshine_advance_position_fp16_kernel(
    int64_t* __restrict__ position,
    int64_t capacity) {
  // Device-owned decode state: advance the fixed position scalar by one after
  // each token step (graph-tail state kernel, C5/§7.3).  The bound keeps the
  // position inside the self-cache capacity so an overlong run cannot
  // overrun the cache; the host decode loop stops at EOS or max positions.
  if (position[0] < capacity) {
    position[0] += 1;
  }
}

__global__ void moonshine_embedding_lookup_fp16_kernel(
    const half_t* __restrict__ embedding,
    const int64_t* __restrict__ token,
    half_t* __restrict__ output,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t token_id = token[0];
  if (index >= hidden_size || token_id < 0 || token_id >= vocab_size) return;
  output[index] = embedding[token_id * hidden_size + index];
}

__global__ void moonshine_residual_fp16_kernel(
    const half_t* __restrict__ hidden,
    const half_t* __restrict__ residual,
    half_t* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  output[index] = static_cast<half_t>(
      static_cast<float>(hidden[index]) + static_cast<float>(residual[index]));
}

__global__ void moonshine_partial_rope_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key,
    const half_t* __restrict__ cos,
    const half_t* __restrict__ sin,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ query_output,
    half_t* __restrict__ key_output,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = heads * head_dim;
  if (index >= elements) return;
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) return;
  const int64_t dimension = index % head_dim;
  const int64_t head_offset = index - dimension;
  query_output[index] = moonshine_rotate_value(
      query, head_offset, dimension, rotary_dim, cos, sin, position);
  key_output[index] = moonshine_rotate_value(
      key, head_offset, dimension, rotary_dim, cos, sin, position);
}

__global__ void moonshine_self_cache_append_fp16_kernel(
    const half_t* __restrict__ key,
    const half_t* __restrict__ value,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ key_cache,
    half_t* __restrict__ value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t capacity) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = heads * head_dim;
  if (index >= elements) return;
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= capacity) return;
  const int64_t head = index / head_dim;
  const int64_t dimension = index - head * head_dim;
  const int64_t cache_offset = (head * capacity + position) * head_dim + dimension;
  key_cache[cache_offset] = key[index];
  value_cache[cache_offset] = value[index];
}

__global__ void moonshine_partial_rope_cache_append_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key,
    const half_t* __restrict__ value,
    const half_t* __restrict__ cos,
    const half_t* __restrict__ sin,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ query_output,
    half_t* __restrict__ key_output,
    half_t* __restrict__ key_cache,
    half_t* __restrict__ value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t capacity,
    int64_t max_positions) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = heads * head_dim;
  if (index >= elements) return;
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= capacity || position >= max_positions) return;
  const int64_t head = index / head_dim;
  const int64_t dimension = index - head * head_dim;
  const int64_t head_offset = head * head_dim;
  const half_t rotated_query = moonshine_rotate_value(
      query, head_offset, dimension, rotary_dim, cos, sin, position);
  const half_t rotated_key = moonshine_rotate_value(
      key, head_offset, dimension, rotary_dim, cos, sin, position);
  query_output[index] = rotated_query;
  key_output[index] = rotated_key;
  const int64_t cache_offset = (head * capacity + position) * head_dim + dimension;
  key_cache[cache_offset] = rotated_key;
  value_cache[cache_offset] = value[index];
}

// -------------------------------------------------------------------------
// C8 phase-1 static-batch variants (grid row == batch row).  Each row runs
// the identical FP32 arithmetic of the single-row kernel, so results are
// bit-exact vs B sequential single-row calls.
// -------------------------------------------------------------------------

__global__ void moonshine_embedding_lookup_batch_fp16_kernel(
    const half_t* __restrict__ embedding,
    const int64_t* __restrict__ token,
    half_t* __restrict__ output,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t batch) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= batch || index >= hidden_size) return;
  const int64_t token_id = token[row];
  if (token_id < 0 || token_id >= vocab_size) return;
  output[row * hidden_size + index] =
      embedding[token_id * hidden_size + index];
}

__global__ void moonshine_partial_rope_cache_append_batch_fp16_kernel(
    const half_t* __restrict__ query,
    const half_t* __restrict__ key,
    const half_t* __restrict__ value,
    const half_t* __restrict__ cos,
    const half_t* __restrict__ sin,
    const int64_t* __restrict__ position_ptr,
    half_t* __restrict__ query_output,
    half_t* __restrict__ key_output,
    half_t* __restrict__ key_cache,
    half_t* __restrict__ value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t capacity,
    int64_t max_positions,
    int64_t batch) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = heads * head_dim;
  if (row >= batch || index >= elements) return;
  const int64_t position = position_ptr[row];
  if (position < 0 || position >= capacity || position >= max_positions) return;
  const int64_t head = index / head_dim;
  const int64_t dimension = index - head * head_dim;
  const int64_t head_offset = head * head_dim;
  const int64_t row_query = row * elements;
  const int64_t row_cache = row * (heads * capacity * head_dim);
  const half_t rotated_query = moonshine_rotate_value(
      query + row_query, head_offset, dimension, rotary_dim, cos, sin, position);
  const half_t rotated_key = moonshine_rotate_value(
      key + row_query, head_offset, dimension, rotary_dim, cos, sin, position);
  query_output[row_query + index] = rotated_query;
  key_output[row_query + index] = rotated_key;
  const int64_t cache_offset =
      row_cache + (head * capacity + position) * head_dim + dimension;
  key_cache[cache_offset] = rotated_key;
  value_cache[cache_offset] = value[row_query + index];
}

__global__ void moonshine_advance_position_batch_fp16_kernel(
    int64_t* __restrict__ position,
    int64_t capacity,
    int64_t batch) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= batch) return;
  if (position[row] < capacity) {
    position[row] += 1;
  }
}

bool valid_reduction_threads(int64_t threads) {
  return threads == 32 || threads == 64 || threads == 128 || threads == 256;
}

bool valid_threads(int64_t threads) {
  return threads > 0 && threads <= 256 && (threads % 32) == 0;
}

bool valid_rope_shape(
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads) {
  return heads > 0 && head_dim > 0 && rotary_dim > 0 &&
      rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
      max_positions > 0 && valid_threads(threads);
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_argmax_fp16(
    const half_t* logits,
    int64_t* output,
    int64_t vocab_size,
    int64_t threads,
    cudaStream_t stream) {
  if (vocab_size <= 0 || !valid_reduction_threads(threads)) {
    return cudaErrorInvalidValue;
  }
  moonshine_argmax_fp16_kernel<<<dim3(1), dim3(threads), 0, stream>>>(
      logits, output, vocab_size);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_advance_position_fp16(
    int64_t* position,
    int64_t capacity,
    cudaStream_t stream) {
  if (position == nullptr || capacity <= 0) {
    return cudaErrorInvalidValue;
  }
  moonshine_advance_position_fp16_kernel<<<dim3(1), dim3(1), 0, stream>>>(
      position, capacity);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_embedding_lookup_fp16(
    const half_t* embedding,
    const int64_t* token,
    half_t* output,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    cudaStream_t stream) {
  if (hidden_size <= 0 || vocab_size <= 0 || !valid_threads(threads)) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = (hidden_size + threads - 1) / threads;
  moonshine_embedding_lookup_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      embedding, token, output, hidden_size, vocab_size);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_residual_fp16(
    const half_t* hidden,
    const half_t* residual,
    half_t* output,
    int64_t elements,
    int64_t threads,
    cudaStream_t stream) {
  if (elements <= 0 || !valid_threads(threads)) return cudaErrorInvalidValue;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_residual_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      hidden, residual, output, elements);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_partial_rope_fp16(
    const half_t* query,
    const half_t* key,
    const half_t* cos,
    const half_t* sin,
    const int64_t* position,
    half_t* query_output,
    half_t* key_output,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    cudaStream_t stream) {
  if (!valid_rope_shape(heads, head_dim, rotary_dim, max_positions, threads)) {
    return cudaErrorInvalidValue;
  }
  const int64_t elements = heads * head_dim;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_partial_rope_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      query, key, cos, sin, position, query_output, key_output, heads,
      head_dim, rotary_dim, max_positions);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_self_cache_append_fp16(
    const half_t* key,
    const half_t* value,
    const int64_t* position,
    half_t* key_cache,
    half_t* value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t capacity,
    int64_t threads,
    cudaStream_t stream) {
  if (heads <= 0 || head_dim <= 0 || capacity <= 0 || !valid_threads(threads)) {
    return cudaErrorInvalidValue;
  }
  const int64_t elements = heads * head_dim;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_self_cache_append_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      key, value, position, key_cache, value_cache, heads, head_dim, capacity);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_fp16(
    const half_t* query,
    const half_t* key,
    const half_t* value,
    const half_t* cos,
    const half_t* sin,
    const int64_t* position,
    half_t* query_output,
    half_t* key_output,
    half_t* key_cache,
    half_t* value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t capacity,
    int64_t max_positions,
    int64_t threads,
    cudaStream_t stream) {
  if (!valid_rope_shape(heads, head_dim, rotary_dim, max_positions, threads) ||
      capacity <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t elements = heads * head_dim;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_partial_rope_cache_append_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      query, key, value, cos, sin, position, query_output, key_output,
      key_cache, value_cache, heads, head_dim, rotary_dim, capacity,
      max_positions);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_embedding_lookup_batch_fp16(
    const half_t* embedding,
    const int64_t* token,
    half_t* output,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t batch,
    int64_t threads,
    cudaStream_t stream) {
  if (hidden_size <= 0 || vocab_size <= 0 || batch <= 0 ||
      !valid_threads(threads)) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = (hidden_size + threads - 1) / threads;
  moonshine_embedding_lookup_batch_fp16_kernel<<<
      dim3(blocks, (unsigned)batch), dim3(threads), 0, stream>>>(
      embedding, token, output, hidden_size, vocab_size, batch);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_partial_rope_cache_append_batch_fp16(
    const half_t* query,
    const half_t* key,
    const half_t* value,
    const half_t* cos,
    const half_t* sin,
    const int64_t* position,
    half_t* query_output,
    half_t* key_output,
    half_t* key_cache,
    half_t* value_cache,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t capacity,
    int64_t max_positions,
    int64_t batch,
    int64_t threads,
    cudaStream_t stream) {
  if (!valid_rope_shape(heads, head_dim, rotary_dim, max_positions, threads) ||
      capacity <= 0 || batch <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t elements = heads * head_dim;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_partial_rope_cache_append_batch_fp16_kernel<<<
      dim3(blocks, (unsigned)batch), dim3(threads), 0, stream>>>(
      query, key, value, cos, sin, position, query_output, key_output,
      key_cache, value_cache, heads, head_dim, rotary_dim, capacity,
      max_positions, batch);
  return cudaGetLastError();
}

extern "C" int hipengine_cuda_sm120a_moonshine_advance_position_batch_fp16(
    int64_t* position,
    int64_t capacity,
    int64_t batch,
    cudaStream_t stream) {
  if (position == nullptr || capacity <= 0 || batch <= 0) {
    return cudaErrorInvalidValue;
  }
  moonshine_advance_position_batch_fp16_kernel<<<
      dim3(1, (unsigned)batch), dim3(1), 0, stream>>>(
      position, capacity, batch);
  return cudaGetLastError();
}
