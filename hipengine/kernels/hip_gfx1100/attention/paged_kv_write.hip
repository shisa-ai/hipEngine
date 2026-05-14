// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 paged KV write kernel bodies are preserved; public wrappers use hipENGINE KVLiveSpans.

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

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

__global__ void qwen35_write_paged_kv_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* block_table,
    int64_t position,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = num_kv_heads * head_dim;
  if (idx >= size) {
    return;
  }
  const int64_t logical_block = position / block_size;
  const int64_t block_offset = position - logical_block * block_size;
  const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
  const int64_t cache_offset = ((physical_block * block_size + block_offset) * num_kv_heads * head_dim) + idx;
  key_cache[cache_offset] = float_to_bf16_bits(key[idx]);
  value_cache[cache_offset] = float_to_bf16_bits(value[idx]);
}

__global__ void qwen35_write_paged_kv_position_tensor_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* block_table,
    const int64_t* position_ptr,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t position = position_ptr[0];
  if (position < 0 || (position / block_size) >= block_table_len) {
    return;
  }
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = num_kv_heads * head_dim;
  if (idx >= size) {
    return;
  }
  const int64_t logical_block = position / block_size;
  const int64_t block_offset = position - logical_block * block_size;
  const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
  const int64_t cache_offset = ((physical_block * block_size + block_offset) * num_kv_heads * head_dim) + idx;
  key_cache[cache_offset] = float_to_bf16_bits(key[idx]);
  value_cache[cache_offset] = float_to_bf16_bits(value[idx]);
}

template <typename value_t>
__global__ void qwen35_write_paged_kv_mixed_value_kernel(
    const float* key,
    const value_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* block_table,
    int64_t position,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = num_kv_heads * head_dim;
  if (idx >= size) {
    return;
  }
  const int64_t logical_block = position / block_size;
  const int64_t block_offset = position - logical_block * block_size;
  const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
  const int64_t cache_offset = ((physical_block * block_size + block_offset) * num_kv_heads * head_dim) + idx;
  key_cache[cache_offset] = float_to_bf16_bits(key[idx]);
  value_cache[cache_offset] = float_to_bf16_bits(scalar_to_float_qwen35(value[idx]));
}

template <typename value_t>
__global__ void qwen35_write_paged_kv_mixed_value_position_tensor_kernel(
    const float* key,
    const value_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* block_table,
    const int64_t* position_ptr,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t position = position_ptr[0];
  if (position < 0 || (position / block_size) >= block_table_len) {
    return;
  }
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = num_kv_heads * head_dim;
  if (idx >= size) {
    return;
  }
  const int64_t logical_block = position / block_size;
  const int64_t block_offset = position - logical_block * block_size;
  const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
  const int64_t cache_offset = ((physical_block * block_size + block_offset) * num_kv_heads * head_dim) + idx;
  key_cache[cache_offset] = float_to_bf16_bits(key[idx]);
  value_cache[cache_offset] = float_to_bf16_bits(scalar_to_float_qwen35(value[idx]));
}

template <typename value_t>
__global__ void qwen35_write_paged_kv_mixed_value_batch_position_tensor_kernel(
    const float* key,
    const value_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* block_table,
    const int64_t* position_ptr,
    int64_t rows,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }
  const int64_t position = position_ptr[row];
  if (position < 0 || (position / block_size) >= block_table_len) {
    return;
  }
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t size = num_kv_heads * head_dim;
  if (idx >= size) {
    return;
  }
  const int64_t logical_block = position / block_size;
  const int64_t block_offset = position - logical_block * block_size;
  const int64_t physical_block = static_cast<int64_t>(block_table[row * block_table_len + logical_block]);
  const int64_t token_offset = ((row * block_table_len + physical_block) * block_size + block_offset) * num_kv_heads * head_dim;
  const int64_t row_value_offset = row * num_kv_heads * head_dim + idx;
  key_cache[token_offset + idx] = float_to_bf16_bits(key[row_value_offset]);
  value_cache[token_offset + idx] = float_to_bf16_bits(scalar_to_float_qwen35(value[row_value_offset]));
}


bool valid_kv_shape(int64_t block_size, int64_t block_table_len, int64_t num_kv_heads, int64_t head_dim) {
  return block_size > 0 && block_table_len > 0 && num_kv_heads > 0 && head_dim > 0;
}

bool valid_kv_batch_shape(int64_t rows, int64_t block_size, int64_t block_table_len, int64_t num_kv_heads, int64_t head_dim) {
  return rows > 0 && valid_kv_shape(block_size, block_table_len, num_kv_heads, head_dim);
}

}  // namespace

extern "C" int hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans(
    const float* key,
    const uint16_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (!valid_kv_shape(block_size, block_table_len, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = num_kv_heads * head_dim;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_write_paged_kv_mixed_value_position_tensor_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_write_paged_kv_mixed_value_bf16_batch_spans(
    const float* key,
    const uint16_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t rows,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (!valid_kv_batch_shape(rows, block_size, block_table_len, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = num_kv_heads * head_dim;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_write_paged_kv_mixed_value_batch_position_tensor_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks), static_cast<unsigned int>(rows)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      rows,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_write_paged_kv_mixed_value_fp16_spans(
    const float* key,
    const half_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (!valid_kv_shape(block_size, block_table_len, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = num_kv_heads * head_dim;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_write_paged_kv_mixed_value_position_tensor_kernel<half_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_write_paged_kv_mixed_value_fp16_batch_spans(
    const float* key,
    const half_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t rows,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (!valid_kv_batch_shape(rows, block_size, block_table_len, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = num_kv_heads * head_dim;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_write_paged_kv_mixed_value_batch_position_tensor_kernel<half_t>),
      dim3(static_cast<unsigned int>(blocks), static_cast<unsigned int>(rows)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      rows,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_write_paged_kv_f32_spans(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (!valid_kv_shape(block_size, block_table_len, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t size = num_kv_heads * head_dim;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_write_paged_kv_position_tensor_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}
