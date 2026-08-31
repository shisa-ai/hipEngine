// hipEngine-native Qwen3.8-Flash-Next QSA strict control primitives.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>

namespace {

__device__ inline int64_t qwen4_exp_mrope_position(
    const int64_t* positions,
    int64_t rows,
    int64_t row,
    int64_t pair,
    bool mrope) {
  if (!mrope) return positions[row];
  int axis = 0;
  if ((pair % 3) == 1 && pair < 33) axis = 1;
  if ((pair % 3) == 2 && pair < 30) axis = 2;
  return positions[static_cast<int64_t>(axis) * rows + row];
}

__global__ void qsa_scatter_index_key_device_position_f32_kernel(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const int32_t* __restrict__ block_table,
    const int64_t* __restrict__ position,
    int64_t block_size,
    int64_t block_table_len,
    int64_t index_dim) {
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= index_dim) return;
  const int64_t logical = position[0];
  if (logical < 0 || logical / block_size >= block_table_len) return;
  const int64_t logical_block = logical / block_size;
  const int64_t block_offset = logical - logical_block * block_size;
  const int64_t physical =
      static_cast<int64_t>(block_table[logical_block]) * block_size
      + block_offset;
  destination[physical * index_dim + column] = source[column];
}

__global__ void qsa_scatter_index_keys_f32_kernel(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const int32_t* __restrict__ block_table,
    int64_t start_position,
    int64_t rows,
    int64_t block_size,
    int64_t index_dim) {
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= rows * index_dim) return;
  const int64_t row = element / index_dim;
  const int64_t column = element - row * index_dim;
  const int64_t logical = start_position + row;
  const int64_t logical_block = logical / block_size;
  const int64_t block_offset = logical - logical_block * block_size;
  const int64_t physical =
      static_cast<int64_t>(block_table[logical_block]) * block_size
      + block_offset;
  destination[physical * index_dim + column] = source[element];
}

template <bool MROPE>
__global__ void qsa_split_norm_rope_f32_kernel(
    const float* __restrict__ q_projected,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int64_t* __restrict__ position,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float* __restrict__ gate_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* values = shared;
  float* partial = values + head_dim;
  const int64_t head = blockIdx.x;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t source_stride = is_query ? 2 * head_dim : head_dim;
  const float* source = is_query
      ? q_projected + local_head * source_stride
      : key + local_head * source_stride;
  const float* weight = is_query ? q_weight : k_weight;
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    const float value = source[dimension];
    values[dimension] = value;
    sumsq += value * value;
    if (is_query) {
      gate_out[local_head * head_dim + dimension] = source[head_dim + dimension];
    }
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  float* output = is_query
      ? query_out + local_head * head_dim
      : key_out + local_head * head_dim;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float result = values[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t half = rotary_dim / 2;
      const int64_t pair = dimension % half;
      const float x = values[pair] * inv_rms * weight[pair];
      const float y = values[pair + half] * inv_rms * weight[pair + half];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(
          qwen4_exp_mrope_position(position, 1, 0, pair, MROPE))
          * powf(theta, exponent);
      const float c = cosf(angle);
      const float s = sinf(angle);
      result = dimension < half ? x * c - y * s : x * s + y * c;
    }
    output[dimension] = result;
  }
}

template <bool MROPE>
__global__ void qsa_norm_rope_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int64_t* __restrict__ position,
    float* __restrict__ output,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* values = shared;
  float* partial = values + head_dim;
  const int64_t head = blockIdx.x;
  const int64_t offset = head * head_dim;
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    const float value = input[offset + dimension];
    values[dimension] = value;
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float result = values[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t half = rotary_dim / 2;
      const int64_t pair = dimension % half;
      const float x = values[pair] * inv_rms * weight[pair];
      const float y = values[pair + half] * inv_rms * weight[pair + half];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(
          qwen4_exp_mrope_position(position, 1, 0, pair, MROPE))
          * powf(theta, exponent);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      result = dimension < half ? x * cosine - y * sine : x * sine + y * cosine;
    }
    output[offset + dimension] = result;
  }
}

template <bool MROPE>
__global__ void qsa_split_norm_rope_rows_f32_kernel(
    const float* __restrict__ q_projected,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int64_t* __restrict__ positions,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float* __restrict__ gate_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* values = shared;
  float* partial = values + head_dim;
  const int64_t row = blockIdx.y;
  const int64_t head = blockIdx.x;
  if (row >= rows) return;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t source_stride = is_query ? 2 * head_dim : head_dim;
  const float* source = is_query
      ? q_projected + (row * num_q_heads + local_head) * source_stride
      : key + (row * num_kv_heads + local_head) * source_stride;
  const float* weight = is_query ? q_weight : k_weight;
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    const float value = source[dimension];
    values[dimension] = value;
    sumsq += value * value;
    if (is_query) {
      gate_out[(row * num_q_heads + local_head) * head_dim + dimension]
          = source[head_dim + dimension];
    }
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  float* output = is_query
      ? query_out + (row * num_q_heads + local_head) * head_dim
      : key_out + (row * num_kv_heads + local_head) * head_dim;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float result = values[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t half = rotary_dim / 2;
      const int64_t pair = dimension % half;
      const float x = values[pair] * inv_rms * weight[pair];
      const float y = values[pair + half] * inv_rms * weight[pair + half];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(
          qwen4_exp_mrope_position(positions, rows, row, pair, MROPE))
          * powf(theta, exponent);
      const float c = cosf(angle);
      const float s = sinf(angle);
      result = dimension < half ? x * c - y * s : x * s + y * c;
    }
    output[dimension] = result;
  }
}

template <bool MROPE>
__global__ void qsa_norm_rope_rows_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int64_t* __restrict__ positions,
    float* __restrict__ output,
    int64_t rows,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* values = shared;
  float* partial = values + head_dim;
  const int64_t row = blockIdx.y;
  const int64_t head = blockIdx.x;
  if (row >= rows || head >= heads) return;
  const int64_t offset = (row * heads + head) * head_dim;
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    const float value = input[offset + dimension];
    values[dimension] = value;
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float result = values[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t half = rotary_dim / 2;
      const int64_t pair = dimension % half;
      const float x = values[pair] * inv_rms * weight[pair];
      const float y = values[pair + half] * inv_rms * weight[pair + half];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(
          qwen4_exp_mrope_position(positions, rows, row, pair, MROPE))
          * powf(theta, exponent);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      result = dimension < half ? x * cosine - y * sine : x * sine + y * cosine;
    }
    output[offset + dimension] = result;
  }
}

__global__ void qsa_gate_context_f32_kernel(
    const float* __restrict__ context,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = context[index] / (1.0f + expf(-gate[index]));
  }
}

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__global__ void qsa_sparse_attention_paged_bf16_f32_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ selected_positions,
    const int32_t* __restrict__ block_table,
    float* __restrict__ output,
    int64_t selected_count,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale) {
  extern __shared__ float shared[];
  float* partial = shared;
  float* stats = partial + blockDim.x;
  const int64_t q_head = blockIdx.x;
  const int64_t kv_head = q_head / (query_heads / kv_heads);
  const int64_t dimension = threadIdx.x;
  float accumulator = 0.0f;
  if (dimension == 0) {
    stats[0] = -INFINITY;
    stats[1] = 0.0f;
  }
  __syncthreads();
  for (int64_t selected_index = 0; selected_index < selected_count; ++selected_index) {
    const int64_t position = selected_positions[selected_index];
    const int64_t logical_block = position / block_size;
    if (position < 0 || logical_block >= block_table_len) {
      continue;
    }
    const int64_t physical_block = block_table[logical_block];
    const int64_t token = physical_block * block_size + position % block_size;
    const int64_t kv_offset = (token * kv_heads + kv_head) * head_dim;
    float product = 0.0f;
    if (dimension < head_dim) {
      product = query[q_head * head_dim + dimension]
          * bf16_bits_to_float(key_cache[kv_offset + dimension]);
    }
    partial[dimension] = product;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (dimension < stride) partial[dimension] += partial[dimension + stride];
      __syncthreads();
    }
    if (dimension == 0) {
      const float score = partial[0] * scale;
      const float old_max = stats[0];
      const float new_max = fmaxf(old_max, score);
      const float alpha = isinf(old_max) ? 0.0f : expf(old_max - new_max);
      const float beta = expf(score - new_max);
      stats[0] = new_max;
      stats[1] = alpha * stats[1] + beta;
      stats[2] = alpha;
      stats[3] = beta;
    }
    __syncthreads();
    if (dimension < head_dim) {
      const float value = bf16_bits_to_float(value_cache[kv_offset + dimension]);
      accumulator = accumulator * stats[2] + value * stats[3];
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    output[q_head * head_dim + dimension] = accumulator / stats[1];
  }
}

__global__ void qsa_sparse_attention_paged_bf16_rows_f32_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ selected_positions,
    const int32_t* __restrict__ selected_counts,
    const int32_t* __restrict__ block_tables,
    float* __restrict__ output,
    int64_t rows,
    int64_t selected_stride,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale) {
  extern __shared__ float shared[];
  float* partial = shared;
  float* stats = partial + blockDim.x;
  const int64_t row = blockIdx.y;
  const int64_t q_head = blockIdx.x;
  if (row >= rows) return;
  const int64_t selected_count = selected_counts[row];
  if (selected_count <= 0 || selected_count > selected_stride) return;
  const int64_t kv_head = q_head / (query_heads / kv_heads);
  const int64_t dimension = threadIdx.x;
  float accumulator = 0.0f;
  if (dimension == 0) {
    stats[0] = -INFINITY;
    stats[1] = 0.0f;
  }
  __syncthreads();
  const float* query_row = query + (row * query_heads + q_head) * head_dim;
  const int64_t* selected_row = selected_positions + row * selected_stride;
  const int32_t* block_table = block_tables + row * block_table_len;
  for (int64_t selected_index = 0; selected_index < selected_count; ++selected_index) {
    const int64_t position = selected_row[selected_index];
    const int64_t logical_block = position / block_size;
    if (position < 0 || logical_block >= block_table_len) continue;
    const int64_t physical_block = block_table[logical_block];
    const int64_t token = physical_block * block_size + position % block_size;
    const int64_t kv_offset = (token * kv_heads + kv_head) * head_dim;
    float product = 0.0f;
    if (dimension < head_dim) {
      product = query_row[dimension]
          * bf16_bits_to_float(key_cache[kv_offset + dimension]);
    }
    partial[dimension] = product;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (dimension < stride) partial[dimension] += partial[dimension + stride];
      __syncthreads();
    }
    if (dimension == 0) {
      const float score = partial[0] * scale;
      const float old_max = stats[0];
      const float next_max = fmaxf(old_max, score);
      const float old_scale = isinf(old_max) ? 0.0f : expf(old_max - next_max);
      const float score_scale = expf(score - next_max);
      stats[0] = next_max;
      stats[1] = stats[1] * old_scale + score_scale;
      partial[0] = old_scale;
      partial[1] = score_scale;
    }
    __syncthreads();
    if (dimension < head_dim) {
      accumulator = accumulator * partial[0]
          + partial[1] * bf16_bits_to_float(value_cache[kv_offset + dimension]);
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    const float denominator = fmaxf(stats[1], 1.0e-20f);
    output[(row * query_heads + q_head) * head_dim + dimension]
        = accumulator / denominator;
  }
}

__global__ __launch_bounds__(32) void qsa_sparse_attention_paged_bf16_wave32_f32_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ selected_positions,
    const int32_t* __restrict__ block_table,
    float* __restrict__ output,
    int64_t selected_count,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    float scale) {
  constexpr int64_t head_dim = 128;
  const int64_t q_head = blockIdx.x;
  const int64_t kv_head = q_head / (query_heads / kv_heads);
  const int lane = threadIdx.x;
  __shared__ float stats[4];
  float accumulator_0 = 0.0f;
  float accumulator_1 = 0.0f;
  float accumulator_2 = 0.0f;
  float accumulator_3 = 0.0f;
  if (lane == 0) {
    stats[0] = -INFINITY;
    stats[1] = 0.0f;
  }
  __syncwarp();
  for (int64_t selected_index = 0; selected_index < selected_count; ++selected_index) {
    const int64_t position = selected_positions[selected_index];
    const int64_t logical_block = position / block_size;
    if (position < 0 || logical_block >= block_table_len) continue;
    const int64_t physical_block = block_table[logical_block];
    const int64_t token = physical_block * block_size + position % block_size;
    const int64_t kv_offset = (token * kv_heads + kv_head) * head_dim;
    const float product_0 = query[q_head * head_dim + lane]
        * bf16_bits_to_float(key_cache[kv_offset + lane]);
    const float product_1 = query[q_head * head_dim + lane + 64]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 64]);
    const float product_2 = query[q_head * head_dim + lane + 32]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 32]);
    const float product_3 = query[q_head * head_dim + lane + 96]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 96]);
    volatile float pair_0 = product_0 + product_1;
    volatile float pair_1 = product_2 + product_3;
    float dot = pair_0 + pair_1;
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down(dot, offset, 32);
    }
    if (lane == 0) {
      const float score = dot * scale;
      const float old_max = stats[0];
      const float next_maximum = fmaxf(old_max, score);
      const float old_scale = isinf(old_max) ? 0.0f : expf(old_max - next_maximum);
      const float score_scale = expf(score - next_maximum);
      stats[0] = next_maximum;
      stats[1] = stats[1] * old_scale + score_scale;
      stats[2] = old_scale;
      stats[3] = score_scale;
    }
    __syncwarp();
    accumulator_0 = accumulator_0 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane]) * stats[3];
    accumulator_1 = accumulator_1 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 64]) * stats[3];
    accumulator_2 = accumulator_2 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 32]) * stats[3];
    accumulator_3 = accumulator_3 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 96]) * stats[3];
    __syncwarp();
  }
  output[q_head * head_dim + lane] = accumulator_0 / stats[1];
  output[q_head * head_dim + lane + 64] = accumulator_1 / stats[1];
  output[q_head * head_dim + lane + 32] = accumulator_2 / stats[1];
  output[q_head * head_dim + lane + 96] = accumulator_3 / stats[1];
}

__global__ __launch_bounds__(32) void qsa_sparse_attention_paged_bf16_rows_wave32_f32_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ selected_positions,
    const int32_t* __restrict__ selected_counts,
    const int32_t* __restrict__ block_tables,
    float* __restrict__ output,
    int64_t rows,
    int64_t selected_stride,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    float scale) {
  constexpr int64_t head_dim = 128;
  const int64_t row = blockIdx.y;
  const int64_t q_head = blockIdx.x;
  if (row >= rows) return;
  const int64_t selected_count = selected_counts[row];
  if (selected_count <= 0 || selected_count > selected_stride) return;
  const int64_t kv_head = q_head / (query_heads / kv_heads);
  const int lane = threadIdx.x;
  const float* query_row = query + (row * query_heads + q_head) * head_dim;
  const int64_t* selected_row = selected_positions + row * selected_stride;
  const int32_t* block_table = block_tables + row * block_table_len;
  __shared__ float stats[4];
  float accumulator_0 = 0.0f;
  float accumulator_1 = 0.0f;
  float accumulator_2 = 0.0f;
  float accumulator_3 = 0.0f;
  if (lane == 0) {
    stats[0] = -INFINITY;
    stats[1] = 0.0f;
  }
  __syncwarp();
  for (int64_t selected_index = 0; selected_index < selected_count; ++selected_index) {
    const int64_t position = selected_row[selected_index];
    const int64_t logical_block = position / block_size;
    if (position < 0 || logical_block >= block_table_len) continue;
    const int64_t physical_block = block_table[logical_block];
    const int64_t token = physical_block * block_size + position % block_size;
    const int64_t kv_offset = (token * kv_heads + kv_head) * head_dim;
    const float product_0 = query_row[lane]
        * bf16_bits_to_float(key_cache[kv_offset + lane]);
    const float product_1 = query_row[lane + 64]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 64]);
    const float product_2 = query_row[lane + 32]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 32]);
    const float product_3 = query_row[lane + 96]
        * bf16_bits_to_float(key_cache[kv_offset + lane + 96]);
    volatile float pair_0 = product_0 + product_1;
    volatile float pair_1 = product_2 + product_3;
    float dot = pair_0 + pair_1;
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down(dot, offset, 32);
    }
    if (lane == 0) {
      const float score = dot * scale;
      const float old_max = stats[0];
      const float next_maximum = fmaxf(old_max, score);
      const float old_scale = isinf(old_max) ? 0.0f : expf(old_max - next_maximum);
      const float score_scale = expf(score - next_maximum);
      stats[0] = next_maximum;
      stats[1] = stats[1] * old_scale + score_scale;
      stats[2] = old_scale;
      stats[3] = score_scale;
    }
    __syncwarp();
    accumulator_0 = accumulator_0 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane]) * stats[3];
    accumulator_1 = accumulator_1 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 64]) * stats[3];
    accumulator_2 = accumulator_2 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 32]) * stats[3];
    accumulator_3 = accumulator_3 * stats[2]
        + bf16_bits_to_float(value_cache[kv_offset + lane + 96]) * stats[3];
    __syncwarp();
  }
  const int64_t output_offset = (row * query_heads + q_head) * head_dim;
  output[output_offset + lane] = accumulator_0 / stats[1];
  output[output_offset + lane + 64] = accumulator_1 / stats[1];
  output[output_offset + lane + 32] = accumulator_2 / stats[1];
  output[output_offset + lane + 96] = accumulator_3 / stats[1];
}

__global__ void qsa_pool_norm_rope_f32_kernel(
    const float* __restrict__ raw_keys,
    const int32_t* __restrict__ member_indices,
    const int64_t* __restrict__ block_starts,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t ratio,
    int64_t index_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* partial = shared + index_dim;
  const int64_t block = blockIdx.x;
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    float value = 0.0f;
    for (int64_t member = 0; member < ratio; ++member) {
      const int32_t physical = member_indices[block * ratio + member];
      value += raw_keys[static_cast<int64_t>(physical) * index_dim + dimension];
    }
    pooled[dimension] = value / static_cast<float>(ratio);
  }
  __syncthreads();
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    sumsq += pooled[dimension] * pooled[dimension];
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(index_dim) + eps);
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    float value = pooled[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t half = rotary_dim / 2;
      const int64_t pair = dimension % half;
      const float x = pooled[pair] * inv_rms * weight[pair];
      const float y = pooled[pair + half] * inv_rms * weight[pair + half];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(block_starts[block]) * powf(theta, exponent);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      value = dimension < half ? x * cosine - y * sine : x * sine + y * cosine;
    }
    output[block * index_dim + dimension] = value;
  }
}

__global__ void qsa_score_f32_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ pooled_keys,
    float* __restrict__ scores,
    int64_t blocks,
    int64_t heads,
    int64_t index_dim) {
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t query = pair / blocks;
  const int64_t block = pair % blocks;
  float value = 0.0f;
  for (int64_t head = threadIdx.x; head < heads; head += blockDim.x) {
    float dot = 0.0f;
    const int64_t query_offset = (query * heads + head) * index_dim;
    const int64_t key_offset = block * index_dim;
    for (int64_t dimension = 0; dimension < index_dim; ++dimension) {
      dot += queries[query_offset + dimension] * pooled_keys[key_offset + dimension];
    }
    value += fmaxf(dot, 0.0f);
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[pair] = partial[0] * rsqrtf(static_cast<float>(index_dim));
  }
}

__global__ __launch_bounds__(1024) void qsa_topk_expand_f32_i64_kernel(
    const float* __restrict__ score_rows,
    const int64_t* __restrict__ query_positions,
    int64_t* __restrict__ selected_rows,
    int32_t* __restrict__ selected_counts,
    int64_t rows,
    int64_t score_stride,
    int64_t output_stride,
    int64_t scalar_blocks,
    int64_t scalar_query_position,
    int64_t ratio,
    int64_t budget) {
  __shared__ int histogram[256];
  __shared__ uint32_t prefix_bits;
  __shared__ uint32_t prefix_mask;
  __shared__ int rank_in_prefix;
  __shared__ int equal_needed;
  __shared__ int equal_seen;
  __shared__ int output_seen;
  __shared__ int wave_counts[32];
  __shared__ int wave_offsets[32];
  __shared__ int equal_wave_counts[32];
  __shared__ int equal_wave_offsets[32];
  __shared__ int tile_count;
  __shared__ int equal_tile_count;

  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t query_position = query_positions
      ? query_positions[row] : scalar_query_position;
  const int64_t blocks = query_positions
      ? (query_position + 1) / ratio : scalar_blocks;
  const float* scores = score_rows + row * score_stride;
  int64_t* selected_positions = selected_rows + row * output_stride;
  int32_t* selected_count = selected_counts + row;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize;
  const int wave = tid / warpSize;
  const int waves = (blockDim.x + warpSize - 1) / warpSize;
  const int64_t output_capacity = budget * ratio + ratio - 1;
  for (int64_t index = tid; index < output_capacity; index += blockDim.x) {
    selected_positions[index] = -1;
  }
  if (tid == 0) {
    prefix_bits = 0;
    prefix_mask = 0;
    rank_in_prefix = static_cast<int>(budget);
  }
  __syncthreads();

  // Scores are finite and non-negative (sum of rectified head dots), so their
  // IEEE-754 bit patterns have the same total order as their values. Four
  // stable radix passes find the exact kth threshold without materializing a
  // host-side sort.
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) histogram[tid] = 0;
    __syncthreads();
    for (int64_t index = tid; index < blocks; index += blockDim.x) {
      const uint32_t bits = __float_as_uint(scores[index]);
      if ((bits & prefix_mask) == prefix_bits) {
        atomicAdd(&histogram[(bits >> shift) & 0xffu], 1);
      }
    }
    __syncthreads();
    if (tid == 0) {
      int higher = 0;
      int chosen = 0;
      for (int bin = 255; bin >= 0; --bin) {
        const int count = histogram[bin];
        if (rank_in_prefix > higher + count) {
          higher += count;
          continue;
        }
        chosen = bin;
        rank_in_prefix -= higher;
        break;
      }
      prefix_bits |= static_cast<uint32_t>(chosen) << shift;
      prefix_mask |= 0xffu << shift;
    }
    __syncthreads();
  }
  if (tid == 0) {
    equal_needed = rank_in_prefix;
    equal_seen = 0;
    output_seen = 0;
  }
  __syncthreads();

  // Compact in increasing block-index order. Exact-threshold ties take the
  // lowest indices, matching the strict host lexsort contract. The output is
  // therefore already in the order consumed by strict online softmax.
  for (int64_t base = 0; base < blocks; base += blockDim.x) {
    const int64_t index = base + tid;
    const bool valid = index < blocks;
    const uint32_t bits = valid ? __float_as_uint(scores[index]) : 0;
    const bool equal = valid && bits == prefix_bits;
    const unsigned long long equal_ballot = __ballot(equal);
    const unsigned long long lower_mask =
        lane == 0 ? 0ull : ((1ull << lane) - 1ull);
    const int equal_lane_offset = __popcll(equal_ballot & lower_mask);
    if (lane == 0) equal_wave_counts[wave] = __popcll(equal_ballot);
    __syncthreads();
    if (tid == 0) {
      int offset = 0;
      for (int current = 0; current < waves; ++current) {
        equal_wave_offsets[current] = offset;
        offset += equal_wave_counts[current];
      }
      equal_tile_count = offset;
    }
    __syncthreads();
    const int equal_rank = equal_seen + equal_wave_offsets[wave] + equal_lane_offset;
    const bool candidate = valid &&
        (bits > prefix_bits || (equal && equal_rank < equal_needed));
    const unsigned long long candidate_ballot = __ballot(candidate);
    const int candidate_lane_offset = __popcll(candidate_ballot & lower_mask);
    if (lane == 0) wave_counts[wave] = __popcll(candidate_ballot);
    __syncthreads();
    if (tid == 0) {
      int offset = 0;
      for (int current = 0; current < waves; ++current) {
        wave_offsets[current] = offset;
        offset += wave_counts[current];
      }
      tile_count = offset;
    }
    __syncthreads();
    if (candidate) {
      const int output_block =
          output_seen + wave_offsets[wave] + candidate_lane_offset;
      for (int member = 0; member < ratio; ++member) {
        selected_positions[static_cast<int64_t>(output_block) * ratio + member]
            = index * ratio + member;
      }
    }
    __syncthreads();
    if (tid == 0) {
      output_seen += tile_count;
      equal_seen += equal_tile_count;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int64_t tail_start = blocks * ratio;
    const int64_t tail_count = query_position + 1 - tail_start;
    for (int64_t tail = 0; tail < tail_count; ++tail) {
      selected_positions[budget * ratio + tail] = tail_start + tail;
    }
    selected_count[0] = static_cast<int32_t>(budget * ratio + tail_count);
  }
}

__global__ void qsa_select_blocks_f32_i64_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ block_starts,
    const int64_t* __restrict__ query_positions,
    int64_t* __restrict__ selected_starts,
    int32_t* __restrict__ selected_counts,
    int64_t blocks,
    int64_t ratio,
    int64_t budget) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t query = blockIdx.x;
  int64_t count = 0;
  for (int64_t slot = 0; slot < budget; ++slot) {
    selected_starts[query * budget + slot] = -1;
  }
  for (int64_t slot = 0; slot < budget; ++slot) {
    float best_score = -FLT_MAX;
    int64_t best_start = INT64_MAX;
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t start = block_starts[block];
      if (start + ratio - 1 > query_positions[query]) {
        continue;
      }
      bool used = false;
      for (int64_t prior = 0; prior < count; ++prior) {
        used = used || selected_starts[query * budget + prior] == start;
      }
      if (used) {
        continue;
      }
      const float score = scores[query * blocks + block];
      if (score > best_score || (score == best_score && start < best_start)) {
        best_score = score;
        best_start = start;
      }
    }
    if (best_start == INT64_MAX) {
      break;
    }
    selected_starts[query * budget + count] = best_start;
    ++count;
  }
  for (int64_t outer = 1; outer < count; ++outer) {
    const int64_t current = selected_starts[query * budget + outer];
    int64_t inner = outer - 1;
    while (inner >= 0 && selected_starts[query * budget + inner] > current) {
      selected_starts[query * budget + inner + 1] = selected_starts[query * budget + inner];
      --inner;
    }
    selected_starts[query * budget + inner + 1] = current;
  }
  selected_counts[query] = static_cast<int32_t>(count);
}

}  // namespace

extern "C" int hipengine_qwen4_exp_qsa_split_norm_rope_f32(
    const void* q_projected,
    const void* key,
    const void* q_weight,
    const void* k_weight,
    const void* position,
    void* query_out,
    void* key_out,
    void* gate_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !position || !query_out ||
      !key_out || !gate_out || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_split_norm_rope_f32_kernel<false>),
      dim3(static_cast<uint32_t>(num_q_heads + num_kv_heads)), dim3(threads), shared, stream,
      static_cast<const float*>(q_projected), static_cast<const float*>(key),
      static_cast<const float*>(q_weight), static_cast<const float*>(k_weight),
      static_cast<const int64_t*>(position), static_cast<float*>(query_out),
      static_cast<float*>(key_out), static_cast<float*>(gate_out),
      num_q_heads, num_kv_heads, head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_norm_rope_f32(
    const void* input,
    const void* weight,
    const void* position,
    void* output,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !position || !output || heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_norm_rope_f32_kernel<false>),
      dim3(static_cast<uint32_t>(heads)), dim3(threads), shared, stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const int64_t*>(position), static_cast<float*>(output),
      head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_split_norm_rope_rows_f32(
    const void* q_projected,
    const void* key,
    const void* q_weight,
    const void* k_weight,
    const void* positions,
    void* query_out,
    void* key_out,
    void* gate_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !positions || !query_out ||
      !key_out || !gate_out || rows <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_split_norm_rope_rows_f32_kernel<false>),
      dim3(static_cast<uint32_t>(num_q_heads + num_kv_heads), static_cast<uint32_t>(rows)),
      dim3(threads), shared, stream,
      static_cast<const float*>(q_projected), static_cast<const float*>(key),
      static_cast<const float*>(q_weight), static_cast<const float*>(k_weight),
      static_cast<const int64_t*>(positions), static_cast<float*>(query_out),
      static_cast<float*>(key_out), static_cast<float*>(gate_out), rows,
      num_q_heads, num_kv_heads, head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_norm_rope_rows_f32(
    const void* input,
    const void* weight,
    const void* positions,
    void* output,
    int64_t rows,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !positions || !output || rows <= 0 || heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_norm_rope_rows_f32_kernel<false>),
      dim3(static_cast<uint32_t>(heads), static_cast<uint32_t>(rows)),
      dim3(threads), shared, stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const int64_t*>(positions), static_cast<float*>(output),
      rows, heads, head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_split_norm_mrope_f32(
    const void* q_projected, const void* key, const void* q_weight,
    const void* k_weight, const void* positions, void* query_out,
    void* key_out, void* gate_out, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim,
    float theta, float eps, hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !positions ||
      !query_out || !key_out || !gate_out || num_q_heads <= 0 ||
      num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_split_norm_rope_f32_kernel<true>),
      dim3(static_cast<uint32_t>(num_q_heads + num_kv_heads)), dim3(threads),
      shared, stream, static_cast<const float*>(q_projected),
      static_cast<const float*>(key), static_cast<const float*>(q_weight),
      static_cast<const float*>(k_weight), static_cast<const int64_t*>(positions),
      static_cast<float*>(query_out), static_cast<float*>(key_out),
      static_cast<float*>(gate_out), num_q_heads, num_kv_heads, head_dim,
      rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_norm_mrope_f32(
    const void* input, const void* weight, const void* positions, void* output,
    int64_t heads, int64_t head_dim, int64_t rotary_dim, float theta,
    float eps, hipStream_t stream) {
  if (!input || !weight || !positions || !output || heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_norm_rope_f32_kernel<true>), dim3(static_cast<uint32_t>(heads)),
      dim3(threads), shared, stream, static_cast<const float*>(input),
      static_cast<const float*>(weight), static_cast<const int64_t*>(positions),
      static_cast<float*>(output), head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_split_norm_mrope_rows_f32(
    const void* q_projected, const void* key, const void* q_weight,
    const void* k_weight, const void* positions, void* query_out,
    void* key_out, void* gate_out, int64_t rows, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim,
    float theta, float eps, hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !positions ||
      !query_out || !key_out || !gate_out || rows <= 0 || num_q_heads <= 0 ||
      num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_split_norm_rope_rows_f32_kernel<true>),
      dim3(static_cast<uint32_t>(num_q_heads + num_kv_heads),
           static_cast<uint32_t>(rows)),
      dim3(threads), shared, stream, static_cast<const float*>(q_projected),
      static_cast<const float*>(key), static_cast<const float*>(q_weight),
      static_cast<const float*>(k_weight), static_cast<const int64_t*>(positions),
      static_cast<float*>(query_out), static_cast<float*>(key_out),
      static_cast<float*>(gate_out), rows, num_q_heads, num_kv_heads,
      head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_norm_mrope_rows_f32(
    const void* input, const void* weight, const void* positions, void* output,
    int64_t rows, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    float theta, float eps, hipStream_t stream) {
  if (!input || !weight || !positions || !output || rows <= 0 || heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qsa_norm_rope_rows_f32_kernel<true>),
      dim3(static_cast<uint32_t>(heads), static_cast<uint32_t>(rows)),
      dim3(threads), shared, stream, static_cast<const float*>(input),
      static_cast<const float*>(weight), static_cast<const int64_t*>(positions),
      static_cast<float*>(output), rows, heads, head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_gate_context_f32(
    const void* context,
    const void* gate,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!context || !gate || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      qsa_gate_context_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(context), static_cast<const float*>(gate),
      static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_f32(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* selected_positions,
    const void* block_table,
    void* output,
    int64_t selected_count,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!query || !key_cache || !value_cache || !selected_positions || !block_table ||
      !output || selected_count <= 0 || block_size <= 0 || block_table_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads ||
      head_dim <= 0 || head_dim > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      qsa_sparse_attention_paged_bf16_f32_kernel,
      dim3(static_cast<uint32_t>(query_heads)), dim3(threads),
      (threads + 4) * sizeof(float), stream,
      static_cast<const float*>(query), static_cast<const uint16_t*>(key_cache),
      static_cast<const uint16_t*>(value_cache),
      static_cast<const int64_t*>(selected_positions),
      static_cast<const int32_t*>(block_table), static_cast<float*>(output),
      selected_count, block_size, block_table_len, query_heads, kv_heads, head_dim, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* selected_positions,
    const void* selected_counts,
    const void* block_tables,
    void* output,
    int64_t rows,
    int64_t selected_stride,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!query || !key_cache || !value_cache || !selected_positions ||
      !selected_counts || !block_tables || !output || rows <= 0 ||
      selected_stride <= 0 || block_size <= 0 || block_table_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads ||
      head_dim <= 0 || head_dim > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      qsa_sparse_attention_paged_bf16_rows_f32_kernel,
      dim3(static_cast<uint32_t>(query_heads), static_cast<uint32_t>(rows)),
      dim3(threads), (threads + 4) * sizeof(float), stream,
      static_cast<const float*>(query), static_cast<const uint16_t*>(key_cache),
      static_cast<const uint16_t*>(value_cache),
      static_cast<const int64_t*>(selected_positions),
      static_cast<const int32_t*>(selected_counts),
      static_cast<const int32_t*>(block_tables), static_cast<float*>(output),
      rows, selected_stride, block_size, block_table_len, query_heads,
      kv_heads, head_dim, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* selected_positions,
    const void* block_table,
    void* output,
    int64_t selected_count,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!query || !key_cache || !value_cache || !selected_positions || !block_table ||
      !output || selected_count <= 0 || block_size <= 0 || block_table_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qsa_sparse_attention_paged_bf16_wave32_f32_kernel,
      dim3(static_cast<uint32_t>(query_heads)), dim3(32), 0, stream,
      static_cast<const float*>(query), static_cast<const uint16_t*>(key_cache),
      static_cast<const uint16_t*>(value_cache),
      static_cast<const int64_t*>(selected_positions),
      static_cast<const int32_t*>(block_table), static_cast<float*>(output),
      selected_count, block_size, block_table_len, query_heads, kv_heads, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* selected_positions,
    const void* selected_counts,
    const void* block_tables,
    void* output,
    int64_t rows,
    int64_t selected_stride,
    int64_t block_size,
    int64_t block_table_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!query || !key_cache || !value_cache || !selected_positions || !selected_counts ||
      !block_tables || !output || rows <= 0 || selected_stride <= 0 || block_size <= 0 ||
      block_table_len <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qsa_sparse_attention_paged_bf16_rows_wave32_f32_kernel,
      dim3(static_cast<uint32_t>(query_heads), static_cast<uint32_t>(rows)),
      dim3(32), 0, stream, static_cast<const float*>(query),
      static_cast<const uint16_t*>(key_cache),
      static_cast<const uint16_t*>(value_cache),
      static_cast<const int64_t*>(selected_positions),
      static_cast<const int32_t*>(selected_counts),
      static_cast<const int32_t*>(block_tables), static_cast<float*>(output),
      rows, selected_stride, block_size, block_table_len, query_heads, kv_heads, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_scatter_index_keys_f32(
    const void* source,
    void* destination,
    const void* block_table,
    int64_t start_position,
    int64_t rows,
    int64_t block_size,
    int64_t index_dim,
    hipStream_t stream) {
  if (!source || !destination || !block_table || start_position < 0 ||
      rows <= 0 || block_size <= 0 || index_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = rows * index_dim;
  hipLaunchKernelGGL(
      qsa_scatter_index_keys_f32_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads)),
      dim3(threads), 0, stream,
      static_cast<const float*>(source), static_cast<float*>(destination),
      static_cast<const int32_t*>(block_table), start_position, rows,
      block_size, index_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_scatter_index_key_device_position_f32(
    const void* source,
    void* destination,
    const void* block_table,
    const void* position,
    int64_t block_size,
    int64_t block_table_len,
    int64_t index_dim,
    hipStream_t stream) {
  if (!source || !destination || !block_table || !position ||
      block_size <= 0 || block_table_len <= 0 || index_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      qsa_scatter_index_key_device_position_f32_kernel,
      dim3(static_cast<uint32_t>((index_dim + threads - 1) / threads)),
      dim3(threads), 0, stream,
      static_cast<const float*>(source), static_cast<float*>(destination),
      static_cast<const int32_t*>(block_table),
      static_cast<const int64_t*>(position), block_size, block_table_len,
      index_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_pool_norm_rope_f32(
    const void* raw_keys,
    const void* member_indices,
    const void* block_starts,
    const void* weight,
    void* output,
    int64_t blocks,
    int64_t ratio,
    int64_t index_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!raw_keys || !member_indices || !block_starts || !weight || !output ||
      blocks <= 0 || ratio <= 0 || index_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > index_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(index_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      qsa_pool_norm_rope_f32_kernel,
      dim3(static_cast<uint32_t>(blocks)), dim3(threads), shared, stream,
      static_cast<const float*>(raw_keys), static_cast<const int32_t*>(member_indices),
      static_cast<const int64_t*>(block_starts), static_cast<const float*>(weight),
      static_cast<float*>(output), ratio, index_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_score_f32(
    const void* queries,
    const void* pooled_keys,
    void* scores,
    int64_t query_count,
    int64_t blocks,
    int64_t heads,
    int64_t index_dim,
    hipStream_t stream) {
  if (!queries || !pooled_keys || !scores || query_count <= 0 || blocks <= 0 ||
      heads <= 0 || index_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 32;
  hipLaunchKernelGGL(
      qsa_score_f32_kernel,
      dim3(static_cast<uint32_t>(query_count * blocks)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(queries), static_cast<const float*>(pooled_keys),
      static_cast<float*>(scores), blocks, heads, index_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_topk_expand_f32_i64(
    const void* scores,
    void* selected_positions,
    void* selected_count,
    int64_t blocks,
    int64_t query_position,
    int64_t ratio,
    int64_t budget,
    hipStream_t stream) {
  if (!scores || !selected_positions || !selected_count || blocks <= 0 ||
      query_position < 0 || ratio <= 0 || budget <= 0 || budget > blocks ||
      query_position + 1 < blocks * ratio ||
      query_position + 1 >= (blocks + 1) * ratio) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 1024;
  hipLaunchKernelGGL(
      qsa_topk_expand_f32_i64_kernel,
      dim3(1), dim3(threads), 0, stream,
      static_cast<const float*>(scores), nullptr,
      static_cast<int64_t*>(selected_positions),
      static_cast<int32_t*>(selected_count), 1, blocks,
      budget * ratio + ratio - 1, blocks, query_position, ratio, budget);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_topk_expand_rows_f32_i64(
    const void* scores,
    const void* query_positions,
    void* selected_positions,
    void* selected_counts,
    int64_t rows,
    int64_t score_stride,
    int64_t output_stride,
    int64_t ratio,
    int64_t budget,
    hipStream_t stream) {
  if (!scores || !query_positions || !selected_positions || !selected_counts ||
      rows <= 0 || score_stride < budget || ratio <= 0 || budget <= 0 ||
      output_stride < budget * ratio + ratio - 1) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 1024;
  hipLaunchKernelGGL(
      qsa_topk_expand_f32_i64_kernel,
      dim3(static_cast<uint32_t>(rows)), dim3(threads), 0, stream,
      static_cast<const float*>(scores),
      static_cast<const int64_t*>(query_positions),
      static_cast<int64_t*>(selected_positions),
      static_cast<int32_t*>(selected_counts), rows, score_stride, output_stride,
      0, 0, ratio, budget);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_select_blocks_f32_i64(
    const void* scores,
    const void* block_starts,
    const void* query_positions,
    void* selected_starts,
    void* selected_counts,
    int64_t query_count,
    int64_t blocks,
    int64_t ratio,
    int64_t budget,
    hipStream_t stream) {
  if (!scores || !block_starts || !query_positions || !selected_starts || !selected_counts ||
      query_count <= 0 || blocks <= 0 || ratio <= 0 || budget <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qsa_select_blocks_f32_i64_kernel,
      dim3(static_cast<uint32_t>(query_count)), dim3(1), 0, stream,
      static_cast<const float*>(scores), static_cast<const int64_t*>(block_starts),
      static_cast<const int64_t*>(query_positions), static_cast<int64_t*>(selected_starts),
      static_cast<int32_t*>(selected_counts), blocks, ratio, budget);
  return static_cast<int>(hipGetLastError());
}
