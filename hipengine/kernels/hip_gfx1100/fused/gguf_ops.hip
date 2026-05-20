// Small BF16 elementwise helpers for GGUF Qwen3.5 bring-up.

#include <hip/hip_runtime.h>

#include <cstdint>

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

__device__ inline float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__global__ void gguf_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ weight,
    uint16_t* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) * inv_rms * weight[idx];
    out[row_base + idx] = float_to_bf16_bits(value);
  }
}

__global__ void gguf_add_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
    residual_out[row_base + idx] = float_to_bf16_bits(value);
    norm_out[row_base + idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

__global__ void gguf_bf16_add_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    out[idx] = float_to_bf16_bits(bf16_bits_to_float(a[idx]) + bf16_bits_to_float(b[idx]));
  }
}

__global__ void gguf_gate_repeat_value_bf16_kernel(
    const uint16_t* __restrict__ q_gate,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim) {
  const int64_t n = head_count * head_dim;
  const int64_t ratio = head_count / head_count_kv;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const int64_t head = idx / head_dim;
    const int64_t dim = idx - head * head_dim;
    const int64_t kv_head = head / ratio;
    const float gate = sigmoidf_stable(bf16_bits_to_float(q_gate[idx]));
    const float v = bf16_bits_to_float(value[kv_head * head_dim + dim]);
    out[idx] = float_to_bf16_bits(gate * v);
  }
}

__global__ void gguf_gate_mul_bf16_kernel(
    const uint16_t* __restrict__ attn,
    const uint16_t* __restrict__ gate,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const float a = bf16_bits_to_float(attn[idx]);
    const float g = sigmoidf_stable(bf16_bits_to_float(gate[idx]));
    out[idx] = float_to_bf16_bits(a * g);
  }
}

__device__ __forceinline__ void gguf_head_rmsnorm_partial_rotary_f32_weight_row(
    const float* __restrict__ src,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim] * inv_rms * (1.0f + weight[dim]);
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * (1.0f + weight[pair_dim]);
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__device__ __forceinline__ void gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
    const uint16_t* __restrict__ src,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = bf16_bits_to_float(src[dim]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float src_value = bf16_bits_to_float(src[dim]);
    const float value = src_value * inv_rms * (1.0f + weight[dim]);
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired_src = bf16_bits_to_float(src[pair_dim]);
      const float paired = paired_src * inv_rms * (1.0f + weight[pair_dim]);
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__global__ void gguf_head_rmsnorm_partial_rotary_position_f32_weight_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ position_ptr,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const int64_t head = blockIdx.x;
  if (head >= num_q_heads + num_kv_heads) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t offset = local_head * head_dim;
  const float* src = (is_query ? query : key) + offset;
  const float* weight = is_query ? q_weight : k_weight;
  float* dst = (is_query ? query_out : key_out) + offset;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  gguf_head_rmsnorm_partial_rotary_f32_weight_row(
      src,
      weight,
      cos,
      sin,
      dst,
      eps,
      head_dim,
      rotary_dim,
      partial);
}

__global__ void gguf_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ position_ptr,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const int64_t head = blockIdx.x;
  if (head >= num_q_heads + num_kv_heads) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t offset = local_head * head_dim;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  if (is_query) {
    gguf_head_rmsnorm_partial_rotary_f32_weight_row(
        query + offset,
        q_weight,
        cos,
        sin,
        query_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  } else {
    gguf_head_rmsnorm_partial_rotary_bf16_input_f32_weight_row(
        key + offset,
        k_weight,
        cos,
        sin,
        key_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
  }
}

__global__ void gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ positions,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float eps,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t head = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (head >= num_q_heads + num_kv_heads || token >= tokens) {
    return;
  }
  const int64_t position = positions[token];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t heads = is_query ? num_q_heads : num_kv_heads;
  const int64_t offset = (token * heads + local_head) * head_dim;
  const float* src = (is_query ? query : key) + offset;
  const float* weight = is_query ? q_weight : k_weight;
  float* dst = (is_query ? query_out : key_out) + offset;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  gguf_head_rmsnorm_partial_rotary_f32_weight_row(
      src,
      weight,
      cos,
      sin,
      dst,
      eps,
      head_dim,
      rotary_dim,
      partial);
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_add_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, add, weight, norm_out, residual_out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_bf16_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_bf16_add_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      a, b, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_gate_repeat_value_bf16(
    const uint16_t* q_gate,
    const uint16_t* value,
    uint16_t* out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  if (head_count <= 0 || head_count_kv <= 0 || head_dim <= 0 || head_count % head_count_kv != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t n = head_count * head_dim;
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_gate_repeat_value_bf16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      q_gate, value, out, head_count, head_count_kv, head_dim);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_gate_mul_bf16(
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_gate_mul_bf16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      attn, gate, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
    const float* query,
    const float* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_position_f32_weight_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight(
    const float* query,
    const uint16_t* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight(
    const float* query,
    const float* key,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* positions,
    float* query_out,
    float* key_out,
    float eps,
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel<<<
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads), static_cast<unsigned int>(tokens)),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      positions,
      query_out,
      key_out,
      eps,
      tokens,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}
