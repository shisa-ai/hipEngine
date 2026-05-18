// DFlash drafter root/query preparation kernels.
//
// The DFlash BF16 drafter reuses the target embedding table for the root token
// plus mask-token query rows.  This kernel materializes the compact native
// input slab on device: token ids, absolute query positions, and BF16 embedding
// rows.  It intentionally does not materialize context K/V; that remains the
// separate DFlash context-cache task.

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

template <typename embed_t>
__device__ inline uint16_t embed_to_bf16(embed_t value) {
  return float_to_bf16_bits(static_cast<float>(value));
}

template <>
__device__ inline uint16_t embed_to_bf16<uint16_t>(uint16_t value) {
  return value;
}

template <typename embed_t>
__global__ void dflash_prepare_noise_inputs_bf16_i32_kernel(
    const int32_t* __restrict__ root_tokens,
    const int32_t* __restrict__ root_positions,
    const embed_t* __restrict__ embed_tokens,
    int32_t* __restrict__ noise_token_ids,
    int32_t* __restrict__ position_ids,
    uint16_t* __restrict__ noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col = static_cast<int64_t>(blockIdx.y);
  if (row >= request_count || col >= block_size) {
    return;
  }
  const int64_t slab_row = row * block_size + col;
  const int32_t token = (col == 0) ? root_tokens[row] : mask_token_id;
  const int32_t position = root_positions[row] + static_cast<int32_t>(col);
  if (threadIdx.x == 0) {
    noise_token_ids[slab_row] = token;
    position_ids[slab_row] = position;
  }
  const bool valid_token = token >= 0 && static_cast<int64_t>(token) < vocab_size;
  for (int64_t h = static_cast<int64_t>(threadIdx.x); h < hidden_size; h += blockDim.x) {
    uint16_t value = 0;
    if (valid_token) {
      value = embed_to_bf16(embed_tokens[static_cast<int64_t>(token) * hidden_size + h]);
    }
    noise_embeddings[slab_row * hidden_size + h] = value;
  }
}

bool valid_shape(int64_t request_count, int64_t block_size, int64_t hidden_size, int64_t vocab_size) {
  return request_count > 0 && block_size > 0 && hidden_size > 0 && vocab_size > 0;
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

__global__ void dflash_dense_bf16_to_f32_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows || out_col >= out_features) {
    return;
  }
  extern __shared__ float partial[];
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;
  float acc = 0.0f;
  for (int64_t dim = static_cast<int64_t>(threadIdx.x); dim < in_features; dim += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(x[x_base + dim]), bf16_bits_to_float(weight[w_base + dim]), acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row * out_features + out_col] = partial[0];
  }
}

__global__ void dflash_gqa_attention_f32_bf16_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t q_row = static_cast<int64_t>(blockIdx.y);
  const int64_t batch = static_cast<int64_t>(blockIdx.z);
  if (batch >= batch_size || q_row >= query_len || q_head >= num_q_heads) {
    return;
  }
  extern __shared__ float shared[];
  float* partial = shared;
  float* scores = shared + blockDim.x;
  const int64_t group_size = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t q_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;

  for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
    const int64_t k_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      dot += query[q_base + dim] * key[k_base + dim];
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[kv_row] = partial[0] * scale;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float max_score = -3.4028234663852886e+38F;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      max_score = fmaxf(max_score, scores[kv_row]);
    }
    float denom = 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      const float prob = expf(scores[kv_row] - max_score);
      scores[kv_row] = prob;
      denom += prob;
    }
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      scores[kv_row] *= inv_denom;
    }
  }
  __syncthreads();

  const int64_t out_base = ((batch * query_len + q_row) * num_q_heads + q_head) * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t kv_row = 0; kv_row < kv_len; ++kv_row) {
      const int64_t v_base = ((batch * kv_len + kv_row) * num_kv_heads + kv_head) * head_dim;
      acc += scores[kv_row] * bf16_bits_to_float(value[v_base + dim]);
    }
    out[out_base + dim] = float_to_bf16_bits(acc);
  }
}

bool valid_dense_shape(int64_t rows, int64_t in_features, int64_t out_features, int64_t threads) {
  return rows > 0 && in_features > 0 && out_features > 0 && valid_threads(threads);
}

bool valid_attention_shape(
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t threads) {
  return batch_size > 0 && query_len > 0 && kv_len > 0 && num_q_heads > 0 && num_kv_heads > 0 &&
         head_dim > 0 && (num_q_heads % num_kv_heads) == 0 && valid_threads(threads);
}

}  // namespace

extern "C" int hipengine_dflash_prepare_noise_inputs_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const uint16_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_dense_bf16_to_f32(
    const uint16_t* x,
    const uint16_t* weight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_dense_shape(rows, in_features, out_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      dflash_dense_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_gqa_attention_f32_bf16(
    const float* query,
    const float* key,
    const uint16_t* value,
    uint16_t* out,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_attention_shape(batch_size, query_len, kv_len, num_q_heads, num_kv_heads, head_dim, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t smem = static_cast<size_t>(threads + kv_len) * sizeof(float);
  hipLaunchKernelGGL(
      dflash_gqa_attention_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(query_len), static_cast<unsigned int>(batch_size)),
      dim3(static_cast<unsigned int>(threads)),
      smem,
      stream,
      query,
      key,
      value,
      out,
      batch_size,
      query_len,
      kv_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dflash_prepare_noise_inputs_f16_to_bf16_i32(
    const int32_t* root_tokens,
    const int32_t* root_positions,
    const half_t* embed_tokens,
    int32_t* noise_token_ids,
    int32_t* position_ids,
    uint16_t* noise_embeddings,
    int64_t request_count,
    int64_t block_size,
    int64_t hidden_size,
    int64_t vocab_size,
    int32_t mask_token_id,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(request_count, block_size, hidden_size, vocab_size) || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dflash_prepare_noise_inputs_bf16_i32_kernel<half_t>),
      dim3(static_cast<unsigned int>(request_count), static_cast<unsigned int>(block_size)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      root_tokens,
      root_positions,
      embed_tokens,
      noise_token_ids,
      position_ids,
      noise_embeddings,
      request_count,
      block_size,
      hidden_size,
      vocab_size,
      mask_token_id);
  return static_cast<int>(hipGetLastError());
}
