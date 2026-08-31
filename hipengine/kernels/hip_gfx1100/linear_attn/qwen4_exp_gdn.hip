// hipEngine-native strict Qwen4Exp GDN decode recurrence with sigmoid output gate.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdint.h>

namespace {

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ inline float softplus_f32(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

__global__ void qwen4_exp_gdn_decode_f32_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ output_gate,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_logits,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ norm_weight,
    float* __restrict__ recurrent_state,
    float* __restrict__ output,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* query = shared;
  float* key = query + head_k_dim;
  float* reductions = key + head_k_dim;
  float* core = reductions + blockDim.x;
  float* norm_reductions = core + head_v_dim;

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = v_head % num_k_heads;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t q_offset = k_head * head_k_dim;
  const int64_t k_offset = key_width + q_offset;
  const int64_t v_offset = 2 * key_width + v_head * head_v_dim;
  float q_sumsq = 0.0f;
  float k_sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < head_k_dim; column += blockDim.x) {
    const float q_value = conv[q_offset + column];
    const float k_value = conv[k_offset + column];
    query[column] = q_value;
    key[column] = k_value;
    q_sumsq += q_value * q_value;
    k_sumsq += k_value * k_value;
  }
  reductions[threadIdx.x] = q_sumsq;
  norm_reductions[threadIdx.x] = k_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float q_scale = rsqrtf(reductions[0] + 1.0e-6f)
      * rsqrtf(static_cast<float>(head_k_dim));
  const float k_scale = rsqrtf(norm_reductions[0] + 1.0e-6f);
  const float beta = sigmoid_f32(beta_logits[v_head]);
  const float decay = expf(a[v_head] * softplus_f32(alpha[v_head] + dt_bias[v_head]));
  const int64_t value_index = threadIdx.x;
  float core_value = 0.0f;
  if (value_index < head_v_dim) {
    const int64_t state_col = v_head * head_k_dim * head_v_dim + value_index;
    float memory = 0.0f;
    for (int64_t column = 0; column < head_k_dim; ++column) {
      memory += key[column] * k_scale
          * recurrent_state[state_col + column * head_v_dim] * decay;
    }
    const float delta = (conv[v_offset + value_index] - memory) * beta;
    for (int64_t column = 0; column < head_k_dim; ++column) {
      const int64_t state_index = state_col + column * head_v_dim;
      const float next = recurrent_state[state_index] * decay
          + key[column] * k_scale * delta;
      recurrent_state[state_index] = next;
      core_value += query[column] * q_scale * next;
    }
    core[value_index] = core_value;
    norm_reductions[threadIdx.x] = core_value * core_value;
  } else {
    norm_reductions[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (value_index < head_v_dim) {
    const int64_t index = v_head * head_v_dim + value_index;
    const float inv_rms = rsqrtf(norm_reductions[0] / static_cast<float>(head_v_dim) + 1.0e-6f);
    output[index] = core[value_index] * inv_rms * norm_weight[value_index]
        * sigmoid_f32(output_gate[index]);
  }
}

__global__ void qwen4_exp_gdn_prefill_prepare_f32_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_logits,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ beta,
    float* __restrict__ decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  float* q_partial = partial;
  float* k_partial = q_partial + blockDim.x;
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  if (token >= tokens) return;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t value_width = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_width + value_width;
  const int64_t conv_base = token * qkv_width;
  if (head < num_k_heads) {
    const int64_t q_base = conv_base + head * head_k_dim;
    const int64_t k_base = conv_base + key_width + head * head_k_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float qv = conv[q_base + idx];
      const float kv = conv[k_base + idx];
      q_sum += qv * qv;
      k_sum += kv * kv;
    }
    q_partial[threadIdx.x] = q_sum;
    k_partial[threadIdx.x] = k_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        q_partial[threadIdx.x] += q_partial[threadIdx.x + stride];
        k_partial[threadIdx.x] += k_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    // The reused peer recurrence applies 1/sqrt(128) to its output, matching
    // strict Qwen4Exp's query-side factor. Prepare only unit-normalizes Q here.
    const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f);
    const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
    const int64_t out_base = (token * num_k_heads + head) * head_k_dim;
    for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      query[out_base + idx] = conv[q_base + idx] * q_scale;
      key[out_base + idx] = conv[k_base + idx] * k_scale;
    }
  }
  if (head < num_v_heads) {
    const int64_t out_base = (token * num_v_heads + head) * head_v_dim;
    const int64_t value_base = conv_base + 2 * key_width + head * head_v_dim;
    for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
      value[out_base + idx] = conv[value_base + idx];
    }
    if (threadIdx.x == 0) {
      const int64_t scalar = token * num_v_heads + head;
      beta[scalar] = sigmoid_f32(beta_logits[scalar]);
      decay[scalar] = expf(
          a[head] * softplus_f32(alpha[scalar] + dt_bias[head]));
    }
  }
}

__global__ void qwen4_exp_gdn_prefill_sigmoid_gate_f32_kernel(
    float* __restrict__ core,
    const float* __restrict__ output_gate,
    const float* __restrict__ norm_weight,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  if (group >= tokens * num_v_heads) return;
  const int64_t base = group * head_v_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    const float v = core[base + idx];
    sumsq += v * v;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(
      partial[0] / static_cast<float>(head_v_dim) + 1.0e-6f);
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    core[base + idx] = core[base + idx] * inv_rms * norm_weight[idx]
        * sigmoid_f32(output_gate[base + idx]);
  }
}

__global__ void qwen4_exp_gdn_prefill_f32_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ output_gate,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_logits,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ norm_weight,
    float* __restrict__ recurrent_state,
    float* __restrict__ output,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* query = shared;
  float* key = query + head_k_dim;
  float* reductions = key + head_k_dim;
  float* core = reductions + blockDim.x;
  float* norm_reductions = core + head_v_dim;

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = v_head % num_k_heads;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_width + num_v_heads * head_v_dim;
  const int64_t q_offset = k_head * head_k_dim;
  const int64_t k_offset = key_width + q_offset;
  const int64_t v_offset = 2 * key_width + v_head * head_v_dim;
  const int64_t core_width = num_v_heads * head_v_dim;
  const int64_t value_index = threadIdx.x;
  const int64_t state_col = v_head * head_k_dim * head_v_dim + value_index;

  for (int64_t token = 0; token < tokens; ++token) {
    const float* conv_row = conv + token * qkv_width;
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    for (int64_t column = threadIdx.x; column < head_k_dim; column += blockDim.x) {
      const float q_value = conv_row[q_offset + column];
      const float k_value = conv_row[k_offset + column];
      query[column] = q_value;
      key[column] = k_value;
      q_sumsq += q_value * q_value;
      k_sumsq += k_value * k_value;
    }
    reductions[threadIdx.x] = q_sumsq;
    norm_reductions[threadIdx.x] = k_sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float q_scale = rsqrtf(reductions[0] + 1.0e-6f)
        * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(norm_reductions[0] + 1.0e-6f);
    const float beta = sigmoid_f32(beta_logits[token * num_v_heads + v_head]);
    const float decay = expf(
        a[v_head] * softplus_f32(alpha[token * num_v_heads + v_head] + dt_bias[v_head]));
    float core_value = 0.0f;
    if (value_index < head_v_dim) {
      float memory = 0.0f;
      for (int64_t column = 0; column < head_k_dim; ++column) {
        memory += key[column] * k_scale
            * recurrent_state[state_col + column * head_v_dim] * decay;
      }
      const float delta = (conv_row[v_offset + value_index] - memory) * beta;
      for (int64_t column = 0; column < head_k_dim; ++column) {
        const int64_t state_index = state_col + column * head_v_dim;
        const float next = recurrent_state[state_index] * decay
            + key[column] * k_scale * delta;
        recurrent_state[state_index] = next;
        core_value += query[column] * q_scale * next;
      }
      core[value_index] = core_value;
      norm_reductions[threadIdx.x] = core_value * core_value;
    } else {
      norm_reductions[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (value_index < head_v_dim) {
      const int64_t index = v_head * head_v_dim + value_index;
      const float inv_rms = rsqrtf(
          norm_reductions[0] / static_cast<float>(head_v_dim) + 1.0e-6f);
      output[token * core_width + index] = core[value_index] * inv_rms
          * norm_weight[value_index]
          * sigmoid_f32(output_gate[token * core_width + index]);
    }
    __syncthreads();
  }
}


// ---------------------------------------------------------------------------
// Column-warp GDN prefill (llama gated_delta_net.cu layout; lineage
// ggml-org/llama.cpp@17252c769). One warp owns one output column; the
// head_k_dim state rows are sharded across 32 lanes in registers; all
// reductions are warp shuffles (no __syncthreads in the token loop); the
// state lives in registers across the whole sequence. The output RMS norm
// crosses columns, so this kernel writes the raw core and a companion tail
// kernel applies rms-norm * weight * sigmoid(gate).
// ---------------------------------------------------------------------------
__device__ inline float warp_sum_f32(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

template <int ROWS_PER_LANE>
__global__ void __launch_bounds__(128, 4)
qwen4_exp_gdn_prefill_columnwarps_f32_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_logits,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ recurrent_state,
    float* __restrict__ core_out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) return;
  const int64_t k_head = v_head % num_k_heads;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t col = static_cast<int64_t>(blockIdx.y) * 4 + warp_id;
  if (col >= head_v_dim) return;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_width + num_v_heads * head_v_dim;
  const int64_t q_offset = k_head * head_k_dim;
  const int64_t k_offset = key_width + q_offset;
  const int64_t v_offset = 2 * key_width + v_head * head_v_dim;
  const int64_t core_width = num_v_heads * head_v_dim;
  const int64_t state_col = v_head * head_k_dim * head_v_dim + col;

  float s_shard[ROWS_PER_LANE];
  int j_idx[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    j_idx[r] = r * 32 + lane;
    s_shard[r] = recurrent_state[state_col + j_idx[r] * head_v_dim];
  }

  for (int64_t token = 0; token < tokens; ++token) {
    const float* conv_row = conv + token * qkv_width;
    float q_r[ROWS_PER_LANE];
    float k_r[ROWS_PER_LANE];
    float q_partial = 0.0f;
    float k_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      q_r[r] = conv_row[q_offset + j_idx[r]];
      k_r[r] = conv_row[k_offset + j_idx[r]];
      q_partial += q_r[r] * q_r[r];
      k_partial += k_r[r] * k_r[r];
    }
    const float q_scale = rsqrtf(warp_sum_f32(q_partial) + 1.0e-6f)
        * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(warp_sum_f32(k_partial) + 1.0e-6f);
    const float beta = sigmoid_f32(beta_logits[token * num_v_heads + v_head]);
    const float decay = expf(
        a[v_head] * softplus_f32(alpha[token * num_v_heads + v_head] + dt_bias[v_head]));

    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      kv_partial += (k_r[r] * k_scale) * s_shard[r];
    }
    const float memory = warp_sum_f32(kv_partial) * decay;
    const float delta = (conv_row[v_offset + col] - memory) * beta;

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      s_shard[r] = s_shard[r] * decay + (k_r[r] * k_scale) * delta;
      attn_partial += (q_r[r] * q_scale) * s_shard[r];
    }
    const float attn = warp_sum_f32(attn_partial);
    core_out[token * core_width + v_head * head_v_dim + col] = attn;
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    recurrent_state[state_col + j_idx[r] * head_v_dim] = s_shard[r];
  }
}

__global__ void qwen4_exp_gdn_state_transpose_f32_kernel(
    const float* __restrict__ source,
    float* __restrict__ destination,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    bool strict_to_transposed) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = heads * key_dim * value_dim;
  if (index >= elements) return;
  const int64_t value = index % value_dim;
  const int64_t key = (index / value_dim) % key_dim;
  const int64_t head = index / (key_dim * value_dim);
  if (strict_to_transposed) {
    destination[(head * value_dim + value) * key_dim + key] = source[index];
  } else {
    destination[index] = source[(head * value_dim + value) * key_dim + key];
  }
}

__global__ void qwen4_exp_gdn_core_norm_gate_f32_kernel(
    float* __restrict__ core,
    const float* __restrict__ output_gate,
    const float* __restrict__ norm_weight,
    int64_t tokens,
    int64_t core_width,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  const int64_t token = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int lane = threadIdx.x;
  __shared__ float reductions[32];
  const int64_t base = token * core_width + v_head * head_v_dim;
  float partial = 0.0f;
  for (int64_t col = lane; col < head_v_dim; col += 32) {
    const float value = core[base + col];
    partial += value * value;
  }
  partial = warp_sum_f32(partial);
  if (lane == 0) reductions[0] = partial;
  __syncthreads();
  const float inv_rms = rsqrtf(
      reductions[0] / static_cast<float>(head_v_dim) + 1.0e-6f);
  for (int64_t col = lane; col < head_v_dim; col += 32) {
    core[base + col] = core[base + col] * inv_rms
        * norm_weight[col]
        * sigmoid_f32(output_gate[base + col]);
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_gdn_decode_f32(
    const void* conv,
    const void* output_gate,
    const void* alpha,
    const void* beta_logits,
    const void* dt_bias,
    const void* a,
    const void* norm_weight,
    void* recurrent_state,
    void* output,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!conv || !output_gate || !alpha || !beta_logits || !dt_bias || !a ||
      !norm_weight || !recurrent_state || !output || num_k_heads <= 0 ||
      num_v_heads <= 0 || num_v_heads % num_k_heads || head_k_dim <= 0 ||
      head_v_dim <= 0 || head_v_dim > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(2 * head_k_dim + 2 * threads + head_v_dim)
      * sizeof(float);
  hipLaunchKernelGGL(
      qwen4_exp_gdn_decode_f32_kernel,
      dim3(static_cast<uint32_t>(num_v_heads)), dim3(threads), shared, stream,
      static_cast<const float*>(conv), static_cast<const float*>(output_gate),
      static_cast<const float*>(alpha), static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a),
      static_cast<const float*>(norm_weight), static_cast<float*>(recurrent_state),
      static_cast<float*>(output), num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gdn_state_transpose_f32(
    const void* source, void* destination, int64_t heads, int64_t key_dim,
    int64_t value_dim, int32_t strict_to_transposed, hipStream_t stream) {
  if (!source || !destination || heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
      (strict_to_transposed != 0 && strict_to_transposed != 1)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = heads * key_dim * value_dim;
  hipLaunchKernelGGL(
      qwen4_exp_gdn_state_transpose_f32_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads)),
      dim3(threads), 0, stream, static_cast<const float*>(source),
      static_cast<float*>(destination), heads, key_dim, value_dim,
      strict_to_transposed != 0);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gdn_prefill_prepare_f32(
    const void* conv,
    const void* alpha,
    const void* beta_logits,
    const void* dt_bias,
    const void* a,
    void* query,
    void* key,
    void* value,
    void* beta,
    void* decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!conv || !alpha || !beta_logits || !dt_bias || !a || !query || !key ||
      !value || !beta || !decay || tokens <= 0 || num_k_heads <= 0 ||
      num_v_heads <= 0 || num_v_heads % num_k_heads || head_k_dim <= 0 ||
      head_v_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 128;
  const int64_t grid_heads = num_k_heads > num_v_heads ? num_k_heads : num_v_heads;
  hipLaunchKernelGGL(
      qwen4_exp_gdn_prefill_prepare_f32_kernel,
      dim3(static_cast<uint32_t>(tokens), static_cast<uint32_t>(grid_heads)),
      dim3(threads), 2 * threads * sizeof(float), stream,
      static_cast<const float*>(conv), static_cast<const float*>(alpha),
      static_cast<const float*>(beta_logits), static_cast<const float*>(dt_bias),
      static_cast<const float*>(a), static_cast<float*>(query),
      static_cast<float*>(key), static_cast<float*>(value),
      static_cast<float*>(beta), static_cast<float*>(decay), tokens, num_k_heads,
      num_v_heads, head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gdn_prefill_sigmoid_gate_f32(
    void* core,
    const void* output_gate,
    const void* norm_weight,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!core || !output_gate || !norm_weight || tokens <= 0 ||
      num_v_heads <= 0 || head_v_dim <= 0 || head_v_dim > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 128;
  hipLaunchKernelGGL(
      qwen4_exp_gdn_prefill_sigmoid_gate_f32_kernel,
      dim3(static_cast<uint32_t>(tokens * num_v_heads)), dim3(threads),
      threads * sizeof(float), stream, static_cast<float*>(core),
      static_cast<const float*>(output_gate),
      static_cast<const float*>(norm_weight), tokens, num_v_heads, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gdn_prefill_f32(
    const void* conv,
    const void* output_gate,
    const void* alpha,
    const void* beta_logits,
    const void* dt_bias,
    const void* a,
    const void* norm_weight,
    void* recurrent_state,
    void* output,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!conv || !output_gate || !alpha || !beta_logits || !dt_bias || !a ||
      !norm_weight || !recurrent_state || !output || tokens <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads ||
      head_k_dim <= 0 || head_v_dim <= 0 || head_v_dim > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 128;
  const size_t shared = static_cast<size_t>(2 * head_k_dim + 2 * threads + head_v_dim)
      * sizeof(float);
  hipLaunchKernelGGL(
      qwen4_exp_gdn_prefill_f32_kernel,
      dim3(static_cast<uint32_t>(num_v_heads)), dim3(threads), shared, stream,
      static_cast<const float*>(conv), static_cast<const float*>(output_gate),
      static_cast<const float*>(alpha), static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a),
      static_cast<const float*>(norm_weight), static_cast<float*>(recurrent_state),
      static_cast<float*>(output), tokens, num_k_heads, num_v_heads,
      head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_qwen4_exp_gdn_prefill_columnwarps_f32(
    const void* conv,
    const void* output_gate,
    const void* alpha,
    const void* beta_logits,
    const void* dt_bias,
    const void* a,
    const void* norm_weight,
    void* recurrent_state,
    void* core,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!conv || !output_gate || !alpha || !beta_logits || !dt_bias || !a ||
      !norm_weight || !recurrent_state || !core ||
      tokens <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      num_v_heads % num_k_heads || head_k_dim != 128 || head_v_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t col_blocks = head_v_dim / 4;  // 4 columns (warps) per block
  hipLaunchKernelGGL(
      (qwen4_exp_gdn_prefill_columnwarps_f32_kernel<4>),
      dim3(static_cast<uint32_t>(num_v_heads),
           static_cast<uint32_t>(col_blocks)),
      dim3(128), 0, stream,
      static_cast<const float*>(conv),
      static_cast<const float*>(alpha),
      static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias),
      static_cast<const float*>(a),
      static_cast<float*>(recurrent_state),
      static_cast<float*>(core),
      tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  const int err = static_cast<int>(hipGetLastError());
  if (err != 0) return err;
  const int64_t core_width = num_v_heads * head_v_dim;
  hipLaunchKernelGGL(
      qwen4_exp_gdn_core_norm_gate_f32_kernel,
      dim3(static_cast<uint32_t>(tokens),
           static_cast<uint32_t>(num_v_heads)),
      dim3(32), 0, stream,
      static_cast<float*>(core),
      static_cast<const float*>(output_gate),
      static_cast<const float*>(norm_weight),
      tokens, core_width, num_v_heads, head_v_dim);
  return static_cast<int>(hipGetLastError());
}
