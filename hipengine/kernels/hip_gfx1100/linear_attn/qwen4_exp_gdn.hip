// hipEngine-native strict Qwen4Exp GDN decode recurrence with sigmoid output gate.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdint.h>
#include <type_traits>

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

__device__ inline void exact_norm128_wave_tail(float* values) {
  if (threadIdx.x < 32) {
    const int t = threadIdx.x;
    float sum = (values[t] + values[t+64]) + (values[t+32] + values[t+96]);
    for (int stride = 16; stride > 0; stride >>= 1)
      sum += __shfl_down(sum, stride, 32);
    if (t == 0) values[0] = sum;
  }
  __syncthreads();
}

template <bool REGISTER_STATE = false, bool WAVE_NORM = false,
          bool NARROW_INDEX = false>
__global__ void __launch_bounds__(REGISTER_STATE ? 128 : 1024)
qwen4_exp_gdn_prefill_f32_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ output_gate,
    const float* __restrict__ alpha,
    const float* __restrict__ beta_logits,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ norm_weight,
    float* __restrict__ recurrent_state,
    float* __restrict__ output,
    std::conditional_t<NARROW_INDEX, int32_t, int64_t> tokens,
    std::conditional_t<NARROW_INDEX, int32_t, int64_t> num_k_heads,
    std::conditional_t<NARROW_INDEX, int32_t, int64_t> num_v_heads,
    std::conditional_t<NARROW_INDEX, int32_t, int64_t> head_k_dim,
    std::conditional_t<NARROW_INDEX, int32_t, int64_t> head_v_dim) {
  using index_t = std::conditional_t<NARROW_INDEX, int32_t, int64_t>;
  if constexpr (REGISTER_STATE) {
    head_k_dim = 128;
    head_v_dim = 128;
  }
  extern __shared__ float shared[];
  float* query = shared;
  float* key = query + head_k_dim;
  float* reductions = key + head_k_dim;
  float* core = reductions + blockDim.x;
  float* norm_reductions = core + head_v_dim;

  const index_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const index_t k_head = v_head % num_k_heads;
  const index_t key_width = num_k_heads * head_k_dim;
  const index_t qkv_width = 2 * key_width + num_v_heads * head_v_dim;
  const index_t q_offset = k_head * head_k_dim;
  const index_t k_offset = key_width + q_offset;
  const index_t v_offset = 2 * key_width + v_head * head_v_dim;
  const index_t core_width = num_v_heads * head_v_dim;
  const index_t value_index = threadIdx.x;
  const index_t state_col = v_head * head_k_dim * head_v_dim + value_index;

  float saved_state[REGISTER_STATE ? 128 : 1];
  if constexpr (REGISTER_STATE) {
#pragma unroll
    for (int column = 0; column < 128; ++column)
      saved_state[column] = recurrent_state[state_col + column * 128];
  }
  for (index_t token = 0; token < tokens; ++token) {
    const float* conv_row = conv + token * qkv_width;
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;
    for (index_t column = threadIdx.x; column < head_k_dim; column += blockDim.x) {
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
    if constexpr (WAVE_NORM) {
      exact_norm128_wave_tail(reductions);
      exact_norm128_wave_tail(norm_reductions);
    } else {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
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
      if constexpr (REGISTER_STATE) {
#pragma unroll
        for (int column = 0; column < 128; ++column)
          memory += key[column] * k_scale * saved_state[column] * decay;
        const float delta = (conv_row[v_offset + value_index] - memory) * beta;
#pragma unroll
        for (int column = 0; column < 128; ++column) {
          float next = saved_state[column] * decay + key[column] * k_scale * delta;
          // Preserve the parent's FP32 state-store boundary before Q accumulation.
          asm volatile("" : "+v"(next));
          saved_state[column] = next;
          core_value += query[column] * q_scale * next;
        }
      } else {
        for (index_t column = 0; column < head_k_dim; ++column) {
          memory += key[column] * k_scale
              * recurrent_state[state_col + column * head_v_dim] * decay;
        }
        const float delta = (conv_row[v_offset + value_index] - memory) * beta;
        for (index_t column = 0; column < head_k_dim; ++column) {
          const index_t state_index = state_col + column * head_v_dim;
          const float next = recurrent_state[state_index] * decay
              + key[column] * k_scale * delta;
          recurrent_state[state_index] = next;
          core_value += query[column] * q_scale * next;
        }
      }
      core[value_index] = core_value;
      norm_reductions[threadIdx.x] = core_value * core_value;
    } else {
      norm_reductions[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if constexpr (WAVE_NORM) {
      exact_norm128_wave_tail(norm_reductions);
    } else {
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_reductions[threadIdx.x] += norm_reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    }
    if (value_index < head_v_dim) {
      const index_t index = v_head * head_v_dim + value_index;
      const float inv_rms = rsqrtf(
          norm_reductions[0] / static_cast<float>(head_v_dim) + 1.0e-6f);
      output[token * core_width + index] = core[value_index] * inv_rms
          * norm_weight[value_index]
          * sigmoid_f32(output_gate[token * core_width + index]);
    }
    __syncthreads();
  }
  if constexpr (REGISTER_STATE) {
#pragma unroll
    for (int column = 0; column < 128; ++column)
      recurrent_state[state_col + column * 128] = saved_state[column];
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
#if defined(HIPENGINE_QWEN4_GDN_DPP) && defined(__gfx1151__)
  // pwilkin/llama.cpp@964c6f2, gated_delta_net.cu: preserve XOR16/8/4/2/1
  // pairing while avoiding the LDS crossbar used by shuffle instructions.
  value += __uint_as_float(__builtin_amdgcn_permlanex16(
      __float_as_uint(value), __float_as_uint(value),
      0x76543210, 0xFEDCBA98, true, false));
  value += __uint_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_uint(value), 0x168, 0xf, 0xf, true));
  value += __uint_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_uint(value), 0x164, 0xf, 0xf, true));
  value += __uint_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_uint(value), 0x162, 0xf, 0xf, true));
  value += __uint_as_float(__builtin_amdgcn_update_dpp(
      0, __float_as_uint(value), 0x161, 0xf, 0xf, true));
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
#endif
  return value;
}

template <int WARPS_PER_BLOCK, int ROWS_PER_LANE>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK == 4 ? 4 : 1)
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
  const int64_t col = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;
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

// PF-5 tile-16 candidate adapted from halo-box PR11 a7ad7b7f
// gated_delta_net_kda_tiled_128_cuda. Unlike the external source, this body
// preserves the in-tree columnwarp arithmetic exactly: one warp computes each
// token's Q/K norms with the same r=0..3 and XOR-shuffle order, then all 16
// column warps reuse those exact scalars and raw Q/K values from LDS. Each
// column's recurrent state remains in registers for the complete sequence.
template <int TOKEN_TILE, int WARPS_PER_BLOCK, int ROWS_PER_LANE, int COLS = 1>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
qwen4_exp_gdn_prefill_tiled_f32_kernel(
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
  constexpr int HEAD_DIM = 128;
  __shared__ float q_shared[TOKEN_TILE][HEAD_DIM];
  __shared__ float k_shared[TOKEN_TILE][HEAD_DIM];
  __shared__ float q_scale_shared[TOKEN_TILE];
  __shared__ float k_scale_shared[TOKEN_TILE];
  __shared__ float beta_shared[TOKEN_TILE];
  __shared__ float decay_shared[TOKEN_TILE];

  const int64_t v_head = blockIdx.x;
  const int64_t k_head = v_head % num_k_heads;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t col = (static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK
      + warp_id) * COLS;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_width + num_v_heads * head_v_dim;
  const int64_t q_offset = k_head * head_k_dim;
  const int64_t k_offset = key_width + q_offset;
  const int64_t v_offset = 2 * key_width + v_head * head_v_dim;
  const int64_t core_width = num_v_heads * head_v_dim;
  const int64_t state_col = v_head * head_k_dim * head_v_dim + col;

  float s_shard[COLS][ROWS_PER_LANE];
  int j_idx[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    j_idx[r] = r * 32 + lane;
    for (int c = 0; c < COLS; ++c) {
      s_shard[c][r] = recurrent_state[state_col + c + j_idx[r] * head_v_dim];
    }
  }

  for (int64_t token_base = 0; token_base < tokens;
       token_base += TOKEN_TILE) {
    const int64_t remaining = tokens - token_base;
    const int tile_size = static_cast<int>(
        remaining < TOKEN_TILE ? remaining : TOKEN_TILE);
    const int tile_elements = tile_size * HEAD_DIM;
    for (int index = threadIdx.x; index < tile_elements;
         index += WARPS_PER_BLOCK * 32) {
      const int tile_token = index / HEAD_DIM;
      const int feature = index - tile_token * HEAD_DIM;
      const float* conv_row = conv + (token_base + tile_token) * qkv_width;
      q_shared[tile_token][feature] = conv_row[q_offset + feature];
      k_shared[tile_token][feature] = conv_row[k_offset + feature];
    }
    __syncthreads();

    for (int norm_token = warp_id; norm_token < tile_size; norm_token += WARPS_PER_BLOCK) {
      float q_partial = 0.0f;
      float k_partial = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        const float q_value = q_shared[norm_token][j_idx[r]];
        const float k_value = k_shared[norm_token][j_idx[r]];
        q_partial += q_value * q_value;
        k_partial += k_value * k_value;
      }
      const float q_scale = rsqrtf(warp_sum_f32(q_partial) + 1.0e-6f)
          * rsqrtf(static_cast<float>(head_k_dim));
      const float k_scale = rsqrtf(warp_sum_f32(k_partial) + 1.0e-6f);
      if (lane == 0) {
        const int64_t token = token_base + norm_token;
        q_scale_shared[norm_token] = q_scale;
        k_scale_shared[norm_token] = k_scale;
        beta_shared[norm_token] = sigmoid_f32(
            beta_logits[token * num_v_heads + v_head]);
        decay_shared[norm_token] = expf(
            a[v_head] * softplus_f32(
                alpha[token * num_v_heads + v_head] + dt_bias[v_head]));
      }
    }
    __syncthreads();

    for (int tile_token = 0; tile_token < tile_size; ++tile_token) {
      float q_r[ROWS_PER_LANE];
      float k_r[ROWS_PER_LANE];
#pragma unroll
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        q_r[r] = q_shared[tile_token][j_idx[r]];
        k_r[r] = k_shared[tile_token][j_idx[r]];
      }
      const float q_scale = q_scale_shared[tile_token];
      const float k_scale = k_scale_shared[tile_token];
      const float beta = beta_shared[tile_token];
      const float decay = decay_shared[tile_token];

#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        float kv_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
          kv_partial += (k_r[r] * k_scale) * s_shard[c][r];
        }
        const float memory = warp_sum_f32(kv_partial) * decay;
        const float* conv_row = conv + (token_base + tile_token) * qkv_width;
        const float delta = (conv_row[v_offset + col + c] - memory) * beta;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
          s_shard[c][r] = s_shard[c][r] * decay + (k_r[r] * k_scale) * delta;
          attn_partial += (q_r[r] * q_scale) * s_shard[c][r];
        }
        const float attn = warp_sum_f32(attn_partial);
        core_out[(token_base + tile_token) * core_width
            + v_head * head_v_dim + col + c] = attn;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    for (int c = 0; c < COLS; ++c) {
      recurrent_state[state_col + c + j_idx[r] * head_v_dim] = s_shard[c][r];
    }
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

__global__ void qwen4_exp_gdn_decode_transposed_f32_kernel(
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ core, int64_t num_k_heads, int64_t num_v_heads,
    int64_t head_k_dim, int64_t head_v_dim) {
  const int64_t v_head = blockIdx.x;
  const int64_t value_index = blockIdx.y;
  const int lane = threadIdx.x;
  if (v_head >= num_v_heads || value_index >= head_v_dim) return;
  const int64_t k_head = v_head % num_k_heads;
  const int64_t state_base = (v_head * head_v_dim + value_index) * head_k_dim;
  float memory = 0.0f;
  for (int64_t column = lane; column < head_k_dim; column += 32) {
    memory += key[k_head * head_k_dim + column]
        * state[state_base + column] * decay[v_head];
  }
  for (int offset = 16; offset > 0; offset >>= 1) memory += __shfl_down(memory, offset, 32);
  float delta = 0.0f;
  if (lane == 0) delta = (value[v_head * head_v_dim + value_index] - memory) * beta[v_head];
  delta = __shfl(delta, 0, 32);
  float result = 0.0f;
  for (int64_t column = lane; column < head_k_dim; column += 32) {
    const float next = state[state_base + column] * decay[v_head]
        + key[k_head * head_k_dim + column] * delta;
    state[state_base + column] = next;
    result += query[k_head * head_k_dim + column] * next;
  }
  for (int offset = 16; offset > 0; offset >>= 1) result += __shfl_down(result, offset, 32);
  if (lane == 0) {
    core[v_head * head_v_dim + value_index] = result
        * rsqrtf(static_cast<float>(head_k_dim));
  }
}

__global__ void qwen4_exp_gdn_state_transpose_inplace_f32_kernel(
    float* state, int64_t matrices, int64_t dimension) {
  const int64_t matrix = blockIdx.y;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = index / dimension;
  const int64_t col = index % dimension;
  if (matrix >= matrices || row >= dimension || col <= row) return;
  const int64_t base = matrix * dimension * dimension;
  const float value = state[base + row * dimension + col];
  state[base + row * dimension + col] = state[base + col * dimension + row];
  state[base + col * dimension + row] = value;
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

extern "C" int hipengine_qwen4_exp_gdn_state_transpose_inplace_f32(
    void* state, int64_t matrices, int64_t dimension, hipStream_t stream) {
  if (!state || matrices <= 0 || dimension <= 0) return static_cast<int>(hipErrorInvalidValue);
  constexpr int threads = 256;
  const int64_t elements = dimension * dimension;
  hipLaunchKernelGGL(qwen4_exp_gdn_state_transpose_inplace_f32_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads), static_cast<uint32_t>(matrices)),
      dim3(threads), 0, stream, static_cast<float*>(state), matrices, dimension);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gdn_decode_transposed_f32(
    const void* query, const void* key, const void* value, const void* beta,
    const void* decay, void* state, void* core, int64_t num_k_heads,
    int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim,
    hipStream_t stream) {
  if (!query || !key || !value || !beta || !decay || !state || !core ||
      num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads ||
      head_k_dim != 128 || head_v_dim != 128) return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(qwen4_exp_gdn_decode_transposed_f32_kernel,
      dim3(static_cast<uint32_t>(num_v_heads), static_cast<uint32_t>(head_v_dim)),
      dim3(32), 0, stream, static_cast<const float*>(query), static_cast<const float*>(key),
      static_cast<const float*>(value), static_cast<const float*>(beta), static_cast<const float*>(decay),
      static_cast<float*>(state), static_cast<float*>(core), num_k_heads, num_v_heads,
      head_k_dim, head_v_dim);
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
      (qwen4_exp_gdn_prefill_f32_kernel<false>),
      dim3(static_cast<uint32_t>(num_v_heads)), dim3(threads), shared, stream,
      static_cast<const float*>(conv), static_cast<const float*>(output_gate),
      static_cast<const float*>(alpha), static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a),
      static_cast<const float*>(norm_weight), static_cast<float*>(recurrent_state),
      static_cast<float*>(output), tokens, num_k_heads, num_v_heads,
      head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}


template <bool WAVE_NORM, bool NARROW_INDEX = false>
int launch_register_gdn(
    const void* conv, const void* output_gate, const void* alpha,
    const void* beta_logits, const void* dt_bias, const void* a,
    const void* norm_weight, void* recurrent_state, void* output,
    int64_t tokens, int64_t num_k_heads, int64_t num_v_heads,
    int64_t head_k_dim, int64_t head_v_dim, hipStream_t stream) {
  if (!conv || !output_gate || !alpha || !beta_logits || !dt_bias || !a ||
      !norm_weight || !recurrent_state || !output || tokens <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads ||
      head_k_dim != 128 || head_v_dim != 128)
    return static_cast<int>(hipErrorInvalidValue);
  if constexpr (NARROW_INDEX) {
    // Division-based admission avoids overflowing even adversarial int64 ABI
    // dimensions. Every element count (not just each dimension) must fit.
    if (num_v_heads > INT32_MAX / (128 * 128))
      return static_cast<int>(hipErrorInvalidValue);
    const int64_t qkv_width = (2 * num_k_heads + num_v_heads) * 128;
    if (tokens > INT32_MAX / qkv_width)
      return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL((qwen4_exp_gdn_prefill_f32_kernel<true, WAVE_NORM, NARROW_INDEX>),
      dim3(static_cast<uint32_t>(num_v_heads)), dim3(128), 640 * sizeof(float), stream,
      static_cast<const float*>(conv), static_cast<const float*>(output_gate),
      static_cast<const float*>(alpha), static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a),
      static_cast<const float*>(norm_weight), static_cast<float*>(recurrent_state),
      static_cast<float*>(output), tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

#define REGISTER_GDN_ENTRY(NAME, WAVE) \
extern "C" int NAME(const void* c, const void* gate, const void* alpha, \
    const void* beta, const void* dt, const void* a, const void* norm, void* state, void* out, \
    int64_t t, int64_t hk, int64_t hv, int64_t dk, int64_t dv, hipStream_t s) { \
  return launch_register_gdn<WAVE>(c,gate,alpha,beta,dt,a,norm,state,out,t,hk,hv,dk,dv,s); \
}
REGISTER_GDN_ENTRY(hipengine_qwen4_exp_gdn_register_prefill_f32, false)
REGISTER_GDN_ENTRY(hipengine_qwen4_exp_gdn_wave_norm_prefill_f32, true)

extern "C" int hipengine_qwen4_exp_gdn_wave_norm_narrow_prefill_f32(
    const void* c, const void* gate, const void* alpha, const void* beta,
    const void* dt, const void* a, const void* norm, void* state, void* out,
    int64_t t, int64_t hk, int64_t hv, int64_t dk, int64_t dv, hipStream_t s) {
  return launch_register_gdn<true, true>(c, gate, alpha, beta, dt, a, norm,
                                         state, out, t, hk, hv, dk, dv, s);
}
#undef REGISTER_GDN_ENTRY


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
      (qwen4_exp_gdn_prefill_columnwarps_f32_kernel<4, 4>),
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

extern "C" int hipengine_qwen4_exp_gdn_prefill_tiled16_f32(
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
      !norm_weight || !recurrent_state || !core || tokens < 16 ||
      num_k_heads != 16 || (num_v_heads != 32 && num_v_heads != 48) ||
      head_k_dim != 128 || head_v_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int token_tile = 16;
#ifdef HIPENGINE_QWEN4_GDN_MULTI_COLUMN
  constexpr int warps_per_block = 8;
  constexpr int cols = 8;
#else
  constexpr int warps_per_block = 16;
  constexpr int cols = 1;
#endif
  constexpr int rows_per_lane = 4;
  const int64_t col_blocks = head_v_dim / (warps_per_block * cols);
  hipLaunchKernelGGL(
      (qwen4_exp_gdn_prefill_tiled_f32_kernel<
          token_tile, warps_per_block, rows_per_lane, cols>),
      dim3(static_cast<uint32_t>(num_v_heads),
           static_cast<uint32_t>(col_blocks)),
      dim3(warps_per_block * 32), 0, stream,
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

// 32-warp/block GDN prefill (PF-5 M6 port, lever 1; halo-box PR11
// gated_delta_net.cu geometry): same column-warp layout and per-warp
// arithmetic as the <4, 4> columnwarps owner, but 32 columns (warps) per
// block (1024 threads), so one block covers a whole 128-wide head. State
// rows stay sharded across 32 lanes in registers with warp-shuffle
// reductions, so the accumulation order is unchanged from the columnwarps
// owner (T0); only the block composition changes.
extern "C" int hipengine_qwen4_exp_gdn_prefill_w32_f32(
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
  const int64_t col_blocks = head_v_dim / 32;  // 32 columns (warps) per block
  hipLaunchKernelGGL(
      (qwen4_exp_gdn_prefill_columnwarps_f32_kernel<32, 4>),
      dim3(static_cast<uint32_t>(num_v_heads),
           static_cast<uint32_t>(col_blocks)),
      dim3(1024), 0, stream,
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


// ---------------------------------------------------------------------------
// Chunked triangular / Woodbury-Young GDN prefill (GD-4).
//
// The UD-Q4_K_XL production lane's recurrence - the wave-norm register route,
// qwen4_exp_gdn_prefill_f32_kernel<true, true> - holds 128 floats of state
// per thread and compiles at VGPR_Count 256, the hardware maximum, which
// permits only 2 waves per SIMD of a 64-wave-per-CU maximum (~12.5%
// occupancy).  It is latency-bound by register-file capacity, confirmed by
// hardware counter.
//
// This body keeps the 128x32 state tile in LDS instead and processes eight
// tokens per step through chunk-level products, so each thread holds 16 state
// elements rather than 128.  The arithmetic is the direct lower-triangular
// form, evaluated in FP32 operands:
//
//   P[t]  = prod_{u<=t} decay[u]                      (prefix decay in chunk)
//   R[t,s]= prod_{u=s+1..t} decay[u],  R[t,t] = 1     (transition)
//   delta[t] = beta[t] * ( v[t] - P[t]*(k_t . H0)
//                          - sum_{s<t} R[t,s]*(k_t . k_s)*delta[s] )
//   out[t]   = P[t]*(q_t . H0) + sum_{s<=t} R[t,s]*(q_t . k_s)*delta[s]
//   H1       = P[C-1]*H0 + sum_s R[C-1,s] * outer(k_s, delta[s])
//
// Measured outcome (GD-4): correct but slower than every route that matters -
// 1.15-1.22x behind the non-multi tiled16 route, 1.15-1.26x behind the
// wave-norm route at 1K/2K, and 1.75-2.26x behind the multi-column tiled16
// route (85 VGPR) that already escapes the register bound.  See
// benchmarks/results/2026-10-01-qwen4exp-gd4-wy-kernel-built/.
//
// FP32 operands, not fp16: the state-drift screen measures this form's fp32
// drift at 0.65-0.80x the serial chain's OWN fp32 drift, while fp16 operands
// cost 1.3e-7 NMSE and chunking does not remove that.  See
// benchmarks/results/2026-09-30-qwen4exp-gd4-wy-chunk-drift/.
//
// Grid: (num_v_heads, head_v_dim / VTILE).  Block: 256 threads.
// The state tile is owned exclusively by one block, so the write-back is
// race-free.  A chunk shorter than CHUNK pads with beta=0 and decay=1, which
// makes delta zero and P[C-1] equal to the true final prefix decay.
// ---------------------------------------------------------------------------
#define QWEN4EXP_WY_CHUNK 8
#define QWEN4EXP_WY_VTILE 32
#define QWEN4EXP_WY_THREADS 256
#define QWEN4EXP_WY_HEAD 128

__device__ inline float qwen4_exp_wy_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

__global__ void __launch_bounds__(QWEN4EXP_WY_THREADS, 2)
qwen4_exp_gdn_prefill_wy_f32_kernel(
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
  constexpr int CHUNK = QWEN4EXP_WY_CHUNK;
  constexpr int VTILE = QWEN4EXP_WY_VTILE;
  constexpr int HEAD = QWEN4EXP_WY_HEAD;

  __shared__ float s_state[HEAD][VTILE];
  __shared__ float s_q[CHUNK][HEAD];
  __shared__ float s_k[CHUNK][HEAD];
  __shared__ float s_qscale[CHUNK];
  __shared__ float s_kscale[CHUNK];
  __shared__ float s_beta[CHUNK];
  __shared__ float s_decay[CHUNK];
  __shared__ float s_p[CHUNK];
  __shared__ float s_r[CHUNK][CHUNK];
  __shared__ float s_keykey[CHUNK][CHUNK];
  __shared__ float s_querykey[CHUNK][CHUNK];
  __shared__ float s_keystate[CHUNK][VTILE];
  __shared__ float s_querystate[CHUNK][VTILE];
  __shared__ float s_delta[CHUNK][VTILE];

  const int64_t v_head = blockIdx.x;
  const int64_t tile = blockIdx.y;
  if (v_head >= num_v_heads || tile * VTILE >= head_v_dim) {
    return;
  }
  const int64_t k_head = v_head % num_k_heads;
  const int64_t key_width = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_width + num_v_heads * head_v_dim;
  const int64_t q_offset = k_head * head_k_dim;
  const int64_t k_offset = key_width + q_offset;
  const int64_t v_base = v_head * head_v_dim + tile * VTILE;
  const int64_t v_offset = 2 * key_width + v_base;
  const int64_t core_width = num_v_heads * head_v_dim;
  const int64_t state_base = v_head * head_k_dim * head_v_dim;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int index = tid; index < HEAD * VTILE; index += QWEN4EXP_WY_THREADS) {
    const int row = index / VTILE;
    const int col = index - row * VTILE;
    s_state[row][col] = recurrent_state[state_base + row * head_v_dim
                                        + tile * VTILE + col];
  }
  __syncthreads();

  for (int64_t chunk_start = 0; chunk_start < tokens; chunk_start += CHUNK) {
    const int active = static_cast<int>(
        (tokens - chunk_start) < CHUNK ? (tokens - chunk_start) : CHUNK);

    // --- one warp per chunk token: stage raw Q/K, reduce the norms ---
    if (warp < CHUNK) {
      if (warp < active) {
        const float* conv_row = conv + (chunk_start + warp) * qkv_width;
        float q_sumsq = 0.0f;
        float k_sumsq = 0.0f;
        for (int feature = lane; feature < HEAD; feature += 32) {
          const float q_value = conv_row[q_offset + feature];
          const float k_value = conv_row[k_offset + feature];
          s_q[warp][feature] = q_value;
          s_k[warp][feature] = k_value;
          q_sumsq += q_value * q_value;
          k_sumsq += k_value * k_value;
        }
        q_sumsq = qwen4_exp_wy_warp_sum(q_sumsq);
        k_sumsq = qwen4_exp_wy_warp_sum(k_sumsq);
        if (lane == 0) {
          // Same scales the production kernel applies.
          s_qscale[warp] = rsqrtf(q_sumsq + 1.0e-6f)
              * rsqrtf(static_cast<float>(head_k_dim));
          s_kscale[warp] = rsqrtf(k_sumsq + 1.0e-6f);
          const int64_t scalar = (chunk_start + warp) * num_v_heads + v_head;
          s_beta[warp] = sigmoid_f32(beta_logits[scalar]);
          s_decay[warp] = expf(
              a[v_head] * softplus_f32(alpha[scalar] + dt_bias[v_head]));
        }
      } else {
        for (int feature = lane; feature < HEAD; feature += 32) {
          s_q[warp][feature] = 0.0f;
          s_k[warp][feature] = 0.0f;
        }
        if (lane == 0) {
          s_qscale[warp] = 0.0f;
          s_kscale[warp] = 0.0f;
          s_beta[warp] = 0.0f;   // delta stays zero for padded tokens
          s_decay[warp] = 1.0f;  // so P[CHUNK-1] is the true final prefix
        }
      }
    }
    __syncthreads();

    for (int index = tid; index < CHUNK * HEAD; index += QWEN4EXP_WY_THREADS) {
      const int token = index / HEAD;
      const int feature = index - token * HEAD;
      s_q[token][feature] *= s_qscale[token];
      s_k[token][feature] *= s_kscale[token];
    }
    if (tid == 0) {
      float product = 1.0f;
      for (int token = 0; token < CHUNK; ++token) {
        product *= s_decay[token];
        s_p[token] = product;
      }
    }
    if (tid < CHUNK * CHUNK) {
      const int target = tid / CHUNK;
      const int source = tid - target * CHUNK;
      if (source == target) {
        s_r[target][source] = 1.0f;
      } else if (source < target) {
        float product = 1.0f;
        for (int step = source + 1; step <= target; ++step) {
          product *= s_decay[step];
        }
        s_r[target][source] = product;
      } else {
        s_r[target][source] = 0.0f;
      }
    }
    __syncthreads();

    // --- intra-chunk coefficients and the token/state projections ---
    if (tid < CHUNK * CHUNK) {
      const int target = tid / CHUNK;
      const int source = tid - target * CHUNK;
      float key_key = 0.0f;
      float query_key = 0.0f;
      for (int feature = 0; feature < HEAD; ++feature) {
        key_key += s_k[target][feature] * s_k[source][feature];
        query_key += s_q[target][feature] * s_k[source][feature];
      }
      s_keykey[target][source] = key_key;
      s_querykey[target][source] = query_key;
    }
    {
      const int token = tid / VTILE;
      const int col = tid - token * VTILE;
      float key_state = 0.0f;
      float query_state = 0.0f;
      for (int row = 0; row < HEAD; ++row) {
        const float state = s_state[row][col];
        key_state += s_k[token][row] * state;
        query_state += s_q[token][row] * state;
      }
      s_keystate[token][col] = key_state;
      s_querystate[token][col] = query_state;
    }
    __syncthreads();

    // --- forward substitution for delta (32 value lanes, sequential in t) ---
    if (tid < VTILE) {
      for (int target = 0; target < CHUNK; ++target) {
        float correction = 0.0f;
        for (int source = 0; source < target; ++source) {
          correction += s_r[target][source] * s_keykey[target][source]
              * s_delta[source][tid];
        }
        const float value = (target < active)
            ? conv[(chunk_start + target) * qkv_width + v_offset + tid]
            : 0.0f;
        s_delta[target][tid] = s_beta[target]
            * (value - s_p[target] * s_keystate[target][tid] - correction);
      }
    }
    __syncthreads();

    // --- outputs, then the chunk-final state ---
    {
      const int token = tid / VTILE;
      const int col = tid - token * VTILE;
      float accumulator = s_p[token] * s_querystate[token][col];
      for (int source = 0; source <= token; ++source) {
        accumulator += s_r[token][source] * s_querykey[token][source]
            * s_delta[source][col];
      }
      if (token < active) {
        core_out[(chunk_start + token) * core_width + v_base + col] = accumulator;
      }
    }
    __syncthreads();
    for (int index = tid; index < HEAD * VTILE; index += QWEN4EXP_WY_THREADS) {
      const int row = index / VTILE;
      const int col = index - row * VTILE;
      float accumulator = s_p[CHUNK - 1] * s_state[row][col];
      for (int source = 0; source < CHUNK; ++source) {
        accumulator += s_r[CHUNK - 1][source] * s_k[source][row]
            * s_delta[source][col];
      }
      s_state[row][col] = accumulator;
    }
    __syncthreads();
  }

  for (int index = tid; index < HEAD * VTILE; index += QWEN4EXP_WY_THREADS) {
    const int row = index / VTILE;
    const int col = index - row * VTILE;
    recurrent_state[state_base + row * head_v_dim + tile * VTILE + col] =
        s_state[row][col];
  }
}

extern "C" int hipengine_qwen4_exp_gdn_prefill_wy_f32(
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
      num_k_heads != 16 || (num_v_heads != 32 && num_v_heads != 48) ||
      head_k_dim != 128 || head_v_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<uint32_t>(num_v_heads),
                  static_cast<uint32_t>(head_v_dim / QWEN4EXP_WY_VTILE));
  hipLaunchKernelGGL(
      qwen4_exp_gdn_prefill_wy_f32_kernel,
      grid,
      dim3(QWEN4EXP_WY_THREADS), 0, stream,
      static_cast<const float*>(conv),
      static_cast<const float*>(alpha),
      static_cast<const float*>(beta_logits),
      static_cast<const float*>(dt_bias),
      static_cast<const float*>(a),
      static_cast<float*>(recurrent_state),
      static_cast<float*>(core),
      tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  // Same two-stage shape as the other prefill owners: the recurrence writes
  // raw core, then the norm/gate pass overwrites it in place.  Without this
  // the route is not a drop-in for the tiled16 interface.
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
