// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// W8A16 linear kernel bodies are preserved; wrappers are hipEngine C ABI.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

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

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

__global__ void shared_gate_sigmoid_fp32_kernel(
    const float* gate_logits,
    float* gate_values,
    int64_t tokens,
    int64_t gate_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= tokens) {
    return;
  }
  const int64_t offset = token * gate_stride;
  const float logit = gate_logits[offset];
  gate_values[offset] = 1.0f / (1.0f + expf(-logit));
}

// M12.2 fused multi-row variant: one block per vocab row, looping over all
// verifier rows so the weight row (5 KB at hidden_size=5120) streams from HBM
// only once per block and is L1-cached for the remaining verifier rows.  The
// stock ``w8a16_linear_kernel`` below uses grid ``(vocab, tokens)`` which
// re-streams the full weight matrix per verifier row; for B+1 = 4 rows this
// nearly quadruples weight HBM traffic vs the theoretical floor.  Verifier
// LM-head time dominates by 2.68 ms/pass at B=3 on gfx1100/W7900 (per the
// 2026-05-21 rocprof artifact) and L2 (4 MB) cannot retain the 780 MB matrix
// across verify-row streams.
__global__ void w8a16_linear_multi_row_kernel(
    const uint16_t* __restrict__ hidden_states,
    const int8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    float* __restrict__ out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t vocab_row = static_cast<int64_t>(blockIdx.x);
  if (vocab_row >= out_features) {
    return;
  }
  const int tid = threadIdx.x;
  const int64_t weight_row_off = vocab_row * hidden_size;
  const float scale = weight_scale[vocab_row];
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t hidden_row = t * hidden_size;
    float acc = 0.0f;
    for (int64_t k = static_cast<int64_t>(tid) * 8; k + 7 < hidden_size; k += vec_stride) {
      const int64_t h_off = hidden_row + k;
      const int64_t w_off = weight_row_off + k;
      acc += bf16_bits_to_float(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
      acc += bf16_bits_to_float(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
      acc += bf16_bits_to_float(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
      acc += bf16_bits_to_float(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
      acc += bf16_bits_to_float(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
      acc += bf16_bits_to_float(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
      acc += bf16_bits_to_float(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
      acc += bf16_bits_to_float(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
    }
    for (int64_t k = (hidden_size & ~7) + tid; k < hidden_size; k += blockDim.x) {
      acc += bf16_bits_to_float(hidden_states[hidden_row + k]) *
          static_cast<float>(static_cast<int>(weight[weight_row_off + k]));
    }
    partial[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[t * out_features + vocab_row] = partial[0] * scale;
    }
    __syncthreads();
  }
}

__global__ void w8a16_linear_kernel(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += bf16_bits_to_float(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += bf16_bits_to_float(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += bf16_bits_to_float(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += bf16_bits_to_float(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += bf16_bits_to_float(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += bf16_bits_to_float(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += bf16_bits_to_float(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += bf16_bits_to_float(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += bf16_bits_to_float(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


template <typename scalar_t>
__global__ void w8a16_linear_lowp_out_kernel(
    const scalar_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    scalar_t* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += scalar_to_float_qwen35(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += scalar_to_float_qwen35(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = static_cast<scalar_t>(partial[0] * weight_scale[row]);
  }
}


__global__ void w8a16_shared_gate_up_silu_fp16_kernel(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t hidden_size,
    int64_t intermediate_size) {
  extern __shared__ float partial[];
  float* gate_partials[4] = {partial, partial + blockDim.x, partial + 2 * blockDim.x, partial + 3 * blockDim.x};
  float* up_partials[4] = {
      partial + 4 * blockDim.x,
      partial + 5 * blockDim.x,
      partial + 6 * blockDim.x,
      partial + 7 * blockDim.x,
  };
  const int64_t j_base = static_cast<int64_t>(blockIdx.x) * 4;
  const int64_t token = blockIdx.y;
  if (j_base >= intermediate_size) {
    return;
  }

  bool active[4];
  int64_t gate_rows[4];
  int64_t up_rows[4];
  float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int64_t j = j_base + slot;
    active[slot] = j < intermediate_size;
    gate_rows[slot] = j * hidden_size;
    up_rows[slot] = (j + intermediate_size) * hidden_size;
  }

  const int64_t hidden_row = token * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const float x0 = static_cast<float>(hidden_states[h_off + 0]);
    const float x1 = static_cast<float>(hidden_states[h_off + 1]);
    const float x2 = static_cast<float>(hidden_states[h_off + 2]);
    const float x3 = static_cast<float>(hidden_states[h_off + 3]);
    const float x4 = static_cast<float>(hidden_states[h_off + 4]);
    const float x5 = static_cast<float>(hidden_states[h_off + 5]);
    const float x6 = static_cast<float>(hidden_states[h_off + 6]);
    const float x7 = static_cast<float>(hidden_states[h_off + 7]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t g_off = gate_rows[slot] + k;
        const int64_t u_off = up_rows[slot] + k;
        gate_acc[slot] += x0 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 0]));
        gate_acc[slot] += x1 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 1]));
        gate_acc[slot] += x2 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 2]));
        gate_acc[slot] += x3 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 3]));
        gate_acc[slot] += x4 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 4]));
        gate_acc[slot] += x5 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 5]));
        gate_acc[slot] += x6 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 6]));
        gate_acc[slot] += x7 * static_cast<float>(static_cast<int>(gate_up_weight[g_off + 7]));
        up_acc[slot] += x0 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 0]));
        up_acc[slot] += x1 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 1]));
        up_acc[slot] += x2 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 2]));
        up_acc[slot] += x3 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 3]));
        up_acc[slot] += x4 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 4]));
        up_acc[slot] += x5 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 5]));
        up_acc[slot] += x6 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 6]));
        up_acc[slot] += x7 * static_cast<float>(static_cast<int>(gate_up_weight[u_off + 7]));
      }
    }
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    const float x = static_cast<float>(hidden_states[hidden_row + k]);
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        gate_acc[slot] += x * static_cast<float>(static_cast<int>(gate_up_weight[gate_rows[slot] + k]));
        up_acc[slot] += x * static_cast<float>(static_cast<int>(gate_up_weight[up_rows[slot] + k]));
      }
    }
  }
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    gate_partials[slot][threadIdx.x] = gate_acc[slot];
    up_partials[slot][threadIdx.x] = up_acc[slot];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        gate_partials[slot][threadIdx.x] += gate_partials[slot][threadIdx.x + stride];
        up_partials[slot][threadIdx.x] += up_partials[slot][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int64_t out_offset = token * intermediate_size;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active[slot]) {
        const int64_t j = j_base + slot;
        const half_t gate_lowp = static_cast<half_t>(gate_partials[slot][0] * gate_up_scale[j]);
        const half_t up_lowp = static_cast<half_t>(up_partials[slot][0] * gate_up_scale[intermediate_size + j]);
        const float gate = static_cast<float>(gate_lowp);
        const float up = static_cast<float>(up_lowp);
        out[out_offset + j] = static_cast<half_t>(silu_f32(gate) * up);
      }
    }
  }
}


template <int TOKEN_TILE>
__global__ void w8a16_shared_gate_up_silu_fp16_token_tiled_kernel(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size) {
  extern __shared__ float partial[];
  const int64_t j_base = static_cast<int64_t>(blockIdx.x) * 4;
  const int64_t token_base = static_cast<int64_t>(blockIdx.y) * TOKEN_TILE;
  if (j_base >= intermediate_size || token_base >= tokens) {
    return;
  }

  bool active_slot[4];
  int64_t gate_rows[4];
  int64_t up_rows[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int64_t j = j_base + slot;
    active_slot[slot] = j < intermediate_size;
    gate_rows[slot] = j * hidden_size;
    up_rows[slot] = (j + intermediate_size) * hidden_size;
  }

  bool active_token[TOKEN_TILE];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
    active_token[t] = token_base + t < tokens;
  }

  float gate_acc[TOKEN_TILE][4];
  float up_acc[TOKEN_TILE][4];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      gate_acc[t][slot] = 0.0f;
      up_acc[t][slot] = 0.0f;
    }
  }

  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active_slot[slot]) {
        const int64_t g_off = gate_rows[slot] + k;
        const int64_t u_off = up_rows[slot] + k;
        const float wg0 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 0]));
        const float wg1 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 1]));
        const float wg2 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 2]));
        const float wg3 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 3]));
        const float wg4 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 4]));
        const float wg5 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 5]));
        const float wg6 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 6]));
        const float wg7 = static_cast<float>(static_cast<int>(gate_up_weight[g_off + 7]));
        const float wu0 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 0]));
        const float wu1 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 1]));
        const float wu2 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 2]));
        const float wu3 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 3]));
        const float wu4 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 4]));
        const float wu5 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 5]));
        const float wu6 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 6]));
        const float wu7 = static_cast<float>(static_cast<int>(gate_up_weight[u_off + 7]));
#pragma unroll
        for (int t = 0; t < TOKEN_TILE; ++t) {
          if (active_token[t]) {
            const int64_t h_off = (token_base + t) * hidden_size + k;
            const float x0 = static_cast<float>(hidden_states[h_off + 0]);
            const float x1 = static_cast<float>(hidden_states[h_off + 1]);
            const float x2 = static_cast<float>(hidden_states[h_off + 2]);
            const float x3 = static_cast<float>(hidden_states[h_off + 3]);
            const float x4 = static_cast<float>(hidden_states[h_off + 4]);
            const float x5 = static_cast<float>(hidden_states[h_off + 5]);
            const float x6 = static_cast<float>(hidden_states[h_off + 6]);
            const float x7 = static_cast<float>(hidden_states[h_off + 7]);
            gate_acc[t][slot] += x0 * wg0;
            gate_acc[t][slot] += x1 * wg1;
            gate_acc[t][slot] += x2 * wg2;
            gate_acc[t][slot] += x3 * wg3;
            gate_acc[t][slot] += x4 * wg4;
            gate_acc[t][slot] += x5 * wg5;
            gate_acc[t][slot] += x6 * wg6;
            gate_acc[t][slot] += x7 * wg7;
            up_acc[t][slot] += x0 * wu0;
            up_acc[t][slot] += x1 * wu1;
            up_acc[t][slot] += x2 * wu2;
            up_acc[t][slot] += x3 * wu3;
            up_acc[t][slot] += x4 * wu4;
            up_acc[t][slot] += x5 * wu5;
            up_acc[t][slot] += x6 * wu6;
            up_acc[t][slot] += x7 * wu7;
          }
        }
      }
    }
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      if (active_slot[slot]) {
        const float wg = static_cast<float>(static_cast<int>(gate_up_weight[gate_rows[slot] + k]));
        const float wu = static_cast<float>(static_cast<int>(gate_up_weight[up_rows[slot] + k]));
#pragma unroll
        for (int t = 0; t < TOKEN_TILE; ++t) {
          if (active_token[t]) {
            const float x = static_cast<float>(hidden_states[(token_base + t) * hidden_size + k]);
            gate_acc[t][slot] += x * wg;
            up_acc[t][slot] += x * wu;
          }
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      partial[(t * 8 + slot) * blockDim.x + threadIdx.x] = gate_acc[t][slot];
      partial[(t * 8 + 4 + slot) * blockDim.x + threadIdx.x] = up_acc[t][slot];
    }
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
          partial[(t * 8 + slot) * blockDim.x + threadIdx.x] +=
              partial[(t * 8 + slot) * blockDim.x + threadIdx.x + stride];
          partial[(t * 8 + 4 + slot) * blockDim.x + threadIdx.x] +=
              partial[(t * 8 + 4 + slot) * blockDim.x + threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t) {
      if (active_token[t]) {
        const int64_t out_offset = (token_base + t) * intermediate_size;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
          if (active_slot[slot]) {
            const int64_t j = j_base + slot;
            const float gate_accum = partial[(t * 8 + slot) * blockDim.x] * gate_up_scale[j];
            const float up_accum = partial[(t * 8 + 4 + slot) * blockDim.x] * gate_up_scale[intermediate_size + j];
            const half_t gate_lowp = static_cast<half_t>(gate_accum);
            const half_t up_lowp = static_cast<half_t>(up_accum);
            const float gate = static_cast<float>(gate_lowp);
            const float up = static_cast<float>(up_lowp);
            out[out_offset + j] = static_cast<half_t>(silu_f32(gate) * up);
          }
        }
      }
    }
  }
}


__global__ void w8a16_shared_down_combine_residual_fp16_kernel(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* shared_gate,
    const half_t* residual,
    half_t* out,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride) {
  extern __shared__ float partial[];
  float* partials[8] = {
      partial,
      partial + blockDim.x,
      partial + 2 * blockDim.x,
      partial + 3 * blockDim.x,
      partial + 4 * blockDim.x,
      partial + 5 * blockDim.x,
      partial + 6 * blockDim.x,
      partial + 7 * blockDim.x,
  };
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * 8;
  const int64_t token = blockIdx.y;
  if (row_base >= hidden_size) {
    return;
  }

  bool active[8];
  int64_t down_rows[8];
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int64_t row = row_base + slot;
    active[slot] = row < hidden_size;
    down_rows[slot] = row * intermediate_size;
  }

  const int64_t mid_offset = token * intermediate_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t j = threadIdx.x * 8; j + 7 < intermediate_size; j += vec_stride) {
    const int64_t m_off = mid_offset + j;
    const float m0 = static_cast<float>(shared_intermediate[m_off + 0]);
    const float m1 = static_cast<float>(shared_intermediate[m_off + 1]);
    const float m2 = static_cast<float>(shared_intermediate[m_off + 2]);
    const float m3 = static_cast<float>(shared_intermediate[m_off + 3]);
    const float m4 = static_cast<float>(shared_intermediate[m_off + 4]);
    const float m5 = static_cast<float>(shared_intermediate[m_off + 5]);
    const float m6 = static_cast<float>(shared_intermediate[m_off + 6]);
    const float m7 = static_cast<float>(shared_intermediate[m_off + 7]);
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (active[slot]) {
        const int64_t w_off = down_rows[slot] + j;
        acc[slot] += m0 * static_cast<float>(static_cast<int>(down_weight[w_off + 0]));
        acc[slot] += m1 * static_cast<float>(static_cast<int>(down_weight[w_off + 1]));
        acc[slot] += m2 * static_cast<float>(static_cast<int>(down_weight[w_off + 2]));
        acc[slot] += m3 * static_cast<float>(static_cast<int>(down_weight[w_off + 3]));
        acc[slot] += m4 * static_cast<float>(static_cast<int>(down_weight[w_off + 4]));
        acc[slot] += m5 * static_cast<float>(static_cast<int>(down_weight[w_off + 5]));
        acc[slot] += m6 * static_cast<float>(static_cast<int>(down_weight[w_off + 6]));
        acc[slot] += m7 * static_cast<float>(static_cast<int>(down_weight[w_off + 7]));
      }
    }
  }
  for (int64_t j = (intermediate_size & ~7) + threadIdx.x; j < intermediate_size; j += blockDim.x) {
    const float mid = static_cast<float>(shared_intermediate[mid_offset + j]);
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (active[slot]) {
        acc[slot] += mid * static_cast<float>(static_cast<int>(down_weight[down_rows[slot] + j]));
      }
    }
  }
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    partials[slot][threadIdx.x] = acc[slot];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int slot = 0; slot < 8; ++slot) {
        partials[slot][threadIdx.x] += partials[slot][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float gate = shared_gate[token * gate_stride];
    const int64_t out_offset = token * hidden_size;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (active[slot]) {
        const int64_t row = row_base + slot;
        const int64_t idx = out_offset + row;
        const half_t shared_lowp = static_cast<half_t>(partials[slot][0] * down_scale[row]);
        const float value = static_cast<float>(residual[idx]) + static_cast<float>(selected[idx]) + gate * static_cast<float>(shared_lowp);
        out[idx] = static_cast<half_t>(value);
      }
    }
  }
}


template <int TOKEN_TILE>
__global__ void w8a16_shared_down_combine_residual_fp16_token_tiled_kernel(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* shared_gate,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride) {
  extern __shared__ float partial[];
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * 8;
  const int64_t token_base = static_cast<int64_t>(blockIdx.y) * TOKEN_TILE;
  if (row_base >= hidden_size || token_base >= tokens) {
    return;
  }

  bool active_row[8];
  int64_t down_rows[8];
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int64_t row = row_base + slot;
    active_row[slot] = row < hidden_size;
    down_rows[slot] = row * intermediate_size;
  }

  bool active_token[TOKEN_TILE];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
    active_token[t] = token_base + t < tokens;
  }

  float acc[TOKEN_TILE][8];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      acc[t][slot] = 0.0f;
    }
  }

  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t j = threadIdx.x * 8; j + 7 < intermediate_size; j += vec_stride) {
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (active_row[slot]) {
        const int64_t w_off = down_rows[slot] + j;
        const float w0 = static_cast<float>(static_cast<int>(down_weight[w_off + 0]));
        const float w1 = static_cast<float>(static_cast<int>(down_weight[w_off + 1]));
        const float w2 = static_cast<float>(static_cast<int>(down_weight[w_off + 2]));
        const float w3 = static_cast<float>(static_cast<int>(down_weight[w_off + 3]));
        const float w4 = static_cast<float>(static_cast<int>(down_weight[w_off + 4]));
        const float w5 = static_cast<float>(static_cast<int>(down_weight[w_off + 5]));
        const float w6 = static_cast<float>(static_cast<int>(down_weight[w_off + 6]));
        const float w7 = static_cast<float>(static_cast<int>(down_weight[w_off + 7]));
#pragma unroll
        for (int t = 0; t < TOKEN_TILE; ++t) {
          if (active_token[t]) {
            const int64_t m_off = (token_base + t) * intermediate_size + j;
            const float m0 = static_cast<float>(shared_intermediate[m_off + 0]);
            const float m1 = static_cast<float>(shared_intermediate[m_off + 1]);
            const float m2 = static_cast<float>(shared_intermediate[m_off + 2]);
            const float m3 = static_cast<float>(shared_intermediate[m_off + 3]);
            const float m4 = static_cast<float>(shared_intermediate[m_off + 4]);
            const float m5 = static_cast<float>(shared_intermediate[m_off + 5]);
            const float m6 = static_cast<float>(shared_intermediate[m_off + 6]);
            const float m7 = static_cast<float>(shared_intermediate[m_off + 7]);
            acc[t][slot] += m0 * w0;
            acc[t][slot] += m1 * w1;
            acc[t][slot] += m2 * w2;
            acc[t][slot] += m3 * w3;
            acc[t][slot] += m4 * w4;
            acc[t][slot] += m5 * w5;
            acc[t][slot] += m6 * w6;
            acc[t][slot] += m7 * w7;
          }
        }
      }
    }
  }
  for (int64_t j = (intermediate_size & ~7) + threadIdx.x; j < intermediate_size; j += blockDim.x) {
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      if (active_row[slot]) {
        const float w = static_cast<float>(static_cast<int>(down_weight[down_rows[slot] + j]));
#pragma unroll
        for (int t = 0; t < TOKEN_TILE; ++t) {
          if (active_token[t]) {
            const float mid = static_cast<float>(shared_intermediate[(token_base + t) * intermediate_size + j]);
            acc[t][slot] += mid * w;
          }
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
      partial[(t * 8 + slot) * blockDim.x + threadIdx.x] = acc[t][slot];
    }
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int t = 0; t < TOKEN_TILE; ++t) {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
          partial[(t * 8 + slot) * blockDim.x + threadIdx.x] +=
              partial[(t * 8 + slot) * blockDim.x + threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t) {
      if (active_token[t]) {
        const int64_t token = token_base + t;
        const float gate = shared_gate[token * gate_stride];
        const int64_t out_offset = token * hidden_size;
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
          if (active_row[slot]) {
            const int64_t row = row_base + slot;
            const int64_t idx = out_offset + row;
            const half_t shared_lowp = static_cast<half_t>(partial[(t * 8 + slot) * blockDim.x] * down_scale[row]);
            const float value = static_cast<float>(residual[idx]) + static_cast<float>(selected[idx]) + gate * static_cast<float>(shared_lowp);
            out[idx] = static_cast<half_t>(value);
          }
        }
      }
    }
  }
}


__global__ void w8a16_linear_f32_kernel(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += hidden_states[h_off + 0] * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += hidden_states[h_off + 1] * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += hidden_states[h_off + 2] * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += hidden_states[h_off + 3] * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += hidden_states[h_off + 4] * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += hidden_states[h_off + 5] * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += hidden_states[h_off + 6] * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += hidden_states[h_off + 7] * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += hidden_states[hidden_row + k] * static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_w8a16_linear_bf16_f32_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

// M12.2: weight-sharing variant that amortizes the LM-head weight reads
// across multiple verifier rows (one block per vocab row instead of one
// block per (vocab, verifier_row) pair).  Returns the same FP32 logits as
// ``hipengine_w8a16_linear_bf16_f32_out`` so the downstream argmax kernel
// is unchanged; the runner gates this path via
// ``HIPENGINE_W8A16_LM_HEAD_MULTI_ROW``.  For ``tokens=1`` the stock
// per-(vocab, row) kernel is preferred (single-row pattern is already
// optimal for the AR decode path).
extern "C" int hipengine_w8a16_linear_bf16_f32_multi_row(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_multi_row_kernel,
      dim3(static_cast<unsigned int>(out_features)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      tokens,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_bf16_lowp_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<hip_bfloat16>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      reinterpret_cast<const hip_bfloat16*>(hidden_states),
      weight,
      weight_scale,
      reinterpret_cast<hip_bfloat16*>(out),
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_fp16_lowp_out(
    const half_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_gate_up_silu_fp16(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_shared_gate_up_silu_fp16_kernel,
      dim3(static_cast<unsigned int>((intermediate_size + 3) / 4), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(8 * threads) * sizeof(float),
      stream,
      hidden_states,
      gate_up_weight,
      gate_up_scale,
      out,
      hidden_size,
      intermediate_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_gate_up_silu_fp16_token_tile2(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2>),
      dim3(
          static_cast<unsigned int>((intermediate_size + 3) / 4),
          static_cast<unsigned int>((tokens + 1) / 2)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(16 * threads) * sizeof(float),
      stream,
      hidden_states,
      gate_up_weight,
      gate_up_scale,
      out,
      tokens,
      hidden_size,
      intermediate_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_gate_up_silu_fp16_token_tile4(
    const half_t* hidden_states,
    const int8_t* gate_up_weight,
    const float* gate_up_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<4>),
      dim3(
          static_cast<unsigned int>((intermediate_size + 3) / 4),
          static_cast<unsigned int>((tokens + 3) / 4)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(32 * threads) * sizeof(float),
      stream,
      hidden_states,
      gate_up_weight,
      gate_up_scale,
      out,
      tokens,
      hidden_size,
      intermediate_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_gate_sigmoid_fp32(
    const float* gate_logits,
    float* gate_values,
    int64_t tokens,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      shared_gate_sigmoid_fp32_kernel,
      dim3(static_cast<unsigned int>((tokens + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate_logits,
      gate_values,
      tokens,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_down_combine_residual_fp16(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* shared_gate,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_shared_down_combine_residual_fp16_kernel,
      dim3(static_cast<unsigned int>((hidden_size + 7) / 8), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(8 * threads) * sizeof(float),
      stream,
      shared_intermediate,
      down_weight,
      down_scale,
      selected,
      shared_gate,
      residual,
      out,
      hidden_size,
      intermediate_size,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_down_combine_residual_fp16_token_tile2(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* shared_gate,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2>),
      dim3(
          static_cast<unsigned int>((hidden_size + 7) / 8),
          static_cast<unsigned int>((tokens + 1) / 2)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(16 * threads) * sizeof(float),
      stream,
      shared_intermediate,
      down_weight,
      down_scale,
      selected,
      shared_gate,
      residual,
      out,
      tokens,
      hidden_size,
      intermediate_size,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_shared_down_combine_residual_fp16_token_tile4(
    const half_t* shared_intermediate,
    const int8_t* down_weight,
    const float* down_scale,
    const half_t* selected,
    const float* shared_gate,
    const half_t* residual,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 || gate_stride <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<4>),
      dim3(
          static_cast<unsigned int>((hidden_size + 7) / 8),
          static_cast<unsigned int>((tokens + 3) / 4)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(32 * threads) * sizeof(float),
      stream,
      shared_intermediate,
      down_weight,
      down_scale,
      selected,
      shared_gate,
      residual,
      out,
      tokens,
      hidden_size,
      intermediate_size,
      gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_f32_f32_out(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_f32_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}
