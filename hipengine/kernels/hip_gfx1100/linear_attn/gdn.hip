// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip.
// Baseline GDN decode/prefill kernels track @ 59195ed; DFlash tree t-loop
// variants track @ b95eaa5. Wrappers are hipEngine raw-pointer C ABI.

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <stdint.h>
#include <type_traits>

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

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ inline float softplus_f32(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

// Block-wide sum via warp shuffles, with a tiny LDS combine only when the block
// spans multiple wavefronts.  For a single-wavefront block (nthreads ==
// warpSize) this is pure shuffle: no __syncthreads, no LDS.  Butterfly xor so
// every lane returns the full block sum and can use it directly.  Sized for
// blocks up to 2 wavefronts (the chain GDN uses 32 or 64 threads).
__device__ inline float block_all_reduce_sum(float v, float* __restrict__ lds, int tid, int nthreads) {
  for (int off = warpSize >> 1; off > 0; off >>= 1) {
    v += __shfl_xor(v, off);
  }
  const int num_warps = nthreads / warpSize;
  if (num_warps <= 1) {
    return v;
  }
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  if (lane == 0) {
    lds[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    if (w < num_warps) total += lds[w];
  }
  __syncthreads();
  return total;
}

inline bool gdn_tloop_c1_exact_enabled() {
  const char* raw = std::getenv("HIPENGINE_GDN_TLOOP_C1_EXACT");
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  return raw[0] == '1' || raw[0] == 't' || raw[0] == 'T' ||
         raw[0] == 'y' || raw[0] == 'Y' || raw[0] == 'o' || raw[0] == 'O';
}


template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

// Recurrent-state storage may be fp32 (strict) or fp16 (production fp16-state
// route, HIPENGINE_GGUF_FP16_RECURRENT_STATE).  Reads always promote to fp32
// for accumulation; writes round the fp32 value back to the storage dtype with
// RNE.  The fp32 instantiation is the identity path, so strict kernels stay
// byte-identical to the legacy float* bodies.  The helpers deduce the pointer's
// cv-qualification so kernels that alias the state through `volatile` (e.g.
// the prefill/segments token loops) keep their exact volatile read/write
// semantics.
template <typename state_t>
__device__ inline float gdn_state_read_qwen35(const state_t* state, int64_t offset) {
  using plain_t = typename std::remove_cv<state_t>::type;
  return scalar_to_float_qwen35<plain_t>(state[offset]);
}

template <typename state_t>
__device__ inline void gdn_state_write_qwen35(state_t* state, int64_t offset, float value) {
  using plain_t = typename std::remove_cv<state_t>::type;
  state[offset] = static_cast<plain_t>(value);
}

__device__ inline int64_t qwen35_gdn_k_head_from_v_head(
    int64_t v_head,
    int64_t num_k_heads,
    int64_t num_v_heads) {
#if defined(HIPENGINE_GDN_GROUPED_HEADS)
  // Canonical HF/PARO tensors group contiguous V heads under each K head.
  return v_head / (num_v_heads / num_k_heads);
#else
  // llama.cpp GGUF conversion reorders V-head tensors to tiled order so GGML's
  // interleaved broadcast maps [v0, v1, ...] to [k0, k1, ..., k0, k1, ...].
  return v_head % num_k_heads;
#endif
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename scalar_t, bool write_bf16 = false, typename state_t = float>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    state_t* recurrent_state,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_offset = 2 * key_dim + v_head * head_v_dim;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  const int64_t q_base = k_head * head_k_dim;
  const int64_t k_base = key_dim + q_base;
  for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
    const float q0 = conv_out[q_base + idx + 0];
    const float q1 = conv_out[q_base + idx + 1];
    const float q2 = conv_out[q_base + idx + 2];
    const float q3 = conv_out[q_base + idx + 3];
    const float q4 = conv_out[q_base + idx + 4];
    const float q5 = conv_out[q_base + idx + 5];
    const float q6 = conv_out[q_base + idx + 6];
    const float q7 = conv_out[q_base + idx + 7];
    const float k0 = conv_out[k_base + idx + 0];
    const float k1 = conv_out[k_base + idx + 1];
    const float k2 = conv_out[k_base + idx + 2];
    const float k3 = conv_out[k_base + idx + 3];
    const float k4 = conv_out[k_base + idx + 4];
    const float k5 = conv_out[k_base + idx + 5];
    const float k6 = conv_out[k_base + idx + 6];
    const float k7 = conv_out[k_base + idx + 7];
    q_shared[idx + 0] = q0;
    q_shared[idx + 1] = q1;
    q_shared[idx + 2] = q2;
    q_shared[idx + 3] = q3;
    q_shared[idx + 4] = q4;
    q_shared[idx + 5] = q5;
    q_shared[idx + 6] = q6;
    q_shared[idx + 7] = q7;
    k_shared[idx + 0] = k0;
    k_shared[idx + 1] = k1;
    k_shared[idx + 2] = k2;
    k_shared[idx + 3] = k3;
    k_shared[idx + 4] = k4;
    k_shared[idx + 5] = k5;
    k_shared[idx + 6] = k6;
    k_shared[idx + 7] = k7;
    q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
    k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
  }
  for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    const float q_value = conv_out[q_base + idx];
    const float k_value = conv_out[k_base + idx];
    q_shared[idx] = q_value;
    k_shared[idx] = k_value;
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
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

  const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
  const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
  const float beta = sigmoid_f32(scalar_to_float_qwen35(b[v_head]));
  const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[v_head]) + dt_bias[v_head]));

  const int64_t value_idx = threadIdx.x;
  float out_acc = 0.0f;
  if (value_idx < head_v_dim) {
    const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
    float kv_mem = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      kv_mem += k_shared[idx + 0] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 0) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 1] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 1) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 2] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 2) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 3] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 3) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 4] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 4) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 5] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 5) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 6] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 6) * head_v_dim) * decay;
      kv_mem += k_shared[idx + 7] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 7) * head_v_dim) * decay;
    }
    for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
      const float k_norm = k_shared[idx] * k_scale;
      const float state_value = gdn_state_read_qwen35(recurrent_state, state_col + idx * head_v_dim) * decay;
      kv_mem += k_norm * state_value;
    }

    const float value = conv_out[value_offset + value_idx];
    const float delta = (value - kv_mem) * beta;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      const float q_norm0 = q_shared[idx + 0] * q_scale;
      const float q_norm1 = q_shared[idx + 1] * q_scale;
      const float q_norm2 = q_shared[idx + 2] * q_scale;
      const float q_norm3 = q_shared[idx + 3] * q_scale;
      const float q_norm4 = q_shared[idx + 4] * q_scale;
      const float q_norm5 = q_shared[idx + 5] * q_scale;
      const float q_norm6 = q_shared[idx + 6] * q_scale;
      const float q_norm7 = q_shared[idx + 7] * q_scale;
      const float k_norm0 = k_shared[idx + 0] * k_scale;
      const float k_norm1 = k_shared[idx + 1] * k_scale;
      const float k_norm2 = k_shared[idx + 2] * k_scale;
      const float k_norm3 = k_shared[idx + 3] * k_scale;
      const float k_norm4 = k_shared[idx + 4] * k_scale;
      const float k_norm5 = k_shared[idx + 5] * k_scale;
      const float k_norm6 = k_shared[idx + 6] * k_scale;
      const float k_norm7 = k_shared[idx + 7] * k_scale;
      const int64_t so0 = state_col + (idx + 0) * head_v_dim;
      const int64_t so1 = state_col + (idx + 1) * head_v_dim;
      const int64_t so2 = state_col + (idx + 2) * head_v_dim;
      const int64_t so3 = state_col + (idx + 3) * head_v_dim;
      const int64_t so4 = state_col + (idx + 4) * head_v_dim;
      const int64_t so5 = state_col + (idx + 5) * head_v_dim;
      const int64_t so6 = state_col + (idx + 6) * head_v_dim;
      const int64_t so7 = state_col + (idx + 7) * head_v_dim;
      const float ns0 = gdn_state_read_qwen35(recurrent_state, so0) * decay + k_norm0 * delta;
      const float ns1 = gdn_state_read_qwen35(recurrent_state, so1) * decay + k_norm1 * delta;
      const float ns2 = gdn_state_read_qwen35(recurrent_state, so2) * decay + k_norm2 * delta;
      const float ns3 = gdn_state_read_qwen35(recurrent_state, so3) * decay + k_norm3 * delta;
      const float ns4 = gdn_state_read_qwen35(recurrent_state, so4) * decay + k_norm4 * delta;
      const float ns5 = gdn_state_read_qwen35(recurrent_state, so5) * decay + k_norm5 * delta;
      const float ns6 = gdn_state_read_qwen35(recurrent_state, so6) * decay + k_norm6 * delta;
      const float ns7 = gdn_state_read_qwen35(recurrent_state, so7) * decay + k_norm7 * delta;
      gdn_state_write_qwen35(recurrent_state, so0, ns0);
      gdn_state_write_qwen35(recurrent_state, so1, ns1);
      gdn_state_write_qwen35(recurrent_state, so2, ns2);
      gdn_state_write_qwen35(recurrent_state, so3, ns3);
      gdn_state_write_qwen35(recurrent_state, so4, ns4);
      gdn_state_write_qwen35(recurrent_state, so5, ns5);
      gdn_state_write_qwen35(recurrent_state, so6, ns6);
      gdn_state_write_qwen35(recurrent_state, so7, ns7);
      out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
               + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
    }
    for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
      const float q_norm = q_shared[idx] * q_scale;
      const float k_norm = k_shared[idx] * k_scale;
      const int64_t state_offset = state_col + idx * head_v_dim;
      const float new_state = gdn_state_read_qwen35(recurrent_state, state_offset) * decay + k_norm * delta;
      gdn_state_write_qwen35(recurrent_state, state_offset, new_state);
      out_acc += q_norm * new_state;
    }
    out_shared[value_idx] = out_acc;
    norm_partial[threadIdx.x] = out_acc * out_acc;
  } else {
    norm_partial[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (value_idx < head_v_dim) {
    const int64_t pos = v_head * head_v_dim + value_idx;
    const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
    const float value = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
        silu_f32(scalar_to_float_qwen35(gate[pos]));
    out[pos] = value;
    if constexpr (write_bf16) {
      out_bf16[pos] = float_to_bf16_bits(value);
    }
  }
}

// Packed AR advances one token per active row.  Keep this sibling separate from
// the arbitrary-length segmented kernel so the compiler sees the same single-
// token recurrence as the c1 kernel while state ownership remains indexed.
template <typename scalar_t, int STATE_CACHE_ROWS = 0, typename state_t = float>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    state_t* recurrent_state,
    float* out,
    const int64_t* __restrict__ state_indices,
    int64_t rows,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;
  float* state_cache = norm_partial + blockDim.x;

  const int64_t row = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (row >= rows || v_head >= num_v_heads) {
    return;
  }
  const int64_t state_slot = state_indices[row];
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t conv_base = row * qkv_width;
  const int64_t q_base = conv_base + k_head * head_k_dim;
  const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;
  const int64_t value_offset = conv_base + 2 * key_dim + v_head * head_v_dim;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
    const float q0 = conv_out[q_base + idx + 0];
    const float q1 = conv_out[q_base + idx + 1];
    const float q2 = conv_out[q_base + idx + 2];
    const float q3 = conv_out[q_base + idx + 3];
    const float q4 = conv_out[q_base + idx + 4];
    const float q5 = conv_out[q_base + idx + 5];
    const float q6 = conv_out[q_base + idx + 6];
    const float q7 = conv_out[q_base + idx + 7];
    const float k0 = conv_out[k_base + idx + 0];
    const float k1 = conv_out[k_base + idx + 1];
    const float k2 = conv_out[k_base + idx + 2];
    const float k3 = conv_out[k_base + idx + 3];
    const float k4 = conv_out[k_base + idx + 4];
    const float k5 = conv_out[k_base + idx + 5];
    const float k6 = conv_out[k_base + idx + 6];
    const float k7 = conv_out[k_base + idx + 7];
    q_shared[idx + 0] = q0;
    q_shared[idx + 1] = q1;
    q_shared[idx + 2] = q2;
    q_shared[idx + 3] = q3;
    q_shared[idx + 4] = q4;
    q_shared[idx + 5] = q5;
    q_shared[idx + 6] = q6;
    q_shared[idx + 7] = q7;
    k_shared[idx + 0] = k0;
    k_shared[idx + 1] = k1;
    k_shared[idx + 2] = k2;
    k_shared[idx + 3] = k3;
    k_shared[idx + 4] = k4;
    k_shared[idx + 5] = k5;
    k_shared[idx + 6] = k6;
    k_shared[idx + 7] = k7;
    q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
    k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
  }
  for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    const float q_value = conv_out[q_base + idx];
    const float k_value = conv_out[k_base + idx];
    q_shared[idx] = q_value;
    k_shared[idx] = k_value;
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
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

  const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
  const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
  const int64_t scalar_idx = row * num_v_heads + v_head;
  const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
  const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

  const int64_t value_idx = threadIdx.x;
  float out_acc = 0.0f;
  if (value_idx < head_v_dim) {
    const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
    const int64_t state_col = state_slot * state_stride + v_head * head_k_dim * head_v_dim + value_idx;
    float kv_mem = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      const int64_t so0 = state_col + (idx + 0) * head_v_dim;
      const int64_t so1 = state_col + (idx + 1) * head_v_dim;
      const int64_t so2 = state_col + (idx + 2) * head_v_dim;
      const int64_t so3 = state_col + (idx + 3) * head_v_dim;
      const int64_t so4 = state_col + (idx + 4) * head_v_dim;
      const int64_t so5 = state_col + (idx + 5) * head_v_dim;
      const int64_t so6 = state_col + (idx + 6) * head_v_dim;
      const int64_t so7 = state_col + (idx + 7) * head_v_dim;
      const float state0 = gdn_state_read_qwen35(recurrent_state, so0);
      const float state1 = gdn_state_read_qwen35(recurrent_state, so1);
      const float state2 = gdn_state_read_qwen35(recurrent_state, so2);
      const float state3 = gdn_state_read_qwen35(recurrent_state, so3);
      const float state4 = gdn_state_read_qwen35(recurrent_state, so4);
      const float state5 = gdn_state_read_qwen35(recurrent_state, so5);
      const float state6 = gdn_state_read_qwen35(recurrent_state, so6);
      const float state7 = gdn_state_read_qwen35(recurrent_state, so7);
      if constexpr (STATE_CACHE_ROWS > 0) {
        if (idx < STATE_CACHE_ROWS) {
          state_cache[(idx + 0) * head_v_dim + value_idx] = state0;
          state_cache[(idx + 1) * head_v_dim + value_idx] = state1;
          state_cache[(idx + 2) * head_v_dim + value_idx] = state2;
          state_cache[(idx + 3) * head_v_dim + value_idx] = state3;
          state_cache[(idx + 4) * head_v_dim + value_idx] = state4;
          state_cache[(idx + 5) * head_v_dim + value_idx] = state5;
          state_cache[(idx + 6) * head_v_dim + value_idx] = state6;
          state_cache[(idx + 7) * head_v_dim + value_idx] = state7;
        }
      }
      kv_mem += k_shared[idx + 0] * k_scale * state0 * decay;
      kv_mem += k_shared[idx + 1] * k_scale * state1 * decay;
      kv_mem += k_shared[idx + 2] * k_scale * state2 * decay;
      kv_mem += k_shared[idx + 3] * k_scale * state3 * decay;
      kv_mem += k_shared[idx + 4] * k_scale * state4 * decay;
      kv_mem += k_shared[idx + 5] * k_scale * state5 * decay;
      kv_mem += k_shared[idx + 6] * k_scale * state6 * decay;
      kv_mem += k_shared[idx + 7] * k_scale * state7 * decay;
    }
    for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
      const float k_norm = k_shared[idx] * k_scale;
      const int64_t state_offset = state_col + idx * head_v_dim;
      const float state_value = gdn_state_read_qwen35(recurrent_state, state_offset);
      if constexpr (STATE_CACHE_ROWS > 0) {
        if (idx < STATE_CACHE_ROWS) {
          state_cache[idx * head_v_dim + value_idx] = state_value;
        }
      }
      kv_mem += k_norm * state_value * decay;
    }

    const float value = conv_out[value_offset + value_idx];
    const float delta = (value - kv_mem) * beta;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      const float q_norm0 = q_shared[idx + 0] * q_scale;
      const float q_norm1 = q_shared[idx + 1] * q_scale;
      const float q_norm2 = q_shared[idx + 2] * q_scale;
      const float q_norm3 = q_shared[idx + 3] * q_scale;
      const float q_norm4 = q_shared[idx + 4] * q_scale;
      const float q_norm5 = q_shared[idx + 5] * q_scale;
      const float q_norm6 = q_shared[idx + 6] * q_scale;
      const float q_norm7 = q_shared[idx + 7] * q_scale;
      const float k_norm0 = k_shared[idx + 0] * k_scale;
      const float k_norm1 = k_shared[idx + 1] * k_scale;
      const float k_norm2 = k_shared[idx + 2] * k_scale;
      const float k_norm3 = k_shared[idx + 3] * k_scale;
      const float k_norm4 = k_shared[idx + 4] * k_scale;
      const float k_norm5 = k_shared[idx + 5] * k_scale;
      const float k_norm6 = k_shared[idx + 6] * k_scale;
      const float k_norm7 = k_shared[idx + 7] * k_scale;
      const int64_t so0 = state_col + (idx + 0) * head_v_dim;
      const int64_t so1 = state_col + (idx + 1) * head_v_dim;
      const int64_t so2 = state_col + (idx + 2) * head_v_dim;
      const int64_t so3 = state_col + (idx + 3) * head_v_dim;
      const int64_t so4 = state_col + (idx + 4) * head_v_dim;
      const int64_t so5 = state_col + (idx + 5) * head_v_dim;
      const int64_t so6 = state_col + (idx + 6) * head_v_dim;
      const int64_t so7 = state_col + (idx + 7) * head_v_dim;
      float state0;
      float state1;
      float state2;
      float state3;
      float state4;
      float state5;
      float state6;
      float state7;
      if constexpr (STATE_CACHE_ROWS > 0) {
        if (idx < STATE_CACHE_ROWS) {
          state0 = state_cache[(idx + 0) * head_v_dim + value_idx];
          state1 = state_cache[(idx + 1) * head_v_dim + value_idx];
          state2 = state_cache[(idx + 2) * head_v_dim + value_idx];
          state3 = state_cache[(idx + 3) * head_v_dim + value_idx];
          state4 = state_cache[(idx + 4) * head_v_dim + value_idx];
          state5 = state_cache[(idx + 5) * head_v_dim + value_idx];
          state6 = state_cache[(idx + 6) * head_v_dim + value_idx];
          state7 = state_cache[(idx + 7) * head_v_dim + value_idx];
        } else {
          state0 = gdn_state_read_qwen35(recurrent_state, so0);
          state1 = gdn_state_read_qwen35(recurrent_state, so1);
          state2 = gdn_state_read_qwen35(recurrent_state, so2);
          state3 = gdn_state_read_qwen35(recurrent_state, so3);
          state4 = gdn_state_read_qwen35(recurrent_state, so4);
          state5 = gdn_state_read_qwen35(recurrent_state, so5);
          state6 = gdn_state_read_qwen35(recurrent_state, so6);
          state7 = gdn_state_read_qwen35(recurrent_state, so7);
        }
      } else {
        state0 = gdn_state_read_qwen35(recurrent_state, so0);
        state1 = gdn_state_read_qwen35(recurrent_state, so1);
        state2 = gdn_state_read_qwen35(recurrent_state, so2);
        state3 = gdn_state_read_qwen35(recurrent_state, so3);
        state4 = gdn_state_read_qwen35(recurrent_state, so4);
        state5 = gdn_state_read_qwen35(recurrent_state, so5);
        state6 = gdn_state_read_qwen35(recurrent_state, so6);
        state7 = gdn_state_read_qwen35(recurrent_state, so7);
      }
      const float ns0 = state0 * decay + k_norm0 * delta;
      const float ns1 = state1 * decay + k_norm1 * delta;
      const float ns2 = state2 * decay + k_norm2 * delta;
      const float ns3 = state3 * decay + k_norm3 * delta;
      const float ns4 = state4 * decay + k_norm4 * delta;
      const float ns5 = state5 * decay + k_norm5 * delta;
      const float ns6 = state6 * decay + k_norm6 * delta;
      const float ns7 = state7 * decay + k_norm7 * delta;
      gdn_state_write_qwen35(recurrent_state, so0, ns0);
      gdn_state_write_qwen35(recurrent_state, so1, ns1);
      gdn_state_write_qwen35(recurrent_state, so2, ns2);
      gdn_state_write_qwen35(recurrent_state, so3, ns3);
      gdn_state_write_qwen35(recurrent_state, so4, ns4);
      gdn_state_write_qwen35(recurrent_state, so5, ns5);
      gdn_state_write_qwen35(recurrent_state, so6, ns6);
      gdn_state_write_qwen35(recurrent_state, so7, ns7);
      out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
               + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
    }
    for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
      const float q_norm = q_shared[idx] * q_scale;
      const float k_norm = k_shared[idx] * k_scale;
      const int64_t state_offset = state_col + idx * head_v_dim;
      float state_value;
      if constexpr (STATE_CACHE_ROWS > 0) {
        state_value = idx < STATE_CACHE_ROWS
            ? state_cache[idx * head_v_dim + value_idx]
            : gdn_state_read_qwen35(recurrent_state, state_offset);
      } else {
        state_value = gdn_state_read_qwen35(recurrent_state, state_offset);
      }
      const float new_state = state_value * decay + k_norm * delta;
      gdn_state_write_qwen35(recurrent_state, state_offset, new_state);
      out_acc += q_norm * new_state;
    }
    out_shared[value_idx] = out_acc;
    norm_partial[threadIdx.x] = out_acc * out_acc;
  } else {
    norm_partial[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (value_idx < head_v_dim) {
    const int64_t pos = v_head * head_v_dim + value_idx;
    const int64_t out_pos = row * num_v_heads * head_v_dim + pos;
    const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
    out[out_pos] = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
        silu_f32(scalar_to_float_qwen35(gate[row * num_v_heads * head_v_dim + pos]));
  }
}

template <typename scalar_t, typename state_t = float, bool write_bf16 = false, bool write_snapshot = false>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const state_t* __restrict__ base_recurrent_state,
    state_t* __restrict__ tree_recurrent_state,
    state_t* __restrict__ initial_recurrent_state_snapshot,
    const int64_t* __restrict__ parent_ids,
    float* __restrict__ out,
    uint16_t* __restrict__ out_bf16,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t conv_stride = (2 * key_dim) + value_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_head_base = v_head * head_k_dim * head_v_dim;
  const float a_log_v = a_log[v_head];
  const float dt_bias_v = dt_bias[v_head];

  for (int64_t t = 0; t < max_nodes; ++t) {
    const int64_t parent = parent_ids == nullptr ? (t - 1) : parent_ids[t];
    const state_t* recurrent_state = parent < 0
        ? base_recurrent_state
        : tree_recurrent_state + parent * state_stride;
    state_t* node_state = tree_recurrent_state + t * state_stride;
    const int64_t conv_row = t * conv_stride;
    const int64_t q_base = conv_row + k_head * head_k_dim;
    const int64_t k_base = conv_row + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_row + 2 * key_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = conv_out[q_base + idx + 0];
      const float q1 = conv_out[q_base + idx + 1];
      const float q2 = conv_out[q_base + idx + 2];
      const float q3 = conv_out[q_base + idx + 3];
      const float q4 = conv_out[q_base + idx + 4];
      const float q5 = conv_out[q_base + idx + 5];
      const float q6 = conv_out[q_base + idx + 6];
      const float q7 = conv_out[q_base + idx + 7];
      const float k0 = conv_out[k_base + idx + 0];
      const float k1 = conv_out[k_base + idx + 1];
      const float k2 = conv_out[k_base + idx + 2];
      const float k3 = conv_out[k_base + idx + 3];
      const float k4 = conv_out[k_base + idx + 4];
      const float k5 = conv_out[k_base + idx + 5];
      const float k6 = conv_out[k_base + idx + 6];
      const float k7 = conv_out[k_base + idx + 7];
      q_shared[idx + 0] = q0;
      q_shared[idx + 1] = q1;
      q_shared[idx + 2] = q2;
      q_shared[idx + 3] = q3;
      q_shared[idx + 4] = q4;
      q_shared[idx + 5] = q5;
      q_shared[idx + 6] = q6;
      q_shared[idx + 7] = q7;
      k_shared[idx + 0] = k0;
      k_shared[idx + 1] = k1;
      k_shared[idx + 2] = k2;
      k_shared[idx + 3] = k3;
      k_shared[idx + 4] = k4;
      k_shared[idx + 5] = k5;
      k_shared[idx + 6] = k6;
      k_shared[idx + 7] = k7;
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_shared[idx] = q_value;
      k_shared[idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
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

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[t * num_v_heads + v_head]));
    const float decay = expf(-expf(a_log_v) * softplus_f32(scalar_to_float_qwen35(a[t * num_v_heads + v_head]) + dt_bias_v));

    const int64_t value_idx = threadIdx.x;
    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = state_head_base + value_idx;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 0) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 1] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 1) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 2] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 2) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 3] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 3) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 4] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 4) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 5] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 5) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 6] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 6) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 7] * k_scale * gdn_state_read_qwen35(recurrent_state, state_col + (idx + 7) * head_v_dim) * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = gdn_state_read_qwen35(recurrent_state, state_col + idx * head_v_dim) * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = conv_out[value_offset + value_idx];
      const float delta = (value - kv_mem) * beta;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float old0 = gdn_state_read_qwen35(recurrent_state, so0);
        const float old1 = gdn_state_read_qwen35(recurrent_state, so1);
        const float old2 = gdn_state_read_qwen35(recurrent_state, so2);
        const float old3 = gdn_state_read_qwen35(recurrent_state, so3);
        const float old4 = gdn_state_read_qwen35(recurrent_state, so4);
        const float old5 = gdn_state_read_qwen35(recurrent_state, so5);
        const float old6 = gdn_state_read_qwen35(recurrent_state, so6);
        const float old7 = gdn_state_read_qwen35(recurrent_state, so7);
        if constexpr (write_snapshot) {
          if (t == 0) {
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so0, old0);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so1, old1);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so2, old2);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so3, old3);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so4, old4);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so5, old5);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so6, old6);
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, so7, old7);
          }
        }
        const float ns0 = old0 * decay + k_norm0 * delta;
        const float ns1 = old1 * decay + k_norm1 * delta;
        const float ns2 = old2 * decay + k_norm2 * delta;
        const float ns3 = old3 * decay + k_norm3 * delta;
        const float ns4 = old4 * decay + k_norm4 * delta;
        const float ns5 = old5 * decay + k_norm5 * delta;
        const float ns6 = old6 * decay + k_norm6 * delta;
        const float ns7 = old7 * decay + k_norm7 * delta;
        gdn_state_write_qwen35(node_state, so0, ns0);
        gdn_state_write_qwen35(node_state, so1, ns1);
        gdn_state_write_qwen35(node_state, so2, ns2);
        gdn_state_write_qwen35(node_state, so3, ns3);
        gdn_state_write_qwen35(node_state, so4, ns4);
        gdn_state_write_qwen35(node_state, so5, ns5);
        gdn_state_write_qwen35(node_state, so6, ns6);
        gdn_state_write_qwen35(node_state, so7, ns7);
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float old_state = gdn_state_read_qwen35(recurrent_state, state_offset);
        if constexpr (write_snapshot) {
          if (t == 0) {
            gdn_state_write_qwen35(initial_recurrent_state_snapshot, state_offset, old_state);
          }
        }
        const float new_state = old_state * decay + k_norm * delta;
        gdn_state_write_qwen35(node_state, state_offset, new_state);
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = (t * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float value = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[pos]));
      out[pos] = value;
      if constexpr (write_bf16) {
        out_bf16[pos] = float_to_bf16_bits(value);
      }
    }
    __syncthreads();
  }
}


template <typename scalar_t, bool write_bf16 = false, bool write_snapshot = false>
__global__ void qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out_kernel(
    const uint16_t* __restrict__ norm,
    const float* __restrict__ alpha_weight,
    const float* __restrict__ beta_weight,
    uint16_t* __restrict__ alpha_out,
    uint16_t* __restrict__ beta_out,
    int64_t in_features,
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const float* __restrict__ base_recurrent_state,
    float* __restrict__ tree_recurrent_state,
    float* __restrict__ initial_recurrent_state_snapshot,
    const int64_t* __restrict__ parent_ids,
    float* __restrict__ out,
    uint16_t* __restrict__ out_bf16,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* projection_partial = shared;
  float* conv_shared = projection_partial + 8 * blockDim.x;
  float* q_partial = conv_shared + 4 * (2 * head_k_dim + head_v_dim);
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  __shared__ uint16_t alpha_shared[4];
  __shared__ uint16_t beta_shared[4];

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t conv_stride = (2 * key_dim) + value_dim;
  const int64_t staged_row_width = 2 * head_k_dim + head_v_dim;
  for (int64_t staged_idx = threadIdx.x;
       staged_idx < max_nodes * staged_row_width;
       staged_idx += blockDim.x) {
    const int64_t row = staged_idx / staged_row_width;
    const int64_t local_idx = staged_idx - row * staged_row_width;
    const int64_t conv_row = row * conv_stride;
    if (local_idx < head_k_dim) {
      conv_shared[staged_idx] =
          conv_out[conv_row + k_head * head_k_dim + local_idx];
    } else if (local_idx < 2 * head_k_dim) {
      conv_shared[staged_idx] = conv_out[
          conv_row + key_dim + k_head * head_k_dim + local_idx - head_k_dim];
    } else {
      conv_shared[staged_idx] = conv_out[
          conv_row + 2 * key_dim + v_head * head_v_dim +
          local_idx - 2 * head_k_dim];
    }
  }
  __syncthreads();

  // Reproduce both scalar local256 dense-F32 GEMV trees for every verifier row,
  // but load each alpha/beta weight once across the rows. The rounded BF16 bits
  // are written for surface parity and consumed directly by this same head.
  float alpha_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float beta_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  constexpr int kValuesPerThread = 8;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * kValuesPerThread;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  const int64_t weight_base = v_head * in_features;
  for (int64_t k_base = static_cast<int64_t>(threadIdx.x) * kValuesPerThread;
       k_base < main_end;
       k_base += vec_stride) {
    const float* __restrict__ alpha_w = alpha_weight + weight_base + k_base;
    const float* __restrict__ beta_w = beta_weight + weight_base + k_base;
#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
      if (local_row < max_nodes) {
        const uint16_t* __restrict__ x = norm +
            static_cast<int64_t>(local_row) * in_features + k_base;
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[0]), alpha_w[0], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[1]), alpha_w[1], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[2]), alpha_w[2], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[3]), alpha_w[3], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[4]), alpha_w[4], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[5]), alpha_w[5], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[6]), alpha_w[6], alpha_acc[local_row]);
        alpha_acc[local_row] = fmaf(bf16_bits_to_float(x[7]), alpha_w[7], alpha_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[0]), beta_w[0], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[1]), beta_w[1], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[2]), beta_w[2], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[3]), beta_w[3], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[4]), beta_w[4], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[5]), beta_w[5], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[6]), beta_w[6], beta_acc[local_row]);
        beta_acc[local_row] = fmaf(bf16_bits_to_float(x[7]), beta_w[7], beta_acc[local_row]);
      }
    }
  }
  for (int64_t in_col = main_end + threadIdx.x;
       in_col < in_features;
       in_col += blockDim.x) {
    const float alpha_w = alpha_weight[weight_base + in_col];
    const float beta_w = beta_weight[weight_base + in_col];
#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
      if (local_row < max_nodes) {
        const float x = bf16_bits_to_float(
            norm[static_cast<int64_t>(local_row) * in_features + in_col]);
        alpha_acc[local_row] = fmaf(x, alpha_w, alpha_acc[local_row]);
        beta_acc[local_row] = fmaf(x, beta_w, beta_acc[local_row]);
      }
    }
  }

#pragma unroll
  for (int local_row = 0; local_row < 4; ++local_row) {
    projection_partial[(2 * local_row + 0) * blockDim.x + threadIdx.x] =
        alpha_acc[local_row];
    projection_partial[(2 * local_row + 1) * blockDim.x + threadIdx.x] =
        beta_acc[local_row];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int local_row = 0; local_row < 4; ++local_row) {
        if (local_row < max_nodes) {
          projection_partial[(2 * local_row + 0) * blockDim.x + threadIdx.x] +=
              projection_partial[(2 * local_row + 0) * blockDim.x + threadIdx.x + stride];
          projection_partial[(2 * local_row + 1) * blockDim.x + threadIdx.x] +=
              projection_partial[(2 * local_row + 1) * blockDim.x + threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int local_row = 0; local_row < 4; ++local_row) {
      if (local_row < max_nodes) {
        const uint16_t alpha_bits = float_to_bf16_bits(
            projection_partial[(2 * local_row + 0) * blockDim.x]);
        const uint16_t beta_bits = float_to_bf16_bits(
            projection_partial[(2 * local_row + 1) * blockDim.x]);
        alpha_shared[local_row] = alpha_bits;
        beta_shared[local_row] = beta_bits;
        alpha_out[static_cast<int64_t>(local_row) * num_v_heads + v_head] = alpha_bits;
        beta_out[static_cast<int64_t>(local_row) * num_v_heads + v_head] = beta_bits;
      }
    }
  }
  __syncthreads();

  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_head_base = v_head * head_k_dim * head_v_dim;
  const float a_log_v = a_log[v_head];
  const float dt_bias_v = dt_bias[v_head];

  for (int64_t t = 0; t < max_nodes; ++t) {
    const int64_t parent = parent_ids == nullptr ? (t - 1) : parent_ids[t];
    const float* recurrent_state = parent < 0
        ? base_recurrent_state
        : tree_recurrent_state + parent * state_stride;
    float* node_state = tree_recurrent_state + t * state_stride;
    const float* __restrict__ staged_row = conv_shared + t * staged_row_width;
    const float* __restrict__ q_shared = staged_row;
    const float* __restrict__ k_shared = staged_row + head_k_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = q_shared[idx + 0];
      const float q1 = q_shared[idx + 1];
      const float q2 = q_shared[idx + 2];
      const float q3 = q_shared[idx + 3];
      const float q4 = q_shared[idx + 4];
      const float q5 = q_shared[idx + 5];
      const float q6 = q_shared[idx + 6];
      const float q7 = q_shared[idx + 7];
      const float k0 = k_shared[idx + 0];
      const float k1 = k_shared[idx + 1];
      const float k2 = k_shared[idx + 2];
      const float k3 = k_shared[idx + 3];
      const float k4 = k_shared[idx + 4];
      const float k5 = k_shared[idx + 5];
      const float k6 = k_shared[idx + 6];
      const float k7 = k_shared[idx + 7];
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = q_shared[idx];
      const float k_value = k_shared[idx];
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
    if (threadIdx.x < 128) {
      q_partial[threadIdx.x] = q_sum;
      k_partial[threadIdx.x] = k_sum;
    }
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        q_partial[threadIdx.x] += q_partial[threadIdx.x + stride];
        k_partial[threadIdx.x] += k_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const float beta = sigmoid_f32(bf16_bits_to_float(beta_shared[t]));
    const float decay = expf(-expf(a_log_v) *
        softplus_f32(bf16_bits_to_float(alpha_shared[t]) + dt_bias_v));

    const int64_t value_idx = threadIdx.x;
    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = state_head_base + value_idx;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * recurrent_state[state_col + (idx + 0) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 1] * k_scale * recurrent_state[state_col + (idx + 1) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 2] * k_scale * recurrent_state[state_col + (idx + 2) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 3] * k_scale * recurrent_state[state_col + (idx + 3) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 4] * k_scale * recurrent_state[state_col + (idx + 4) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 5] * k_scale * recurrent_state[state_col + (idx + 5) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 6] * k_scale * recurrent_state[state_col + (idx + 6) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 7] * k_scale * recurrent_state[state_col + (idx + 7) * head_v_dim] * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = recurrent_state[state_col + idx * head_v_dim] * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = staged_row[2 * head_k_dim + value_idx];
      const float delta = (value - kv_mem) * beta;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float old0 = recurrent_state[so0];
        const float old1 = recurrent_state[so1];
        const float old2 = recurrent_state[so2];
        const float old3 = recurrent_state[so3];
        const float old4 = recurrent_state[so4];
        const float old5 = recurrent_state[so5];
        const float old6 = recurrent_state[so6];
        const float old7 = recurrent_state[so7];
        if constexpr (write_snapshot) {
          if (t == 0) {
            initial_recurrent_state_snapshot[so0] = old0;
            initial_recurrent_state_snapshot[so1] = old1;
            initial_recurrent_state_snapshot[so2] = old2;
            initial_recurrent_state_snapshot[so3] = old3;
            initial_recurrent_state_snapshot[so4] = old4;
            initial_recurrent_state_snapshot[so5] = old5;
            initial_recurrent_state_snapshot[so6] = old6;
            initial_recurrent_state_snapshot[so7] = old7;
          }
        }
        const float ns0 = old0 * decay + k_norm0 * delta;
        const float ns1 = old1 * decay + k_norm1 * delta;
        const float ns2 = old2 * decay + k_norm2 * delta;
        const float ns3 = old3 * decay + k_norm3 * delta;
        const float ns4 = old4 * decay + k_norm4 * delta;
        const float ns5 = old5 * decay + k_norm5 * delta;
        const float ns6 = old6 * decay + k_norm6 * delta;
        const float ns7 = old7 * decay + k_norm7 * delta;
        node_state[so0] = ns0;
        node_state[so1] = ns1;
        node_state[so2] = ns2;
        node_state[so3] = ns3;
        node_state[so4] = ns4;
        node_state[so5] = ns5;
        node_state[so6] = ns6;
        node_state[so7] = ns7;
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float old_state = recurrent_state[state_offset];
        if constexpr (write_snapshot) {
          if (t == 0) {
            initial_recurrent_state_snapshot[state_offset] = old_state;
          }
        }
        const float new_state = old_state * decay + k_norm * delta;
        node_state[state_offset] = new_state;
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = (t * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float value = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[pos]));
      out[pos] = value;
      if constexpr (write_bf16) {
        out_bf16[pos] = float_to_bf16_bits(value);
      }
    }
    __syncthreads();
  }
}


__global__ void qwen35_gdn_prefill_recurrent_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx = blockIdx.y;
  const int64_t k_idx = threadIdx.x;
  if (v_head >= num_v_heads || value_idx >= head_v_dim || k_idx >= head_k_dim) {
    return;
  }

  const int64_t state_offset = (v_head * head_k_dim + k_idx) * head_v_dim + value_idx;
  float state_value = recurrent_state[state_offset];
  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const float key_value = key[head_offset * head_k_dim + k_idx];
    state_value *= decay[head_offset];

    partial[k_idx] = key_value * state_value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - partial[0]) * beta[head_offset];
    state_value += key_value * delta;
    __syncthreads();

    partial[k_idx] = query[head_offset * head_k_dim + k_idx] * state_value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[head_offset * head_v_dim + value_idx] = partial[0];
    }
    __syncthreads();
  }
  recurrent_state[state_offset] = state_value;
}

__global__ void qwen35_gdn_prefill_recurrent_k2_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  __shared__ float partial_update[2];
  __shared__ float partial_query[2];
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx = blockIdx.y;
  // Wrapper validates head_k_dim == 128 and launches 64 threads, so this k2
  // specialization is always two wave32 warps. Keep the same partial[0] +
  // partial[1] accumulation order while avoiding the per-token num_warps loop.
  const int lane_id = threadIdx.x & (warpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int64_t k_base = static_cast<int64_t>(threadIdx.x) * 2;
  if (v_head >= num_v_heads || value_idx >= head_v_dim || k_base + 1 >= head_k_dim) {
    return;
  }

  const int64_t state_offset0 = (v_head * head_k_dim + k_base) * head_v_dim + value_idx;
  const int64_t state_offset1 = state_offset0 + head_v_dim;
  float state_value0 = recurrent_state[state_offset0];
  float state_value1 = recurrent_state[state_offset1];
  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t key_offset = head_offset * head_k_dim + k_base;
    const float key_value0 = key[key_offset];
    const float key_value1 = key[key_offset + 1];
    const float decay_value = decay[head_offset];
    state_value0 *= decay_value;
    state_value1 *= decay_value;

    float acc = key_value0 * state_value0 + key_value1 * state_value1;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      partial_update[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      partial_update[0] += partial_update[1];
    }
    __syncthreads();
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - partial_update[0]) * beta[head_offset];
    state_value0 += key_value0 * delta;
    state_value1 += key_value1 * delta;

    const int64_t query_offset = head_offset * head_k_dim + k_base;
    acc = query[query_offset] * state_value0 + query[query_offset + 1] * state_value1;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      partial_query[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      out[head_offset * head_v_dim + value_idx] = partial_query[0] + partial_query[1];
    }
    __syncthreads();
  }
  recurrent_state[state_offset0] = state_value0;
  recurrent_state[state_offset1] = state_value1;
}

// State-parallel prefill with the c=1 decode reduction contract.  The ordinary
// K2 kernel tree-reduces two K terms per lane; that changes F32 association and
// can move BF16 layer outputs.  Keep the same 64-thread / (v_head, value_idx)
// grid and register-resident recurrent state, but materialize the 128 products
// in K order and let lane 0 accumulate them in the same order as decode.
__launch_bounds__(64, 4)
__global__ void qwen35_gdn_prefill_recurrent_k2_decode_order_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  __shared__ float reduction_lhs[128];
  __shared__ float reduction_rhs[128];
  __shared__ float reduction_total;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx = blockIdx.y;
  const int64_t k_base = static_cast<int64_t>(threadIdx.x) * 2;
  if (v_head >= num_v_heads || value_idx >= head_v_dim || k_base + 1 >= head_k_dim) {
    return;
  }

  const int64_t state_offset0 = (v_head * head_k_dim + k_base) * head_v_dim + value_idx;
  const int64_t state_offset1 = state_offset0 + head_v_dim;
  float state_value0 = recurrent_state[state_offset0];
  float state_value1 = recurrent_state[state_offset1];
  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t key_offset = head_offset * head_k_dim + k_base;
    const float key_value0 = key[key_offset];
    const float key_value1 = key[key_offset + 1];
    const float decay_value = decay[head_offset];
    const float previous_state0 = state_value0;
    const float previous_state1 = state_value1;

    // The decode kernel rounds k_norm*state, then fuses decay*product into
    // the serial accumulator.  Materializing the decay product here would
    // introduce one extra F32 rounding before the accumulation.
    reduction_lhs[k_base] = key_value0 * previous_state0;
    reduction_lhs[k_base + 1] = key_value1 * previous_state1;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int64_t idx = 0; idx < head_k_dim; ++idx) {
        total = fmaf(decay_value, reduction_lhs[idx], total);
      }
      reduction_total = total;
    }
    __syncthreads();

    const float delta =
        (value[head_offset * head_v_dim + value_idx] - reduction_total) * beta[head_offset];
    state_value0 = fmaf(key_value0, delta, previous_state0 * decay_value);
    state_value1 = fmaf(key_value1, delta, previous_state1 * decay_value);

    const int64_t query_offset = head_offset * head_k_dim + k_base;
    reduction_lhs[k_base] = query[query_offset];
    reduction_lhs[k_base + 1] = query[query_offset + 1];
    reduction_rhs[k_base] = state_value0;
    reduction_rhs[k_base + 1] = state_value1;
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        // Match the compiler's decode-order lowering for
        // ``q0*s0 + ... + q7*s7``: round term 1, fuse term 0 into it,
        // then fuse terms 2..7 before adding the eight-term group.
        float group = reduction_lhs[idx + 1] * reduction_rhs[idx + 1];
        group = fmaf(reduction_lhs[idx + 0], reduction_rhs[idx + 0], group);
        group = fmaf(reduction_lhs[idx + 2], reduction_rhs[idx + 2], group);
        group = fmaf(reduction_lhs[idx + 3], reduction_rhs[idx + 3], group);
        group = fmaf(reduction_lhs[idx + 4], reduction_rhs[idx + 4], group);
        group = fmaf(reduction_lhs[idx + 5], reduction_rhs[idx + 5], group);
        group = fmaf(reduction_lhs[idx + 6], reduction_rhs[idx + 6], group);
        group = fmaf(reduction_lhs[idx + 7], reduction_rhs[idx + 7], group);
        total += group;
      }
      out[head_offset * head_v_dim + value_idx] = total;
    }
    __syncthreads();
  }
  recurrent_state[state_offset0] = state_value0;
  recurrent_state[state_offset1] = state_value1;
}

// Exact decode-order recurrence with one wave owning 32 adjacent value
// columns.  The state lives in a conflict-free [K][value] LDS tile across the
// serial token loop.  Every value lane preserves the K2 exact kernel's scalar
// K contraction, state update, and eight-term output grouping; only independent
// value columns are interleaved so Q/K/scalar loads can be wave-uniform.
__launch_bounds__(32, 4)
__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  constexpr int kHeadKDim = 128;
  constexpr int kValueTile = 32;
  __shared__ float state_tile[kHeadKDim * kValueTile];

  const int64_t v_head = blockIdx.x;
  const int value_lane = static_cast<int>(threadIdx.x);
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * kValueTile + value_lane;
  if (v_head >= num_v_heads || value_idx >= head_v_dim ||
      head_k_dim != kHeadKDim) {
    return;
  }

  const int64_t state_col =
      v_head * kHeadKDim * head_v_dim + value_idx;
#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    state_tile[idx * kValueTile + value_lane] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float decay_value = decay[head_offset];

    float kv_mem = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; ++idx) {
      // Match K2 exact: round key*previous_state before the decay FMA.
      const float key_state =
          key[qk_base + idx] * state_tile[idx * kValueTile + value_lane];
      kv_mem = fmaf(decay_value, key_state, kv_mem);
    }

    const float delta =
        (value[head_offset * head_v_dim + value_idx] - kv_mem) *
        beta[head_offset];
    float out_acc = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; idx += 8) {
      const float q0 = query[qk_base + idx + 0];
      const float q1 = query[qk_base + idx + 1];
      const float q2 = query[qk_base + idx + 2];
      const float q3 = query[qk_base + idx + 3];
      const float q4 = query[qk_base + idx + 4];
      const float q5 = query[qk_base + idx + 5];
      const float q6 = query[qk_base + idx + 6];
      const float q7 = query[qk_base + idx + 7];
      const float k0 = key[qk_base + idx + 0];
      const float k1 = key[qk_base + idx + 1];
      const float k2 = key[qk_base + idx + 2];
      const float k3 = key[qk_base + idx + 3];
      const float k4 = key[qk_base + idx + 4];
      const float k5 = key[qk_base + idx + 5];
      const float k6 = key[qk_base + idx + 6];
      const float k7 = key[qk_base + idx + 7];
      const int so0 = (idx + 0) * kValueTile + value_lane;
      const int so1 = (idx + 1) * kValueTile + value_lane;
      const int so2 = (idx + 2) * kValueTile + value_lane;
      const int so3 = (idx + 3) * kValueTile + value_lane;
      const int so4 = (idx + 4) * kValueTile + value_lane;
      const int so5 = (idx + 5) * kValueTile + value_lane;
      const int so6 = (idx + 6) * kValueTile + value_lane;
      const int so7 = (idx + 7) * kValueTile + value_lane;
      const float ns0 = fmaf(k0, delta, state_tile[so0] * decay_value);
      const float ns1 = fmaf(k1, delta, state_tile[so1] * decay_value);
      const float ns2 = fmaf(k2, delta, state_tile[so2] * decay_value);
      const float ns3 = fmaf(k3, delta, state_tile[so3] * decay_value);
      const float ns4 = fmaf(k4, delta, state_tile[so4] * decay_value);
      const float ns5 = fmaf(k5, delta, state_tile[so5] * decay_value);
      const float ns6 = fmaf(k6, delta, state_tile[so6] * decay_value);
      const float ns7 = fmaf(k7, delta, state_tile[so7] * decay_value);
      state_tile[so0] = ns0;
      state_tile[so1] = ns1;
      state_tile[so2] = ns2;
      state_tile[so3] = ns3;
      state_tile[so4] = ns4;
      state_tile[so5] = ns5;
      state_tile[so6] = ns6;
      state_tile[so7] = ns7;

      // Preserve the decode compiler's group-of-eight contraction exactly.
      float group = q1 * ns1;
      group = fmaf(q0, ns0, group);
      group = fmaf(q2, ns2, group);
      group = fmaf(q3, ns3, group);
      group = fmaf(q4, ns4, group);
      group = fmaf(q5, ns5, group);
      group = fmaf(q6, ns6, group);
      group = fmaf(q7, ns7, group);
      out_acc += group;
    }
    out[head_offset * head_v_dim + value_idx] = out_acc;
  }

#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_tile[idx * kValueTile + value_lane];
  }
}

__global__ void qwen35_gdn_prefill_recurrent_k2_segments_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  __shared__ float partial[64];
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx = blockIdx.z;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  const int64_t k_base = static_cast<int64_t>(threadIdx.x) * 2;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim || k_base + 1 >= head_k_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_offset0 = state_slot * state_stride + (v_head * head_k_dim + k_base) * head_v_dim + value_idx;
  const int64_t state_offset1 = state_offset0 + head_v_dim;
  float state_value0 = recurrent_state[state_offset0];
  float state_value1 = recurrent_state[state_offset1];
  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t key_offset = head_offset * head_k_dim + k_base;
    const float key_value0 = key[key_offset];
    const float key_value1 = key[key_offset + 1];
    const float decay_value = decay[head_offset];
    state_value0 *= decay_value;
    state_value1 *= decay_value;

    float acc = key_value0 * state_value0 + key_value1 * state_value1;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      partial[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int warp = 0; warp < num_warps; ++warp) {
        total += partial[warp];
      }
      partial[0] = total;
    }
    __syncthreads();
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - partial[0]) * beta[head_offset];
    state_value0 += key_value0 * delta;
    state_value1 += key_value1 * delta;

    const int64_t query_offset = head_offset * head_k_dim + k_base;
    acc = query[query_offset] * state_value0 + query[query_offset + 1] * state_value1;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      partial[warp_id] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
      for (int warp = 0; warp < num_warps; ++warp) {
        total += partial[warp];
      }
      out[head_offset * head_v_dim + value_idx] = total;
    }
    __syncthreads();
  }
  recurrent_state[state_offset0] = state_value0;
  recurrent_state[state_offset1] = state_value1;
}

template <typename scalar_t, bool apply_output_scale = true>
__global__ void qwen35_linear_attn_prefill_prepare_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
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
  const int64_t v_head = blockIdx.y;
  if (token >= tokens || v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_dim + value_dim;
  const int64_t conv_base = token * qkv_width;
  const int64_t q_base = conv_base + k_head * head_k_dim;
  const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    const float q_value = conv_out[q_base + idx];
    const float k_value = conv_out[k_base + idx];
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
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

  const float q_norm_scale = 1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f);
  const float q_scale = apply_output_scale
      ? q_norm_scale * rsqrtf(static_cast<float>(head_k_dim))
      : q_norm_scale;
  const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
  const int64_t head_base = (token * num_v_heads + v_head) * head_k_dim;
  for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    query[head_base + idx] = conv_out[q_base + idx] * q_scale;
    key[head_base + idx] = conv_out[k_base + idx] * k_scale;
  }
  const int64_t value_base = (token * num_v_heads + v_head) * head_v_dim;
  const int64_t conv_value_base = conv_base + 2 * key_dim + v_head * head_v_dim;
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    value[value_base + idx] = conv_out[conv_value_base + idx];
  }
  if (threadIdx.x == 0) {
    const int64_t scalar_idx = token * num_v_heads + v_head;
    beta[scalar_idx] = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    decay[scalar_idx] = expf(
        -expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));
  }
}

// GPF compact-peer screen: materialize normalized Q/K once per K head while
// retaining per-V-head value/beta/decay.  The Q/K block uses the same 128-thread
// reduction tree as the admitted peer prepare, so every V head mapped to a K
// head observes the same normalized bits as the duplicated layout.
template <typename scalar_t>
__global__ void qwen35_linear_attn_prefill_prepare_compact_peer_normalized_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
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
  if (token >= tokens) {
    return;
  }
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_dim + value_dim;
  const int64_t conv_base = token * qkv_width;

  if (head < num_k_heads) {
    const int64_t q_base = conv_base + head * head_k_dim;
    const int64_t k_base = conv_base + key_dim + head * head_k_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
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
    const float q_scale = 1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f);
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const int64_t head_base = (token * num_k_heads + head) * head_k_dim;
    for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      query[head_base + idx] = conv_out[q_base + idx] * q_scale;
      key[head_base + idx] = conv_out[k_base + idx] * k_scale;
    }
  }

  if (head < num_v_heads) {
    const int64_t value_base = (token * num_v_heads + head) * head_v_dim;
    const int64_t conv_value_base = conv_base + 2 * key_dim + head * head_v_dim;
    for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
      value[value_base + idx] = conv_out[conv_value_base + idx];
    }
    if (threadIdx.x == 0) {
      const int64_t scalar_idx = token * num_v_heads + head;
      beta[scalar_idx] = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
      decay[scalar_idx] = expf(
          -expf(a_log[head]) *
          softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[head]));
    }
  }
}

template <typename scalar_t>
__global__ void qwen35_linear_attn_prefill_prepare_raw_scales_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ query_raw,
    float* __restrict__ key_raw,
    float* __restrict__ value,
    float* __restrict__ beta,
    float* __restrict__ decay,
    float* __restrict__ query_scale,
    float* __restrict__ key_scale,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  float* q_partial = partial;
  float* k_partial = q_partial + blockDim.x;

  const int64_t token = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (token >= tokens || v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_dim + value_dim;
  const int64_t conv_base = token * qkv_width;
  const int64_t q_base = conv_base + k_head * head_k_dim;
  const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;

  // Keep the exact eight-wide local accumulation and tree used by the fused
  // decode-order kernel. Raw Q/K and the two scales remain separate so the
  // recurrent kernel can preserve the fused multiplication/contraction order.
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
    const float q0 = conv_out[q_base + idx + 0];
    const float q1 = conv_out[q_base + idx + 1];
    const float q2 = conv_out[q_base + idx + 2];
    const float q3 = conv_out[q_base + idx + 3];
    const float q4 = conv_out[q_base + idx + 4];
    const float q5 = conv_out[q_base + idx + 5];
    const float q6 = conv_out[q_base + idx + 6];
    const float q7 = conv_out[q_base + idx + 7];
    const float k0 = conv_out[k_base + idx + 0];
    const float k1 = conv_out[k_base + idx + 1];
    const float k2 = conv_out[k_base + idx + 2];
    const float k3 = conv_out[k_base + idx + 3];
    const float k4 = conv_out[k_base + idx + 4];
    const float k5 = conv_out[k_base + idx + 5];
    const float k6 = conv_out[k_base + idx + 6];
    const float k7 = conv_out[k_base + idx + 7];
    q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
    k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
  }
  for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    const float q_value = conv_out[q_base + idx];
    const float k_value = conv_out[k_base + idx];
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
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

  const int64_t head_base = (token * num_v_heads + v_head) * head_k_dim;
  for (int64_t idx = threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
    query_raw[head_base + idx] = conv_out[q_base + idx];
    key_raw[head_base + idx] = conv_out[k_base + idx];
  }
  const int64_t value_base = (token * num_v_heads + v_head) * head_v_dim;
  const int64_t conv_value_base = conv_base + 2 * key_dim + v_head * head_v_dim;
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    value[value_base + idx] = conv_out[conv_value_base + idx];
  }
  if (threadIdx.x == 0) {
    const int64_t scalar_idx = token * num_v_heads + v_head;
    query_scale[scalar_idx] =
        (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    key_scale[scalar_idx] = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    beta[scalar_idx] = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    decay[scalar_idx] = expf(
        -expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));
  }
}

// GPF-2E: the exact LDS32 recurrence can read Q/K/V directly from conv_out.
// Only normalization scales and the per-V-head recurrence scalars need
// materialization. Q/K are shared by every V head mapping to the same K head,
// so compute each norm once and store scales as [token, k_head]. The reduction
// tree and eight-wide accumulation remain identical to raw_scales above.
template <typename scalar_t>
__global__ void qwen35_linear_attn_prefill_prepare_compact_scales_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ beta,
    float* __restrict__ decay,
    float* __restrict__ query_scale,
    float* __restrict__ key_scale,
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
  if (token >= tokens) {
    return;
  }

  if (head >= num_k_heads) {
    if (head < num_v_heads && threadIdx.x == 0) {
      const int64_t scalar_idx = token * num_v_heads + head;
      beta[scalar_idx] = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
      decay[scalar_idx] = expf(
          -expf(a_log[head]) *
          softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[head]));
    }
    return;
  }

  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_dim + value_dim;
  const int64_t conv_base = token * qkv_width;
  const int64_t q_base = conv_base + head * head_k_dim;
  const int64_t k_base = conv_base + key_dim + head * head_k_dim;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim;
       idx += blockDim.x * 8) {
    const float q0 = conv_out[q_base + idx + 0];
    const float q1 = conv_out[q_base + idx + 1];
    const float q2 = conv_out[q_base + idx + 2];
    const float q3 = conv_out[q_base + idx + 3];
    const float q4 = conv_out[q_base + idx + 4];
    const float q5 = conv_out[q_base + idx + 5];
    const float q6 = conv_out[q_base + idx + 6];
    const float q7 = conv_out[q_base + idx + 7];
    const float k0 = conv_out[k_base + idx + 0];
    const float k1 = conv_out[k_base + idx + 1];
    const float k2 = conv_out[k_base + idx + 2];
    const float k3 = conv_out[k_base + idx + 3];
    const float k4 = conv_out[k_base + idx + 4];
    const float k5 = conv_out[k_base + idx + 5];
    const float k6 = conv_out[k_base + idx + 6];
    const float k7 = conv_out[k_base + idx + 7];
    q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
    k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
  }
  for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim;
       idx += blockDim.x) {
    const float q_value = conv_out[q_base + idx];
    const float k_value = conv_out[k_base + idx];
    q_sum += q_value * q_value;
    k_sum += k_value * k_value;
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

  if (threadIdx.x == 0) {
    const int64_t scale_idx = token * num_k_heads + head;
    query_scale[scale_idx] =
        (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) *
        rsqrtf(static_cast<float>(head_k_dim));
    key_scale[scale_idx] = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    if (head < num_v_heads) {
      const int64_t scalar_idx = token * num_v_heads + head;
      beta[scalar_idx] = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
      decay[scalar_idx] = expf(
          -expf(a_log[head]) *
          softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[head]));
    }
  }
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
  volatile float* recurrent_state_v = recurrent_state;
  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * head_k_dim;
    const float q_scale = query_scale[head_offset];
    const float k_scale = key_scale[head_offset];
    const float decay_value = decay[head_offset];
    float kv_mem = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      kv_mem += key_raw[qk_base + idx + 0] * k_scale * recurrent_state_v[state_col + (idx + 0) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 1] * k_scale * recurrent_state_v[state_col + (idx + 1) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 2] * k_scale * recurrent_state_v[state_col + (idx + 2) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 3] * k_scale * recurrent_state_v[state_col + (idx + 3) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 4] * k_scale * recurrent_state_v[state_col + (idx + 4) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 5] * k_scale * recurrent_state_v[state_col + (idx + 5) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 6] * k_scale * recurrent_state_v[state_col + (idx + 6) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 7] * k_scale * recurrent_state_v[state_col + (idx + 7) * head_v_dim] * decay_value;
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - kv_mem) * beta[head_offset];
    float out_acc = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      const float q0 = query_raw[qk_base + idx + 0] * q_scale;
      const float q1 = query_raw[qk_base + idx + 1] * q_scale;
      const float q2 = query_raw[qk_base + idx + 2] * q_scale;
      const float q3 = query_raw[qk_base + idx + 3] * q_scale;
      const float q4 = query_raw[qk_base + idx + 4] * q_scale;
      const float q5 = query_raw[qk_base + idx + 5] * q_scale;
      const float q6 = query_raw[qk_base + idx + 6] * q_scale;
      const float q7 = query_raw[qk_base + idx + 7] * q_scale;
      const float k0 = key_raw[qk_base + idx + 0] * k_scale;
      const float k1 = key_raw[qk_base + idx + 1] * k_scale;
      const float k2 = key_raw[qk_base + idx + 2] * k_scale;
      const float k3 = key_raw[qk_base + idx + 3] * k_scale;
      const float k4 = key_raw[qk_base + idx + 4] * k_scale;
      const float k5 = key_raw[qk_base + idx + 5] * k_scale;
      const float k6 = key_raw[qk_base + idx + 6] * k_scale;
      const float k7 = key_raw[qk_base + idx + 7] * k_scale;
      const int64_t so0 = state_col + (idx + 0) * head_v_dim;
      const int64_t so1 = state_col + (idx + 1) * head_v_dim;
      const int64_t so2 = state_col + (idx + 2) * head_v_dim;
      const int64_t so3 = state_col + (idx + 3) * head_v_dim;
      const int64_t so4 = state_col + (idx + 4) * head_v_dim;
      const int64_t so5 = state_col + (idx + 5) * head_v_dim;
      const int64_t so6 = state_col + (idx + 6) * head_v_dim;
      const int64_t so7 = state_col + (idx + 7) * head_v_dim;
      const float ns0 = recurrent_state_v[so0] * decay_value + k0 * delta;
      const float ns1 = recurrent_state_v[so1] * decay_value + k1 * delta;
      const float ns2 = recurrent_state_v[so2] * decay_value + k2 * delta;
      const float ns3 = recurrent_state_v[so3] * decay_value + k3 * delta;
      const float ns4 = recurrent_state_v[so4] * decay_value + k4 * delta;
      const float ns5 = recurrent_state_v[so5] * decay_value + k5 * delta;
      const float ns6 = recurrent_state_v[so6] * decay_value + k6 * delta;
      const float ns7 = recurrent_state_v[so7] * decay_value + k7 * delta;
      recurrent_state_v[so0] = ns0;
      recurrent_state_v[so1] = ns1;
      recurrent_state_v[so2] = ns2;
      recurrent_state_v[so3] = ns3;
      recurrent_state_v[so4] = ns4;
      recurrent_state_v[so5] = ns5;
      recurrent_state_v[so6] = ns6;
      recurrent_state_v[so7] = ns7;
      out_acc += q0 * ns0 + q1 * ns1 + q2 * ns2 + q3 * ns3
                 + q4 * ns4 + q5 * ns5 + q6 * ns6 + q7 * ns7;
    }
    out[head_offset * head_v_dim + value_idx] = out_acc;
  }
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_segments_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * blockDim.x + threadIdx.x;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * head_k_dim * head_v_dim + value_idx;
  volatile float* recurrent_state_v = recurrent_state;
  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * head_k_dim;
    const float q_scale = query_scale[head_offset];
    const float k_scale = key_scale[head_offset];
    const float decay_value = decay[head_offset];
    float kv_mem = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      kv_mem += key_raw[qk_base + idx + 0] * k_scale * recurrent_state_v[state_col + (idx + 0) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 1] * k_scale * recurrent_state_v[state_col + (idx + 1) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 2] * k_scale * recurrent_state_v[state_col + (idx + 2) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 3] * k_scale * recurrent_state_v[state_col + (idx + 3) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 4] * k_scale * recurrent_state_v[state_col + (idx + 4) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 5] * k_scale * recurrent_state_v[state_col + (idx + 5) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 6] * k_scale * recurrent_state_v[state_col + (idx + 6) * head_v_dim] * decay_value;
      kv_mem += key_raw[qk_base + idx + 7] * k_scale * recurrent_state_v[state_col + (idx + 7) * head_v_dim] * decay_value;
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - kv_mem) * beta[head_offset];
    float out_acc = 0.0f;
    for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
      const float q0 = query_raw[qk_base + idx + 0] * q_scale;
      const float q1 = query_raw[qk_base + idx + 1] * q_scale;
      const float q2 = query_raw[qk_base + idx + 2] * q_scale;
      const float q3 = query_raw[qk_base + idx + 3] * q_scale;
      const float q4 = query_raw[qk_base + idx + 4] * q_scale;
      const float q5 = query_raw[qk_base + idx + 5] * q_scale;
      const float q6 = query_raw[qk_base + idx + 6] * q_scale;
      const float q7 = query_raw[qk_base + idx + 7] * q_scale;
      const float k0 = key_raw[qk_base + idx + 0] * k_scale;
      const float k1 = key_raw[qk_base + idx + 1] * k_scale;
      const float k2 = key_raw[qk_base + idx + 2] * k_scale;
      const float k3 = key_raw[qk_base + idx + 3] * k_scale;
      const float k4 = key_raw[qk_base + idx + 4] * k_scale;
      const float k5 = key_raw[qk_base + idx + 5] * k_scale;
      const float k6 = key_raw[qk_base + idx + 6] * k_scale;
      const float k7 = key_raw[qk_base + idx + 7] * k_scale;
      const int64_t so0 = state_col + (idx + 0) * head_v_dim;
      const int64_t so1 = state_col + (idx + 1) * head_v_dim;
      const int64_t so2 = state_col + (idx + 2) * head_v_dim;
      const int64_t so3 = state_col + (idx + 3) * head_v_dim;
      const int64_t so4 = state_col + (idx + 4) * head_v_dim;
      const int64_t so5 = state_col + (idx + 5) * head_v_dim;
      const int64_t so6 = state_col + (idx + 6) * head_v_dim;
      const int64_t so7 = state_col + (idx + 7) * head_v_dim;
      const float ns0 = recurrent_state_v[so0] * decay_value + k0 * delta;
      const float ns1 = recurrent_state_v[so1] * decay_value + k1 * delta;
      const float ns2 = recurrent_state_v[so2] * decay_value + k2 * delta;
      const float ns3 = recurrent_state_v[so3] * decay_value + k3 * delta;
      const float ns4 = recurrent_state_v[so4] * decay_value + k4 * delta;
      const float ns5 = recurrent_state_v[so5] * decay_value + k5 * delta;
      const float ns6 = recurrent_state_v[so6] * decay_value + k6 * delta;
      const float ns7 = recurrent_state_v[so7] * decay_value + k7 * delta;
      recurrent_state_v[so0] = ns0;
      recurrent_state_v[so1] = ns1;
      recurrent_state_v[so2] = ns2;
      recurrent_state_v[so3] = ns3;
      recurrent_state_v[so4] = ns4;
      recurrent_state_v[so5] = ns5;
      recurrent_state_v[so6] = ns6;
      recurrent_state_v[so7] = ns7;
      out_acc += q0 * ns0 + q1 * ns1 + q2 * ns2 + q3 * ns3
                 + q4 * ns4 + q5 * ns5 + q6 * ns6 + q7 * ns7;
    }
    out[head_offset * head_v_dim + value_idx] = out_acc;
  }
}

// GPF-2D: preserve the scalar exact recurrence owned by one thread/value
// column, but keep a wave-aligned tile of recurrent state in LDS across the
// complete serial token loop. Row-major [state_row][value_column] storage gives
// each wave conflict-free contiguous LDS accesses without changing any
// contraction, update, or token ordering.
template <int kValueTile>
__device__ __forceinline__ void qwen35_gdn_prefill_recurrent_decode_order_exact_lds_body(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    volatile float* state_tile,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  static_assert(kValueTile == 32 || kValueTile == 64, "unsupported LDS value tile");
  constexpr int kHeadKDim = 128;
  const int value_lane = static_cast<int>(threadIdx.x);

#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    state_tile[idx * kValueTile + value_lane] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float q_scale = query_scale[head_offset];
    const float k_scale = key_scale[head_offset];
    const float decay_value = decay[head_offset];
    float kv_mem = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; idx += 8) {
      kv_mem += key_raw[qk_base + idx + 0] * k_scale * state_tile[(idx + 0) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 1] * k_scale * state_tile[(idx + 1) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 2] * k_scale * state_tile[(idx + 2) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 3] * k_scale * state_tile[(idx + 3) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 4] * k_scale * state_tile[(idx + 4) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 5] * k_scale * state_tile[(idx + 5) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 6] * k_scale * state_tile[(idx + 6) * kValueTile + value_lane] * decay_value;
      kv_mem += key_raw[qk_base + idx + 7] * k_scale * state_tile[(idx + 7) * kValueTile + value_lane] * decay_value;
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - kv_mem) * beta[head_offset];
    float out_acc = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; idx += 8) {
      const float q0 = query_raw[qk_base + idx + 0] * q_scale;
      const float q1 = query_raw[qk_base + idx + 1] * q_scale;
      const float q2 = query_raw[qk_base + idx + 2] * q_scale;
      const float q3 = query_raw[qk_base + idx + 3] * q_scale;
      const float q4 = query_raw[qk_base + idx + 4] * q_scale;
      const float q5 = query_raw[qk_base + idx + 5] * q_scale;
      const float q6 = query_raw[qk_base + idx + 6] * q_scale;
      const float q7 = query_raw[qk_base + idx + 7] * q_scale;
      const float k0 = key_raw[qk_base + idx + 0] * k_scale;
      const float k1 = key_raw[qk_base + idx + 1] * k_scale;
      const float k2 = key_raw[qk_base + idx + 2] * k_scale;
      const float k3 = key_raw[qk_base + idx + 3] * k_scale;
      const float k4 = key_raw[qk_base + idx + 4] * k_scale;
      const float k5 = key_raw[qk_base + idx + 5] * k_scale;
      const float k6 = key_raw[qk_base + idx + 6] * k_scale;
      const float k7 = key_raw[qk_base + idx + 7] * k_scale;
      const int so0 = (idx + 0) * kValueTile + value_lane;
      const int so1 = (idx + 1) * kValueTile + value_lane;
      const int so2 = (idx + 2) * kValueTile + value_lane;
      const int so3 = (idx + 3) * kValueTile + value_lane;
      const int so4 = (idx + 4) * kValueTile + value_lane;
      const int so5 = (idx + 5) * kValueTile + value_lane;
      const int so6 = (idx + 6) * kValueTile + value_lane;
      const int so7 = (idx + 7) * kValueTile + value_lane;
      const float ns0 = state_tile[so0] * decay_value + k0 * delta;
      const float ns1 = state_tile[so1] * decay_value + k1 * delta;
      const float ns2 = state_tile[so2] * decay_value + k2 * delta;
      const float ns3 = state_tile[so3] * decay_value + k3 * delta;
      const float ns4 = state_tile[so4] * decay_value + k4 * delta;
      const float ns5 = state_tile[so5] * decay_value + k5 * delta;
      const float ns6 = state_tile[so6] * decay_value + k6 * delta;
      const float ns7 = state_tile[so7] * decay_value + k7 * delta;
      state_tile[so0] = ns0;
      state_tile[so1] = ns1;
      state_tile[so2] = ns2;
      state_tile[so3] = ns3;
      state_tile[so4] = ns4;
      state_tile[so5] = ns5;
      state_tile[so6] = ns6;
      state_tile[so7] = ns7;
      out_acc += q0 * ns0 + q1 * ns1 + q2 * ns2 + q3 * ns3
                 + q4 * ns4 + q5 * ns5 + q6 * ns6 + q7 * ns7;
    }
    out[head_offset * head_v_dim + value_idx] = out_acc;
  }

#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_tile[idx * kValueTile + value_lane];
  }
}

template <int kValueTile>
__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_lds_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * kValueTile];
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * kValueTile + threadIdx.x;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds_body<kValueTile>(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      state_tile,
      0,
      tokens,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim);
}

template <int kValueTile>
__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * kValueTile];
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * kValueTile + threadIdx.x;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds_body<kValueTile>(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      state_tile,
      start,
      end,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim);
}

// GPF-2E candidate: retain the GPF-2D scalar recurrence and LDS-resident state,
// but read raw Q/K/V from the canonical conv_out tensor. Compact Q/K scales are
// indexed by [token, k_head], while beta/decay and output remain indexed by
// [token, v_head]. This is intentionally separate from the retained GPF-2D
// body so the baseline kernel's code generation cannot drift during the A/B.
template <typename StateTilePtr>
__device__ __forceinline__ void
qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_body(
    const float* __restrict__ conv_out,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    StateTilePtr state_tile,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kValueTile = 32;
  constexpr int kHeadKDim = 128;
  const int value_lane = static_cast<int>(threadIdx.x);
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * kHeadKDim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t qkv_width = 2 * key_dim + value_dim;

#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    state_tile[idx * kValueTile + value_lane] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t scale_offset = token * num_k_heads + k_head;
    const int64_t conv_base = token * qkv_width;
    const int64_t q_base = conv_base + k_head * kHeadKDim;
    const int64_t k_base = conv_base + key_dim + k_head * kHeadKDim;
    const int64_t value_base =
        conv_base + 2 * key_dim + v_head * head_v_dim;
    const float q_scale = query_scale[scale_offset];
    const float k_scale = key_scale[scale_offset];
    const float decay_value = decay[head_offset];
    float kv_mem = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; idx += 8) {
      kv_mem += conv_out[k_base + idx + 0] * k_scale * state_tile[(idx + 0) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 1] * k_scale * state_tile[(idx + 1) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 2] * k_scale * state_tile[(idx + 2) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 3] * k_scale * state_tile[(idx + 3) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 4] * k_scale * state_tile[(idx + 4) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 5] * k_scale * state_tile[(idx + 5) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 6] * k_scale * state_tile[(idx + 6) * kValueTile + value_lane] * decay_value;
      kv_mem += conv_out[k_base + idx + 7] * k_scale * state_tile[(idx + 7) * kValueTile + value_lane] * decay_value;
    }
    const float delta =
        (conv_out[value_base + value_idx] - kv_mem) * beta[head_offset];
    float out_acc = 0.0f;
#pragma unroll 1
    for (int idx = 0; idx < kHeadKDim; idx += 8) {
      const float q0 = conv_out[q_base + idx + 0] * q_scale;
      const float q1 = conv_out[q_base + idx + 1] * q_scale;
      const float q2 = conv_out[q_base + idx + 2] * q_scale;
      const float q3 = conv_out[q_base + idx + 3] * q_scale;
      const float q4 = conv_out[q_base + idx + 4] * q_scale;
      const float q5 = conv_out[q_base + idx + 5] * q_scale;
      const float q6 = conv_out[q_base + idx + 6] * q_scale;
      const float q7 = conv_out[q_base + idx + 7] * q_scale;
      const float k0 = conv_out[k_base + idx + 0] * k_scale;
      const float k1 = conv_out[k_base + idx + 1] * k_scale;
      const float k2 = conv_out[k_base + idx + 2] * k_scale;
      const float k3 = conv_out[k_base + idx + 3] * k_scale;
      const float k4 = conv_out[k_base + idx + 4] * k_scale;
      const float k5 = conv_out[k_base + idx + 5] * k_scale;
      const float k6 = conv_out[k_base + idx + 6] * k_scale;
      const float k7 = conv_out[k_base + idx + 7] * k_scale;
      const int so0 = (idx + 0) * kValueTile + value_lane;
      const int so1 = (idx + 1) * kValueTile + value_lane;
      const int so2 = (idx + 2) * kValueTile + value_lane;
      const int so3 = (idx + 3) * kValueTile + value_lane;
      const int so4 = (idx + 4) * kValueTile + value_lane;
      const int so5 = (idx + 5) * kValueTile + value_lane;
      const int so6 = (idx + 6) * kValueTile + value_lane;
      const int so7 = (idx + 7) * kValueTile + value_lane;
      const float ns0 = state_tile[so0] * decay_value + k0 * delta;
      const float ns1 = state_tile[so1] * decay_value + k1 * delta;
      const float ns2 = state_tile[so2] * decay_value + k2 * delta;
      const float ns3 = state_tile[so3] * decay_value + k3 * delta;
      const float ns4 = state_tile[so4] * decay_value + k4 * delta;
      const float ns5 = state_tile[so5] * decay_value + k5 * delta;
      const float ns6 = state_tile[so6] * decay_value + k6 * delta;
      const float ns7 = state_tile[so7] * decay_value + k7 * delta;
      state_tile[so0] = ns0;
      state_tile[so1] = ns1;
      state_tile[so2] = ns2;
      state_tile[so3] = ns3;
      state_tile[so4] = ns4;
      state_tile[so5] = ns5;
      state_tile[so6] = ns6;
      state_tile[so7] = ns7;
      out_acc += q0 * ns0 + q1 * ns1 + q2 * ns2 + q3 * ns3
                 + q4 * ns4 + q5 * ns5 + q6 * ns6 + q7 * ns7;
    }
    out[head_offset * head_v_dim + value_idx] = out_acc;
  }

#pragma unroll 1
  for (int idx = 0; idx < kHeadKDim; ++idx) {
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_tile[idx * kValueTile + value_lane];
  }
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * 32];
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx = static_cast<int64_t>(blockIdx.y) * 32 + threadIdx.x;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_body<
      volatile float*>(
      conv_out, beta, decay, query_scale, key_scale, recurrent_state, out,
      state_tile, 0, tokens, v_head, value_idx, state_col, num_k_heads,
      num_v_heads, head_v_dim);
}

__global__ void
qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * 32];
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx = static_cast<int64_t>(blockIdx.y) * 32 + threadIdx.x;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_body<float*>(
      conv_out, beta, decay, query_scale, key_scale, recurrent_state, out,
      state_tile, 0, tokens, v_head, value_idx, state_col, num_k_heads,
      num_v_heads, head_v_dim);
}

__global__ void
qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * 32];
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx = static_cast<int64_t>(blockIdx.z) * 32 + threadIdx.x;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_body<
      volatile float*>(
      conv_out, beta, decay, query_scale, key_scale, recurrent_state, out,
      state_tile, start, end, v_head, value_idx, state_col, num_k_heads,
      num_v_heads, head_v_dim);
}

__global__ void
qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  __shared__ float state_tile[128 * 32];
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx = static_cast<int64_t>(blockIdx.z) * 32 + threadIdx.x;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_body<float*>(
      conv_out, beta, decay, query_scale, key_scale, recurrent_state, out,
      state_tile, start, end, v_head, value_idx, state_col, num_k_heads,
      num_v_heads, head_v_dim);
}

// GPF-2: one logical wave32 owns one value column. The 128 recurrent-state
// rows are sharded four-per-lane for loads and updates, while lane 0 rebuilds
// the exact 0..127 contraction order through source-lane shuffles. Keeping the
// key/state operands separate until lane 0 evaluates the expression preserves
// the same contraction/FMA opportunity as the scalar exact kernel. Each lane
// keeps its four state rows in registers across the complete token loop; this
// changes state lifetime only, not contraction or update order.
__device__ __forceinline__ void qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_body(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int lane) {
  constexpr int kWaveWidth = 32;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kWaveWidth;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    state_shard[row] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float q_scale = query_scale[head_offset];
    const float k_scale = key_scale[head_offset];
    const float decay_value = decay[head_offset];
    float kv_mem = 0.0f;

#pragma unroll
    for (int state_group = 0; state_group < kHeadKDim; state_group += kWaveWidth) {
      const int idx = state_group + lane;
      const float lane_key = key_raw[qk_base + idx];
      const float lane_state = state_shard[state_group / kWaveWidth];
#pragma unroll
      for (int source_lane = 0; source_lane < kWaveWidth; ++source_lane) {
        const float ordered_key = __shfl(lane_key, source_lane, kWaveWidth);
        const float ordered_state = __shfl(lane_state, source_lane, kWaveWidth);
        if (lane == 0) {
          const float scaled_state = ordered_key * k_scale * ordered_state;
          kv_mem = fmaf(scaled_state, decay_value, kv_mem);
        }
      }
    }
    kv_mem = __shfl(kv_mem, 0, kWaveWidth);
    float delta = 0.0f;
    if (lane == 0) {
      delta =
          (value[head_offset * head_v_dim + value_idx] - kv_mem) * beta[head_offset];
    }
    delta = __shfl(delta, 0, kWaveWidth);

    float out_acc = 0.0f;
#pragma unroll
    for (int state_group = 0; state_group < kHeadKDim; state_group += kWaveWidth) {
      const int idx = state_group + lane;
      const int row = state_group / kWaveWidth;
      const float q_lane = query_raw[qk_base + idx] * q_scale;
      const float k_lane = key_raw[qk_base + idx] * k_scale;
      const float decayed_state = state_shard[row] * decay_value;
      const float next_state = fmaf(k_lane, delta, decayed_state);
      state_shard[row] = next_state;

#pragma unroll
      for (int eight = 0; eight < kWaveWidth; eight += 8) {
        const float q0 = __shfl(q_lane, eight + 0, kWaveWidth);
        const float q1 = __shfl(q_lane, eight + 1, kWaveWidth);
        const float q2 = __shfl(q_lane, eight + 2, kWaveWidth);
        const float q3 = __shfl(q_lane, eight + 3, kWaveWidth);
        const float q4 = __shfl(q_lane, eight + 4, kWaveWidth);
        const float q5 = __shfl(q_lane, eight + 5, kWaveWidth);
        const float q6 = __shfl(q_lane, eight + 6, kWaveWidth);
        const float q7 = __shfl(q_lane, eight + 7, kWaveWidth);
        const float ns0 = __shfl(next_state, eight + 0, kWaveWidth);
        const float ns1 = __shfl(next_state, eight + 1, kWaveWidth);
        const float ns2 = __shfl(next_state, eight + 2, kWaveWidth);
        const float ns3 = __shfl(next_state, eight + 3, kWaveWidth);
        const float ns4 = __shfl(next_state, eight + 4, kWaveWidth);
        const float ns5 = __shfl(next_state, eight + 5, kWaveWidth);
        const float ns6 = __shfl(next_state, eight + 6, kWaveWidth);
        const float ns7 = __shfl(next_state, eight + 7, kWaveWidth);
        if (lane == 0) {
          out_acc += q0 * ns0 + q1 * ns1 + q2 * ns2 + q3 * ns3
                     + q4 * ns4 + q5 * ns5 + q6 * ns6 + q7 * ns7;
        }
      }
    }
    if (lane == 0) {
      out[head_offset * head_v_dim + value_idx] = out_acc;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_shard[row];
  }
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * waves_per_block + wave;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col =
      v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_body(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      0,
      tokens,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim,
      lane);
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * waves_per_block + wave;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_body(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      start,
      end,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim,
      lane);
}

// GPF-2 relaxed diagnostic: one wave32 per value column with a conventional
// tree reduction. Each lane keeps its four recurrent-state rows in registers
// across the complete serial token loop, matching the critical residency
// property of llama.cpp's schedule. This exposes the available state-row
// parallelism but changes contraction order, so it is registered separately
// from every exact variant.
__device__ __forceinline__ void qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_body(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int lane) {
  constexpr int kWaveWidth = 32;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kWaveWidth;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    state_shard[row] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float q_scale = query_scale[head_offset];
    const float k_scale = key_scale[head_offset];
    const float decay_value = decay[head_offset];
    float q_reg[kRowsPerLane];
    float k_reg[kRowsPerLane];
    float kv_local = 0.0f;

#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const int idx = row * kWaveWidth + lane;
      q_reg[row] = query_raw[qk_base + idx] * q_scale;
      k_reg[row] = key_raw[qk_base + idx] * k_scale;
      const float scaled_state = k_reg[row] * state_shard[row];
      kv_local = fmaf(scaled_state, decay_value, kv_local);
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      kv_local += __shfl_down(kv_local, offset, kWaveWidth);
    }
    float delta = 0.0f;
    if (lane == 0) {
      delta =
          (value[head_offset * head_v_dim + value_idx] - kv_local) * beta[head_offset];
    }
    delta = __shfl(delta, 0, kWaveWidth);

    float out_local = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const float decayed_state = state_shard[row] * decay_value;
      state_shard[row] = fmaf(k_reg[row], delta, decayed_state);
      out_local = fmaf(q_reg[row], state_shard[row], out_local);
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      out_local += __shfl_down(out_local, offset, kWaveWidth);
    }
    if (lane == 0) {
      out[head_offset * head_v_dim + value_idx] = out_local;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_shard[row];
  }
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * waves_per_block + wave;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_body(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      0,
      tokens,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim,
      lane);
}

__global__ void qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_kernel(
    const float* __restrict__ query_raw,
    const float* __restrict__ key_raw,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ query_scale,
    const float* __restrict__ key_scale,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * waves_per_block + wave;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_body(
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      start,
      end,
      v_head,
      value_idx,
      state_col,
      num_v_heads,
      head_v_dim,
      lane);
}

// GPF-9C: llama.cpp HIP's gfx1100 recurrence schedule. The normalized Q/K
// tensors are materialized by the existing prepare kernel. One wave32 owns one
// value column, keeps four state rows per lane resident across the serial token
// loop, reduces with the peer's XOR tree, and applies scalar decay after the
// S^T@k reduction. Source schedule: llama.cpp 1ebf790cda38,
// ggml/src/ggml-cuda/gated_delta_net.cu.
template <typename state_t = float>
__device__ __forceinline__ void qwen35_gdn_prefill_recurrent_normalized_wave32_xor_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    state_t* recurrent_state,
    float* __restrict__ out,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int lane) {
  constexpr int kWaveWidth = 32;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kWaveWidth;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    state_shard[row] = gdn_state_read_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim);
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float decay_value = decay[head_offset];
    float q_reg[kRowsPerLane];
    float k_reg[kRowsPerLane];
    float kv_shard = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const int idx = row * kWaveWidth + lane;
      k_reg[row] = key[qk_base + idx];
      q_reg[row] = query[qk_base + idx];
      kv_shard += state_shard[row] * k_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      kv_shard += __shfl_xor(kv_shard, offset, kWaveWidth);
    }

    const float delta =
        (value[head_offset * head_v_dim + value_idx] - decay_value * kv_shard) *
        beta[head_offset];
    float attn_partial = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      state_shard[row] =
          decay_value * state_shard[row] + k_reg[row] * delta;
      attn_partial += state_shard[row] * q_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      attn_partial += __shfl_xor(attn_partial, offset, kWaveWidth);
    }
    if (lane == 0) {
      constexpr float kOutputScale = 0.08838834764831845f;
      out[head_offset * head_v_dim + value_idx] = attn_partial * kOutputScale;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    gdn_state_write_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim,
        state_shard[row]);
  }
}

template <typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    state_t* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kWaveWidth;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * waves_per_block + wave;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t state_col = v_head * kHeadKDim * head_v_dim + value_idx;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    state_shard[row] = gdn_state_read_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim);
  }

  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = (token * num_k_heads + k_head) * kHeadKDim;
    const float decay_value = decay[head_offset];
    float q_reg[kRowsPerLane];
    float k_reg[kRowsPerLane];
    float kv_shard = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const int idx = row * kWaveWidth + lane;
      k_reg[row] = key[qk_base + idx];
      q_reg[row] = query[qk_base + idx];
      kv_shard += state_shard[row] * k_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      kv_shard += __shfl_xor(kv_shard, offset, kWaveWidth);
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - decay_value * kv_shard) *
        beta[head_offset];
    float attn_partial = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      state_shard[row] =
          decay_value * state_shard[row] + k_reg[row] * delta;
      attn_partial += state_shard[row] * q_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      attn_partial += __shfl_xor(attn_partial, offset, kWaveWidth);
    }
    if (lane == 0) {
      constexpr float kOutputScale = 0.08838834764831845f;
      out[head_offset * head_v_dim + value_idx] = attn_partial * kOutputScale;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    gdn_state_write_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim,
        state_shard[row]);
  }
}

template <typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    state_t* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kWaveWidth;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * waves_per_block + wave;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t k_head =
      qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t state_stride = num_v_heads * kHeadKDim * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * kHeadKDim * head_v_dim + value_idx;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    state_shard[row] = gdn_state_read_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim);
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = (token * num_k_heads + k_head) * kHeadKDim;
    const float decay_value = decay[head_offset];
    float q_reg[kRowsPerLane];
    float k_reg[kRowsPerLane];
    float kv_shard = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const int idx = row * kWaveWidth + lane;
      k_reg[row] = key[qk_base + idx];
      q_reg[row] = query[qk_base + idx];
      kv_shard += state_shard[row] * k_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      kv_shard += __shfl_xor(kv_shard, offset, kWaveWidth);
    }
    const float delta =
        (value[head_offset * head_v_dim + value_idx] - decay_value * kv_shard) *
        beta[head_offset];
    float attn_partial = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      state_shard[row] =
          decay_value * state_shard[row] + k_reg[row] * delta;
      attn_partial += state_shard[row] * q_reg[row];
    }
#pragma unroll
    for (int offset = kWaveWidth / 2; offset > 0; offset >>= 1) {
      attn_partial += __shfl_xor(attn_partial, offset, kWaveWidth);
    }
    if (lane == 0) {
      constexpr float kOutputScale = 0.08838834764831845f;
      out[head_offset * head_v_dim + value_idx] = attn_partial * kOutputScale;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kWaveWidth + lane;
    gdn_state_write_qwen35(
        recurrent_state, state_col + static_cast<int64_t>(idx) * head_v_dim,
        state_shard[row]);
  }
}

template <typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_normalized_wave32_xor_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    state_t* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * waves_per_block + wave;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_normalized_wave32_xor_body<state_t>(
      query, key, value, beta, decay, recurrent_state, out, 0, tokens, v_head,
      value_idx, state_col, num_v_heads, head_v_dim, lane);
}

template <typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    state_t* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kWaveWidth = 32;
  const int wave = static_cast<int>(threadIdx.x) / kWaveWidth;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveWidth - 1);
  const int waves_per_block = static_cast<int>(blockDim.x) / kWaveWidth;
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * waves_per_block + wave;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_normalized_wave32_xor_body<state_t>(
      query, key, value, beta, decay, recurrent_state, out, start, end, v_head,
      value_idx, state_col, num_v_heads, head_v_dim, lane);
}

// GPF-9D: llama.cpp Vulkan's S_v=128 clustered schedule, expressed with
// contiguous eight-lane HIP shuffle clusters. Each lane retains 16 recurrent
// rows; four columns share each physical wave32 and 32 columns share a
// 256-thread block. Source schedule: llama.cpp Vulkan 263cc04a5405,
// ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp.
__device__ __forceinline__ void qwen35_gdn_prefill_recurrent_normalized_cluster8_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t start,
    int64_t end,
    int64_t v_head,
    int64_t value_idx,
    int64_t state_col,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int cluster_lane) {
  constexpr int kLanesPerColumn = 8;
  constexpr int kHeadKDim = 128;
  constexpr int kRowsPerLane = kHeadKDim / kLanesPerColumn;
  float state_shard[kRowsPerLane];
#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kLanesPerColumn + cluster_lane;
    state_shard[row] =
        recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim];
  }

  for (int64_t token = start; token < end; ++token) {
    const int64_t head_offset = token * num_v_heads + v_head;
    const int64_t qk_base = head_offset * kHeadKDim;
    const float decay_value = decay[head_offset];
    float q_reg[kRowsPerLane];
    float k_reg[kRowsPerLane];
    float kv_shard = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      const int idx = row * kLanesPerColumn + cluster_lane;
      k_reg[row] = key[qk_base + idx];
      q_reg[row] = query[qk_base + idx];
      kv_shard += decay_value * state_shard[row] * k_reg[row];
    }
#pragma unroll
    for (int offset = kLanesPerColumn / 2; offset > 0; offset >>= 1) {
      kv_shard += __shfl_xor(kv_shard, offset, kLanesPerColumn);
    }

    const float delta =
        (value[head_offset * head_v_dim + value_idx] - kv_shard) *
        beta[head_offset];
    float attn_partial = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
      state_shard[row] =
          decay_value * state_shard[row] + k_reg[row] * delta;
      attn_partial += state_shard[row] * q_reg[row];
    }
#pragma unroll
    for (int offset = kLanesPerColumn / 2; offset > 0; offset >>= 1) {
      attn_partial += __shfl_xor(attn_partial, offset, kLanesPerColumn);
    }
    if (cluster_lane == 0) {
      constexpr float kOutputScale = 0.08838834764831845f;
      out[head_offset * head_v_dim + value_idx] = attn_partial * kOutputScale;
    }
  }

#pragma unroll
  for (int row = 0; row < kRowsPerLane; ++row) {
    const int idx = row * kLanesPerColumn + cluster_lane;
    recurrent_state[state_col + static_cast<int64_t>(idx) * head_v_dim] =
        state_shard[row];
  }
}

__global__ void qwen35_gdn_prefill_recurrent_normalized_cluster8_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* recurrent_state,
    float* __restrict__ out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kLanesPerColumn = 8;
  const int cluster_lane = static_cast<int>(threadIdx.x) & (kLanesPerColumn - 1);
  const int columns_per_block = static_cast<int>(blockDim.x) / kLanesPerColumn;
  const int64_t v_head = blockIdx.x;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.y) * columns_per_block +
      static_cast<int>(threadIdx.x) / kLanesPerColumn;
  if (v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t state_col = v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_normalized_cluster8_body(
      query, key, value, beta, decay, recurrent_state, out, 0, tokens, v_head,
      value_idx, state_col, num_v_heads, head_v_dim, cluster_lane);
}

__global__ void qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* recurrent_state,
    float* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  constexpr int kLanesPerColumn = 8;
  const int cluster_lane = static_cast<int>(threadIdx.x) & (kLanesPerColumn - 1);
  const int columns_per_block = static_cast<int>(blockDim.x) / kLanesPerColumn;
  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int64_t value_idx =
      static_cast<int64_t>(blockIdx.z) * columns_per_block +
      static_cast<int>(threadIdx.x) / kLanesPerColumn;
  if (segment >= segments || v_head >= num_v_heads || value_idx >= head_v_dim) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t state_stride = num_v_heads * 128 * head_v_dim;
  const int64_t state_col =
      state_slot * state_stride + v_head * 128 * head_v_dim + value_idx;
  qwen35_gdn_prefill_recurrent_normalized_cluster8_body(
      query, key, value, beta, decay, recurrent_state, out, start, end, v_head,
      value_idx, state_col, num_v_heads, head_v_dim, cluster_lane);
}

__global__ void qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel(
    const float* __restrict__ recurrent,
    const uint16_t* __restrict__ gate,
    const float* __restrict__ norm_weight,
    uint16_t* __restrict__ out,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  const int64_t token = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (token >= tokens || v_head >= num_v_heads) {
    return;
  }
  const int64_t base = (token * num_v_heads + v_head) * head_v_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    const float x = recurrent[base + idx];
    sumsq += x * x;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_v_dim)) + eps);
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    const float gated = recurrent[base + idx] * inv_rms * norm_weight[idx] * silu_f32(bf16_bits_to_float(gate[base + idx]));
    out[base + idx] = float_to_bf16_bits(gated);
  }
}

__global__ void qwen35_gdn_prefill_rmsnorm_gate_fp16_kernel(
    const float* __restrict__ recurrent,
    const half_t* __restrict__ gate,
    const float* __restrict__ norm_weight,
    half_t* __restrict__ out,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  extern __shared__ float partial[];
  const int64_t token = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (token >= tokens || v_head >= num_v_heads) {
    return;
  }
  const int64_t base = (token * num_v_heads + v_head) * head_v_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    const float x = recurrent[base + idx];
    sumsq += x * x;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_v_dim)) + eps);
  for (int64_t idx = threadIdx.x; idx < head_v_dim; idx += blockDim.x) {
    const float gated = recurrent[base + idx] * inv_rms * norm_weight[idx] * silu_f32(static_cast<float>(gate[base + idx]));
    out[base + idx] = static_cast<half_t>(gated);
  }
}

__global__ void qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16_kernel(
    const float* __restrict__ recurrent,
    const half_t* __restrict__ gate,
    const float* __restrict__ norm_weight,
    half_t* __restrict__ out_rot,
    const int16_t* __restrict__ pairs,
    const half_t* __restrict__ theta,
    const half_t* __restrict__ scales,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int group_size,
    int krot) {
  extern __shared__ float shared[];
  float* partial = shared;
  float* buf = partial + blockDim.x;
  const int64_t token = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  const int lane = threadIdx.x;
  if (token >= tokens || v_head >= num_v_heads) {
    return;
  }

  const int half_group = group_size / 2;
  const int64_t recurrent_base = (token * num_v_heads + v_head) * head_v_dim;
  float sumsq = 0.0f;
  for (int64_t idx = lane; idx < head_v_dim; idx += blockDim.x) {
    const float x = recurrent[recurrent_base + idx];
    sumsq += x * x;
  }
  partial[lane] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_v_dim)) + eps);
  const int group = static_cast<int>(v_head);
  const int base = group * group_size;
  const int64_t hidden = num_v_heads * head_v_dim;
  const int64_t row_base = token * hidden;
  if (lane < half_group) {
    const int local0 = lane;
    const int local1 = lane + half_group;
    const int col0 = base + local0;
    const int col1 = base + local1;
    const float gated0 = recurrent[recurrent_base + local0] * inv_rms * norm_weight[local0] *
        silu_f32(static_cast<float>(gate[recurrent_base + local0]));
    const float gated1 = recurrent[recurrent_base + local1] * inv_rms * norm_weight[local1] *
        silu_f32(static_cast<float>(gate[recurrent_base + local1]));
    const half_t rounded0 = static_cast<half_t>(gated0);
    const half_t rounded1 = static_cast<half_t>(gated1);
    buf[local0] = static_cast<float>(rounded0) * static_cast<float>(scales[col0]);
    buf[local1] = static_cast<float>(rounded1) * static_cast<float>(scales[col1]);
  }
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    if (lane < half_group) {
      const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
      const int i = static_cast<int>(pairs[pair_base + 0]);
      const int j = static_cast<int>(pairs[pair_base + 1]);
      const float angle = static_cast<float>(theta[r * (hidden / 2) + group * half_group + lane]);
      float s, c;
      sincosf(angle, &s, &c);
      const float xi = buf[i];
      const float xj = buf[j];
      buf[i] = fmaf(xj, s, xi * c);
      buf[j] = fmaf(xi, -s, xj * c);
    }
    __syncthreads();
  }

  if (lane < half_group) {
    const int col0 = base + lane;
    const int col1 = col0 + half_group;
    out_rot[row_base + col0] = static_cast<half_t>(buf[lane]);
    out_rot[row_base + col1] = static_cast<half_t>(buf[lane + half_group]);
  }
}

// One-launch parent-indexed tree GatedDeltaNet recurrence.
//
// Ported from nano-vllm-amd@b95eaa5. Each block owns one (v_head, dv) pair
// and walks t=0..N in topological order. RMSNorm+gate finalization is a second
// raw-pointer kernel over the fixed acc_buf supplied by the runtime.
template <typename scalar_t>
__global__ void qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ /* gate */,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ /* norm_weight */,
    const float* __restrict__ base_recurrent_state,
    float* __restrict__ tree_recurrent_state,
    const int64_t* __restrict__ parent_ids,
    float* __restrict__ acc_buf,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  constexpr int BLOCK = 64;
  constexpr int DK_MAX = 4;
  __shared__ float partial[BLOCK];

  const int64_t v_head = blockIdx.x;
  const int64_t dv_idx = blockIdx.y;
  if (v_head >= num_v_heads || dv_idx >= head_v_dim) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t conv_stride = (2 * key_dim) + value_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_head_base = v_head * head_k_dim * head_v_dim;
  const float a_log_v = a_log[v_head];
  const float dt_bias_v = dt_bias[v_head];
  const int tid = static_cast<int>(threadIdx.x);
  const int dk_per_thread = static_cast<int>(head_k_dim / BLOCK);

  for (int64_t t = 0; t < max_nodes; ++t) {
    const int64_t parent = parent_ids[t];
    const int64_t conv_row = t * conv_stride;
    const int64_t q_base = conv_row + k_head * head_k_dim;
    const int64_t k_base = conv_row + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_row + 2 * key_dim + v_head * head_v_dim;
    const float* parent_state = parent < 0
        ? base_recurrent_state + state_head_base
        : tree_recurrent_state + parent * state_stride + state_head_base;
    float* node_state = tree_recurrent_state + t * state_stride + state_head_base;

    float q_local[DK_MAX];
    float k_local[DK_MAX];
    float ps_local[DK_MAX];
    float q_sum_local = 0.0f;
    float k_sum_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < DK_MAX; ++i) {
      const int dk = tid * dk_per_thread + i;
      if (i < dk_per_thread && dk < head_k_dim) {
        q_local[i] = conv_out[q_base + dk];
        k_local[i] = conv_out[k_base + dk];
        ps_local[i] = parent_state[dk * head_v_dim + dv_idx];
        q_sum_local += q_local[i] * q_local[i];
        k_sum_local += k_local[i] * k_local[i];
      } else {
        q_local[i] = 0.0f;
        k_local[i] = 0.0f;
        ps_local[i] = 0.0f;
      }
    }

    partial[tid] = q_sum_local;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    const float q_sum = partial[0];
    __syncthreads();
    partial[tid] = k_sum_local;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    const float k_sum = partial[0];
    __syncthreads();

    const float q_scale = (1.0f / fmaxf(sqrtf(q_sum), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_sum), 1.0e-6f);
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[t * num_v_heads + v_head]));
    const float decay = expf(-expf(a_log_v) * softplus_f32(scalar_to_float_qwen35(a[t * num_v_heads + v_head]) + dt_bias_v));

    float kv_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < DK_MAX; ++i) {
      if (i < dk_per_thread) {
        const float kn = k_local[i] * k_scale;
        const float ds = ps_local[i] * decay;
        kv_local += kn * ds;
      }
    }
    partial[tid] = kv_local;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    const float kv_mem = partial[0];
    __syncthreads();

    const float value = conv_out[value_offset + dv_idx];
    const float delta = (value - kv_mem) * beta;
    float out_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < DK_MAX; ++i) {
      const int dk = tid * dk_per_thread + i;
      if (i < dk_per_thread && dk < head_k_dim) {
        const float new_state = ps_local[i] * decay + k_local[i] * k_scale * delta;
        node_state[dk * head_v_dim + dv_idx] = new_state;
        out_local += new_state * (q_local[i] * q_scale);
      }
    }
    partial[tid] = out_local;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      acc_buf[(t * num_v_heads + v_head) * head_v_dim + dv_idx] = partial[0];
    }
    __syncthreads();
  }
}

// Single-chain verifier variant.  For parent_ids == [-1,0,1,...], keep the
// recurrent state for each (v_head, dv, dk) lane in registers across t, but
// materialize every row so partial accepts can commit without a replay.
// dv-tiling: each block owns VTILE consecutive dv columns of one v_head, so the
// per-(v_head,t) work (q/k load, q_sum/k_sum reductions, q_scale/k_scale, and
// the beta/decay transcendentals) is computed ONCE and reused across the tile
// instead of redundantly by every single-dv block. Grid drops num_v_heads*head_v_dim
// -> num_v_heads*(head_v_dim/VTILE) (lower dispatch), and the VTILE consecutive
// dv writes into chain_recurrent_state[...+dk*head_v_dim+dv_idx] form a contiguous
// span (coalesced) instead of single strided floats. Per-dv math is unchanged.
template <typename scalar_t, int VTILE>
__global__ void qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ /* gate */,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ /* norm_weight */,
    const float* __restrict__ base_recurrent_state,
    float* __restrict__ chain_recurrent_state,
    float* __restrict__ acc_buf,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int BLOCK = static_cast<int>(blockDim.x);
  constexpr int DK_MAX = 4;
  __shared__ float lds[2];

  const int64_t v_head = blockIdx.x;
  const int64_t dv_base = static_cast<int64_t>(blockIdx.y) * VTILE;
  if (v_head >= num_v_heads || dv_base + VTILE > head_v_dim) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t conv_stride = (2 * key_dim) + value_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t state_head_base = v_head * head_k_dim * head_v_dim;
  const float a_log_v = a_log[v_head];
  const float dt_bias_v = dt_bias[v_head];
  const int tid = static_cast<int>(threadIdx.x);
  const int dk_per_thread = static_cast<int>(head_k_dim / BLOCK);

  float ps_local[DK_MAX][VTILE];
  #pragma unroll
  for (int i = 0; i < DK_MAX; ++i) {
    const int dk = tid * dk_per_thread + i;
    const bool dk_ok = (i < dk_per_thread && dk < head_k_dim);
    #pragma unroll
    for (int j = 0; j < VTILE; ++j) {
      ps_local[i][j] = dk_ok
          ? base_recurrent_state[state_head_base + dk * head_v_dim + (dv_base + j)]
          : 0.0f;
    }
  }

  for (int64_t t = 0; t < max_nodes; ++t) {
    const int64_t conv_row = t * conv_stride;
    const int64_t q_base = conv_row + k_head * head_k_dim;
    const int64_t k_base = conv_row + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_row + 2 * key_dim + v_head * head_v_dim;

    float q_local[DK_MAX];
    float k_local[DK_MAX];
    float q_sum_local = 0.0f;
    float k_sum_local = 0.0f;
    #pragma unroll
    for (int i = 0; i < DK_MAX; ++i) {
      const int dk = tid * dk_per_thread + i;
      if (i < dk_per_thread && dk < head_k_dim) {
        q_local[i] = conv_out[q_base + dk];
        k_local[i] = conv_out[k_base + dk];
        q_sum_local += q_local[i] * q_local[i];
        k_sum_local += k_local[i] * k_local[i];
      } else {
        q_local[i] = 0.0f;
        k_local[i] = 0.0f;
      }
    }

    const float q_sum = block_all_reduce_sum(q_sum_local, lds, tid, BLOCK);
    const float k_sum = block_all_reduce_sum(k_sum_local, lds, tid, BLOCK);

    const float q_scale = (1.0f / fmaxf(sqrtf(q_sum), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_sum), 1.0e-6f);
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[t * num_v_heads + v_head]));
    const float decay = expf(-expf(a_log_v) * softplus_f32(scalar_to_float_qwen35(a[t * num_v_heads + v_head]) + dt_bias_v));

    #pragma unroll
    for (int j = 0; j < VTILE; ++j) {
      const int64_t dv_idx = dv_base + j;
      float kv_local = 0.0f;
      #pragma unroll
      for (int i = 0; i < DK_MAX; ++i) {
        if (i < dk_per_thread) {
          const float kn = k_local[i] * k_scale;
          const float ds = ps_local[i][j] * decay;
          kv_local += kn * ds;
        }
      }
      const float kv_mem = block_all_reduce_sum(kv_local, lds, tid, BLOCK);

      const float value = conv_out[value_offset + dv_idx];
      const float delta = (value - kv_mem) * beta;
      float out_local = 0.0f;
      #pragma unroll
      for (int i = 0; i < DK_MAX; ++i) {
        const int dk = tid * dk_per_thread + i;
        if (i < dk_per_thread && dk < head_k_dim) {
          const float new_state = ps_local[i][j] * decay + k_local[i] * k_scale * delta;
          ps_local[i][j] = new_state;
          chain_recurrent_state[t * state_stride + state_head_base + dk * head_v_dim + dv_idx] = new_state;
          out_local += new_state * (q_local[i] * q_scale);
        }
      }
      const float out_red = block_all_reduce_sum(out_local, lds, tid, BLOCK);
      if (tid == 0) {
        acc_buf[(t * num_v_heads + v_head) * head_v_dim + dv_idx] = out_red;
      }
    }
  }
}


template <typename scalar_t>
__global__ void qwen35_gdn_tree_rmsnorm_gate_finalize_kernel(
    const float* __restrict__ acc_buf,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ norm_weight,
    float* __restrict__ out,
    float eps,
    int64_t num_v_heads,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* partial = shared;
  const int64_t v_head = blockIdx.x;
  const int64_t t = blockIdx.y;
  const int64_t tid = threadIdx.x;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const float* row = acc_buf + (t * num_v_heads + v_head) * head_v_dim;
  const float val = (tid < head_v_dim) ? row[tid] : 0.0f;
  partial[tid] = val * val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid < head_v_dim) {
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_v_dim) + eps);
    const int64_t pos = v_head * head_v_dim + tid;
    out[t * value_dim + pos] = val * inv_rms * norm_weight[tid] * silu_f32(scalar_to_float_qwen35(gate[t * value_dim + pos]));
  }
}

bool valid_gdn_shape(int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim) {
  return num_k_heads > 0 && num_v_heads > 0 && (num_v_heads % num_k_heads) == 0 &&
      head_k_dim > 0 && head_v_dim > 0 && head_v_dim <= 128;
}

bool valid_prefill_shape(int64_t tokens, int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim) {
  return tokens > 0 && num_k_heads > 0 && num_v_heads > 0 && (num_v_heads % num_k_heads) == 0 &&
      head_k_dim > 0 && head_v_dim > 0;
}

template <bool mutate_state, typename state_t>
__device__ __forceinline__ float qwen35_gdn_state_read(
    const state_t* state_read_base,
    volatile state_t* recurrent_state_v,
    int64_t offset) {
  return mutate_state ? scalar_to_float_qwen35<state_t>(recurrent_state_v[offset])
                      : scalar_to_float_qwen35<state_t>(state_read_base[offset]);
}

template <typename scalar_t, bool mutate_state, typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const state_t* recurrent_state_initial,
    state_t* recurrent_state_mutable,
    state_t* recurrent_state_rows,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t v_head = blockIdx.x;
  if (v_head >= num_v_heads) {
    return;
  }
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t value_idx = threadIdx.x;
  volatile state_t* recurrent_state_v = recurrent_state_mutable;

  for (int64_t token = 0; token < tokens; ++token) {
    const int64_t conv_base = token * qkv_width;
    const int64_t q_base = conv_base + k_head * head_k_dim;
    const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_base + 2 * key_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = conv_out[q_base + idx + 0];
      const float q1 = conv_out[q_base + idx + 1];
      const float q2 = conv_out[q_base + idx + 2];
      const float q3 = conv_out[q_base + idx + 3];
      const float q4 = conv_out[q_base + idx + 4];
      const float q5 = conv_out[q_base + idx + 5];
      const float q6 = conv_out[q_base + idx + 6];
      const float q7 = conv_out[q_base + idx + 7];
      const float k0 = conv_out[k_base + idx + 0];
      const float k1 = conv_out[k_base + idx + 1];
      const float k2 = conv_out[k_base + idx + 2];
      const float k3 = conv_out[k_base + idx + 3];
      const float k4 = conv_out[k_base + idx + 4];
      const float k5 = conv_out[k_base + idx + 5];
      const float k6 = conv_out[k_base + idx + 6];
      const float k7 = conv_out[k_base + idx + 7];
      q_shared[idx + 0] = q0;
      q_shared[idx + 1] = q1;
      q_shared[idx + 2] = q2;
      q_shared[idx + 3] = q3;
      q_shared[idx + 4] = q4;
      q_shared[idx + 5] = q5;
      q_shared[idx + 6] = q6;
      q_shared[idx + 7] = q7;
      k_shared[idx + 0] = k0;
      k_shared[idx + 1] = k1;
      k_shared[idx + 2] = k2;
      k_shared[idx + 3] = k3;
      k_shared[idx + 4] = k4;
      k_shared[idx + 5] = k5;
      k_shared[idx + 6] = k6;
      k_shared[idx + 7] = k7;
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_shared[idx] = q_value;
      k_shared[idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
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

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
      const state_t* state_read_base = mutate_state
          ? recurrent_state_initial
          : (token == 0 ? recurrent_state_initial : recurrent_state_rows + (token - 1) * state_stride);
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 0) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 1] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 1) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 2] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 2) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 3] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 3) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 4] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 4) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 5] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 5) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 6] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 6) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 7] * k_scale * qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + (idx + 7) * head_v_dim) * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_col + idx * head_v_dim) * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = conv_out[value_offset + value_idx];
      const float delta = (value - kv_mem) * beta;
      const int64_t row_state_base = token * state_stride;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float ns0 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so0) * decay + k_norm0 * delta;
        const float ns1 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so1) * decay + k_norm1 * delta;
        const float ns2 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so2) * decay + k_norm2 * delta;
        const float ns3 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so3) * decay + k_norm3 * delta;
        const float ns4 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so4) * decay + k_norm4 * delta;
        const float ns5 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so5) * decay + k_norm5 * delta;
        const float ns6 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so6) * decay + k_norm6 * delta;
        const float ns7 = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, so7) * decay + k_norm7 * delta;
        if (mutate_state) {
          gdn_state_write_qwen35(recurrent_state_v, so0, ns0);
          gdn_state_write_qwen35(recurrent_state_v, so1, ns1);
          gdn_state_write_qwen35(recurrent_state_v, so2, ns2);
          gdn_state_write_qwen35(recurrent_state_v, so3, ns3);
          gdn_state_write_qwen35(recurrent_state_v, so4, ns4);
          gdn_state_write_qwen35(recurrent_state_v, so5, ns5);
          gdn_state_write_qwen35(recurrent_state_v, so6, ns6);
          gdn_state_write_qwen35(recurrent_state_v, so7, ns7);
        }
        if (recurrent_state_rows != nullptr) {
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so0, ns0);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so1, ns1);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so2, ns2);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so3, ns3);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so4, ns4);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so5, ns5);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so6, ns6);
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so7, ns7);
        }
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = qwen35_gdn_state_read<mutate_state, state_t>(state_read_base, recurrent_state_v, state_offset) * decay + k_norm * delta;
        if (mutate_state) {
          gdn_state_write_qwen35(recurrent_state_v, state_offset, new_state);
        }
        if (recurrent_state_rows != nullptr) {
          gdn_state_write_qwen35(recurrent_state_rows, row_state_base + state_offset, new_state);
        }
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = v_head * head_v_dim + value_idx;
      const int64_t out_pos = (token * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float gated = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[token * num_v_heads * head_v_dim + pos]));
      out[out_pos] = float_to_bf16_bits(gated);
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename state_t = float>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    state_t* recurrent_state,
    state_t* recurrent_state_rows,
    float* out,
    uint16_t* out_bf16,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    int64_t segments,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (segment >= segments || v_head >= num_v_heads) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t value_idx = threadIdx.x;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  volatile state_t* recurrent_state_v = recurrent_state;

  for (int64_t token = start; token < end; ++token) {
    const int64_t conv_base = token * qkv_width;
    const int64_t q_base = conv_base + k_head * head_k_dim;
    const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_base + 2 * key_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = conv_out[q_base + idx + 0];
      const float q1 = conv_out[q_base + idx + 1];
      const float q2 = conv_out[q_base + idx + 2];
      const float q3 = conv_out[q_base + idx + 3];
      const float q4 = conv_out[q_base + idx + 4];
      const float q5 = conv_out[q_base + idx + 5];
      const float q6 = conv_out[q_base + idx + 6];
      const float q7 = conv_out[q_base + idx + 7];
      const float k0 = conv_out[k_base + idx + 0];
      const float k1 = conv_out[k_base + idx + 1];
      const float k2 = conv_out[k_base + idx + 2];
      const float k3 = conv_out[k_base + idx + 3];
      const float k4 = conv_out[k_base + idx + 4];
      const float k5 = conv_out[k_base + idx + 5];
      const float k6 = conv_out[k_base + idx + 6];
      const float k7 = conv_out[k_base + idx + 7];
      q_shared[idx + 0] = q0;
      q_shared[idx + 1] = q1;
      q_shared[idx + 2] = q2;
      q_shared[idx + 3] = q3;
      q_shared[idx + 4] = q4;
      q_shared[idx + 5] = q5;
      q_shared[idx + 6] = q6;
      q_shared[idx + 7] = q7;
      k_shared[idx + 0] = k0;
      k_shared[idx + 1] = k1;
      k_shared[idx + 2] = k2;
      k_shared[idx + 3] = k3;
      k_shared[idx + 4] = k4;
      k_shared[idx + 5] = k5;
      k_shared[idx + 6] = k6;
      k_shared[idx + 7] = k7;
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_shared[idx] = q_value;
      k_shared[idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
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

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_local_col = v_head * head_k_dim * head_v_dim + value_idx;
      const int64_t state_col = state_slot * state_stride + state_local_col;
      const int64_t row_state_col = token * state_stride + state_local_col;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 0) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 1] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 1) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 2] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 2) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 3] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 3) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 4] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 4) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 5] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 5) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 6] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 6) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 7] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 7) * head_v_dim) * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = gdn_state_read_qwen35(recurrent_state_v, state_col + idx * head_v_dim) * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = conv_out[value_offset + value_idx];
      const float delta = (value - kv_mem) * beta;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float ns0 = gdn_state_read_qwen35(recurrent_state_v, so0) * decay + k_norm0 * delta;
        const float ns1 = gdn_state_read_qwen35(recurrent_state_v, so1) * decay + k_norm1 * delta;
        const float ns2 = gdn_state_read_qwen35(recurrent_state_v, so2) * decay + k_norm2 * delta;
        const float ns3 = gdn_state_read_qwen35(recurrent_state_v, so3) * decay + k_norm3 * delta;
        const float ns4 = gdn_state_read_qwen35(recurrent_state_v, so4) * decay + k_norm4 * delta;
        const float ns5 = gdn_state_read_qwen35(recurrent_state_v, so5) * decay + k_norm5 * delta;
        const float ns6 = gdn_state_read_qwen35(recurrent_state_v, so6) * decay + k_norm6 * delta;
        const float ns7 = gdn_state_read_qwen35(recurrent_state_v, so7) * decay + k_norm7 * delta;
        gdn_state_write_qwen35(recurrent_state_v, so0, ns0);
        gdn_state_write_qwen35(recurrent_state_v, so1, ns1);
        gdn_state_write_qwen35(recurrent_state_v, so2, ns2);
        gdn_state_write_qwen35(recurrent_state_v, so3, ns3);
        gdn_state_write_qwen35(recurrent_state_v, so4, ns4);
        gdn_state_write_qwen35(recurrent_state_v, so5, ns5);
        gdn_state_write_qwen35(recurrent_state_v, so6, ns6);
        gdn_state_write_qwen35(recurrent_state_v, so7, ns7);
        if (recurrent_state_rows != nullptr) {
          const int64_t row_base = row_state_col + idx * head_v_dim;
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 0 * head_v_dim, ns0);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 1 * head_v_dim, ns1);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 2 * head_v_dim, ns2);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 3 * head_v_dim, ns3);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 4 * head_v_dim, ns4);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 5 * head_v_dim, ns5);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 6 * head_v_dim, ns6);
          gdn_state_write_qwen35(recurrent_state_rows, row_base + 7 * head_v_dim, ns7);
        }
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = gdn_state_read_qwen35(recurrent_state_v, state_offset) * decay + k_norm * delta;
        gdn_state_write_qwen35(recurrent_state_v, state_offset, new_state);
        if (recurrent_state_rows != nullptr) {
          gdn_state_write_qwen35(recurrent_state_rows, row_state_col + idx * head_v_dim, new_state);
        }
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = v_head * head_v_dim + value_idx;
      const int64_t out_pos = (token * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float gated = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[token * num_v_heads * head_v_dim + pos]));
      if (out != nullptr) {
        out[out_pos] = gated;
      }
      if (out_bf16 != nullptr) {
        out_bf16[out_pos] = float_to_bf16_bits(gated);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename state_t = float, bool WAVE_REDUCE = false>
__global__ void qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_state_rows_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const state_t* recurrent_state_initial,
    state_t* recurrent_state_rows,
    uint16_t* out,
    float* out_f32,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    float eps,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (segment >= segments || v_head >= num_v_heads) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t value_idx = threadIdx.x;

  for (int64_t token = start; token < end; ++token) {
    const int64_t conv_base = token * qkv_width;
    const int64_t q_base = conv_base + k_head * head_k_dim;
    const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_base + 2 * key_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = conv_out[q_base + idx + 0];
      const float q1 = conv_out[q_base + idx + 1];
      const float q2 = conv_out[q_base + idx + 2];
      const float q3 = conv_out[q_base + idx + 3];
      const float q4 = conv_out[q_base + idx + 4];
      const float q5 = conv_out[q_base + idx + 5];
      const float q6 = conv_out[q_base + idx + 6];
      const float q7 = conv_out[q_base + idx + 7];
      const float k0 = conv_out[k_base + idx + 0];
      const float k1 = conv_out[k_base + idx + 1];
      const float k2 = conv_out[k_base + idx + 2];
      const float k3 = conv_out[k_base + idx + 3];
      const float k4 = conv_out[k_base + idx + 4];
      const float k5 = conv_out[k_base + idx + 5];
      const float k6 = conv_out[k_base + idx + 6];
      const float k7 = conv_out[k_base + idx + 7];
      q_shared[idx + 0] = q0;
      q_shared[idx + 1] = q1;
      q_shared[idx + 2] = q2;
      q_shared[idx + 3] = q3;
      q_shared[idx + 4] = q4;
      q_shared[idx + 5] = q5;
      q_shared[idx + 6] = q6;
      q_shared[idx + 7] = q7;
      k_shared[idx + 0] = k0;
      k_shared[idx + 1] = k1;
      k_shared[idx + 2] = k2;
      k_shared[idx + 3] = k3;
      k_shared[idx + 4] = k4;
      k_shared[idx + 5] = k5;
      k_shared[idx + 6] = k6;
      k_shared[idx + 7] = k7;
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_shared[idx] = q_value;
      k_shared[idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
    }
    if constexpr (WAVE_REDUCE) {
      // Qwen3.8's K128 load gives lanes 0..15 one ordered eight-value
      // subtotal and lanes 16..31 zero. The first two local128 tree stages
      // therefore only add zero. A wave32 shuffle tree reproduces the
      // remaining 16/8/4/2/1 additions exactly and removes seven workgroup
      // barriers while the final barrier still publishes Q/K and the sums.
      if (threadIdx.x < 32) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          q_sum += __shfl_down(q_sum, offset, 32);
          k_sum += __shfl_down(k_sum, offset, 32);
        }
        if (threadIdx.x == 0) {
          q_partial[0] = q_sum;
          k_partial[0] = k_sum;
        }
      }
      __syncthreads();
    } else {
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
    }

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
      const state_t* state_read_base = token == start
          ? recurrent_state_initial + state_slot * state_stride
          : recurrent_state_rows + (token - 1) * state_stride;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 0) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 1] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 1) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 2] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 2) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 3] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 3) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 4] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 4) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 5] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 5) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 6] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 6) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 7] * k_scale * gdn_state_read_qwen35(state_read_base, state_col + (idx + 7) * head_v_dim) * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = gdn_state_read_qwen35(state_read_base, state_col + idx * head_v_dim) * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = conv_out[value_offset + value_idx];
      const float delta = (value - kv_mem) * beta;
      const int64_t row_state_base = token * state_stride;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float ns0 = gdn_state_read_qwen35(state_read_base, so0) * decay + k_norm0 * delta;
        const float ns1 = gdn_state_read_qwen35(state_read_base, so1) * decay + k_norm1 * delta;
        const float ns2 = gdn_state_read_qwen35(state_read_base, so2) * decay + k_norm2 * delta;
        const float ns3 = gdn_state_read_qwen35(state_read_base, so3) * decay + k_norm3 * delta;
        const float ns4 = gdn_state_read_qwen35(state_read_base, so4) * decay + k_norm4 * delta;
        const float ns5 = gdn_state_read_qwen35(state_read_base, so5) * decay + k_norm5 * delta;
        const float ns6 = gdn_state_read_qwen35(state_read_base, so6) * decay + k_norm6 * delta;
        const float ns7 = gdn_state_read_qwen35(state_read_base, so7) * decay + k_norm7 * delta;
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so0, ns0);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so1, ns1);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so2, ns2);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so3, ns3);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so4, ns4);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so5, ns5);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so6, ns6);
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + so7, ns7);
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = gdn_state_read_qwen35(state_read_base, state_offset) * decay + k_norm * delta;
        gdn_state_write_qwen35(recurrent_state_rows, row_state_base + state_offset, new_state);
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if constexpr (WAVE_REDUCE) {
      // Rebuild the parent's exact stride-64 and stride-32 subtrees from LDS,
      // then finish its 16/8/4/2/1 stages in wave 0.
      if (threadIdx.x < 32) {
        float norm_sum =
            norm_partial[threadIdx.x] + norm_partial[threadIdx.x + 64];
        const float norm_hi =
            norm_partial[threadIdx.x + 32] + norm_partial[threadIdx.x + 96];
        norm_sum += norm_hi;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          norm_sum += __shfl_down(norm_sum, offset, 32);
        }
        if (threadIdx.x == 0) {
          norm_partial[0] = norm_sum;
        }
      }
      __syncthreads();
    } else {
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
        }
        __syncthreads();
      }
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = v_head * head_v_dim + value_idx;
      const int64_t out_pos = (token * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float gated = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[token * num_v_heads * head_v_dim + pos]));
      out[out_pos] = float_to_bf16_bits(gated);
      if (out_f32 != nullptr) {
        out_f32[out_pos] = gated;
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename state_t = float>
__global__ void qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    state_t* recurrent_state,
    uint16_t* out,
    const int32_t* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices,
    float eps,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + head_k_dim;
  float* q_partial = k_shared + head_k_dim;
  float* k_partial = q_partial + blockDim.x;
  float* out_shared = k_partial + blockDim.x;
  float* norm_partial = out_shared + head_v_dim;

  const int64_t segment = blockIdx.x;
  const int64_t v_head = blockIdx.y;
  if (segment >= segments || v_head >= num_v_heads) {
    return;
  }
  const int64_t start = static_cast<int64_t>(cu_seqlens[segment]);
  const int64_t end = static_cast<int64_t>(cu_seqlens[segment + 1]);
  if (end <= start) {
    return;
  }
  const int64_t state_slot = state_indices[segment];
  const int64_t k_head = qwen35_gdn_k_head_from_v_head(v_head, num_k_heads, num_v_heads);
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  const int64_t value_idx = threadIdx.x;
  volatile state_t* recurrent_state_v = recurrent_state;

  for (int64_t token = start; token < end; ++token) {
    const int64_t conv_base = token * qkv_width;
    const int64_t q_base = conv_base + k_head * head_k_dim;
    const int64_t k_base = conv_base + key_dim + k_head * head_k_dim;
    const int64_t value_offset = conv_base + 2 * key_dim + v_head * head_v_dim;

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int64_t idx = threadIdx.x * 8; idx + 7 < head_k_dim; idx += blockDim.x * 8) {
      const float q0 = conv_out[q_base + idx + 0];
      const float q1 = conv_out[q_base + idx + 1];
      const float q2 = conv_out[q_base + idx + 2];
      const float q3 = conv_out[q_base + idx + 3];
      const float q4 = conv_out[q_base + idx + 4];
      const float q5 = conv_out[q_base + idx + 5];
      const float q6 = conv_out[q_base + idx + 6];
      const float q7 = conv_out[q_base + idx + 7];
      const float k0 = conv_out[k_base + idx + 0];
      const float k1 = conv_out[k_base + idx + 1];
      const float k2 = conv_out[k_base + idx + 2];
      const float k3 = conv_out[k_base + idx + 3];
      const float k4 = conv_out[k_base + idx + 4];
      const float k5 = conv_out[k_base + idx + 5];
      const float k6 = conv_out[k_base + idx + 6];
      const float k7 = conv_out[k_base + idx + 7];
      q_shared[idx + 0] = q0;
      q_shared[idx + 1] = q1;
      q_shared[idx + 2] = q2;
      q_shared[idx + 3] = q3;
      q_shared[idx + 4] = q4;
      q_shared[idx + 5] = q5;
      q_shared[idx + 6] = q6;
      q_shared[idx + 7] = q7;
      k_shared[idx + 0] = k0;
      k_shared[idx + 1] = k1;
      k_shared[idx + 2] = k2;
      k_shared[idx + 3] = k3;
      k_shared[idx + 4] = k4;
      k_shared[idx + 5] = k5;
      k_shared[idx + 6] = k6;
      k_shared[idx + 7] = k7;
      q_sum += q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4 + q5*q5 + q6*q6 + q7*q7;
      k_sum += k0*k0 + k1*k1 + k2*k2 + k3*k3 + k4*k4 + k5*k5 + k6*k6 + k7*k7;
    }
    for (int64_t idx = (head_k_dim & ~7) + threadIdx.x; idx < head_k_dim; idx += blockDim.x) {
      const float q_value = conv_out[q_base + idx];
      const float k_value = conv_out[k_base + idx];
      q_shared[idx] = q_value;
      k_shared[idx] = k_value;
      q_sum += q_value * q_value;
      k_sum += k_value * k_value;
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

    const float q_scale = (1.0f / fmaxf(sqrtf(q_partial[0]), 1.0e-6f)) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = 1.0f / fmaxf(sqrtf(k_partial[0]), 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = state_slot * state_stride + v_head * head_k_dim * head_v_dim + value_idx;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 0) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 1] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 1) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 2] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 2) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 3] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 3) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 4] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 4) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 5] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 5) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 6] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 6) * head_v_dim) * decay;
        kv_mem += k_shared[idx + 7] * k_scale * gdn_state_read_qwen35(recurrent_state_v, state_col + (idx + 7) * head_v_dim) * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = gdn_state_read_qwen35(recurrent_state_v, state_col + idx * head_v_dim) * decay;
        kv_mem += k_norm * state_value;
      }

      const float value = conv_out[value_offset + value_idx];
      const float delta = (value - kv_mem) * beta;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        const float q_norm0 = q_shared[idx + 0] * q_scale;
        const float q_norm1 = q_shared[idx + 1] * q_scale;
        const float q_norm2 = q_shared[idx + 2] * q_scale;
        const float q_norm3 = q_shared[idx + 3] * q_scale;
        const float q_norm4 = q_shared[idx + 4] * q_scale;
        const float q_norm5 = q_shared[idx + 5] * q_scale;
        const float q_norm6 = q_shared[idx + 6] * q_scale;
        const float q_norm7 = q_shared[idx + 7] * q_scale;
        const float k_norm0 = k_shared[idx + 0] * k_scale;
        const float k_norm1 = k_shared[idx + 1] * k_scale;
        const float k_norm2 = k_shared[idx + 2] * k_scale;
        const float k_norm3 = k_shared[idx + 3] * k_scale;
        const float k_norm4 = k_shared[idx + 4] * k_scale;
        const float k_norm5 = k_shared[idx + 5] * k_scale;
        const float k_norm6 = k_shared[idx + 6] * k_scale;
        const float k_norm7 = k_shared[idx + 7] * k_scale;
        const int64_t so0 = state_col + (idx + 0) * head_v_dim;
        const int64_t so1 = state_col + (idx + 1) * head_v_dim;
        const int64_t so2 = state_col + (idx + 2) * head_v_dim;
        const int64_t so3 = state_col + (idx + 3) * head_v_dim;
        const int64_t so4 = state_col + (idx + 4) * head_v_dim;
        const int64_t so5 = state_col + (idx + 5) * head_v_dim;
        const int64_t so6 = state_col + (idx + 6) * head_v_dim;
        const int64_t so7 = state_col + (idx + 7) * head_v_dim;
        const float ns0 = gdn_state_read_qwen35(recurrent_state_v, so0) * decay + k_norm0 * delta;
        const float ns1 = gdn_state_read_qwen35(recurrent_state_v, so1) * decay + k_norm1 * delta;
        const float ns2 = gdn_state_read_qwen35(recurrent_state_v, so2) * decay + k_norm2 * delta;
        const float ns3 = gdn_state_read_qwen35(recurrent_state_v, so3) * decay + k_norm3 * delta;
        const float ns4 = gdn_state_read_qwen35(recurrent_state_v, so4) * decay + k_norm4 * delta;
        const float ns5 = gdn_state_read_qwen35(recurrent_state_v, so5) * decay + k_norm5 * delta;
        const float ns6 = gdn_state_read_qwen35(recurrent_state_v, so6) * decay + k_norm6 * delta;
        const float ns7 = gdn_state_read_qwen35(recurrent_state_v, so7) * decay + k_norm7 * delta;
        gdn_state_write_qwen35(recurrent_state_v, so0, ns0);
        gdn_state_write_qwen35(recurrent_state_v, so1, ns1);
        gdn_state_write_qwen35(recurrent_state_v, so2, ns2);
        gdn_state_write_qwen35(recurrent_state_v, so3, ns3);
        gdn_state_write_qwen35(recurrent_state_v, so4, ns4);
        gdn_state_write_qwen35(recurrent_state_v, so5, ns5);
        gdn_state_write_qwen35(recurrent_state_v, so6, ns6);
        gdn_state_write_qwen35(recurrent_state_v, so7, ns7);
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = gdn_state_read_qwen35(recurrent_state_v, state_offset) * decay + k_norm * delta;
        gdn_state_write_qwen35(recurrent_state_v, state_offset, new_state);
        out_acc += q_norm * new_state;
      }
      out_shared[value_idx] = out_acc;
      norm_partial[threadIdx.x] = out_acc * out_acc;
    } else {
      norm_partial[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        norm_partial[threadIdx.x] += norm_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (value_idx < head_v_dim) {
      const int64_t pos = v_head * head_v_dim + value_idx;
      const int64_t out_pos = (token * num_v_heads + v_head) * head_v_dim + value_idx;
      const float inv_rms = rsqrtf((norm_partial[0] / static_cast<float>(head_v_dim)) + eps);
      const float gated = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[token * num_v_heads * head_v_dim + pos]));
      out[out_pos] = float_to_bf16_bits(gated);
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      nullptr,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    float* out,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t, false, half_t>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      nullptr,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!out_bf16 ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      out_bf16,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!out_bf16 ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t, true, half_t>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      out_bf16,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(
    const float* conv_out,
    const half_t* gate,
    const half_t* a,
    const half_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<half_t>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      nullptr,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* tree_recurrent_state,
    const int64_t* parent_ids,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (gdn_tloop_c1_exact_enabled()) {
    const int threads = 128;
    const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
    hipLaunchKernelGGL(
        (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<scalar_t, float>),
        dim3(static_cast<unsigned int>(num_v_heads)),
        dim3(threads),
        shared_bytes,
        stream,
        conv_out,
        gate,
        a,
        b,
        dt_bias,
        a_log,
        norm_weight,
        base_recurrent_state,
        tree_recurrent_state,
        nullptr,
        parent_ids,
        out,
        nullptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim);
    return static_cast<int>(hipGetLastError());
  }
  const int threads = 64;
  hipLaunchKernelGGL(
      (qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_kernel<scalar_t>),
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(head_v_dim)),
      dim3(threads),
      0,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      tree_recurrent_state,
      parent_ids,
      acc_buf,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int finalize_threads = 128;
  const size_t finalize_shared = static_cast<size_t>(finalize_threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_tree_rmsnorm_gate_finalize_kernel<scalar_t>),
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(max_nodes)),
      dim3(finalize_threads),
      finalize_shared,
      stream,
      acc_buf,
      gate,
      norm_weight,
      out,
      eps,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (gdn_tloop_c1_exact_enabled()) {
    const int threads = 128;
    const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
    hipLaunchKernelGGL(
        (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<scalar_t, float>),
        dim3(static_cast<unsigned int>(num_v_heads)),
        dim3(threads),
        shared_bytes,
        stream,
        conv_out,
        gate,
        a,
        b,
        dt_bias,
        a_log,
        norm_weight,
        base_recurrent_state,
        leaf_recurrent_state,
        nullptr,
        nullptr,
        out,
        nullptr,
        eps,
        max_nodes,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim);
    return static_cast<int>(hipGetLastError());
  }
  // One wavefront/block (gfx1100 warpSize=32) when head_k_dim<=128 so the
  // per-token reductions are pure warp shuffles (no LDS, no __syncthreads);
  // fall back to 64 for head_k_dim in {192,256} to keep dk_per_thread<=DK_MAX.
  const int threads = (head_k_dim <= 128) ? 32 : 64;
  // dv-tiling: collapse the grid by VTILE (4096 -> 1024 blocks at 32x128) when
  // head_v_dim is divisible; VTILE=1 is the bit-identical unfused fallback.
  constexpr int kChainVTile = 4;
  int requested_vtile = kChainVTile;
  if (const char* raw_vtile = std::getenv("HIPENGINE_GDN_CHAIN_TLOOP_VTILE")) {
    const int parsed = std::atoi(raw_vtile);
    if (parsed == 1 || parsed == kChainVTile) {
      requested_vtile = parsed;
    }
  }
  const bool use_tile = requested_vtile == kChainVTile && (head_v_dim % kChainVTile) == 0;
#define HIPENGINE_LAUNCH_GDN_CHAIN_TLOOP(VT, GRID_Y)                              \
  hipLaunchKernelGGL(                                                            \
      (qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_kernel<scalar_t, VT>), \
      dim3(static_cast<unsigned int>(num_v_heads),                              \
           static_cast<unsigned int>(GRID_Y)),                                  \
      dim3(threads),                                                            \
      0,                                                                        \
      stream,                                                                   \
      conv_out, gate, a, b, dt_bias, a_log, norm_weight,                        \
      base_recurrent_state, leaf_recurrent_state, acc_buf,                      \
      max_nodes, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
  if (use_tile) {
    HIPENGINE_LAUNCH_GDN_CHAIN_TLOOP(kChainVTile, head_v_dim / kChainVTile);
  } else {
    HIPENGINE_LAUNCH_GDN_CHAIN_TLOOP(1, head_v_dim);
  }
#undef HIPENGINE_LAUNCH_GDN_CHAIN_TLOOP
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int finalize_threads = 128;
  const size_t finalize_shared = static_cast<size_t>(finalize_threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_tree_rmsnorm_gate_finalize_kernel<scalar_t>),
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(max_nodes)),
      dim3(finalize_threads),
      finalize_shared,
      stream,
      acc_buf,
      gate,
      norm_weight,
      out,
      eps,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* tree_recurrent_state,
    const int64_t* parent_ids,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop<uint16_t>(
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      tree_recurrent_state,
      parent_ids,
      acc_buf,
      out,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16(
    const float* conv_out,
    const half_t* gate,
    const half_t* a,
    const half_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* tree_recurrent_state,
    const int64_t* parent_ids,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop<half_t>(
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      tree_recurrent_state,
      parent_ids,
      acc_buf,
      out,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop<uint16_t>(
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      acc_buf,
      out,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* /* acc_buf */,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<uint16_t, float>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      nullptr,
      nullptr,
      out,
      nullptr,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_bf16_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const half_t* base_recurrent_state,
    half_t* leaf_recurrent_state,
    float* /* acc_buf */,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (max_nodes <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<uint16_t, half_t, false, false>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      nullptr,
      nullptr,
      out,
      nullptr,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_f32_bf16_out(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* /* acc_buf */,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!out_bf16 || max_nodes <= 0 ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<
          uint16_t, float, true>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      nullptr,
      nullptr,
      out,
      out_bf16,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* initial_recurrent_state_snapshot,
    float* /* acc_buf */,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!initial_recurrent_state_snapshot || max_nodes <= 0 ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<
          uint16_t, float, false, true>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      initial_recurrent_state_snapshot,
      nullptr,
      out,
      nullptr,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_c1_exact_snapshot_tloop_f32_bf16_out(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* initial_recurrent_state_snapshot,
    float* /* acc_buf */,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!initial_recurrent_state_snapshot || !out_bf16 || max_nodes <= 0 ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim > 256 || (head_k_dim % 64) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<
          uint16_t, float, true, true>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      initial_recurrent_state_snapshot,
      nullptr,
      out,
      out_bf16,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out(
    const uint16_t* norm,
    const float* alpha_weight,
    const float* beta_weight,
    uint16_t* alpha_out,
    uint16_t* beta_out,
    const float* conv_out,
    const uint16_t* gate,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* initial_recurrent_state_snapshot,
    float* out,
    uint16_t* out_bf16,
    float eps,
    int64_t rows,
    int64_t in_features,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!norm || !alpha_weight || !beta_weight || !alpha_out || !beta_out ||
      !conv_out || !gate || !dt_bias || !a_log || !norm_weight ||
      !base_recurrent_state || !leaf_recurrent_state ||
      !initial_recurrent_state_snapshot || !out || !out_bf16 ||
      rows < 1 || rows > 4 || in_features != 5120 || num_k_heads != 16 ||
      num_v_heads != 48 || head_k_dim != 128 || head_v_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(
      (8 * threads) + 4 * (2 * head_k_dim + head_v_dim) +
      (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_alpha_beta_gdn_chain_snapshot_f32_bf16_out_kernel<uint16_t, true, true>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      norm,
      alpha_weight,
      beta_weight,
      alpha_out,
      beta_out,
      in_features,
      conv_out,
      gate,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      initial_recurrent_state_snapshot,
      nullptr,
      out,
      out_bf16,
      eps,
      rows,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16(
    const float* conv_out,
    const half_t* gate,
    const half_t* a,
    const half_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* base_recurrent_state,
    float* leaf_recurrent_state,
    float* acc_buf,
    float* out,
    float eps,
    int64_t max_nodes,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop<half_t>(
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      base_recurrent_state,
      leaf_recurrent_state,
      acc_buf,
      out,
      eps,
      max_nodes,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 || head_k_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = static_cast<int>(head_k_dim);
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_kernel,
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(head_v_dim)),
      dim3(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_k2_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 || head_k_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = static_cast<int>(head_k_dim / 2);
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_k2_kernel,
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(head_v_dim)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_k2_decode_order_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 || head_v_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 64;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_k2_decode_order_kernel,
      dim3(static_cast<unsigned int>(num_v_heads), static_cast<unsigned int>(head_v_dim)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_segments_k2_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0 || head_k_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = static_cast<int>(head_k_dim / 2);
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_k2_segments_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(head_v_dim)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_f32_bf16(
    const float* conv_out,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    float* query,
    float* key,
    float* value,
    float* beta,
    float* decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      query,
      key,
      value,
      beta,
      decay,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16(
    const float* conv_out,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    float* query,
    float* key,
    float* value,
    float* beta,
    float* decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      query,
      key,
      value,
      beta,
      decay,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16(
    const float* conv_out,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    float* query,
    float* key,
    float* value,
    float* beta,
    float* decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const int64_t grid_heads = num_k_heads > num_v_heads ? num_k_heads : num_v_heads;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_compact_peer_normalized_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(grid_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      query,
      key,
      value,
      beta,
      decay,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_f32_fp16(
    const float* conv_out,
    const half_t* a,
    const half_t* b,
    const float* dt_bias,
    const float* a_log,
    float* query,
    float* key,
    float* value,
    float* beta,
    float* decay,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_kernel<half_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      query,
      key,
      value,
      beta,
      decay,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16(
    const float* conv_out,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    float* query_raw,
    float* key_raw,
    float* value,
    float* beta,
    float* decay,
    float* query_scale,
    float* key_scale,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim != 128 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_raw_scales_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16(
    const float* conv_out,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    float* beta,
    float* decay,
    float* query_scale,
    float* key_scale,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      head_k_dim != 128 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const int64_t grid_heads = num_k_heads > num_v_heads ? num_k_heads : num_v_heads;
  const size_t shared_bytes = static_cast<size_t>(2 * threads) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_prefill_prepare_compact_scales_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(grid_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      a,
      b,
      dt_bias,
      a_log,
      beta,
      decay,
      query_scale,
      key_scale,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

static int launch_qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int value_tile,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128 ||
      (value_tile != 32 && value_tile != 64 && value_tile != 128)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + value_tile - 1) / value_tile)),
      dim3(static_cast<unsigned int>(value_tile)),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

template <int kValueTile>
static int launch_qwen35_gdn_prefill_recurrent_decode_order_exact_lds_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          qwen35_gdn_prefill_recurrent_decode_order_exact_lds_kernel<kValueTile>),
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + kValueTile - 1) / kValueTile)),
      dim3(kValueTile),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, tokens, num_v_heads, head_k_dim, head_v_dim, 128,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, tokens, num_v_heads, head_k_dim, head_v_dim, 64,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, tokens, num_v_heads, head_k_dim, head_v_dim, 32,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_lds_f32<64>(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, tokens, num_v_heads, head_k_dim, head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_lds_f32<32>(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, tokens, num_v_heads, head_k_dim, head_v_dim,
      stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32(
    const float* conv_out,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + 31) / 32)),
      dim3(32),
      0,
      stream,
      conv_out,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32(
    const float* conv_out,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + 31) / 32)),
      dim3(32),
      0,
      stream,
      conv_out,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

static int launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int value_tile,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128 ||
      (value_tile != 32 && value_tile != 64 && value_tile != 128)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_segments_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + value_tile - 1) / value_tile)),
      dim3(static_cast<unsigned int>(value_tile)),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

template <int kValueTile>
static int launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds_kernel<
              kValueTile>),
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + kValueTile - 1) / kValueTile)),
      dim3(kValueTile),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, cu_seqlens, state_indices, total_tokens, segments,
      num_v_heads, head_k_dim, head_v_dim, 128, stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, cu_seqlens, state_indices, total_tokens, segments,
      num_v_heads, head_k_dim, head_v_dim, 64, stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, cu_seqlens, state_indices, total_tokens, segments,
      num_v_heads, head_k_dim, head_v_dim, 32, stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds_f32<
      64>(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, cu_seqlens, state_indices, total_tokens, segments,
      num_v_heads, head_k_dim, head_v_dim, stream);
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  return launch_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds_f32<
      32>(
      query_raw, key_raw, value, beta, decay, query_scale, key_scale,
      recurrent_state, out, cu_seqlens, state_indices, total_tokens, segments,
      num_v_heads, head_k_dim, head_v_dim, stream);
}

extern "C" int
hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32(
    const float* conv_out,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_k_heads <= 0 ||
      num_v_heads <= 0 || head_k_dim != 128 || head_v_dim <= 0 ||
      head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + 31) / 32)),
      dim3(32),
      0,
      stream,
      conv_out,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32(
    const float* conv_out,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_k_heads <= 0 ||
      num_v_heads <= 0 || head_k_dim != 128 || head_v_dim <= 0 ||
      head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>((head_v_dim + 31) / 32)),
      dim3(32),
      0,
      stream,
      conv_out,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32(
    const float* query_raw,
    const float* key_raw,
    const float* value,
    const float* beta,
    const float* decay,
    const float* query_scale,
    const float* key_scale,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query_raw,
      key_raw,
      value,
      beta,
      decay,
      query_scale,
      key_scale,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_normalized_wave32_xor_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_fp16state(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    half_t* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_kernel<half_t>,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_k_heads <= 0 ||
      num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128 ||
      cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_kernel<
          float>),
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_fp16state(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    half_t* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_k_heads <= 0 ||
      num_v_heads <= 0 || (num_v_heads % num_k_heads) != 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128 ||
      cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_compact_normalized_segments_wave32_xor_kernel<
          half_t>),
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_k_heads,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 128;
  constexpr int kWavesPerBlock = kThreads / 32;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kWavesPerBlock - 1) / kWavesPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_normalized_cluster8_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kColumnsPerBlock = kThreads / 8;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_normalized_cluster8_kernel,
      dim3(
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kColumnsPerBlock - 1) / kColumnsPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || head_v_dim <= 0 || head_v_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kThreads = 256;
  constexpr int kColumnsPerBlock = kThreads / 8;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_kernel,
      dim3(
          static_cast<unsigned int>(segments),
          static_cast<unsigned int>(num_v_heads),
          static_cast<unsigned int>(
              (head_v_dim + kColumnsPerBlock - 1) / kColumnsPerBlock)),
      dim3(kThreads),
      0,
      stream,
      query,
      key,
      value,
      beta,
      decay,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      segments,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_rmsnorm_gate_bf16(
    const float* recurrent,
    const uint16_t* gate,
    const float* norm_weight,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_v_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel,
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      recurrent,
      gate,
      norm_weight,
      out,
      eps,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_rmsnorm_gate_fp16(
    const float* recurrent,
    const half_t* gate,
    const float* norm_weight,
    half_t* out,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_v_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_rmsnorm_gate_fp16_kernel,
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      recurrent,
      gate,
      norm_weight,
      out,
      eps,
      tokens,
      num_v_heads,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16(
    const float* recurrent,
    const half_t* gate,
    const float* norm_weight,
    half_t* out_rot,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    float eps,
    int64_t tokens,
    int64_t num_v_heads,
    int64_t head_v_dim,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (tokens <= 0 || num_v_heads <= 0 || head_v_dim <= 0 || group_size <= 0 || krot < 0 ||
      group_size != head_v_dim || (group_size % 2) != 0 || group_size > 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = (static_cast<size_t>(threads) + static_cast<size_t>(group_size)) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16_kernel,
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      recurrent,
      gate,
      norm_weight,
      out_rot,
      pairs,
      theta,
      scales,
      eps,
      tokens,
      num_v_heads,
      head_v_dim,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    const int64_t* state_indices,
    int64_t rows,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (rows <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel<uint16_t, 0, float>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      state_indices,
      rows,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    float* out,
    const int64_t* state_indices,
    int64_t rows,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (rows <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel<uint16_t, 0, half_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      state_indices,
      rows,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    const int64_t* state_indices,
    int64_t rows,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (rows <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  constexpr int state_cache_rows = 24;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim +
      state_cache_rows * head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel<
          uint16_t, state_cache_rows, float>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      state_indices,
      rows,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    float* out,
    const int64_t* state_indices,
    int64_t rows,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (rows <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  constexpr int state_cache_rows = 24;
  const size_t shared_bytes = static_cast<size_t>(
      (2 * head_k_dim) + (3 * threads) + head_v_dim +
      state_cache_rows * head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel<
          uint16_t, state_cache_rows, half_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      state_indices,
      rows,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      static_cast<float*>(nullptr),
      out,
      static_cast<uint16_t*>(nullptr),
      cu_seqlens,
      state_indices,
      segments,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16(
    const float* conv_out, const uint16_t* gate, const uint16_t* a, const uint16_t* b,
    const float* dt_bias, const float* a_log, const float* norm_weight,
    float* recurrent_state, float* recurrent_state_rows, float* out, uint16_t* out_bf16,
    const int32_t* cu_seqlens, const int64_t* state_indices,
    int64_t total_tokens, int64_t segments, float eps, int64_t num_k_heads,
    int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim, hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || recurrent_state_rows == nullptr ||
      !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads), shared_bytes, stream, conv_out, gate, a, b, dt_bias, a_log,
      norm_weight, recurrent_state, recurrent_state_rows, out, out_bf16, cu_seqlens,
      state_indices, segments, eps, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel<uint16_t, half_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      static_cast<half_t*>(nullptr),
      out,
      static_cast<uint16_t*>(nullptr),
      cu_seqlens,
      state_indices,
      segments,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16(
    const float* conv_out,
    const half_t* gate,
    const half_t* a,
    const half_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    int64_t total_tokens,
    int64_t segments,
    float eps,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (total_tokens <= 0 || segments <= 0 || !valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel<half_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      static_cast<float*>(nullptr),
      out,
      static_cast<uint16_t*>(nullptr),
      cu_seqlens,
      state_indices,
      segments,
      eps,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<uint16_t, true, float>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      recurrent_state,
      nullptr,
      out,
      eps,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* recurrent_state_rows,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_rows == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<uint16_t, true, float>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      recurrent_state,
      recurrent_state_rows,
      out,
      eps,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* recurrent_state_initial,
    float* recurrent_state_rows,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_initial == nullptr || recurrent_state_rows == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<uint16_t, false, float>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state_initial,
      nullptr,
      recurrent_state_rows,
      out,
      eps,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const half_t* recurrent_state_initial,
    half_t* recurrent_state_rows,
    uint16_t* out,
    float eps,
    int64_t tokens,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_initial == nullptr || recurrent_state_rows == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<uint16_t, false, half_t>),
      dim3(static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state_initial,
      nullptr,
      recurrent_state_rows,
      out,
      eps,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* recurrent_state_initial,
    float* recurrent_state_rows,
    uint16_t* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    float eps,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) || segments <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_initial == nullptr || recurrent_state_rows == nullptr || cu_seqlens == nullptr ||
      state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_state_rows_kernel<uint16_t, float>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state_initial,
      recurrent_state_rows,
      out,
      static_cast<float*>(nullptr),
      cu_seqlens,
      state_indices,
      eps,
      segments,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const float* recurrent_state_initial,
    float* recurrent_state_rows,
    uint16_t* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    float eps,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      segments <= 0 || head_k_dim != 128 || head_v_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_initial == nullptr || recurrent_state_rows == nullptr ||
      cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 128;
  const size_t shared_bytes =
      static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) *
      sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_state_rows_kernel<uint16_t, float, true>),
      dim3(static_cast<unsigned int>(segments),
           static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state_initial,
      recurrent_state_rows,
      out,
      static_cast<float*>(nullptr),
      cu_seqlens,
      state_indices,
      eps,
      segments,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32(
    const float* conv_out, const uint16_t* gate, const uint16_t* a, const uint16_t* b,
    const float* dt_bias, const float* a_log, const float* norm_weight,
    const float* recurrent_state_initial, float* recurrent_state_rows,
    uint16_t* out, float* out_f32, const int32_t* cu_seqlens,
    const int64_t* state_indices, float eps, int64_t total_tokens, int64_t segments,
    int64_t num_k_heads, int64_t num_v_heads, int64_t head_k_dim, int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) ||
      segments <= 0 || recurrent_state_initial == nullptr || recurrent_state_rows == nullptr ||
      out == nullptr || out_f32 == nullptr || cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_state_rows_kernel<uint16_t, float>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads), shared_bytes, stream, conv_out, gate, a, b, dt_bias, a_log,
      norm_weight, recurrent_state_initial, recurrent_state_rows, out, out_f32,
      cu_seqlens, state_indices, eps, segments, num_k_heads, num_v_heads,
      head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    const half_t* recurrent_state_initial,
    half_t* recurrent_state_rows,
    uint16_t* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    float eps,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) || segments <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state_initial == nullptr || recurrent_state_rows == nullptr || cu_seqlens == nullptr ||
      state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_state_rows_kernel<uint16_t, half_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state_initial,
      recurrent_state_rows,
      out,
      static_cast<float*>(nullptr),
      cu_seqlens,
      state_indices,
      eps,
      segments,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    uint16_t* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    float eps,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) || segments <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state == nullptr || cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_kernel<uint16_t, float>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      eps,
      segments,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_fp16state(
    const float* conv_out,
    const uint16_t* gate,
    const uint16_t* a,
    const uint16_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    half_t* recurrent_state,
    uint16_t* out,
    const int32_t* cu_seqlens,
    const int64_t* state_indices,
    float eps,
    int64_t total_tokens,
    int64_t segments,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!valid_prefill_shape(total_tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim) || segments <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (recurrent_state == nullptr || cu_seqlens == nullptr || state_indices == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 128;
  const size_t shared_bytes = static_cast<size_t>((2 * head_k_dim) + (3 * threads) + head_v_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_segments_kernel<uint16_t, half_t>),
      dim3(static_cast<unsigned int>(segments), static_cast<unsigned int>(num_v_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      conv_out,
      gate,
      a,
      b,
      dt_bias,
      a_log,
      norm_weight,
      recurrent_state,
      out,
      cu_seqlens,
      state_indices,
      eps,
      segments,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}
