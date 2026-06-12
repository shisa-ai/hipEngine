// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip.
// Baseline GDN decode/prefill kernels track @ 59195ed; DFlash tree t-loop
// variants track @ b95eaa5. Wrappers are hipEngine raw-pointer C ABI.

#include <hip/hip_runtime.h>
#include <cstdlib>
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

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename scalar_t>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
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

  const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
  const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
  const float beta = sigmoid_f32(scalar_to_float_qwen35(b[v_head]));
  const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[v_head]) + dt_bias[v_head]));

  const int64_t value_idx = threadIdx.x;
  float out_acc = 0.0f;
  if (value_idx < head_v_dim) {
    const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
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
      const float ns0 = recurrent_state[so0] * decay + k_norm0 * delta;
      const float ns1 = recurrent_state[so1] * decay + k_norm1 * delta;
      const float ns2 = recurrent_state[so2] * decay + k_norm2 * delta;
      const float ns3 = recurrent_state[so3] * decay + k_norm3 * delta;
      const float ns4 = recurrent_state[so4] * decay + k_norm4 * delta;
      const float ns5 = recurrent_state[so5] * decay + k_norm5 * delta;
      const float ns6 = recurrent_state[so6] * decay + k_norm6 * delta;
      const float ns7 = recurrent_state[so7] * decay + k_norm7 * delta;
      recurrent_state[so0] = ns0;
      recurrent_state[so1] = ns1;
      recurrent_state[so2] = ns2;
      recurrent_state[so3] = ns3;
      recurrent_state[so4] = ns4;
      recurrent_state[so5] = ns5;
      recurrent_state[so6] = ns6;
      recurrent_state[so7] = ns7;
      out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
               + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
    }
    for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
      const float q_norm = q_shared[idx] * q_scale;
      const float k_norm = k_shared[idx] * k_scale;
      const int64_t state_offset = state_col + idx * head_v_dim;
      const float new_state = recurrent_state[state_offset] * decay + k_norm * delta;
      recurrent_state[state_offset] = new_state;
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
    out[pos] = out_shared[value_idx] * inv_rms * norm_weight[value_idx] * silu_f32(scalar_to_float_qwen35(gate[pos]));
  }
}

template <typename scalar_t>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel(
    const float* __restrict__ conv_out,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ norm_weight,
    const float* __restrict__ base_recurrent_state,
    float* __restrict__ tree_recurrent_state,
    const int64_t* __restrict__ parent_ids,
    float* __restrict__ out,
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t value_dim = num_v_heads * head_v_dim;
  const int64_t conv_stride = (2 * key_dim) + value_dim;
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

    const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[t * num_v_heads + v_head]));
    const float decay = expf(-expf(a_log_v) * softplus_f32(scalar_to_float_qwen35(a[t * num_v_heads + v_head]) + dt_bias_v));

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
        const float ns0 = recurrent_state[so0] * decay + k_norm0 * delta;
        const float ns1 = recurrent_state[so1] * decay + k_norm1 * delta;
        const float ns2 = recurrent_state[so2] * decay + k_norm2 * delta;
        const float ns3 = recurrent_state[so3] * decay + k_norm3 * delta;
        const float ns4 = recurrent_state[so4] * decay + k_norm4 * delta;
        const float ns5 = recurrent_state[so5] * decay + k_norm5 * delta;
        const float ns6 = recurrent_state[so6] * decay + k_norm6 * delta;
        const float ns7 = recurrent_state[so7] * decay + k_norm7 * delta;
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
        const float new_state = recurrent_state[state_offset] * decay + k_norm * delta;
        node_state[state_offset] = new_state;
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
      out[pos] = out_shared[value_idx] * inv_rms * norm_weight[value_idx] * silu_f32(scalar_to_float_qwen35(gate[pos]));
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

template <typename scalar_t>
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
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

  const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
  const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
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

    const float q_scale = rsqrtf(q_sum + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(k_sum + 1.0e-6f);
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
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

    const float q_scale = rsqrtf(q_sum + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(k_sum + 1.0e-6f);
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


template <typename scalar_t>
__global__ void qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t value_idx = threadIdx.x;
  volatile float* recurrent_state_v = recurrent_state;

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

    const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = v_head * head_k_dim * head_v_dim + value_idx;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * recurrent_state_v[state_col + (idx + 0) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 1] * k_scale * recurrent_state_v[state_col + (idx + 1) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 2] * k_scale * recurrent_state_v[state_col + (idx + 2) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 3] * k_scale * recurrent_state_v[state_col + (idx + 3) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 4] * k_scale * recurrent_state_v[state_col + (idx + 4) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 5] * k_scale * recurrent_state_v[state_col + (idx + 5) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 6] * k_scale * recurrent_state_v[state_col + (idx + 6) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 7] * k_scale * recurrent_state_v[state_col + (idx + 7) * head_v_dim] * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = recurrent_state_v[state_col + idx * head_v_dim] * decay;
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
        const float ns0 = recurrent_state_v[so0] * decay + k_norm0 * delta;
        const float ns1 = recurrent_state_v[so1] * decay + k_norm1 * delta;
        const float ns2 = recurrent_state_v[so2] * decay + k_norm2 * delta;
        const float ns3 = recurrent_state_v[so3] * decay + k_norm3 * delta;
        const float ns4 = recurrent_state_v[so4] * decay + k_norm4 * delta;
        const float ns5 = recurrent_state_v[so5] * decay + k_norm5 * delta;
        const float ns6 = recurrent_state_v[so6] * decay + k_norm6 * delta;
        const float ns7 = recurrent_state_v[so7] * decay + k_norm7 * delta;
        recurrent_state_v[so0] = ns0;
        recurrent_state_v[so1] = ns1;
        recurrent_state_v[so2] = ns2;
        recurrent_state_v[so3] = ns3;
        recurrent_state_v[so4] = ns4;
        recurrent_state_v[so5] = ns5;
        recurrent_state_v[so6] = ns6;
        recurrent_state_v[so7] = ns7;
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = recurrent_state_v[state_offset] * decay + k_norm * delta;
        recurrent_state_v[state_offset] = new_state;
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

template <typename scalar_t>
__global__ void qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel(
    const float* conv_out,
    const scalar_t* gate,
    const scalar_t* a,
    const scalar_t* b,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* recurrent_state,
    float* out,
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
  const int64_t repeat = num_v_heads / num_k_heads;
  const int64_t k_head = v_head / repeat;
  const int64_t key_dim = num_k_heads * head_k_dim;
  const int64_t qkv_width = 2 * key_dim + num_v_heads * head_v_dim;
  const int64_t value_idx = threadIdx.x;
  const int64_t state_stride = num_v_heads * head_k_dim * head_v_dim;
  volatile float* recurrent_state_v = recurrent_state;

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

    const float q_scale = rsqrtf(q_partial[0] + 1.0e-6f) * rsqrtf(static_cast<float>(head_k_dim));
    const float k_scale = rsqrtf(k_partial[0] + 1.0e-6f);
    const int64_t scalar_idx = token * num_v_heads + v_head;
    const float beta = sigmoid_f32(scalar_to_float_qwen35(b[scalar_idx]));
    const float decay = expf(-expf(a_log[v_head]) * softplus_f32(scalar_to_float_qwen35(a[scalar_idx]) + dt_bias[v_head]));

    float out_acc = 0.0f;
    if (value_idx < head_v_dim) {
      const int64_t state_col = state_slot * state_stride + v_head * head_k_dim * head_v_dim + value_idx;
      float kv_mem = 0.0f;
      for (int64_t idx = 0; idx + 7 < head_k_dim; idx += 8) {
        kv_mem += k_shared[idx + 0] * k_scale * recurrent_state_v[state_col + (idx + 0) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 1] * k_scale * recurrent_state_v[state_col + (idx + 1) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 2] * k_scale * recurrent_state_v[state_col + (idx + 2) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 3] * k_scale * recurrent_state_v[state_col + (idx + 3) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 4] * k_scale * recurrent_state_v[state_col + (idx + 4) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 5] * k_scale * recurrent_state_v[state_col + (idx + 5) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 6] * k_scale * recurrent_state_v[state_col + (idx + 6) * head_v_dim] * decay;
        kv_mem += k_shared[idx + 7] * k_scale * recurrent_state_v[state_col + (idx + 7) * head_v_dim] * decay;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float k_norm = k_shared[idx] * k_scale;
        const float state_value = recurrent_state_v[state_col + idx * head_v_dim] * decay;
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
        const float ns0 = recurrent_state_v[so0] * decay + k_norm0 * delta;
        const float ns1 = recurrent_state_v[so1] * decay + k_norm1 * delta;
        const float ns2 = recurrent_state_v[so2] * decay + k_norm2 * delta;
        const float ns3 = recurrent_state_v[so3] * decay + k_norm3 * delta;
        const float ns4 = recurrent_state_v[so4] * decay + k_norm4 * delta;
        const float ns5 = recurrent_state_v[so5] * decay + k_norm5 * delta;
        const float ns6 = recurrent_state_v[so6] * decay + k_norm6 * delta;
        const float ns7 = recurrent_state_v[so7] * decay + k_norm7 * delta;
        recurrent_state_v[so0] = ns0;
        recurrent_state_v[so1] = ns1;
        recurrent_state_v[so2] = ns2;
        recurrent_state_v[so3] = ns3;
        recurrent_state_v[so4] = ns4;
        recurrent_state_v[so5] = ns5;
        recurrent_state_v[so6] = ns6;
        recurrent_state_v[so7] = ns7;
        out_acc += q_norm0 * ns0 + q_norm1 * ns1 + q_norm2 * ns2 + q_norm3 * ns3
                 + q_norm4 * ns4 + q_norm5 * ns5 + q_norm6 * ns6 + q_norm7 * ns7;
      }
      for (int64_t idx = (head_k_dim & ~7); idx < head_k_dim; ++idx) {
        const float q_norm = q_shared[idx] * q_scale;
        const float k_norm = k_shared[idx] * k_scale;
        const int64_t state_offset = state_col + idx * head_v_dim;
        const float new_state = recurrent_state_v[state_offset] * decay + k_norm * delta;
        recurrent_state_v[state_offset] = new_state;
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
      out[out_pos] = out_shared[value_idx] * inv_rms * norm_weight[value_idx] *
          silu_f32(scalar_to_float_qwen35(gate[token * num_v_heads * head_v_dim + pos]));
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
        (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<scalar_t>),
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
        parent_ids,
        out,
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
        (qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel<scalar_t>),
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
        out,
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
      out,
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
      (qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel<uint16_t>),
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
      eps,
      tokens,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim);
  return static_cast<int>(hipGetLastError());
}
