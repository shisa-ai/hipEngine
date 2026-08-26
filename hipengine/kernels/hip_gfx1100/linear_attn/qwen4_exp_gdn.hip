// hipEngine-native strict Qwen4Exp GDN decode recurrence with sigmoid output gate.

#include <hip/hip_runtime.h>
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
    const float* __restrict__ a_log,
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
  const float decay = expf(-expf(a_log[v_head]) * softplus_f32(alpha[v_head] + dt_bias[v_head]));
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

}  // namespace

extern "C" int hipengine_qwen4_exp_gdn_decode_f32(
    const void* conv,
    const void* output_gate,
    const void* alpha,
    const void* beta_logits,
    const void* dt_bias,
    const void* a_log,
    const void* norm_weight,
    void* recurrent_state,
    void* output,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    hipStream_t stream) {
  if (!conv || !output_gate || !alpha || !beta_logits || !dt_bias || !a_log ||
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
      static_cast<const float*>(dt_bias), static_cast<const float*>(a_log),
      static_cast<const float*>(norm_weight), static_cast<float*>(recurrent_state),
      static_cast<float*>(output), num_k_heads, num_v_heads, head_k_dim, head_v_dim);
  return static_cast<int>(hipGetLastError());
}
