// Qwen3-VL-compatible vision primitives for Qwen4Exp basic image support.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__global__ void vision_layernorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t rows,
    int64_t features,
    float eps) {
  __shared__ float sums[256];
  __shared__ float squares[256];
  __shared__ float mean;
  __shared__ float inv_std;
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  float square = 0.0f;
  for (int64_t col = threadIdx.x; col < features; col += blockDim.x) {
    const float value = input[row * features + col];
    sum += value;
    square += value * value;
  }
  sums[threadIdx.x] = sum;
  squares[threadIdx.x] = square;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      squares[threadIdx.x] += squares[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    mean = sums[0] / static_cast<float>(features);
    const float variance = squares[0] / static_cast<float>(features) - mean * mean;
    inv_std = rsqrtf(fmaxf(variance, 0.0f) + eps);
  }
  __syncthreads();
  for (int64_t col = threadIdx.x; col < features; col += blockDim.x) {
    const float value = (input[row * features + col] - mean) * inv_std;
    output[row * features + col] = value * weight[col] + bias[col];
  }
}

__global__ void vision_add_bias_residual_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int64_t elements,
    int64_t features) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = input[index] + bias[index % features];
  if (residual) value += residual[index];
  output[index] = value;
}

__global__ void vision_bias_gelu_tanh_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t elements,
    int64_t features) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float x = input[index] + bias[index % features];
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  output[index] = 0.5f * x * (1.0f + tanhf(inner));
}

__device__ inline void rotate_pair(
    float x0,
    float x1,
    int pair,
    int pos_h,
    int pos_w,
    float* y0,
    float* y1) {
  const int local = pair < 18 ? pair : pair - 18;
  const int position = pair < 18 ? pos_h : pos_w;
  const float theta_scale = powf(10000.0f, -2.0f / 36.0f);
  const float theta = static_cast<float>(position) * powf(theta_scale, static_cast<float>(local));
  const float c = cosf(theta);
  const float s = sinf(theta);
  *y0 = x0 * c - x1 * s;
  *y1 = x0 * s + x1 * c;
}

// Correctness-first full non-causal attention. One thread owns one (query,head)
// row so FP32 score/softmax/value order is explicit and independent.
__global__ void vision_attention_f32_kernel(
    const float* __restrict__ qkv,
    const float* __restrict__ qkv_bias,
    const int32_t* __restrict__ pos_h,
    const int32_t* __restrict__ pos_w,
    float* __restrict__ output,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    float scale) {
  if (threadIdx.x != 0) return;
  const int64_t query = blockIdx.x / heads;
  const int64_t head = blockIdx.x % heads;
  if (query >= tokens || head_dim != 72) return;
  const int64_t hidden = heads * head_dim;
  const int64_t q_base = query * 3 * hidden + head * head_dim;
  float maximum = -INFINITY;
  for (int64_t key = 0; key < tokens; ++key) {
    const int64_t k_base = key * 3 * hidden + hidden + head * head_dim;
    float dot = 0.0f;
    for (int pair = 0; pair < 36; ++pair) {
      float q0, q1, k0, k1;
      rotate_pair(
          qkv[q_base + pair] + qkv_bias[head * head_dim + pair],
          qkv[q_base + pair + 36] + qkv_bias[head * head_dim + pair + 36],
          pair, pos_h[query], pos_w[query], &q0, &q1);
      rotate_pair(
          qkv[k_base + pair] + qkv_bias[hidden + head * head_dim + pair],
          qkv[k_base + pair + 36] + qkv_bias[hidden + head * head_dim + pair + 36],
          pair, pos_h[key], pos_w[key], &k0, &k1);
      dot += q0 * k0 + q1 * k1;
    }
    maximum = fmaxf(maximum, dot * scale);
  }
  float denominator = 0.0f;
  float accum[72] = {};
  for (int64_t key = 0; key < tokens; ++key) {
    const int64_t k_base = key * 3 * hidden + hidden + head * head_dim;
    const int64_t v_base = key * 3 * hidden + 2 * hidden + head * head_dim;
    float dot = 0.0f;
    for (int pair = 0; pair < 36; ++pair) {
      float q0, q1, k0, k1;
      rotate_pair(
          qkv[q_base + pair] + qkv_bias[head * head_dim + pair],
          qkv[q_base + pair + 36] + qkv_bias[head * head_dim + pair + 36],
          pair, pos_h[query], pos_w[query], &q0, &q1);
      rotate_pair(
          qkv[k_base + pair] + qkv_bias[hidden + head * head_dim + pair],
          qkv[k_base + pair + 36] + qkv_bias[hidden + head * head_dim + pair + 36],
          pair, pos_h[key], pos_w[key], &k0, &k1);
      dot += q0 * k0 + q1 * k1;
    }
    const float probability = expf(dot * scale - maximum);
    denominator += probability;
    for (int dim = 0; dim < 72; ++dim) {
      accum[dim] += probability *
          (qkv[v_base + dim] + qkv_bias[2 * hidden + head * head_dim + dim]);
    }
  }
  for (int dim = 0; dim < 72; ++dim) {
    output[query * hidden + head * head_dim + dim] = accum[dim] / denominator;
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_vision_layernorm_f32(
    const void* input, const void* weight, const void* bias, void* output,
    int64_t rows, int64_t features, float eps, hipStream_t stream) {
  if (!input || !weight || !bias || !output || rows <= 0 || features <= 0 || eps <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  vision_layernorm_f32_kernel<<<dim3(static_cast<uint32_t>(rows)), dim3(256), 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const float*>(bias), static_cast<float*>(output), rows, features, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_vision_add_bias_residual_f32(
    const void* input, const void* bias, const void* residual, void* output,
    int64_t rows, int64_t features, hipStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = rows * features;
  vision_add_bias_residual_f32_kernel<<<dim3(static_cast<uint32_t>((elements + 255) / 256)), dim3(256), 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(bias),
      static_cast<const float*>(residual), static_cast<float*>(output), elements, features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_vision_bias_gelu_tanh_f32(
    const void* input, const void* bias, void* output,
    int64_t rows, int64_t features, hipStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t elements = rows * features;
  vision_bias_gelu_tanh_f32_kernel<<<dim3(static_cast<uint32_t>((elements + 255) / 256)), dim3(256), 0, stream>>>(
      static_cast<const float*>(input), static_cast<const float*>(bias),
      static_cast<float*>(output), elements, features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_vision_attention_f32(
    const void* qkv, const void* qkv_bias, const void* pos_h, const void* pos_w,
    void* output, int64_t tokens, int64_t heads, int64_t head_dim,
    float scale, hipStream_t stream) {
  if (!qkv || !qkv_bias || !pos_h || !pos_w || !output || tokens <= 0 ||
      heads <= 0 || head_dim != 72 || scale <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  vision_attention_f32_kernel<<<dim3(static_cast<uint32_t>(tokens * heads)), dim3(1), 0, stream>>>(
      static_cast<const float*>(qkv), static_cast<const float*>(qkv_bias),
      static_cast<const int32_t*>(pos_h), static_cast<const int32_t*>(pos_w),
      static_cast<float*>(output), tokens, heads, head_dim, scale);
  return static_cast<int>(hipGetLastError());
}
