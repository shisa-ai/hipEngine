// hipEngine-native Qwen3.8-Flash-Next gated-residual strict primitives.
// No external kernel body is ported; formulas follow the frozen qwen4exp oracle.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union { float f32; uint32_t u32; } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__global__ void repeat_bf16_branches_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t branches,
    int64_t hidden) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = rows * branches * hidden;
  if (index < elements) {
    const int64_t row = index / (branches * hidden);
    output[index] = input[row * hidden + index % hidden];
  }
}

__global__ void grouped_rmsnorm_bf16_f32_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column]);
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  const int64_t branch = group % branches;
  const int64_t weight_offset = branch * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] =
        bf16_bits_to_float(residual[offset + column]) * inv_rms * weight[weight_offset + column];
  }
}

// DP-5 diagnostic sibling: preserve the parent's tree and F32 publication.
// Materialize the identical half rounding used by the standalone staging cast.
__global__ void grouped_rmsnorm_bf16_f32_f16_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    _Float16* __restrict__ output_f16,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column]);
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  const int64_t branch = group % branches;
  const int64_t weight_offset = branch * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value =
        bf16_bits_to_float(residual[offset + column]) * inv_rms * weight[weight_offset + column];
    output[offset + column] = value;
    // Keep conversion on the exact F32 register that is published. The plain
    // language cast was observed to differ on signed zero and halfway inputs
    // even while the F32 output matched the parent; pin the conversion itself.
    uint32_t half_bits;
    asm volatile("v_cvt_f16_f32 %0, %1" : "=v"(half_bits) : "v"(value));
    reinterpret_cast<uint16_t*>(output_f16)[offset + column] = static_cast<uint16_t>(half_bits);
  }
}

// The F16 plane and the per-group scale, without the FP32 plane. A consumer
// that needs an FP32 normalized value recomputes
// bf16(residual) * inv_rms * weight, which is the expression stored by the
// kernels above, so it reads half the bytes and gets the same bits.
__global__ void grouped_rmsnorm_bf16_f16_scale_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ weight,
    _Float16* __restrict__ output_f16,
    float* __restrict__ inv_rms_out,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column]);
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  if (threadIdx.x == 0) {
    inv_rms_out[group] = inv_rms;
  }
  const int64_t branch = group % branches;
  const int64_t weight_offset = branch * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value =
        bf16_bits_to_float(residual[offset + column]) * inv_rms * weight[weight_offset + column];
    uint32_t half_bits;
    asm volatile("v_cvt_f16_f32 %0, %1" : "=v"(half_bits) : "v"(value));
    reinterpret_cast<uint16_t*>(output_f16)[offset + column] = static_cast<uint16_t>(half_bits);
  }
}

// Hyper-connection inject logits from the BF16 residual: out[row][b] =
// sum_k normalized[row][k] * inject_weight[b][k] over the four-branch row,
// with normalized recomputed as above. One block per row computes all four
// logits; each keeps the scalar GEMV's order (thread t owns k = 8t + 2048j,
// eight sequential FMAs per step, then the 256-wide halving tree).
__global__ void gr_inject_bf16_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ norm_weight,
    const float* __restrict__ inv_rms,
    const float* __restrict__ inject_weight,
    float* __restrict__ out,
    int64_t hidden) {
  __shared__ float sdata[4][256];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t width = 4 * hidden;
  const uint16_t* __restrict__ r = residual + row * width;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int64_t k = static_cast<int64_t>(tid) * 8; k < width; k += 2048) {
    const float inv = inv_rms[row * 4 + k / hidden];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float x = bf16_bits_to_float(r[k + j]) * inv * norm_weight[k + j];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        acc[b] = fmaf(x, inject_weight[b * width + k + j], acc[b]);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < 4; ++b) sdata[b][tid] = acc[b];
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) {
#pragma unroll
      for (int b = 0; b < 4; ++b) sdata[b][tid] += sdata[b][tid + s];
    }
    __syncthreads();
  }
  if (tid < 4) {
    out[row * 4 + tid] = sdata[tid][0];
  }
}

__global__ void grouped_rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = input[offset + column];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  const int64_t branch = group % branches;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] = input[offset + column] * inv_rms
        * weight[branch * hidden + column];
  }
}

__global__ void gated_mean_f32_kernel(
    const float* __restrict__ normalized,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t row = blockIdx.x;
  const int64_t residual_width = branches * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    float value = 0.0f;
    for (int64_t branch = 0; branch < branches; ++branch) {
      const int64_t index = row * residual_width + branch * hidden + column;
      value += normalized[index] * gate[index];
    }
    output[row * hidden + column] = value / static_cast<float>(branches);
  }
}

// STORE_GATE false leaves gate_logits untouched, for callers that do not
// read the gate afterwards; the mixed output is the same either way.
template <bool STORE_GATE>
__global__ void gated_mean_sigmoid_f32_kernel(
    const float* __restrict__ normalized,
    float* __restrict__ gate_logits,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t column_tiles = (hidden + blockDim.x - 1) / blockDim.x;
  const int64_t row = blockIdx.x / column_tiles;
  const int64_t column = (blockIdx.x % column_tiles) * blockDim.x + threadIdx.x;
  if (column < hidden) {
    const int64_t residual_width = branches * hidden;
    float value = 0.0f;
    for (int64_t branch = 0; branch < branches; ++branch) {
      const int64_t index = row * residual_width + branch * hidden + column;
      const float gate = 1.0f / (1.0f + expf(-gate_logits[index]));
      if constexpr (STORE_GATE) {
        gate_logits[index] = gate;
      }
      value += normalized[index] * gate;
    }
    output[row * hidden + column] = value / static_cast<float>(branches);
  }
}

__global__ void gr_write_bf16_f32_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ block_output,
    const float* __restrict__ inject_logits,
    uint16_t* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int64_t branch = group % branches;
  const float inject = inject_logits[row * branches + branch] / static_cast<float>(branches);
  const float scatter = 2.0f / (1.0f + expf(-inject));
  const int64_t offset = group * hidden;
  const int64_t block_offset = row * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column])
        + scatter * block_output[block_offset + column];
    output[offset + column] = float_to_bf16_bits(value);
  }
}

__global__ void scaled_silu_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t elements,
    float scale) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = input[index] * scale;
    output[index] = value / (1.0f + expf(-value));
  }
}

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
  }
}

__global__ void sigmoid_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = 1.0f / (1.0f + expf(-input[index]));
  }
}

__global__ void sigmoid_gated_rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t head_dim,
    float eps) {
  extern __shared__ float partial[];
  const int64_t head = blockIdx.x;
  const int64_t offset = head * head_dim;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const float value = input[offset + column];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const int64_t index = offset + column;
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate[index]));
    output[index] = input[index] * inv_rms * weight[column] * sigmoid_gate;
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_repeat_bf16_branches(
    const void* input,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!input || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = rows * branches * hidden;
  hipLaunchKernelGGL(
      repeat_bf16_branches_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads)), dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input), static_cast<uint16_t*>(output), rows, branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_bf16_f32(
    const void* residual,
    const void* weight,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!residual || !weight || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_bf16_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(weight),
      static_cast<float*>(output), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
    const void* residual,
    const void* weight,
    void* output,
    void* output_f16,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!residual || !weight || !output || !output_f16 || output == output_f16 ||
      rows <= 0 || branches <= 0 || hidden <= 0 ||
      rows > UINT32_MAX / branches ||
      rows * branches > INT64_MAX / hidden) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_bf16_f32_f16_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(weight),
      static_cast<float*>(output), static_cast<_Float16*>(output_f16), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_bf16_f16_scale(
    const void* residual,
    const void* weight,
    void* output_f16,
    void* inv_rms,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!residual || !weight || !output_f16 || !inv_rms || rows <= 0 ||
      branches <= 0 || hidden <= 0 || rows > UINT32_MAX / branches ||
      rows * branches > INT64_MAX / hidden) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_bf16_f16_scale_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(weight),
      static_cast<_Float16*>(output_f16), static_cast<float*>(inv_rms), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gr_inject_bf16_f32(
    const void* residual,
    const void* norm_weight,
    const void* inv_rms,
    const void* inject_weight,
    void* out,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!residual || !norm_weight || !inv_rms || !inject_weight || !out ||
      rows <= 0 || rows > UINT32_MAX || branches != 4 || hidden <= 0 ||
      (hidden % 8) != 0 || ((4 * hidden) % 2048) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gr_inject_bf16_kernel, dim3(static_cast<uint32_t>(rows)), dim3(256), 0, stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(norm_weight),
      static_cast<const float*>(inv_rms), static_cast<const float*>(inject_weight),
      static_cast<float*>(out), hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_f32(
    const void* input,
    const void* weight,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<float*>(output), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gated_mean_f32(
    const void* normalized,
    const void* gate,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!normalized || !gate || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      gated_mean_f32_kernel,
      dim3(static_cast<uint32_t>(rows)), dim3(threads), 0, stream,
      static_cast<const float*>(normalized), static_cast<const float*>(gate),
      static_cast<float*>(output), branches, hidden);
  return static_cast<int>(hipGetLastError());
}

#define QWEN4_EXP_GATED_MEAN_SIGMOID_ENTRY(NAME, STORE_GATE) \
extern "C" int NAME( \
    const void* normalized, \
    void* gate_logits, \
    void* output, \
    int64_t rows, \
    int64_t branches, \
    int64_t hidden, \
    hipStream_t stream) { \
  if (!normalized || !gate_logits || !output || \
      rows <= 0 || branches <= 0 || hidden <= 0) { \
    return static_cast<int>(hipErrorInvalidValue); \
  } \
  constexpr int threads = 256; \
  const uint32_t column_tiles = static_cast<uint32_t>((hidden + threads - 1) / threads); \
  hipLaunchKernelGGL( \
      gated_mean_sigmoid_f32_kernel<STORE_GATE>, \
      dim3(static_cast<uint32_t>(rows) * column_tiles), dim3(threads), 0, stream, \
      static_cast<const float*>(normalized), static_cast<float*>(gate_logits), \
      static_cast<float*>(output), branches, hidden); \
  return static_cast<int>(hipGetLastError()); \
}
QWEN4_EXP_GATED_MEAN_SIGMOID_ENTRY(hipengine_qwen4_exp_gated_mean_sigmoid_f32, true)
QWEN4_EXP_GATED_MEAN_SIGMOID_ENTRY(hipengine_qwen4_exp_gated_mean_sigmoid_no_gate_f32, false)
#undef QWEN4_EXP_GATED_MEAN_SIGMOID_ENTRY

extern "C" int hipengine_qwen4_exp_gr_write_bf16_f32(
    const void* residual,
    const void* block_output,
    const void* inject_logits,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!residual || !block_output || !inject_logits || !output ||
      rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      gr_write_bf16_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), 0, stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(block_output),
      static_cast<const float*>(inject_logits), static_cast<uint16_t*>(output),
      branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_scaled_silu_f32(
    const void* input,
    void* output,
    int64_t elements,
    float scale,
    hipStream_t stream) {
  if (!input || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      scaled_silu_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<float*>(output), elements, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_silu_mul_f32(
    const void* gate,
    const void* up,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!gate || !up || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      silu_mul_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(gate), static_cast<const float*>(up),
      static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_sigmoid_f32(
    const void* input,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!input || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      sigmoid_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_sigmoid_gated_rmsnorm_f32(
    const void* input,
    const void* weight,
    const void* gate,
    void* output,
    int64_t heads,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !gate || !output || heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      sigmoid_gated_rmsnorm_f32_kernel,
      dim3(static_cast<uint32_t>(heads)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const float*>(gate), static_cast<float*>(output), head_dim, eps);
  return static_cast<int>(hipGetLastError());
}
