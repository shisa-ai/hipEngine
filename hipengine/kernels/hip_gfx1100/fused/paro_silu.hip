// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// PARO selected-expert SiLU and down-rotation kernel bodies are preserved; wrappers are hipEngine C ABI.

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

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t>
__global__ void silu_mul_dual_out_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * features;
  if (idx >= total) return;
  const int64_t row = idx / features;
  const int64_t col = idx - row * features;
  const int64_t base = row * features * 2;
  const float gate = scalar_to_float(gate_up[base + col]);
  const float up = scalar_to_float(gate_up[base + features + col]);
  const float sig = 1.0f / (1.0f + expf(-gate));
  out[idx] = float_to_scalar<scalar_t>(gate * sig * up);
}

template <typename scalar_t>
__global__ void silu_mul_separate_out_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * features;
  if (idx >= total) return;
  const float g = scalar_to_float(gate[idx]);
  const float u = scalar_to_float(up[idx]);
  const float sig = 1.0f / (1.0f + expf(-g));
  out[idx] = float_to_scalar<scalar_t>(g * sig * u);
}


template <typename scalar_t>
__global__ void silu_mul_dual_rotate_out_kernel(
    const scalar_t* __restrict__ gate_up,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= rows || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * features;
  const int64_t gate_up_base = row_base * 2;

  const float gate0 = scalar_to_float(gate_up[gate_up_base + col0]);
  const float gate1 = scalar_to_float(gate_up[gate_up_base + col1]);
  const float up0 = scalar_to_float(gate_up[gate_up_base + features + col0]);
  const float up1 = scalar_to_float(gate_up[gate_up_base + features + col1]);
  const scalar_t act0 = float_to_scalar<scalar_t>(gate0 * (1.0f / (1.0f + expf(-gate0))) * up0);
  const scalar_t act1 = float_to_scalar<scalar_t>(gate1 * (1.0f / (1.0f + expf(-gate1))) * up1);

  buf[lane] = scalar_to_float(act0) * scalar_to_float(scales[col0]);
  buf[lane + half_group] = scalar_to_float(act1) * scalar_to_float(scales[col1]);
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(features) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (features / 2) + group * half_group + lane]);
    float s, c;
    sincosf(angle, &s, &c);
    const float xi = buf[i];
    const float xj = buf[j];
    buf[i] = fmaf(xj, s, xi * c);
    buf[j] = fmaf(xi, -s, xj * c);
    __syncthreads();
  }

  out[row_base + col0] = float_to_scalar<scalar_t>(buf[lane]);
  out[row_base + col1] = float_to_scalar<scalar_t>(buf[lane + half_group]);
}


template <typename scalar_t>
__global__ void silu_mul_pair_rotate_out_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t features,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= rows || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * features;

  const float gate0 = scalar_to_float(gate[row_base + col0]);
  const float gate1 = scalar_to_float(gate[row_base + col1]);
  const float up0 = scalar_to_float(up[row_base + col0]);
  const float up1 = scalar_to_float(up[row_base + col1]);
  const scalar_t act0 = float_to_scalar<scalar_t>(gate0 * (1.0f / (1.0f + expf(-gate0))) * up0);
  const scalar_t act1 = float_to_scalar<scalar_t>(gate1 * (1.0f / (1.0f + expf(-gate1))) * up1);

  buf[lane] = scalar_to_float(act0) * scalar_to_float(scales[col0]);
  buf[lane + half_group] = scalar_to_float(act1) * scalar_to_float(scales[col1]);
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(features) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (features / 2) + group * half_group + lane]);
    float s, c;
    sincosf(angle, &s, &c);
    const float xi = buf[i];
    const float xj = buf[j];
    buf[i] = fmaf(xj, s, xi * c);
    buf[j] = fmaf(xi, -s, xj * c);
    __syncthreads();
  }

  out[row_base + col0] = float_to_scalar<scalar_t>(buf[lane]);
  out[row_base + col1] = float_to_scalar<scalar_t>(buf[lane + half_group]);
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

}  // namespace

extern "C" int hipengine_silu_mul_dual_out_bf16(
    const uint16_t* gate_up,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * features;
  hipLaunchKernelGGL(
      (silu_mul_dual_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>((total + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate_up,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_dual_out_fp16(
    const half_t* gate_up,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * features;
  hipLaunchKernelGGL(
      (silu_mul_dual_out_kernel<half_t>),
      dim3(static_cast<unsigned int>((total + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate_up,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_separate_out_bf16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * features;
  hipLaunchKernelGGL(
      (silu_mul_separate_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>((total + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate,
      up,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_separate_out_fp16(
    const half_t* gate,
    const half_t* up,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * features;
  hipLaunchKernelGGL(
      (silu_mul_separate_out_kernel<half_t>),
      dim3(static_cast<unsigned int>((total + threads - 1) / threads)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      gate,
      up,
      out,
      rows,
      features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_dual_rotate_out_bf16(
    const uint16_t* gate_up,
    const int16_t* pairs,
    const uint16_t* theta,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || group_size <= 0 || krot <= 0 ||
      (group_size % 2) != 0 || (features % group_size) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (silu_mul_dual_rotate_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(features / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<size_t>(group_size) * sizeof(float),
      stream,
      gate_up,
      pairs,
      theta,
      scales,
      out,
      rows,
      features,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_dual_rotate_out_fp16(
    const half_t* gate_up,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || group_size <= 0 || krot <= 0 ||
      (group_size % 2) != 0 || (features % group_size) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (silu_mul_dual_rotate_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(features / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<size_t>(group_size) * sizeof(float),
      stream,
      gate_up,
      pairs,
      theta,
      scales,
      out,
      rows,
      features,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_pair_rotate_out_bf16(
    const uint16_t* gate,
    const uint16_t* up,
    const int16_t* pairs,
    const uint16_t* theta,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t features,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || group_size <= 0 || krot <= 0 ||
      (group_size % 2) != 0 || (features % group_size) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (silu_mul_pair_rotate_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(features / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<size_t>(group_size) * sizeof(float),
      stream,
      gate,
      up,
      pairs,
      theta,
      scales,
      out,
      rows,
      features,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_silu_mul_pair_rotate_out_fp16(
    const half_t* gate,
    const half_t* up,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t features,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0 || group_size <= 0 || krot <= 0 ||
      (group_size % 2) != 0 || (features % group_size) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (silu_mul_pair_rotate_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(features / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<size_t>(group_size) * sizeof(float),
      stream,
      gate,
      up,
      pairs,
      theta,
      scales,
      out,
      rows,
      features,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}
