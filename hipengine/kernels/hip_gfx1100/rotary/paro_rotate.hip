// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// PARO pairwise rotate2/rotate3 kernel bodies are preserved; rotate1 is the same single-output specialization.
// Wrappers are hipEngine C ABI.

#include <hip/hip_runtime.h>
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

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t>
__global__ void paro_rotate1_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;

  float v0 = scalar_to_float(x[row_base + col0]) * scalar_to_float(scales[col0]);
  float v1 = scalar_to_float(x[row_base + col1]) * scalar_to_float(scales[col1]);
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (hidden / 2) + group * half_group + lane]);
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

__global__ void paro_rotate1_f32_to_fp16_kernel(
    const float* __restrict__ x,
    half_t* __restrict__ out,
    const int16_t* __restrict__ pairs,
    const half_t* __restrict__ theta,
    const half_t* __restrict__ scales,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;

  const half_t h0 = static_cast<half_t>(x[row_base + col0]);
  const half_t h1 = static_cast<half_t>(x[row_base + col1]);
  float v0 = static_cast<float>(h0) * static_cast<float>(scales[col0]);
  float v1 = static_cast<float>(h1) * static_cast<float>(scales[col1]);
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
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
    __syncthreads();
  }

  out[row_base + col0] = static_cast<half_t>(buf[lane]);
  out[row_base + col1] = static_cast<half_t>(buf[lane + half_group]);
}

__global__ void paro_rotate1_bf16_gate_fp16_kernel(
    const uint16_t* __restrict__ x,
    const half_t* __restrict__ gate,
    half_t* __restrict__ out,
    const int16_t* __restrict__ pairs,
    const half_t* __restrict__ theta,
    const half_t* __restrict__ scales,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;

  const int64_t offset0 = row_base + col0;
  const int64_t offset1 = row_base + col1;
  const half_t gated0 = static_cast<half_t>(
      bf16_bits_to_float(x[offset0]) * sigmoid_f32(static_cast<float>(gate[offset0])));
  const half_t gated1 = static_cast<half_t>(
      bf16_bits_to_float(x[offset1]) * sigmoid_f32(static_cast<float>(gate[offset1])));
  float v0 = static_cast<float>(gated0) * static_cast<float>(scales[col0]);
  float v1 = static_cast<float>(gated1) * static_cast<float>(scales[col1]);
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
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
    __syncthreads();
  }

  out[row_base + col0] = static_cast<half_t>(buf[lane]);
  out[row_base + col1] = static_cast<half_t>(buf[lane + half_group]);
}

template <typename scalar_t>
__global__ void paro_rotate2_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out0,
    scalar_t* __restrict__ out1,
    const int16_t* __restrict__ pairs0,
    const int16_t* __restrict__ pairs1,
    const scalar_t* __restrict__ theta0,
    const scalar_t* __restrict__ theta1,
    const scalar_t* __restrict__ scales0,
    const scalar_t* __restrict__ scales1,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int rotation = blockIdx.z;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;
  const int16_t* __restrict__ pairs = (rotation == 0) ? pairs0 : pairs1;
  const scalar_t* __restrict__ theta = (rotation == 0) ? theta0 : theta1;
  const scalar_t* __restrict__ scales = (rotation == 0) ? scales0 : scales1;
  scalar_t* __restrict__ out = (rotation == 0) ? out0 : out1;

  float v0 = scalar_to_float(x[row_base + col0]) * scalar_to_float(scales[col0]);
  float v1 = scalar_to_float(x[row_base + col1]) * scalar_to_float(scales[col1]);
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (hidden / 2) + group * half_group + lane]);
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
__global__ void paro_rotate3_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out0,
    scalar_t* __restrict__ out1,
    scalar_t* __restrict__ out2,
    const int16_t* __restrict__ pairs0,
    const int16_t* __restrict__ pairs1,
    const int16_t* __restrict__ pairs2,
    const scalar_t* __restrict__ theta0,
    const scalar_t* __restrict__ theta1,
    const scalar_t* __restrict__ theta2,
    const scalar_t* __restrict__ scales0,
    const scalar_t* __restrict__ scales1,
    const scalar_t* __restrict__ scales2,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int rotation = blockIdx.z;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;
  const int16_t* __restrict__ pairs = pairs0;
  const scalar_t* __restrict__ theta = theta0;
  const scalar_t* __restrict__ scales = scales0;
  scalar_t* __restrict__ out = out0;
  if (rotation == 1) {
    pairs = pairs1;
    theta = theta1;
    scales = scales1;
    out = out1;
  } else if (rotation == 2) {
    pairs = pairs2;
    theta = theta2;
    scales = scales2;
    out = out2;
  }

  float v0 = scalar_to_float(x[row_base + col0]) * scalar_to_float(scales[col0]);
  float v1 = scalar_to_float(x[row_base + col1]) * scalar_to_float(scales[col1]);
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (hidden / 2) + group * half_group + lane]);
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


bool valid_rotate_shape(int64_t tokens, int64_t hidden, int64_t group_size, int64_t krot) {
  return tokens > 0 && hidden > 0 && group_size > 0 && krot >= 0 &&
      (group_size % 2) == 0 && (hidden % group_size) == 0 && (group_size / 2) <= 1024;
}

// M15.4: fused RMSNorm + paro_rotate2.  One block per row, 256 threads.
// Phase 1 reproduces ``paro_rmsnorm_out_kernel`` byte-for-byte (same vec8 fp32
// accumulation + tree reduction at 256 threads), Phase 2 rounds the normed row
// to ``scalar_t`` in LDS (matching the HBM round-trip the standalone pair pays),
// Phase 3 reproduces ``paro_rotate2_kernel``'s per-group butterfly.  Each output
// element is therefore bit-identical to rmsnorm-then-rotate2.  Requires
// ``blockDim.x == 256`` and ``(hidden / group_size) % (256 / (group_size/2)) == 0``.
constexpr int kRmsRotateThreads = 256;
template <typename scalar_t>
__launch_bounds__(kRmsRotateThreads, 1)
__global__ void paro_rmsnorm_rotate2_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ ln_weight,
    scalar_t* __restrict__ out_norm,
    scalar_t* __restrict__ out0,
    scalar_t* __restrict__ out1,
    const int16_t* __restrict__ pairs0,
    const int16_t* __restrict__ pairs1,
    const scalar_t* __restrict__ theta0,
    const scalar_t* __restrict__ theta1,
    const scalar_t* __restrict__ scales0,
    const scalar_t* __restrict__ scales1,
    float eps,
    int64_t hidden,
    int group_size,
    int krot) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int64_t row_base = row * hidden;
  const int half_group = group_size / 2;
  const int groups = static_cast<int>(hidden / group_size);
  const int groups_per_batch = nthreads / half_group;

  // --- Phase 1: RMSNorm reduction, identical to paro_rmsnorm_out_kernel. ---
  float sumsq = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(nthreads) * 8;
  const int64_t main_end = (hidden / vec_stride) * vec_stride;
  for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
    const int64_t off = row_base + idx;
    const float v0 = scalar_to_float(x[off + 0]);
    const float v1 = scalar_to_float(x[off + 1]);
    const float v2 = scalar_to_float(x[off + 2]);
    const float v3 = scalar_to_float(x[off + 3]);
    const float v4 = scalar_to_float(x[off + 4]);
    const float v5 = scalar_to_float(x[off + 5]);
    const float v6 = scalar_to_float(x[off + 6]);
    const float v7 = scalar_to_float(x[off + 7]);
    sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
  }
  for (int64_t idx = main_end + tid; idx < hidden; idx += nthreads) {
    const float value = scalar_to_float(x[row_base + idx]);
    sumsq += value * value;
  }
  __shared__ float partial[kRmsRotateThreads];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);

  // --- Phase 2: normed = round_scalar(x * inv_rms * ln_weight) into LDS. ---
  extern __shared__ char smem[];
  scalar_t* normed = reinterpret_cast<scalar_t*>(smem);
  float* buf = reinterpret_cast<float*>(smem + hidden * sizeof(scalar_t));
  for (int64_t idx = tid; idx < hidden; idx += nthreads) {
    const scalar_t v = float_to_scalar<scalar_t>(scalar_to_float(x[row_base + idx]) * inv_rms *
                                                 scalar_to_float(ln_weight[idx]));
    normed[idx] = v;
    if (out_norm != nullptr) {
      out_norm[row_base + idx] = v;
    }
  }
  __syncthreads();

  // --- Phase 3: two paro_rotate2 passes over the LDS normed row. ---
  const int gib = tid / half_group;
  const int lane = tid % half_group;
  for (int rot = 0; rot < 2; ++rot) {
    const int16_t* __restrict__ pairs = (rot == 0) ? pairs0 : pairs1;
    const scalar_t* __restrict__ theta = (rot == 0) ? theta0 : theta1;
    const scalar_t* __restrict__ scales = (rot == 0) ? scales0 : scales1;
    scalar_t* __restrict__ out = (rot == 0) ? out0 : out1;
    for (int gbatch = 0; gbatch < groups; gbatch += groups_per_batch) {
      const int group = gbatch + gib;
      const int base = group * group_size;
      const int col0 = base + lane;
      const int col1 = col0 + half_group;
      float* gbuf = buf + gib * group_size;
      __syncthreads();
      gbuf[lane] = scalar_to_float(normed[col0]) * scalar_to_float(scales[col0]);
      gbuf[lane + half_group] = scalar_to_float(normed[col1]) * scalar_to_float(scales[col1]);
      __syncthreads();
      for (int r = 0; r < krot; ++r) {
        const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (static_cast<int>(hidden) / 2) + group * half_group + lane]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = gbuf[i];
        const float xj = gbuf[j];
        gbuf[i] = fmaf(xj, s, xi * c);
        gbuf[j] = fmaf(xi, -s, xj * c);
        __syncthreads();
      }
      out[row_base + col0] = float_to_scalar<scalar_t>(gbuf[lane]);
      out[row_base + col1] = float_to_scalar<scalar_t>(gbuf[lane + half_group]);
    }
  }
}

bool valid_rmsnorm_rotate2_shape(int64_t tokens, int64_t hidden, int64_t group_size, int64_t krot) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return false;
  }
  const int half_group = static_cast<int>(group_size) / 2;
  if (half_group <= 0 || (kRmsRotateThreads % half_group) != 0) {
    return false;
  }
  const int groups_per_batch = kRmsRotateThreads / half_group;
  const int groups = static_cast<int>(hidden / group_size);
  return (groups % groups_per_batch) == 0;
}

}  // namespace

extern "C" int hipengine_paro_rotate1_bf16(
    const uint16_t* x,
    uint16_t* out,
    const int16_t* pairs,
    const uint16_t* theta,
    const uint16_t* scales,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate1_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out,
      pairs,
      theta,
      scales,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate2_bf16(
    const uint16_t* x,
    uint16_t* out0,
    uint16_t* out1,
    const int16_t* pairs0,
    const int16_t* pairs1,
    const uint16_t* theta0,
    const uint16_t* theta1,
    const uint16_t* scales0,
    const uint16_t* scales1,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate2_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size), 2U),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out0,
      out1,
      pairs0,
      pairs1,
      theta0,
      theta1,
      scales0,
      scales1,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate3_bf16(
    const uint16_t* x,
    uint16_t* out0,
    uint16_t* out1,
    uint16_t* out2,
    const int16_t* pairs0,
    const int16_t* pairs1,
    const int16_t* pairs2,
    const uint16_t* theta0,
    const uint16_t* theta1,
    const uint16_t* theta2,
    const uint16_t* scales0,
    const uint16_t* scales1,
    const uint16_t* scales2,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate3_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size), 3U),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out0,
      out1,
      out2,
      pairs0,
      pairs1,
      pairs2,
      theta0,
      theta1,
      theta2,
      scales0,
      scales1,
      scales2,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate1_fp16(
    const half_t* x,
    half_t* out,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate1_kernel<half_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out,
      pairs,
      theta,
      scales,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate1_f32_to_fp16(
    const float* x,
    half_t* out,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_rotate1_f32_to_fp16_kernel,
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out,
      pairs,
      theta,
      scales,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate1_bf16_gate_fp16(
    const uint16_t* x,
    const half_t* gate,
    half_t* out,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* scales,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_rotate1_bf16_gate_fp16_kernel,
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size)),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      gate,
      out,
      pairs,
      theta,
      scales,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate2_fp16(
    const half_t* x,
    half_t* out0,
    half_t* out1,
    const int16_t* pairs0,
    const int16_t* pairs1,
    const half_t* theta0,
    const half_t* theta1,
    const half_t* scales0,
    const half_t* scales1,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate2_kernel<half_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size), 2U),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out0,
      out1,
      pairs0,
      pairs1,
      theta0,
      theta1,
      scales0,
      scales1,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rotate3_fp16(
    const half_t* x,
    half_t* out0,
    half_t* out1,
    half_t* out2,
    const int16_t* pairs0,
    const int16_t* pairs1,
    const int16_t* pairs2,
    const half_t* theta0,
    const half_t* theta1,
    const half_t* theta2,
    const half_t* scales0,
    const half_t* scales1,
    const half_t* scales2,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rotate_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (paro_rotate3_kernel<half_t>),
      dim3(static_cast<unsigned int>(tokens), static_cast<unsigned int>(hidden / group_size), 3U),
      dim3(static_cast<unsigned int>(group_size / 2)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      out0,
      out1,
      out2,
      pairs0,
      pairs1,
      pairs2,
      theta0,
      theta1,
      theta2,
      scales0,
      scales1,
      scales2,
      tokens,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rmsnorm_rotate2_fp16(
    const half_t* x,
    const half_t* ln_weight,
    half_t* out_norm,
    half_t* out0,
    half_t* out1,
    const int16_t* pairs0,
    const int16_t* pairs1,
    const half_t* theta0,
    const half_t* theta1,
    const half_t* scales0,
    const half_t* scales1,
    float eps,
    int64_t tokens,
    int64_t hidden,
    int64_t group_size,
    int64_t krot,
    hipStream_t stream) {
  if (!valid_rmsnorm_rotate2_shape(tokens, hidden, group_size, krot)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int lds_bytes = static_cast<unsigned int>(
      hidden * sizeof(half_t) +
      (kRmsRotateThreads / (group_size / 2)) * group_size * sizeof(float));
  hipLaunchKernelGGL(
      (paro_rmsnorm_rotate2_kernel<half_t>),
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(kRmsRotateThreads)),
      lds_bytes,
      stream,
      x,
      ln_weight,
      out_norm,
      out0,
      out1,
      pairs0,
      pairs1,
      theta0,
      theta1,
      scales0,
      scales1,
      eps,
      hidden,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}
