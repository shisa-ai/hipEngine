// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// dense_gemv_out kernel body is preserved; wrappers are hipENGINE C ABI.

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
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float value) {
  return float_to_bf16_bits(value);
}

template <typename scalar_t>
__global__ void dense_gemv_out_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t x_base = row * in_features;
  const int64_t w_base = out_col * in_features;

  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const scalar_t* x_ptr = x + x_base + k_base;
    const scalar_t* w_ptr = weight + w_base + k_base;
    acc = fmaf(scalar_to_float(x_ptr[0]), scalar_to_float(w_ptr[0]), acc);
    acc = fmaf(scalar_to_float(x_ptr[1]), scalar_to_float(w_ptr[1]), acc);
    acc = fmaf(scalar_to_float(x_ptr[2]), scalar_to_float(w_ptr[2]), acc);
    acc = fmaf(scalar_to_float(x_ptr[3]), scalar_to_float(w_ptr[3]), acc);
    acc = fmaf(scalar_to_float(x_ptr[4]), scalar_to_float(w_ptr[4]), acc);
    acc = fmaf(scalar_to_float(x_ptr[5]), scalar_to_float(w_ptr[5]), acc);
    acc = fmaf(scalar_to_float(x_ptr[6]), scalar_to_float(w_ptr[6]), acc);
    acc = fmaf(scalar_to_float(x_ptr[7]), scalar_to_float(w_ptr[7]), acc);
  }
  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    acc = fmaf(scalar_to_float(x[x_base + in_col]), scalar_to_float(weight[w_base + in_col]), acc);
  }

  __shared__ float sdata[256];
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row * out_features + out_col] = float_to_scalar<scalar_t>(sdata[0]);
  }
}


template <typename scalar_t, int TILE_M, int TILE_N, int TILE_K>
__global__ void dense_prefill_gemm_out_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int local_col = threadIdx.x;
  const int local_row = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * TILE_M + local_row;
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) * TILE_N + local_col;
  const int linear_tid = local_row * blockDim.x + local_col;

  __shared__ scalar_t sx[TILE_M][TILE_K];
  __shared__ scalar_t sw[TILE_N][TILE_K];

  float acc = 0.0f;
  for (int64_t k0 = 0; k0 < in_features; k0 += TILE_K) {
    for (int idx = linear_tid; idx < TILE_M * TILE_K; idx += TILE_M * TILE_N) {
      const int r = idx / TILE_K;
      const int k = idx - r * TILE_K;
      const int64_t global_row = static_cast<int64_t>(blockIdx.y) * TILE_M + r;
      const int64_t global_k = k0 + k;
      sx[r][k] = (global_row < rows && global_k < in_features) ? x[global_row * in_features + global_k] : scalar_t{0};
    }
    for (int idx = linear_tid; idx < TILE_N * TILE_K; idx += TILE_M * TILE_N) {
      const int c = idx / TILE_K;
      const int k = idx - c * TILE_K;
      const int64_t global_col = static_cast<int64_t>(blockIdx.x) * TILE_N + c;
      const int64_t global_k = k0 + k;
      sw[c][k] = (global_col < out_features && global_k < in_features) ? weight[global_col * in_features + global_k] : scalar_t{0};
    }
    __syncthreads();

    if (row < rows && out_col < out_features) {
      const int64_t remaining_k = in_features - k0;
      const int tile_k = remaining_k < TILE_K ? static_cast<int>(remaining_k) : TILE_K;
      for (int k = 0; k < tile_k; ++k) {
        acc = fmaf(scalar_to_float(sx[local_row][k]), scalar_to_float(sw[local_col][k]), acc);
      }
    }
    __syncthreads();
  }
  if (row < rows && out_col < out_features) {
    out[row * out_features + out_col] = float_to_scalar<scalar_t>(acc);
  }
}


template <typename scalar_t>
__global__ void dense_dual_gemv_out_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight_a,
    const scalar_t* __restrict__ weight_b,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int64_t global_col = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t out_features = out_features_a + out_features_b;
  const scalar_t* __restrict__ weight = use_b ? weight_b : weight_a;
  const int64_t x_base = row * in_features;
  const int64_t w_base = local_col * in_features;

  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const scalar_t* x_ptr = x + x_base + k_base;
    const scalar_t* w_ptr = weight + w_base + k_base;
    acc = fmaf(scalar_to_float(x_ptr[0]), scalar_to_float(w_ptr[0]), acc);
    acc = fmaf(scalar_to_float(x_ptr[1]), scalar_to_float(w_ptr[1]), acc);
    acc = fmaf(scalar_to_float(x_ptr[2]), scalar_to_float(w_ptr[2]), acc);
    acc = fmaf(scalar_to_float(x_ptr[3]), scalar_to_float(w_ptr[3]), acc);
    acc = fmaf(scalar_to_float(x_ptr[4]), scalar_to_float(w_ptr[4]), acc);
    acc = fmaf(scalar_to_float(x_ptr[5]), scalar_to_float(w_ptr[5]), acc);
    acc = fmaf(scalar_to_float(x_ptr[6]), scalar_to_float(w_ptr[6]), acc);
    acc = fmaf(scalar_to_float(x_ptr[7]), scalar_to_float(w_ptr[7]), acc);
  }
  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    acc = fmaf(scalar_to_float(x[x_base + in_col]), scalar_to_float(weight[w_base + in_col]), acc);
  }

  __shared__ float sdata[256];
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row * out_features + global_col] = float_to_scalar<scalar_t>(sdata[0]);
  }
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

}  // namespace

extern "C" int hipengine_dense_gemv_out_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_gemv_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dense_prefill_gemm_out_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  (void)threads;
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 8;
  constexpr int TILE_K = 32;
  hipLaunchKernelGGL(
      (dense_prefill_gemm_out_kernel<uint16_t, TILE_M, TILE_N, TILE_K>),
      dim3(
          static_cast<unsigned int>((out_features + TILE_N - 1) / TILE_N),
          static_cast<unsigned int>((rows + TILE_M - 1) / TILE_M)),
      dim3(TILE_N, TILE_M),
      0,
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dense_gemv_out_fp16(
    const half_t* x,
    const half_t* weight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_gemv_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      weight,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dense_dual_gemv_out_bf16(
    const uint16_t* x,
    const uint16_t* weight_a,
    const uint16_t* weight_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features_a <= 0 || out_features_b <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_dual_gemv_out_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(out_features_a + out_features_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      weight_a,
      weight_b,
      out,
      rows,
      in_features,
      out_features_a,
      out_features_b);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dense_dual_gemv_out_fp16(
    const half_t* x,
    const half_t* weight_a,
    const half_t* weight_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features_a <= 0 || out_features_b <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (dense_dual_gemv_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_features_a + out_features_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      weight_a,
      weight_b,
      out,
      rows,
      in_features,
      out_features_a,
      out_features_b);
  return static_cast<int>(hipGetLastError());
}
