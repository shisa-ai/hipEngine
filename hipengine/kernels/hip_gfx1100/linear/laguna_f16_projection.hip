// Laguna source-F16 dense projections with BF16/F32 activations and outputs.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

using half_t = _Float16;

__device__ inline float bf16_to_f32(uint16_t value) {
  union { uint32_t u; float f; } out;
  out.u = static_cast<uint32_t>(value) << 16;
  return out.f;
}

__device__ inline uint16_t f32_to_bf16(float value) {
  union { float f; uint32_t u; } in;
  in.f = value;
  in.u += 0x7FFFu + ((in.u >> 16) & 1u);
  return static_cast<uint16_t>(in.u >> 16);
}

template <typename T>
__device__ inline float load_scalar(T value) { return static_cast<float>(value); }

template <>
__device__ inline float load_scalar<uint16_t>(uint16_t value) { return bf16_to_f32(value); }

template <typename T>
__device__ inline void store_scalar(T* out, int64_t idx, float value) { out[idx] = static_cast<T>(value); }

template <>
__device__ inline void store_scalar<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = f32_to_bf16(value);
}

__device__ inline float block_sum(float value) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset);
  __shared__ float partial[8];
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) total += partial[i];
    partial[0] = total;
  }
  __syncthreads();
  return partial[0];
}

template <typename x_t, typename out_t>
__global__ void laguna_f16w_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || col >= out_features) return;
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) store_scalar(out, row * out_features + col, total);
}

template <typename x_t>
__global__ void laguna_f16w_dual_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t total_cols = out_a_features + out_b_features;
  if (row >= rows || col >= total_cols) return;
  const bool a = col < out_a_features;
  const int64_t local_col = a ? col : col - out_a_features;
  const half_t* weight = a ? weight_a : weight_b;
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[local_col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) {
    if (a) out_a[row * out_a_features + local_col] = total;
    else out_b[row * out_b_features + local_col] = total;
  }
}

template <typename x_t>
__global__ void laguna_f16w_triple_gemv_kernel(
    const x_t* __restrict__ x,
    const half_t* __restrict__ weight_a,
    const half_t* __restrict__ weight_b,
    const half_t* __restrict__ weight_c,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    float* __restrict__ out_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_a_features,
    int64_t out_b_features,
    int64_t out_c_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows || col >= out_a_features + out_b_features + out_c_features) return;
  const half_t* weight;
  float* out;
  int64_t local_col;
  int64_t width;
  if (col < out_a_features) {
    weight = weight_a; out = out_a; local_col = col; width = out_a_features;
  } else if (col < out_a_features + out_b_features) {
    weight = weight_b; out = out_b; local_col = col - out_a_features; width = out_b_features;
  } else {
    weight = weight_c; out = out_c; local_col = col - out_a_features - out_b_features; width = out_c_features;
  }
  float acc = 0.0f;
  for (int64_t k = threadIdx.x; k < in_features; k += blockDim.x) {
    acc += load_scalar(x[row * in_features + k]) * static_cast<float>(weight[local_col * in_features + k]);
  }
  const float total = block_sum(acc);
  if (threadIdx.x == 0) out[row * width + local_col] = total;
}

template <typename x_t, typename out_t>
int launch_single(const x_t* x, const half_t* weight, out_t* out, int64_t rows,
                  int64_t in_features, int64_t out_features, int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_gemv_kernel<<<dim3(out_features, rows), dim3(threads), 0, stream>>>(x, weight, out, rows, in_features, out_features);
  return hipGetLastError();
}

template <typename x_t>
int launch_dual(const x_t* x, const half_t* wa, const half_t* wb, float* oa, float* ob,
                int64_t rows, int64_t in_features, int64_t out_a, int64_t out_b,
                int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_a <= 0 || out_b <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_dual_gemv_kernel<<<dim3(out_a + out_b, rows), dim3(threads), 0, stream>>>(x, wa, wb, oa, ob, rows, in_features, out_a, out_b);
  return hipGetLastError();
}

template <typename x_t>
int launch_triple(const x_t* x, const half_t* wa, const half_t* wb, const half_t* wc,
                  float* oa, float* ob, float* oc, int64_t rows, int64_t in_features,
                  int64_t out_a, int64_t out_b, int64_t out_c, int64_t threads, hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_a <= 0 || out_b <= 0 || out_c <= 0 || threads <= 0 || threads > 256 || threads % 32) return hipErrorInvalidValue;
  laguna_f16w_triple_gemv_kernel<<<dim3(out_a + out_b + out_c, rows), dim3(threads), 0, stream>>>(x, wa, wb, wc, oa, ob, oc, rows, in_features, out_a, out_b, out_c);
  return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_laguna_f16w_gemv_bf16_f32_out(const uint16_t* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_bf16_bf16_out(const uint16_t* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_f32_f32_out(const float* x, const half_t* w, float* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_gemv_f32_bf16_out(const float* x, const half_t* w, uint16_t* out, int64_t r, int64_t k, int64_t n, int64_t t, hipStream_t s) { return launch_single(x,w,out,r,k,n,t,s); }
extern "C" int hipengine_laguna_f16w_dual_gemv_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, float* oa, float* ob, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t t, hipStream_t s) { return launch_dual(x,wa,wb,oa,ob,r,k,na,nb,t,s); }
extern "C" int hipengine_laguna_f16w_triple_gemv_bf16_f32_out(const uint16_t* x, const half_t* wa, const half_t* wb, const half_t* wc, float* oa, float* ob, float* oc, int64_t r, int64_t k, int64_t na, int64_t nb, int64_t nc, int64_t t, hipStream_t s) { return launch_triple(x,wa,wb,wc,oa,ob,oc,r,k,na,nb,nc,t,s); }
