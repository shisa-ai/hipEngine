// Laguna unfused FP32 per-head softplus attention gating.

#include <hip/hip_runtime.h>
#include <cstdint>

using half_t = _Float16;

namespace {

__device__ inline uint16_t f32_to_bf16(float value) {
  union { float f; uint32_t u; } in;
  in.f = value;
  in.u += 0x7FFFu + ((in.u >> 16) & 1u);
  return static_cast<uint16_t>(in.u >> 16);
}

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u;
    float f;
  } out;
  out.u = static_cast<uint32_t>(value) << 16;
  return out.f;
}

__device__ inline float softplus(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

template <typename out_t>
__device__ inline void store(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = f32_to_bf16(value);
}

template <typename out_t, bool VIA_BF16 = false>
__global__ void laguna_softplus_head_gate_kernel(
    const float* __restrict__ context,
    const float* __restrict__ gate,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t heads,
    int64_t head_dim) {
  const int64_t size = rows * heads * head_dim;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += stride) {
    const int64_t head_row = idx / head_dim;
    const float value = context[idx] * softplus(gate[head_row]);
    if constexpr (VIA_BF16) {
      out[idx] = static_cast<out_t>(
          bf16_bits_to_float(f32_to_bf16(value)));
    } else {
      store(out, idx, value);
    }
  }
}

template <typename out_t, bool VIA_BF16 = false>
int launch(const float* context, const float* gate, out_t* out, int64_t rows,
           int64_t heads, int64_t head_dim, int64_t threads, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0 || threads <= 0 || threads > 256 || threads % 32) {
    return hipErrorInvalidValue;
  }
  const int64_t size = rows * heads * head_dim;
  const int64_t blocks = (size + threads - 1) / threads;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          laguna_softplus_head_gate_kernel<out_t, VIA_BF16>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      context,
      gate,
      out,
      rows,
      heads,
      head_dim);
  return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_laguna_softplus_head_gate_f32_out(
    const float* context, const float* gate, float* out, int64_t rows,
    int64_t heads, int64_t head_dim, int64_t threads, hipStream_t stream) {
  return launch(context, gate, out, rows, heads, head_dim, threads, stream);
}

extern "C" int hipengine_laguna_softplus_head_gate_f32_bf16_out(
    const float* context, const float* gate, uint16_t* out, int64_t rows,
    int64_t heads, int64_t head_dim, int64_t threads, hipStream_t stream) {
  return launch(context, gate, out, rows, heads, head_dim, threads, stream);
}

extern "C" int
hipengine_laguna_softplus_head_gate_f32_fp16_via_bf16_out(
    const float* context,
    const float* gate,
    half_t* out,
    int64_t rows,
    int64_t heads,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  return launch<half_t, true>(
      context,
      gate,
      out,
      rows,
      heads,
      head_dim,
      threads,
      stream);
}
