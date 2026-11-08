// DSpine adjacent-causal-injection bridge: elementwise glue kernels.
//
// Clean-room implementation of the DSpine code bridge (arXiv:2609.36173). The
// heavy projections (read R_l, packed [G_q;G_p;A], write W_l) run on the reused
// DFlash dense WMMA kernels; these kernels handle the normalization, code
// formation, gated predecessor message, and residual write between them. The
// runtime orchestrates: rms_normalize -> dense(R_l) -> codes -> dense(packed)
// -> gate_message -> dense(W_l) -> residual -> next input norm.
//
// Block geometry: T rows (anchor at row 0), hidden H. The predecessor shift is
// row t-1; row 0 (anchor) takes its code from the codebook and receives no
// message. BF16 activations, F32 accumulation where the reference uses it.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kWarp = 32;

__device__ inline float bf16_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}
__device__ inline uint16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>(u >> 16);
}

// One workgroup per row; lanes stride over H. Computes rms = sqrt(mean(raw^2))
// (f32, clamped) and normalized = raw/(rms+eps) (BF16). Emits rms per row.
__global__ void dspine_rms_normalize_kernel(
    const uint16_t* __restrict__ raw, uint16_t* __restrict__ normalized,
    float* __restrict__ rms_out, int64_t T, int64_t H, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= T) return;
  const uint16_t* r = raw + row * H;
  __shared__ float red[32];
  float acc = 0.f;
  for (int64_t d = threadIdx.x; d < H; d += blockDim.x) {
    float v = bf16_to_f32(r[d]);
    acc += v * v;
  }
  // warp reduce
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    if (threadIdx.x == 0) {
      float mean = v / static_cast<float>(H);
      red[0] = sqrtf(fmaxf(mean, 1.1754943508222875e-38f));
      rms_out[row] = red[0];
    }
  }
  __syncthreads();
  float rms = red[0];
  uint16_t* n = normalized + row * H;
  for (int64_t d = threadIdx.x; d < H; d += blockDim.x) {
    n[d] = f32_to_bf16(bf16_to_f32(r[d]) / (rms + eps));
  }
}

// projected [T,C] (f32) -> codes [T,C] (BF16): codes = sqrt(C)*p/(||p||+eps);
// row 0 replaced by sqrt(C)*codebook[anchor]. One workgroup per row.
__global__ void dspine_codes_kernel(
    const float* __restrict__ projected, const uint16_t* __restrict__ codebook,
    uint16_t* __restrict__ codes, int64_t anchor, int64_t T, int64_t C,
    float eps, float sqrt_c) {
  const int64_t row = blockIdx.x;
  if (row >= T) return;
  __shared__ float red[32];
  if (row == 0) {
    const uint16_t* cb = codebook + anchor * C;
    uint16_t* out = codes;
    for (int64_t d = threadIdx.x; d < C; d += blockDim.x)
      out[d] = f32_to_bf16(sqrt_c * bf16_to_f32(cb[d]));
    return;
  }
  const float* p = projected + row * C;
  float acc = 0.f;
  for (int64_t d = threadIdx.x; d < C; d += blockDim.x) acc += p[d] * p[d];
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    if (threadIdx.x == 0) red[0] = sqrtf(v);
  }
  __syncthreads();
  float norm = red[0];
  uint16_t* out = codes + row * C;
  for (int64_t d = threadIdx.x; d < C; d += blockDim.x)
    out[d] = f32_to_bf16(sqrt_c * p[d] / (norm + eps));
}

// packed [T,3R] (f32) = [q|s|v] -> message [T,R] (BF16):
// gate_t = sigmoid(q_t + s_{t-1} + bias); message_t = gate_t * v_{t-1}; row0=0.
// One workgroup per row; lanes stride over R.
__global__ void dspine_gate_message_kernel(
    const float* __restrict__ packed, const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ message, int64_t T, int64_t R) {
  const int64_t row = blockIdx.x;
  if (row >= T) return;
  uint16_t* m = message + row * R;
  if (row == 0) {
    for (int64_t d = threadIdx.x; d < R; d += blockDim.x) m[d] = f32_to_bf16(0.f);
    return;
  }
  const float* cur = packed + row * (3 * R);          // q at [0,R)
  const float* prev = packed + (row - 1) * (3 * R);   // s at [R,2R), v at [2R,3R)
  for (int64_t d = threadIdx.x; d < R; d += blockDim.x) {
    float q = cur[d];
    float s = prev[R + d];
    float v = prev[2 * R + d];
    float g = 1.f / (1.f + __expf(-(q + s + bf16_to_f32(bias[d]))));
    m[d] = f32_to_bf16(g * v);
  }
}

// out = raw + rms*delta (non-anchor rows); row0 = raw. delta [T,H] f32.
__global__ void dspine_residual_kernel(
    const uint16_t* __restrict__ raw, const float* __restrict__ rms,
    const float* __restrict__ delta, uint16_t* __restrict__ out,
    int64_t T, int64_t H) {
  const int64_t row = blockIdx.x;
  if (row >= T) return;
  const uint16_t* r = raw + row * H;
  uint16_t* o = out + row * H;
  if (row == 0) {
    for (int64_t d = threadIdx.x; d < H; d += blockDim.x) o[d] = r[d];
    return;
  }
  float rmsv = rms[row];
  const float* dl = delta + row * H;
  for (int64_t d = threadIdx.x; d < H; d += blockDim.x)
    o[d] = f32_to_bf16(bf16_to_f32(r[d]) + rmsv * dl[d]);
}

}  // namespace

extern "C" int hipengine_dspine_rms_normalize(
    const void* raw, void* normalized, void* rms_out,
    int64_t T, int64_t H, float eps, hipStream_t stream) {
  if (T <= 0 || H <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dspine_rms_normalize_kernel, dim3((unsigned)T), dim3(256),
                     0, stream, (const uint16_t*)raw, (uint16_t*)normalized,
                     (float*)rms_out, T, H, eps);
  return hipGetLastError();
}
extern "C" int hipengine_dspine_codes(
    const void* projected, const void* codebook, void* codes, int64_t anchor,
    int64_t T, int64_t C, float eps, float sqrt_c, hipStream_t stream) {
  if (T <= 0 || C <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dspine_codes_kernel, dim3((unsigned)T), dim3(256), 0, stream,
                     (const float*)projected, (const uint16_t*)codebook,
                     (uint16_t*)codes, anchor, T, C, eps, sqrt_c);
  return hipGetLastError();
}
extern "C" int hipengine_dspine_gate_message(
    const void* packed, const void* bias, void* message,
    int64_t T, int64_t R, hipStream_t stream) {
  if (T <= 0 || R <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dspine_gate_message_kernel, dim3((unsigned)T), dim3(256),
                     0, stream, (const float*)packed, (const uint16_t*)bias,
                     (uint16_t*)message, T, R);
  return hipGetLastError();
}
extern "C" int hipengine_dspine_residual(
    const void* raw, const void* rms, const void* delta, void* out,
    int64_t T, int64_t H, hipStream_t stream) {
  if (T <= 0 || H <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(dspine_residual_kernel, dim3((unsigned)T), dim3(256), 0,
                     stream, (const uint16_t*)raw, (const float*)rms,
                     (const float*)delta, (uint16_t*)out, T, H);
  return hipGetLastError();
}