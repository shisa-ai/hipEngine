// Dense decode GEMV for raw GGUF Q8_0 weights using q8_1-quantized activations
// and the RDNA3 int8 dot (v_dot4 via __builtin_amdgcn_sudot4).
//
// Task #13: the dense Q8_0 attention projections (q/k/v/o/gate) are the top GPU
// cost (~35-47%) of the GGUF MTP verifier. Q8_0 is already int8 and symmetric
// (no zero-point), so the q8_1+dp4a formula is simply
//
//     out[r,c] = sum_blocks  x_d[r,b] * w_d[c,b] * dp4a(w_int8x4, x_int8x4)
//
// with no dmin/min correction term (unlike Q4_K). The raw Q8_0 block stores its
// 32 int8 contiguously, so 4 consecutive-K weights are a contiguous uint32 -- no
// repack is needed (unlike the stride-16 T16 layout).
//
// Grid: one block (one wavefront) computes one (row, out_feature). Accumulation
// is FP32; the output scalar type matches the Q8_0 t16 decode wrappers (bf16).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int Q8_0_BLOCK = 32;       // elements per Q8_0 / q8_1 block
constexpr int Q8_0_RAW_BYTES = 34;   // 2-byte fp16 scale + 32 int8 quants
constexpr int Q8_1_BLOCK = 32;

struct gguf_q8_1_block {
  half_t d;                 // activation block scale (amax / 127)
  half_t s;                 // sum of activations (unused for symmetric Q8_0)
  int8_t qs[Q8_1_BLOCK];    // int8 quantized activations
};

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

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

template <typename out_t>
__device__ inline out_t float_to_scalar(float v) {
  return static_cast<out_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

// Signed int8 4-wide dot product (both operands signed). Mirrors the proven
// codebase idiom; CPU fallback keeps the same signed semantics for tests.
__device__ inline int q8_0_dp4a(uint32_t w, uint32_t x, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(w), true, static_cast<int>(x), c, false);
#else
  int acc = c;
  for (int i = 0; i < 4; ++i) {
    acc += static_cast<int>(static_cast<int8_t>((w >> (8 * i)) & 0xFFu)) *
           static_cast<int>(static_cast<int8_t>((x >> (8 * i)) & 0xFFu));
  }
  return acc;
#endif
}

#define Q8_0_DP4A_WAVE_REDUCE(acc) \
  do {                             \
    acc += __shfl_down(acc, 16);   \
    acc += __shfl_down(acc, 8);    \
    acc += __shfl_down(acc, 4);    \
    acc += __shfl_down(acc, 2);    \
    acc += __shfl_down(acc, 1);    \
  } while (0)

template <typename out_t>
__global__ __launch_bounds__(32) void q8_0_dp4a_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (col >= out_features || row >= rows) return;

  const int64_t blocks = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ wrow = qweight + col * blocks * Q8_0_RAW_BYTES;
  const gguf_q8_1_block* __restrict__ xrow = xq + row * blocks;

  float acc = 0.0f;
  for (int64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
    const uint8_t* __restrict__ wb = wrow + b * Q8_0_RAW_BYTES;
    const float wd = fp16_bytes_to_float(wb);
    const int8_t* __restrict__ wq = reinterpret_cast<const int8_t*>(wb + 2);
    const gguf_q8_1_block* __restrict__ xb = xrow + b;
    const float xd = static_cast<float>(xb->d);
    int dot = 0;
#pragma unroll
    for (int g = 0; g < Q8_0_BLOCK / 4; ++g) {
      const uint32_t w4 = *reinterpret_cast<const uint32_t*>(wq + g * 4);
      const uint32_t x4 = *reinterpret_cast<const uint32_t*>(xb->qs + g * 4);
      dot = q8_0_dp4a(w4, x4, dot);
    }
    acc += wd * xd * static_cast<float>(dot);
  }

  Q8_0_DP4A_WAVE_REDUCE(acc);
  if (threadIdx.x == 0) {
    out[row * out_features + col] = float_to_scalar<out_t>(acc);
  }
}

}  // namespace

extern "C" int hipengine_gguf_q8_0_dp4a_gemv_bf16_bf16_out(
    const void* xq,
    const void* qweight,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  (void)threads;
  if (rows <= 0 || in_features <= 0 || out_features <= 0) return hipErrorInvalidValue;
  if (in_features % Q8_0_BLOCK != 0) return hipErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(out_features), static_cast<unsigned>(rows), 1);
  const dim3 block(32);
  hipLaunchKernelGGL((q8_0_dp4a_gemv_kernel<uint16_t>), grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const uint8_t*>(qweight),
                     reinterpret_cast<uint16_t*>(out),
                     rows, in_features, out_features);
  return hipGetLastError();
}
