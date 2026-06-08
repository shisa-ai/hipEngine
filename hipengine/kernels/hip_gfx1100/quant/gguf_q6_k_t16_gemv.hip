// Dense decode-shaped GEMV for GGUF Q6_K T16 replacement layout.
//
// P9.H3 extension for the qwen35moe Q6_K lm-head fallback: consumes the
// byte-neutral Q6T16 tile layout produced by ``hipengine.quant.gguf_t16``.
// One block computes one 16-column output tile for one input row.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q6_GROUPS16 = 16;
constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);

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

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename out_t>
__device__ inline out_t float_to_output(float v) {
  return static_cast<out_t>(v);
}

template <>
__device__ inline uint16_t float_to_output<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ inline int t16_q6_quant(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const uint8_t low_packed = tile[Q6_T16_QL_OFFSET + k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q6_T16_QH_OFFSET + k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high = (high_packed >> (2 * (col & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q6_k_t16_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const scalar_t* __restrict__ x_row = x + row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i = static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        acc[col] += xv * (d * static_cast<float>(scale_i * q_i));
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);

  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[sub_id * T16_COLS + col] = acc[col];
  }
  __syncthreads();

  if (tid == 0) {
    out_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_output<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_q6_t16(const void* x,
                  const void* tiles,
                  void* out,
                  int64_t rows,
                  int64_t in_features,
                  int64_t out_features,
                  hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q6_k_t16_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<out_t*>(out), rows, in_features, out_features);
  return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_gguf_q6_k_t16_gemv_decode_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q6_t16<uint16_t, float>(x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q6_t16<uint16_t, uint16_t>(x, tiles, out, rows, in_features, out_features, stream);
}
