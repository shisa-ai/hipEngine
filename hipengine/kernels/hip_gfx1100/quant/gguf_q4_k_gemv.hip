// GGUF block_q4_K GEMV spike.  This kernel consumes raw GGUF Q4_K tensor
// bytes directly and preserves GGML Q4_K scale/min math; it is intentionally
// separate from PARO/AWQ kernels.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
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
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

__device__ inline uint8_t gguf_q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline float gguf_q4_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * 32 + lane];
  const uint8_t q = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const float scale = static_cast<float>(gguf_q4_k_scale(scales, subblock));
  const float minv = static_cast<float>(gguf_q4_k_min(scales, subblock));
  return d * scale * static_cast<float>(q) - dmin * minv;
}

__device__ inline float reduce_block_sum(float acc) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) {
    wave_sums[wave] = acc;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      total += wave_sums[i];
    }
  }
  return total;
}

template <typename scalar_t>
__launch_bounds__(256, 2)
__global__ void gguf_q4_k_gemv_f32_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int row_bytes = (in_features / QK_K) * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ x_row =
      x + static_cast<int64_t>(row) * in_features;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    const float wv = gguf_q4_k_weight(weight_row, k);
    acc += xv * wv;
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] = total;
  }
}

template <typename scalar_t>
int launch_gguf_q4_k_gemv_f32_out(
    const scalar_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || (in_features % QK_K) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_q4_k_gemv_f32_out_kernel<scalar_t><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_gemv_f32_f32_out(
    const float* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_f32_out<float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_gemv_fp16_f32_out(
    const half_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_f32_out<half_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_gemv_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_f32_out<uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}
