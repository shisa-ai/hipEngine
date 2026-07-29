// Exact-value raw Q5_K to F32 expansion for transient rocBLAS SGEMM.
//
// The dequantized plane is caller-owned projection-local scratch.  This keeps
// hipEngine's resident BF16 activation values and the established raw-Q5 F32
// scale/min expression; only the downstream SGEMM reduction association may
// differ from the exact coltile fallback.  No persistent weight sidecar exists.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q5_K_THREADS = 64;

__device__ __forceinline__ float fp16_bytes_to_float(const uint8_t* ptr) {
  union {
    uint16_t u16;
    half_t f16;
  } value;
  value.u16 = static_cast<uint16_t>(ptr[0]) |
              (static_cast<uint16_t>(ptr[1]) << 8);
  return static_cast<float>(value.f16);
}

__device__ __forceinline__ uint8_t q5_k_scale(
    const uint8_t* __restrict__ scales,
    int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int index = subblock - 4;
  return (scales[8 + index] & 0x0Fu) |
         ((scales[index] >> 2) & 0x30u);
}

__device__ __forceinline__ uint8_t q5_k_min(
    const uint8_t* __restrict__ scales,
    int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int index = subblock - 4;
  return (scales[8 + index] >> 4) |
         ((scales[4 + index] >> 2) & 0x30u);
}

__device__ __forceinline__ float q5_k_value(
    const uint8_t* __restrict__ block,
    int within) {
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* __restrict__ scales = block + 4;
  const uint8_t* __restrict__ qh = block + 16;
  const uint8_t* __restrict__ qs = block + 48;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane];
  const uint8_t low =
      (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane] >> subblock) & 0x01u;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const float scale = static_cast<float>(q5_k_scale(scales, subblock));
  const float min_value = static_cast<float>(q5_k_min(scales, subblock));
  const float quant = static_cast<float>(low | (high << 4));
  return d * scale * quant - dmin * min_value;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u32;
    float f32;
  } value;
  value.u32 = static_cast<uint32_t>(bits) << 16;
  return value.f32;
}

__device__ __forceinline__ void expand_q5_block(
    const uint8_t* __restrict__ block,
    float* __restrict__ out,
    int tid) {
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int within = tid + item * Q5_K_THREADS;
    out[within] = q5_k_value(block, within);
  }
}

__global__ void gguf_q5_k_dequantize_f32_exact_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out) {
  const int64_t block_id = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const uint8_t* __restrict__ block =
      qweight + block_id * Q5_K_BLOCK_BYTES;
  float* __restrict__ dst = out + block_id * QK_K;
  expand_q5_block(block, dst, tid);
}

__global__ void gguf_q5_k_dequantize_bf16_to_f32_exact_fused_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ weight_out,
    const uint16_t* __restrict__ input,
    float* __restrict__ input_out,
    int64_t weight_blocks,
    int64_t input_count) {
  const int64_t task = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (task >= weight_blocks) {
    const int64_t element =
        (task - weight_blocks) * Q5_K_THREADS + tid;
    if (element < input_count) {
      input_out[element] = bf16_bits_to_float(input[element]);
    }
    return;
  }

  const uint8_t* __restrict__ block =
      qweight + task * Q5_K_BLOCK_BYTES;
  float* __restrict__ dst = weight_out + task * QK_K;
  expand_q5_block(block, dst, tid);
}

bool valid_shape(int64_t in_features, int64_t out_features) {
  return in_features > 0 && in_features % QK_K == 0 && out_features > 0;
}

}  // namespace

extern "C" int hipengine_gguf_q5_k_dequantize_f32_exact(
    const uint8_t* qweight,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !out || !valid_shape(in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_blocks = out_features * (in_features / QK_K);
  hipLaunchKernelGGL(
      gguf_q5_k_dequantize_f32_exact_kernel,
      dim3(static_cast<unsigned int>(total_blocks)),
      dim3(Q5_K_THREADS),
      0,
      stream,
      qweight,
      out);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
    const uint8_t* qweight,
    float* weight_out,
    const uint16_t* input,
    float* input_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !weight_out || !input || !input_out || rows <= 0 ||
      !valid_shape(in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t weight_blocks = out_features * (in_features / QK_K);
  const int64_t input_count = rows * in_features;
  const int64_t input_blocks =
      (input_count + Q5_K_THREADS - 1) / Q5_K_THREADS;
  hipLaunchKernelGGL(
      gguf_q5_k_dequantize_bf16_to_f32_exact_fused_kernel,
      dim3(static_cast<unsigned int>(weight_blocks + input_blocks)),
      dim3(Q5_K_THREADS),
      0,
      stream,
      qweight,
      weight_out,
      input,
      input_out,
      weight_blocks,
      input_count);
  return static_cast<int>(hipGetLastError());
}
