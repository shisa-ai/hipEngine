// Source-faithful raw Q6_K to F16 dequantization for the bounded rocBLAS path.
//
// Source lineage: llama.cpp HIP c0bc8591e, ggml-cuda/dequantize.cuh and
// convert.cu. One local64 workgroup expands one raw 210-byte K256 block to
// contiguous row-major F16. The dequantized bytes are caller-owned transient
// scratch; no persistent weight sidecar is created.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q6_K_THREADS = 64;

__global__ void gguf_q6_k_dequantize_f16_source_kernel(
    const uint8_t* __restrict__ qweight,
    half_t* __restrict__ out) {
  const int64_t block_id = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int ip = tid >> 5;
  const int il = tid & 31;
  const int scale_index = 8 * ip + (il >> 4);

  const uint8_t* block = qweight + block_id * Q6_K_BLOCK_BYTES;
  const uint8_t* ql = block + 64 * ip + il;
  const uint8_t qh = block[128 + 32 * ip + il];
  const int8_t* scales =
      reinterpret_cast<const int8_t*>(block + 192) + scale_index;
  const float d = static_cast<float>(
      *reinterpret_cast<const half_t*>(block + 208));
  half_t* dst = out + block_id * QK_K + 128 * ip + il;

  dst[0] = static_cast<half_t>(
      d * scales[0] * (static_cast<int>((ql[0] & 0x0Fu) |
                                        (((qh >> 0) & 3u) << 4)) -
                         32));
  dst[32] = static_cast<half_t>(
      d * scales[2] * (static_cast<int>((ql[32] & 0x0Fu) |
                                        (((qh >> 2) & 3u) << 4)) -
                         32));
  dst[64] = static_cast<half_t>(
      d * scales[4] * (static_cast<int>((ql[0] >> 4) |
                                        (((qh >> 4) & 3u) << 4)) -
                         32));
  dst[96] = static_cast<half_t>(
      d * scales[6] * (static_cast<int>((ql[32] >> 4) |
                                        (((qh >> 6) & 3u) << 4)) -
                         32));
}

__global__ void gguf_q6_k_dequantize_bf16_to_f16_source_fused_kernel(
    const uint8_t* __restrict__ qweight,
    half_t* __restrict__ weight_out,
    const uint16_t* __restrict__ input,
    half_t* __restrict__ input_out,
    int64_t weight_blocks,
    int64_t input_count) {
  const int64_t task = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (task >= weight_blocks) {
    const int64_t element = (task - weight_blocks) * Q6_K_THREADS + tid;
    if (element < input_count) {
      union {
        uint32_t u32;
        float f32;
      } value;
      value.u32 = static_cast<uint32_t>(input[element]) << 16;
      input_out[element] = static_cast<half_t>(value.f32);
    }
    return;
  }

  const int ip = tid >> 5;
  const int il = tid & 31;
  const int scale_index = 8 * ip + (il >> 4);
  const uint8_t* block = qweight + task * Q6_K_BLOCK_BYTES;
  const uint8_t* ql = block + 64 * ip + il;
  const uint8_t qh = block[128 + 32 * ip + il];
  const int8_t* scales =
      reinterpret_cast<const int8_t*>(block + 192) + scale_index;
  const float d = static_cast<float>(
      *reinterpret_cast<const half_t*>(block + 208));
  half_t* dst = weight_out + task * QK_K + 128 * ip + il;
  dst[0] = static_cast<half_t>(
      d * scales[0] * (static_cast<int>((ql[0] & 0x0Fu) |
                                        (((qh >> 0) & 3u) << 4)) - 32));
  dst[32] = static_cast<half_t>(
      d * scales[2] * (static_cast<int>((ql[32] & 0x0Fu) |
                                        (((qh >> 2) & 3u) << 4)) - 32));
  dst[64] = static_cast<half_t>(
      d * scales[4] * (static_cast<int>((ql[0] >> 4) |
                                        (((qh >> 4) & 3u) << 4)) - 32));
  dst[96] = static_cast<half_t>(
      d * scales[6] * (static_cast<int>((ql[32] >> 4) |
                                        (((qh >> 6) & 3u) << 4)) - 32));
}

bool valid_shape(int64_t in_features, int64_t out_features) {
  return in_features > 0 && in_features % QK_K == 0 && out_features > 0;
}

}  // namespace

extern "C" int hipengine_gguf_q6_k_dequantize_f16_source(
    const uint8_t* qweight,
    half_t* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !out || !valid_shape(in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_blocks = out_features * (in_features / QK_K);
  hipLaunchKernelGGL(
      gguf_q6_k_dequantize_f16_source_kernel,
      dim3(static_cast<unsigned int>(total_blocks)),
      dim3(Q6_K_THREADS),
      0,
      stream,
      qweight,
      out);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q6_k_dequantize_bf16_to_f16_source_fused(
    const uint8_t* qweight,
    half_t* weight_out,
    const uint16_t* input,
    half_t* input_out,
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
      (input_count + Q6_K_THREADS - 1) / Q6_K_THREADS;
  hipLaunchKernelGGL(
      gguf_q6_k_dequantize_bf16_to_f16_source_fused_kernel,
      dim3(static_cast<unsigned int>(weight_blocks + input_blocks)),
      dim3(Q6_K_THREADS),
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
