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
constexpr int Q4_K_THREADS = 64;
constexpr int Q4_T16_COLS = 16;
constexpr int Q4_T16_SUBBLOCK = 32;
constexpr int Q4_T16_SUBBLOCKS = 8;
constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + Q4_T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET =
    Q4_T16_DMIN_OFFSET + Q4_T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET =
    Q4_T16_SCALE_OFFSET + Q4_T16_SUBBLOCKS * Q4_T16_COLS;
constexpr int Q4_T16_Q_OFFSET =
    Q4_T16_MIN_OFFSET + Q4_T16_SUBBLOCKS * Q4_T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES =
    Q4_T16_Q_OFFSET +
    Q4_T16_SUBBLOCKS * Q4_T16_SUBBLOCK * (Q4_T16_COLS / 2);
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q6_K_THREADS = 64;
constexpr int Q6_T16_COLS = 16;
constexpr int Q6_T16_GROUPS = 16;
constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_COLS * 2;
constexpr int Q6_T16_QMICRO_OFFSET =
    Q6_T16_SCALE_OFFSET + Q6_T16_GROUPS * Q6_T16_COLS;
constexpr int Q6_T16_QMICRO_RECORD_BYTES = 12;
constexpr int Q6_T16_BLOCK_BYTES = Q6_K_BLOCK_BYTES * Q6_T16_COLS;

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

__global__ void gguf_q4_k_t16_dequantize_f16_tile_kernel(
    const uint8_t* __restrict__ tiles,
    half_t* __restrict__ out,
    int64_t blocks_per_row,
    int64_t out_features,
    int64_t col_start,
    int64_t col_count) {
  const int64_t task = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t block = task % blocks_per_row;
  const int64_t local_col = task / blocks_per_row;
  if (local_col >= col_count) {
    return;
  }
  const int64_t out_col = col_start + local_col;
  const int64_t col_tile = out_col / Q4_T16_COLS;
  const int col16 = static_cast<int>(out_col % Q4_T16_COLS);
  const uint8_t* tile =
      tiles + (col_tile * blocks_per_row + block) * Q4_T16_BLOCK_BYTES;
  const float d = static_cast<float>(
      *reinterpret_cast<const half_t*>(tile + Q4_T16_D_OFFSET + col16 * 2));
  const float dmin = static_cast<float>(
      *reinterpret_cast<const half_t*>(tile + Q4_T16_DMIN_OFFSET + col16 * 2));

  for (int k = tid; k < QK_K; k += Q4_K_THREADS) {
    const int subblock = k >> 5;
    const int lane32 = k & 31;
    const uint8_t packed = tile[
        Q4_T16_Q_OFFSET +
        (subblock * Q4_T16_SUBBLOCK + lane32) * (Q4_T16_COLS / 2) +
        (col16 >> 1)];
    const uint8_t quant =
        (col16 & 1) ? (packed >> 4) : (packed & 0x0Fu);
    const float value =
        d * static_cast<float>(
                tile[Q4_T16_SCALE_OFFSET +
                     subblock * Q4_T16_COLS + col16]) *
            static_cast<float>(quant) -
        dmin * static_cast<float>(
                   tile[Q4_T16_MIN_OFFSET +
                        subblock * Q4_T16_COLS + col16]);
    out[local_col * (blocks_per_row * QK_K) + block * QK_K + k] =
        static_cast<half_t>(value);
  }
}

__global__ void gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_kernel(
    const uint8_t* __restrict__ tiles,
    half_t* __restrict__ out,
    int64_t blocks_per_row,
    int64_t out_features,
    int64_t col_start,
    int64_t col_count) {
  const int64_t task = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t block = task % blocks_per_row;
  const int64_t local_col = task / blocks_per_row;
  if (local_col >= col_count) {
    return;
  }
  const int64_t out_col = col_start + local_col;
  const int64_t col_tile = out_col / Q6_T16_COLS;
  const int col16 = static_cast<int>(out_col % Q6_T16_COLS);
  const uint8_t* tile =
      tiles + (col_tile * blocks_per_row + block) * Q6_T16_BLOCK_BYTES;
  const float d = static_cast<float>(
      *reinterpret_cast<const half_t*>(tile + Q6_T16_D_OFFSET + col16 * 2));
  const int8_t* scales = reinterpret_cast<const int8_t*>(
      tile + Q6_T16_SCALE_OFFSET);

  for (int k = tid; k < QK_K; k += Q6_K_THREADS) {
    const int subblock = k >> 5;
    const int pack = (k & 31) >> 2;
    const int k4 = k & 3;
    const int col_quartet = col16 >> 2;
    const int col4 = col16 & 3;
    const int record_index =
        (subblock * (Q6_T16_COLS / 4) + col_quartet) * 8 + pack;
    const uint8_t* record =
        tile + Q6_T16_QMICRO_OFFSET +
        record_index * Q6_T16_QMICRO_RECORD_BYTES;
    const uint8_t low_byte = record[(col4 >> 1) * 4 + k4];
    const uint8_t low =
        (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
    const uint8_t high = (record[8 + k4] >> (2 * col4)) & 0x03u;
    const int quant = static_cast<int>(low | (high << 4)) - 32;
    const float value =
        d * static_cast<float>(scales[(k >> 4) * Q6_T16_COLS + col16]) *
        static_cast<float>(quant);
    out[local_col * (blocks_per_row * QK_K) + block * QK_K + k] =
        static_cast<half_t>(value);
  }
}

bool valid_shape(int64_t in_features, int64_t out_features) {
  return in_features > 0 && in_features % QK_K == 0 && out_features > 0;
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_t16_dequantize_f16_tile(
    const uint8_t* tiles,
    half_t* out,
    int64_t in_features,
    int64_t out_features,
    int64_t col_start,
    int64_t col_count,
    hipStream_t stream) {
  if (!tiles || !out || !valid_shape(in_features, out_features) ||
      out_features % Q4_T16_COLS != 0 || col_start < 0 || col_count <= 0 ||
      col_start % Q4_T16_COLS != 0 || col_start + col_count > out_features) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t tasks = col_count * blocks_per_row;
  hipLaunchKernelGGL(
      gguf_q4_k_t16_dequantize_f16_tile_kernel,
      dim3(static_cast<unsigned int>(tasks)),
      dim3(Q4_K_THREADS),
      0,
      stream,
      tiles,
      out,
      blocks_per_row,
      out_features,
      col_start,
      col_count);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile(
    const uint8_t* tiles,
    half_t* out,
    int64_t in_features,
    int64_t out_features,
    int64_t col_start,
    int64_t col_count,
    hipStream_t stream) {
  if (!tiles || !out || !valid_shape(in_features, out_features) ||
      out_features % Q6_T16_COLS != 0 || col_start < 0 || col_count <= 0 ||
      col_start % Q6_T16_COLS != 0 || col_start + col_count > out_features) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t tasks = col_count * blocks_per_row;
  hipLaunchKernelGGL(
      gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_kernel,
      dim3(static_cast<unsigned int>(tasks)),
      dim3(Q6_K_THREADS),
      0,
      stream,
      tiles,
      out,
      blocks_per_row,
      out_features,
      col_start,
      col_count);
  return static_cast<int>(hipGetLastError());
}

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
