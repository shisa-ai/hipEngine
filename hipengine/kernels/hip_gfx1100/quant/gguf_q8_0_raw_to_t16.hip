// SH6-P1 byte-lossless raw GGUF Q8_0 -> Q8T16 pair repacker.
//
// One workgroup converts one (16 output columns x one 32-K block) record:
//
//   raw:  16 independently-strided 34-byte Q8_0 blocks
//   T16:  16 fp16 scales followed by a [32 K, 16 output] int8 plane
//
// The transform changes only byte order. A and B share one combined grid but
// keep independent input/output pointers so callers can use one contiguous
// pair-sized scratch allocation without retaining a duplicate weight owner.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int Q8_0_QK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int T16_COLS = 16;
constexpr int T16_D_BYTES = T16_COLS * 2;
constexpr int T16_BLOCK_BYTES = T16_D_BYTES + Q8_0_QK * T16_COLS;

__global__ __launch_bounds__(256, 2) void gguf_q8_0_raw_pair_to_t16_kernel(
    const uint8_t* __restrict__ raw_a,
    const uint8_t* __restrict__ raw_b,
    uint8_t* __restrict__ t16_a,
    uint8_t* __restrict__ t16_b,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  extern __shared__ uint8_t raw_tile[];

  const int64_t blocks_per_row = in_features / Q8_0_QK;
  const int64_t records_a = (out_features_a / T16_COLS) * blocks_per_row;
  const int64_t record = static_cast<int64_t>(blockIdx.x);
  const bool is_a = record < records_a;
  const int64_t local_record = is_a ? record : record - records_a;
  const uint8_t* raw = is_a ? raw_a : raw_b;
  uint8_t* t16 = is_a ? t16_a : t16_b;

  const int64_t out_tile = local_record / blocks_per_row;
  const int64_t block_index = local_record - out_tile * blocks_per_row;
  const int64_t bytes_per_row = blocks_per_row * Q8_0_BLOCK_BYTES;

  // Stage each source row's contiguous block. This turns the output transpose
  // into coalesced sequential stores while using only 544 bytes of LDS.
  for (int index = static_cast<int>(threadIdx.x); index < T16_BLOCK_BYTES;
       index += static_cast<int>(blockDim.x)) {
    const int col = index / Q8_0_BLOCK_BYTES;
    const int byte = index - col * Q8_0_BLOCK_BYTES;
    const int64_t out_col = out_tile * T16_COLS + col;
    raw_tile[index] = raw[
        out_col * bytes_per_row + block_index * Q8_0_BLOCK_BYTES + byte];
  }
  __syncthreads();

  uint8_t* dst = t16 + local_record * T16_BLOCK_BYTES;
  for (int index = static_cast<int>(threadIdx.x); index < T16_BLOCK_BYTES;
       index += static_cast<int>(blockDim.x)) {
    if (index < T16_D_BYTES) {
      const int col = index >> 1;
      const int byte = index & 1;
      dst[index] = raw_tile[col * Q8_0_BLOCK_BYTES + byte];
    } else {
      const int q_index = index - T16_D_BYTES;
      const int k = q_index / T16_COLS;
      const int col = q_index - k * T16_COLS;
      dst[index] = raw_tile[col * Q8_0_BLOCK_BYTES + 2 + k];
    }
  }
}

}  // namespace

extern "C" int hipengine_gguf_q8_0_raw_pair_to_t16(
    const uint8_t* raw_a,
    const uint8_t* raw_b,
    uint8_t* t16_a,
    uint8_t* t16_b,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t threads,
    hipStream_t stream) {
  if (raw_a == nullptr || raw_b == nullptr || t16_a == nullptr ||
      t16_b == nullptr || in_features <= 0 || out_features_a <= 0 ||
      out_features_b <= 0 || in_features % Q8_0_QK != 0 ||
      out_features_a % T16_COLS != 0 || out_features_b % T16_COLS != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = in_features / Q8_0_QK;
  const int64_t records =
      ((out_features_a + out_features_b) / T16_COLS) * blocks_per_row;
  hipLaunchKernelGGL(
      gguf_q8_0_raw_pair_to_t16_kernel,
      dim3(static_cast<unsigned int>(records)),
      dim3(static_cast<unsigned int>(threads)),
      T16_BLOCK_BYTES,
      stream,
      raw_a,
      raw_b,
      t16_a,
      t16_b,
      in_features,
      out_features_a,
      out_features_b);
  return static_cast<int>(hipGetLastError());
}
