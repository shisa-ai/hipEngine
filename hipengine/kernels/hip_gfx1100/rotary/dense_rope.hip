// Standard rotate-half RoPE over prefill rows for dense Llama-family decoders.
//
// Q is [rows, num_q_heads, head_dim] and K is [rows, num_kv_heads, head_dim],
// both BF16 and rotated in place. cos/sin are FP32 [rows, rotary_dim] tables
// built on the host for exactly the rows being rotated, so a caller can rotate a
// chunk at arbitrary absolute positions without materializing a table for the
// whole context window.
//
// One thread owns one (row, head, dim-pair) triple and writes both halves of the
// pair, which is what keeps an in-place rotation free of the read/write race a
// per-element mapping would have.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float dense_rope_bf16_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__device__ __forceinline__ uint16_t dense_rope_float_to_bf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int lsb = (bits >> 16) & 1u;
  const unsigned int rounded = bits + 0x7FFFu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
}

__device__ __forceinline__ void dense_rope_apply_pair(
    uint16_t* __restrict__ base,
    int64_t head_offset,
    int64_t half,
    float cosine,
    float sine) {
  const float first = dense_rope_bf16_to_float(base[head_offset]);
  const float second = dense_rope_bf16_to_float(base[head_offset + half]);
  base[head_offset] = dense_rope_float_to_bf16(first * cosine - second * sine);
  base[head_offset + half] = dense_rope_float_to_bf16(second * cosine + first * sine);
}

__global__ void dense_rope_positions_bf16_inplace_kernel(
    uint16_t* __restrict__ query,
    uint16_t* __restrict__ key,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t rows) {
  const int64_t half = rotary_dim >> 1;
  const int64_t q_pairs = rows * num_q_heads * half;
  const int64_t total_pairs = q_pairs + rows * num_kv_heads * half;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total_pairs;
       index += stride) {
    const bool is_query = index < q_pairs;
    const int64_t heads = is_query ? num_q_heads : num_kv_heads;
    const int64_t local = is_query ? index : index - q_pairs;
    const int64_t row = local / (heads * half);
    const int64_t within_row = local - row * heads * half;
    const int64_t head = within_row / half;
    const int64_t pair = within_row - head * half;

    uint16_t* base = is_query ? query : key;
    const int64_t row_offset = row * heads * head_dim;
    const int64_t head_offset = row_offset + head * head_dim + pair;
    const int64_t table_offset = row * rotary_dim + pair;

    dense_rope_apply_pair(
        base,
        head_offset,
        half,
        cos_table[table_offset],
        sin_table[table_offset]);
  }
}

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

}  // namespace

extern "C" int hipengine_dense_rope_positions_bf16_inplace(
    uint16_t* query,
    uint16_t* key,
    const float* cos_table,
    const float* sin_table,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t rows,
    int64_t threads,
    hipStream_t stream) {
  if (query == nullptr || key == nullptr || cos_table == nullptr || sin_table == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (!valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t pairs =
      rows * (num_q_heads + num_kv_heads) * (rotary_dim >> 1);
  const int64_t blocks = (pairs + threads - 1) / threads;
  const int64_t capped_blocks = blocks > 65535 ? 65535 : blocks;
  hipLaunchKernelGGL(
      dense_rope_positions_bf16_inplace_kernel,
      dim3(static_cast<unsigned int>(capped_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      query,
      key,
      cos_table,
      sin_table,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      rows);
  return static_cast<int>(hipGetLastError());
}
