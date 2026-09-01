// Selected MoE decode-shaped GEMV for GGUF K-family T16 replacement layouts.
//
// P9.H3 slice: consumes resident T16 tiles instead of raw row-major GGUF
// expert bytes.  Q4_K uses the dual gate/up selected ABI and can also use the
// single-output down-projection ABI for Q4_K_S; Q5_K/Q6_K use the same
// single-output selected down-projection ABI.  The compact-MoE scheduler ABI
// is the same as the P9.B raw pack8 kernels:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   tiles                [E, out_tiles16, blocks_per_row, tile_bytes]
//   out                  row-major compact output
//
// One block computes one 16-column output tile for one compact row.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>

namespace {

using half_t = _Float16;
typedef half_t half2_t __attribute__((ext_vector_type(2)));

constexpr int QK_K = 256;
constexpr int Q8_1_BLOCK = 32;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q6_GROUPS16 = 16;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_1_BLOCK];
};

static_assert(sizeof(gguf_q8_1_block) == 36, "GGUF q8_1 block layout must match GGML");

constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES = Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

constexpr int Q4_QMICRO_T16_D_OFFSET = 0;
constexpr int Q4_QMICRO_T16_DMIN_OFFSET = Q4_QMICRO_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_META_OFFSET = Q4_QMICRO_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_META_BYTES = 2 * Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_QMICRO_T16_Q_OFFSET = Q4_QMICRO_T16_META_OFFSET + 2 * Q4_SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q4_QMICRO_T16_BLOCK_BYTES = Q4_QMICRO_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

constexpr int Q4_T16_DUAL_D_OFFSET = 0;
constexpr int Q4_T16_DUAL_DMIN_OFFSET =
    Q4_T16_DUAL_D_OFFSET + (T16_COLS / 2) * 8;
constexpr int Q4_T16_DUAL_SCALE_OFFSET =
    Q4_T16_DUAL_DMIN_OFFSET + (T16_COLS / 2) * 8;
constexpr int Q4_T16_DUAL_MIN_OFFSET =
    Q4_T16_DUAL_SCALE_OFFSET + Q4_SUBBLOCKS * (T16_COLS / 2) * 4;
constexpr int Q4_T16_DUAL_Q_OFFSET =
    Q4_T16_DUAL_MIN_OFFSET + Q4_SUBBLOCKS * (T16_COLS / 2) * 4;
constexpr int Q4_T16_DUAL_BLOCK_BYTES =
    Q4_T16_DUAL_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * 2 * 8;

constexpr int Q5_T16_D_OFFSET = 0;
constexpr int Q5_T16_DMIN_OFFSET = Q5_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_SCALE_OFFSET = Q5_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_MIN_OFFSET = Q5_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QL_OFFSET = Q5_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QH_OFFSET = Q5_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_T16_BLOCK_BYTES = Q5_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);

constexpr int Q5_QMICRO_T16_D_OFFSET = 0;
constexpr int Q5_QMICRO_T16_DMIN_OFFSET = Q5_QMICRO_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_T16_META_OFFSET = Q5_QMICRO_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_T16_META_BYTES = 2 * Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_QMICRO_T16_QL_OFFSET = Q5_QMICRO_T16_META_OFFSET + 2 * Q4_SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q5_QMICRO_T16_QH_OFFSET = Q5_QMICRO_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_QMICRO_T16_BLOCK_BYTES = Q5_QMICRO_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);

constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);
constexpr int Q6_T16_QMICRO_OFFSET = Q6_T16_QL_OFFSET;
constexpr int Q6_T16_QMICRO_RECORD_BYTES = 12;

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

__device__ inline float fp16_bits_to_float(uint16_t bits) {
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

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float round_trip_scalar_to_float(float v) {
  return scalar_to_float<scalar_t>(float_to_scalar<scalar_t>(v));
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ inline int64_t find_expert_for_compact_row(
    const int64_t* __restrict__ expert_start_compact,
    int64_t compact_row,
    int64_t num_experts) {
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      return static_cast<int64_t>(e);
    }
  }
  return -1;
}

template <int tile_bytes>
__device__ inline const uint8_t* expert_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t expert_id,
    int64_t out_tile,
    int64_t block_idx,
    int64_t out_tiles,
    int64_t blocks_per_row) {
  const int64_t tile_index = (expert_id * out_tiles + out_tile) * blocks_per_row + block_idx;
  return tiles + tile_index * tile_bytes;
}

__device__ inline uint8_t t16_q4_quant(const uint8_t* __restrict__ tile, int sb, int lane32, int col) {
  const uint8_t packed = tile[Q4_T16_Q_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline uint32_t t16_q4_pack4(const uint8_t* __restrict__ tile, int sb, int lane4, int col) {
  const uint32_t q0 = static_cast<uint32_t>(t16_q4_quant(tile, sb, lane4 + 0, col));
  const uint32_t q1 = static_cast<uint32_t>(t16_q4_quant(tile, sb, lane4 + 1, col));
  const uint32_t q2 = static_cast<uint32_t>(t16_q4_quant(tile, sb, lane4 + 2, col));
  const uint32_t q3 = static_cast<uint32_t>(t16_q4_quant(tile, sb, lane4 + 3, col));
  return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

__device__ inline int t16_dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true, static_cast<int>(b), c, false);
#else
  const int8_t b0 = static_cast<int8_t>(b & 0xFFu);
  const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFFu);
  const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFFu);
  const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFFu);
  return c + static_cast<int>(a & 0xFFu) * static_cast<int>(b0) +
         static_cast<int>((a >> 8) & 0xFFu) * static_cast<int>(b1) +
         static_cast<int>((a >> 16) & 0xFFu) * static_cast<int>(b2) +
         static_cast<int>((a >> 24) & 0xFFu) * static_cast<int>(b3);
#endif
}

__device__ inline uint32_t qmicro_unpack_record(
    const uint8_t* __restrict__ record) {
  const uint32_t word = static_cast<uint32_t>(record[0]) |
                        (static_cast<uint32_t>(record[1]) << 8) |
                        (static_cast<uint32_t>(record[2]) << 16);
  return (word & 0x3Fu) |
         (((word >> 6) & 0x3Fu) << 8) |
         (((word >> 12) & 0x3Fu) << 16) |
         (((word >> 18) & 0x3Fu) << 24);
}

__device__ inline uint8_t q4_qmicro_t16_quant(
    const uint8_t* __restrict__ tile,
    int sb,
    int lane32,
    int col) {
  const uint8_t packed = tile[
      Q4_QMICRO_T16_Q_OFFSET +
      (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline uint32_t q4_qmicro_t16_pack4(
    const uint8_t* __restrict__ tile,
    int sb,
    int lane4,
    int col) {
  const uint32_t q0 = static_cast<uint32_t>(
      q4_qmicro_t16_quant(tile, sb, lane4 + 0, col));
  const uint32_t q1 = static_cast<uint32_t>(
      q4_qmicro_t16_quant(tile, sb, lane4 + 1, col));
  const uint32_t q2 = static_cast<uint32_t>(
      q4_qmicro_t16_quant(tile, sb, lane4 + 2, col));
  const uint32_t q3 = static_cast<uint32_t>(
      q4_qmicro_t16_quant(tile, sb, lane4 + 3, col));
  return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

__device__ inline float q4_qmicro_t16_weight(
    const uint8_t* __restrict__ tile,
    const uint8_t* __restrict__ metadata,
    int k_in_block,
    int col) {
  const int sb = k_in_block >> 5;
  const int lane32 = k_in_block & 31;
  const float d = fp16_bytes_to_float(tile + Q4_QMICRO_T16_D_OFFSET + col * 2);
  const float dmin = fp16_bytes_to_float(tile + Q4_QMICRO_T16_DMIN_OFFSET + col * 2);
  const float scale = static_cast<float>(metadata[sb * T16_COLS + col]);
  const float minv = static_cast<float>(metadata[Q4_SUBBLOCKS * T16_COLS + sb * T16_COLS + col]);
  const float q = static_cast<float>(q4_qmicro_t16_quant(tile, sb, lane32, col));
  return d * scale * q - dmin * minv;
}

__device__ inline uint8_t q5_qmicro_t16_quant(
    const uint8_t* __restrict__ tile,
    int sb,
    int lane32,
    int col) {
  const uint8_t low_packed = tile[
      Q5_QMICRO_T16_QL_OFFSET +
      (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[
      Q5_QMICRO_T16_QH_OFFSET +
      (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 8) + (col >> 3)];
  const uint8_t high = (high_packed >> (col & 7)) & 0x01u;
  return low | (high << 4);
}

__device__ inline uint8_t t16_q5_quant(const uint8_t* __restrict__ tile, int sb, int lane32, int col) {
  const uint8_t low_packed = tile[Q5_T16_QL_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q5_T16_QH_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 8) + (col >> 3)];
  const uint8_t high = (high_packed >> (col & 7)) & 0x01u;
  return low | (high << 4);
}

__device__ inline uint32_t t16_q5_pack4(const uint8_t* __restrict__ tile, int sb, int lane4, int col) {
  const uint32_t q0 = static_cast<uint32_t>(t16_q5_quant(tile, sb, lane4 + 0, col));
  const uint32_t q1 = static_cast<uint32_t>(t16_q5_quant(tile, sb, lane4 + 1, col));
  const uint32_t q2 = static_cast<uint32_t>(t16_q5_quant(tile, sb, lane4 + 2, col));
  const uint32_t q3 = static_cast<uint32_t>(t16_q5_quant(tile, sb, lane4 + 3, col));
  return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

int parse_selected_t16_dp4a_threads(const char* env, int fallback) {
  if (env == nullptr || env[0] == '\0') {
    return fallback;
  }
  const int value = atoi(env);
  return value == 32 || value == 64 || value == 128 ? value : fallback;
}

int selected_t16_dp4a_threads() {
  return parse_selected_t16_dp4a_threads(
      getenv("HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS"), 64);
}

int selected_t16_q5_dp4a_threads() {
  const char* q5_env = getenv("HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS");
  if (q5_env != nullptr && q5_env[0] != '\0') {
    return parse_selected_t16_dp4a_threads(q5_env, 64);
  }
  return selected_t16_dp4a_threads();
}

__device__ inline float t16_q4_weight(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const int sb = k_in_block >> 5;
  const int lane32 = k_in_block & 31;
  const float d = fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col * 2);
  const float dmin = fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col * 2);
  const float scale = static_cast<float>(tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
  const float minv = static_cast<float>(tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
  const float q = static_cast<float>(t16_q4_quant(tile, sb, lane32, col));
  return d * scale * q - dmin * minv;
}

__device__ inline float t16_q4_weight_from_quant(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col,
    uint8_t quant) {
  const int sb = k_in_block >> 5;
  const float d =
      fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col * 2);
  const float dmin =
      fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col * 2);
  const float scale = static_cast<float>(
      tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
  const float minv = static_cast<float>(
      tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
  return d * scale * static_cast<float>(quant) - dmin * minv;
}

__device__ inline uint8_t t16_q6_quant_unsigned(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const uint8_t low_packed = tile[Q6_T16_QL_OFFSET + k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q6_T16_QH_OFFSET + k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high = (high_packed >> (2 * (col & 3))) & 0x03u;
  return low | (high << 4);
}

__device__ inline int t16_q6_quant(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  return static_cast<int>(t16_q6_quant_unsigned(tile, k_in_block, col)) - 32;
}

__device__ inline int t16_q6_qmicro_quant(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const int subblock = k_in_block >> 5;
  const int pack = (k_in_block & 31) >> 2;
  const int k4 = k_in_block & 3;
  const int col_quartet = col >> 2;
  const int col4 = col & 3;
  const int record_index =
      (subblock * (T16_COLS / 4) + col_quartet) * 8 + pack;
  const uint8_t* __restrict__ record =
      tile + Q6_T16_QMICRO_OFFSET +
      record_index * Q6_T16_QMICRO_RECORD_BYTES;
  const uint8_t low_packed = record[k4 * 2 + (col4 >> 1)];
  const uint8_t low =
      (col4 & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = record[8 + k4];
  const uint8_t high =
      (high_packed >> (2 * col4)) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

__device__ inline int t16_q6_qmicro_planar_quant(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const int subblock = k_in_block >> 5;
  const int pack = (k_in_block & 31) >> 2;
  const int k4 = k_in_block & 3;
  const int col_quartet = col >> 2;
  const int col4 = col & 3;
  const int record_index =
      (subblock * (T16_COLS / 4) + col_quartet) * 8 + pack;
  const uint8_t* __restrict__ record =
      tile + Q6_T16_QMICRO_OFFSET +
      record_index * Q6_T16_QMICRO_RECORD_BYTES;
  const uint8_t low_packed =
      record[(col4 >> 1) * 4 + k4];
  const uint8_t low =
      (col4 & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = record[8 + k4];
  const uint8_t high =
      (high_packed >> (2 * col4)) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void q4_k_t16_selected_dual_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int64_t global_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t out_tiles_a = out_features_a / T16_COLS;
  const int64_t out_tiles_b = out_features_b / T16_COLS;
  const bool use_b = global_tile >= out_tiles_a;
  const int64_t local_tile = use_b ? (global_tile - out_tiles_a) : global_tile;
  if ((!use_b && local_tile >= out_tiles_a) || (use_b && local_tile >= out_tiles_b)) return;

  const int64_t expert_id = find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_stride = out_features_a + out_features_b;
  const int64_t out_base = use_b ? out_features_a : 0;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const int64_t out_tiles = use_b ? out_tiles_b : out_tiles_a;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  const scalar_t* __restrict__ x_row = x + compact_row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles, expert_id, local_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv = t16_q4_weight(tile, k_in_block, col);
        acc[col] += xv * wv;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);

  __shared__ float xchg[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[sub_id * T16_COLS + col] = acc[col];
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + compact_row * out_stride + out_base + local_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      if (local_tile * T16_COLS + col < local_out_features) {
        out_row[col] = float_to_scalar<scalar_t>(total);
      }
    }
  }
}

template <typename scalar_t, int qtype>
__global__ __launch_bounds__(128, 4) void qk_t16_selected_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  const int64_t out_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t expert_id = find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + compact_row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      if constexpr (qtype == 4) {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv = t16_q4_weight(tile, k_in_block, col);
          acc[col] += xv * wv;
        }
      } else if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc[col] += xv * (scale * q - minv);
        }
      } else {
        const int g16 = k_in_block >> 4;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          acc[col] += xv * (scale * q);
        }
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
    scalar_t* __restrict__ out_row = out + compact_row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}


template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void q4_k_t16_selected_dual_pairreuse_compact_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int64_t global_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t expert_id =
      find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;
  const int64_t group_start = expert_start_compact[expert_id];
  const int64_t group_end = expert_start_compact[expert_id + 1];
  if (((compact_row - group_start) & 1) != 0) return;
  const int64_t paired_row = compact_row + 1 < group_end ? compact_row + 1 : -1;

  const int64_t out_tiles_a = out_features_a / T16_COLS;
  const int64_t out_tiles_b = out_features_b / T16_COLS;
  const bool use_b = global_tile >= out_tiles_a;
  const int64_t local_tile = use_b ? global_tile - out_tiles_a : global_tile;
  if ((!use_b && local_tile >= out_tiles_a) || (use_b && local_tile >= out_tiles_b)) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_stride = out_features_a + out_features_b;
  const int64_t out_base = use_b ? out_features_a : 0;
  const int64_t out_tiles = use_b ? out_tiles_b : out_tiles_a;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  const scalar_t* __restrict__ x0 = x + compact_row * in_features;
  const scalar_t* __restrict__ x1 = paired_row >= 0 ? x + paired_row * in_features : nullptr;

  float acc0[T16_COLS];
  float acc1[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles, expert_id, local_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x0[k_base + k_in_block]);
      const float xv1 = paired_row >= 0
                            ? scalar_to_float<scalar_t>(x1[k_base + k_in_block])
                            : 0.0f;
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv = t16_q4_weight(tile, k_in_block, col);
        acc0[col] += xv0 * wv;
        if (paired_row >= 0) acc1[col] += xv1 * wv;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
  if (paired_row >= 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);
  }

  __shared__ float xchg0[4 * T16_COLS];
  __shared__ float xchg1[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      if (paired_row >= 0) xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out0 =
        out + compact_row * out_stride + out_base + local_tile * T16_COLS;
    scalar_t* __restrict__ out1 = paired_row >= 0
                                      ? out + paired_row * out_stride + out_base +
                                            local_tile * T16_COLS
                                      : nullptr;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        if (paired_row >= 0) total1 += xchg1[g * T16_COLS + col];
      }
      out0[col] = float_to_scalar<scalar_t>(total0);
      if (paired_row >= 0) out1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
}

template <typename scalar_t, int qtype>
__global__ __launch_bounds__(128, 2) void qk_t16_selected_pairreuse_compact_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  const int64_t out_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t expert_id =
      find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;
  const int64_t group_start = expert_start_compact[expert_id];
  const int64_t group_end = expert_start_compact[expert_id + 1];
  if (((compact_row - group_start) & 1) != 0) return;
  const int64_t paired_row = compact_row + 1 < group_end ? compact_row + 1 : -1;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x0 = x + compact_row * in_features;
  const scalar_t* __restrict__ x1 = paired_row >= 0 ? x + paired_row * in_features : nullptr;

  float acc0[T16_COLS];
  float acc1[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x0[k_base + k_in_block]);
      const float xv1 = paired_row >= 0
                            ? scalar_to_float<scalar_t>(x1[k_base + k_in_block])
                            : 0.0f;
      if constexpr (qtype == 4) {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv = t16_q4_weight(tile, k_in_block, col);
          acc0[col] += xv0 * wv;
          if (paired_row >= 0) acc1[col] += xv1 * wv;
        }
      } else if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale =
              d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv =
              dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          const float wv = scale * q - minv;
          acc0[col] += xv0 * wv;
          if (paired_row >= 0) acc1[col] += xv1 * wv;
        }
      } else {
        const int g16 = k_in_block >> 4;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales =
              reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          const float wv = scale * q;
          acc0[col] += xv0 * wv;
          if (paired_row >= 0) acc1[col] += xv1 * wv;
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
  if (paired_row >= 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);
  }

  __shared__ float xchg0[4 * T16_COLS];
  __shared__ float xchg1[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      if (paired_row >= 0) xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out0 =
        out + compact_row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out1 = paired_row >= 0
                                      ? out + paired_row * out_features + out_tile * T16_COLS
                                      : nullptr;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        if (paired_row >= 0) total1 += xchg1[g * T16_COLS + col];
      }
      out0[col] = float_to_scalar<scalar_t>(total0);
      if (paired_row >= 0) out1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
}

constexpr int GROUPED_SMALLM_ROWS = 4;
constexpr int GROUPED_SMALLM_COLS = 16;
constexpr int GROUPED_SMALLM_WAVES = 4;
constexpr int GROUPED_SMALLM_PERSISTENT_BLOCKS = 1048576;

// Exact small-M grouped Q4 gate/up. One block owns one expert and four output
// columns, then reuses each decoded weight across up to four compact rows. Each
// row retains the direct kernel's K order, wave32 shuffle tree, and serial
// wave-0..3 block reduction; only independent rows/columns are interleaved.
template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void q4_k_t16_selected_dual_grouped_smallm_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ active_experts,
    const int64_t* __restrict__ active_count_ptr,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_groups = out_features / GROUPED_SMALLM_COLS;
  const int64_t active_count = active_count_ptr[0];
  if (active_count <= 0 || active_count > num_experts) return;
  const int64_t total_tasks = out_groups * active_count;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  __shared__ float wave_sums_a[
      GROUPED_SMALLM_ROWS * GROUPED_SMALLM_WAVES * GROUPED_SMALLM_COLS];
  __shared__ float wave_sums_b[
      GROUPED_SMALLM_ROWS * GROUPED_SMALLM_WAVES * GROUPED_SMALLM_COLS];

  for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
    const int64_t active_idx = task / out_groups;
    const int64_t out_group = task - active_idx * out_groups;
    const int64_t expert_id = active_experts[active_idx];
    if (expert_id < 0 || expert_id >= num_experts) continue;
    const int64_t begin = expert_start_compact[expert_id];
    const int64_t end = expert_start_compact[expert_id + 1];
    if (begin >= end || begin < 0 || end > compact_rows) continue;
    const int64_t col_base = out_group * GROUPED_SMALLM_COLS;
    const int64_t out_tile = col_base / T16_COLS;
    const int col_in_tile = static_cast<int>(col_base % T16_COLS);

    for (int64_t row_base = begin; row_base < end; row_base += GROUPED_SMALLM_ROWS) {
    float acc_a[GROUPED_SMALLM_ROWS][GROUPED_SMALLM_COLS] = {};
    float acc_b[GROUPED_SMALLM_ROWS][GROUPED_SMALLM_COLS] = {};

    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      const int64_t k_base = blk * QK_K;
      for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
        float xv[GROUPED_SMALLM_ROWS] = {};
#pragma unroll
        for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
          const int64_t row = row_base + row_idx;
          if (row < end) {
            xv[row_idx] = scalar_to_float<scalar_t>(
                x[row * in_features + k_base + k_in_block]);
          }
        }
#pragma unroll
        for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
          const int col = col_in_tile + local_col;
          const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
          const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
#pragma unroll
          for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
            if (row_base + row_idx < end) {
              acc_a[row_idx][local_col] += xv[row_idx] * wv_a;
              acc_b[row_idx][local_col] += xv[row_idx] * wv_b;
            }
          }
        }
      }
    }

#pragma unroll
    for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
#pragma unroll
      for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
        GGUF_T16_WAVE_REDUCE(acc_a[row_idx][local_col]);
        GGUF_T16_WAVE_REDUCE(acc_b[row_idx][local_col]);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
#pragma unroll
        for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
          const int idx =
              (row_idx * GROUPED_SMALLM_WAVES + wave) * GROUPED_SMALLM_COLS +
              local_col;
          wave_sums_a[idx] = acc_a[row_idx][local_col];
          wave_sums_b[idx] = acc_b[row_idx][local_col];
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
#pragma unroll
          for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
            float total_a = 0.0f;
            float total_b = 0.0f;
#pragma unroll
            for (int wave_idx = 0; wave_idx < GROUPED_SMALLM_WAVES; ++wave_idx) {
              const int idx =
                  (row_idx * GROUPED_SMALLM_WAVES + wave_idx) *
                      GROUPED_SMALLM_COLS +
                  local_col;
              total_a += wave_sums_a[idx];
              total_b += wave_sums_b[idx];
            }
            const int64_t out_idx = row * out_features + col_base + local_col;
            out_a[out_idx] = float_to_scalar<scalar_t>(total_a);
            out_b[out_idx] = float_to_scalar<scalar_t>(total_b);
          }
        }
      }
    }
      __syncthreads();
    }
  }
}

// Exact grouped Q4/Q6 single projection used by Laguna's down path. Four-row
// buckets share each decoded 16-column T16 weight tile while preserving the
// direct kernel's per-row accumulation/reduction association.
template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false>
__global__ __launch_bounds__(128, 2) void qk_t16_selected_grouped_smallm_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ active_experts,
    const int64_t* __restrict__ active_count_ptr,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 6, "grouped small-M supports Q4 or Q6");
  static_assert(!qmicro_planar || qmicro,
                "planar qmicro requires qmicro layout");
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_groups = out_features / GROUPED_SMALLM_COLS;
  const int64_t active_count = active_count_ptr[0];
  if (active_count <= 0 || active_count > num_experts) return;
  const int64_t total_tasks = out_groups * active_count;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  __shared__ float wave_sums[
      GROUPED_SMALLM_ROWS * GROUPED_SMALLM_WAVES * GROUPED_SMALLM_COLS];

  for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
    const int64_t active_idx = task / out_groups;
    const int64_t out_group = task - active_idx * out_groups;
    const int64_t expert_id = active_experts[active_idx];
    if (expert_id < 0 || expert_id >= num_experts) continue;
    const int64_t begin = expert_start_compact[expert_id];
    const int64_t end = expert_start_compact[expert_id + 1];
    if (begin >= end || begin < 0 || end > compact_rows) continue;
    const int64_t col_base = out_group * GROUPED_SMALLM_COLS;
    const int64_t out_tile = col_base / T16_COLS;
    const int col_in_tile = static_cast<int>(col_base % T16_COLS);

    for (int64_t row_base = begin; row_base < end; row_base += GROUPED_SMALLM_ROWS) {
    float acc[GROUPED_SMALLM_ROWS][GROUPED_SMALLM_COLS] = {};
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ tile;
      if constexpr (qtype == 4) {
        tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      } else {
        tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(
            tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      }
      const int64_t k_base = blk * QK_K;
      for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
        float xv[GROUPED_SMALLM_ROWS] = {};
#pragma unroll
        for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
          const int64_t row = row_base + row_idx;
          if (row < end) {
            xv[row_idx] = scalar_to_float<scalar_t>(
                x[row * in_features + k_base + k_in_block]);
          }
        }
#pragma unroll
        for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
          const int col = col_in_tile + local_col;
          float wv = 0.0f;
          if constexpr (qtype == 4) {
            wv = t16_q4_weight(tile, k_in_block, col);
          } else {
            const int g16 = k_in_block >> 4;
            const float d =
                fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
            const int8_t* __restrict__ scales =
                reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
            const float scale =
                d * static_cast<float>(scales[g16 * T16_COLS + col]);
            int q;
            if constexpr (qmicro_planar) {
              q = t16_q6_qmicro_planar_quant(
                  tile, k_in_block, col);
            } else if constexpr (qmicro) {
              q = t16_q6_qmicro_quant(tile, k_in_block, col);
            } else {
              q = t16_q6_quant(tile, k_in_block, col);
            }
            wv = scale * q;
          }
#pragma unroll
          for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
            if (row_base + row_idx < end) {
              acc[row_idx][local_col] += xv[row_idx] * wv;
            }
          }
        }
      }
    }

#pragma unroll
    for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
#pragma unroll
      for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
        GGUF_T16_WAVE_REDUCE(acc[row_idx][local_col]);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
#pragma unroll
        for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
          const int idx =
              (row_idx * GROUPED_SMALLM_WAVES + wave) * GROUPED_SMALLM_COLS +
              local_col;
          wave_sums[idx] = acc[row_idx][local_col];
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < GROUPED_SMALLM_ROWS; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
#pragma unroll
          for (int local_col = 0; local_col < GROUPED_SMALLM_COLS; ++local_col) {
            float total = 0.0f;
#pragma unroll
            for (int wave_idx = 0; wave_idx < GROUPED_SMALLM_WAVES; ++wave_idx) {
              const int idx =
                  (row_idx * GROUPED_SMALLM_WAVES + wave_idx) *
                      GROUPED_SMALLM_COLS +
                  local_col;
              total += wave_sums[idx];
            }
            out[row * out_features + col_base + local_col] =
                float_to_scalar<scalar_t>(total);
          }
        }
      }
    }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void q4_k_t16_selected_dual_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
        const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
        acc_a[col] += xv * wv_a;
        acc_b[col] += xv * wv_b;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row_a = out_a + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row_b = out_b + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      out_row_a[col] = float_to_scalar<scalar_t>(total_a);
      out_row_b[col] = float_to_scalar<scalar_t>(total_b);
    }
  }
}

// Exact Laguna c=1 specialization: one input row expands to ten selected
// expert lanes, K3072 -> N1024 for both gate and up. The workgroup grid,
// per-thread K ownership, FMA order, wave tree, wave-0..3 block reduction, and
// BF16 stores match q4_k_t16_selected_dual_direct_gemv_kernel.
template <typename scalar_t>
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_natural_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t num_experts) {
  constexpr int kRows = 10;
  constexpr int kInFeatures = 3072;
  constexpr int kOutFeatures = 1024;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= kRows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile_a =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_a, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const uint8_t* __restrict__ tile_b =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_b, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const float xv =
          scalar_to_float<scalar_t>(x[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv_a =
            t16_q4_weight(tile_a, k_in_block, col);
        const float wv_b =
            t16_q4_weight(tile_b, k_in_block, col);
        acc_a[col] += xv * wv_a;
        acc_b[col] += xv * wv_b;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc_a[col]);
  }
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc_b[col]);
  }

  __shared__ float xchg_a[4 * T16_COLS];
  __shared__ float xchg_b[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row_a =
        out_a + row * kOutFeatures + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row_b =
        out_b + row * kOutFeatures + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      out_row_a[col] = float_to_scalar<scalar_t>(total_a);
      out_row_b[col] = float_to_scalar<scalar_t>(total_b);
    }
  }
}

// Exact Laguna c=1 tile8 screen: split each resident 16-column T16 tile
// across two workgroups. Each output column retains the natural kernel's
// thread K ownership, FMA order, wave tree, wave-0..3 reduction, and BF16
// store while halving the live gate/up accumulator set per workgroup.
template <
    typename scalar_t,
    bool parallel_tail = false,
    bool fuse_silu = false,
    bool pair_q_payload = false,
    bool pair_coeff_payload = false>
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_natural_tile8_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t num_experts) {
  constexpr int kRows = 10;
  constexpr int kInFeatures = 3072;
  constexpr int kOutFeatures = 1024;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= kRows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile_a =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_a, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const uint8_t* __restrict__ tile_b =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_b, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const float xv =
          scalar_to_float<scalar_t>(x[k_base + k_in_block]);
      if constexpr (pair_q_payload) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int local_pair = 0;
             local_pair < kTileCols / 2;
             ++local_pair) {
          const int local_col0 = local_pair * 2;
          const int local_col1 = local_col0 + 1;
          const int col0 = col_base + local_col0;
          const int col1 = col0 + 1;
          const int q_index =
              Q4_T16_Q_OFFSET +
              (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
              (col0 >> 1);
          const uint8_t packed_a = tile_a[q_index];
          const uint8_t packed_b = tile_b[q_index];
          float wv_a0;
          float wv_a1;
          float wv_b0;
          float wv_b1;
          if constexpr (pair_coeff_payload) {
            const uint32_t d_a = *reinterpret_cast<const uint32_t*>(
                tile_a + Q4_T16_D_OFFSET + col0 * 2);
            const uint32_t dmin_a = *reinterpret_cast<const uint32_t*>(
                tile_a + Q4_T16_DMIN_OFFSET + col0 * 2);
            const uint16_t scale_a = *reinterpret_cast<const uint16_t*>(
                tile_a + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
            const uint16_t min_a = *reinterpret_cast<const uint16_t*>(
                tile_a + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
            const uint32_t d_b = *reinterpret_cast<const uint32_t*>(
                tile_b + Q4_T16_D_OFFSET + col0 * 2);
            const uint32_t dmin_b = *reinterpret_cast<const uint32_t*>(
                tile_b + Q4_T16_DMIN_OFFSET + col0 * 2);
            const uint16_t scale_b = *reinterpret_cast<const uint16_t*>(
                tile_b + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
            const uint16_t min_b = *reinterpret_cast<const uint16_t*>(
                tile_b + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
            const float d_a0 = fp16_bits_to_float(
                static_cast<uint16_t>(d_a));
            const float d_a1 = fp16_bits_to_float(
                static_cast<uint16_t>(d_a >> 16));
            const float dmin_a0 = fp16_bits_to_float(
                static_cast<uint16_t>(dmin_a));
            const float dmin_a1 = fp16_bits_to_float(
                static_cast<uint16_t>(dmin_a >> 16));
            const float d_b0 = fp16_bits_to_float(
                static_cast<uint16_t>(d_b));
            const float d_b1 = fp16_bits_to_float(
                static_cast<uint16_t>(d_b >> 16));
            const float dmin_b0 = fp16_bits_to_float(
                static_cast<uint16_t>(dmin_b));
            const float dmin_b1 = fp16_bits_to_float(
                static_cast<uint16_t>(dmin_b >> 16));
            wv_a0 =
                d_a0 * static_cast<float>(scale_a & 0xFFu) *
                    static_cast<float>(packed_a & 0x0Fu) -
                dmin_a0 * static_cast<float>(min_a & 0xFFu);
            wv_a1 =
                d_a1 * static_cast<float>(scale_a >> 8) *
                    static_cast<float>(packed_a >> 4) -
                dmin_a1 * static_cast<float>(min_a >> 8);
            wv_b0 =
                d_b0 * static_cast<float>(scale_b & 0xFFu) *
                    static_cast<float>(packed_b & 0x0Fu) -
                dmin_b0 * static_cast<float>(min_b & 0xFFu);
            wv_b1 =
                d_b1 * static_cast<float>(scale_b >> 8) *
                    static_cast<float>(packed_b >> 4) -
                dmin_b1 * static_cast<float>(min_b >> 8);
          } else {
            wv_a0 = t16_q4_weight_from_quant(
                tile_a, k_in_block, col0, packed_a & 0x0Fu);
            wv_a1 = t16_q4_weight_from_quant(
                tile_a, k_in_block, col1, packed_a >> 4);
            wv_b0 = t16_q4_weight_from_quant(
                tile_b, k_in_block, col0, packed_b & 0x0Fu);
            wv_b1 = t16_q4_weight_from_quant(
                tile_b, k_in_block, col1, packed_b >> 4);
          }
          acc_a[local_col0] += xv * wv_a0;
          acc_a[local_col1] += xv * wv_a1;
          acc_b[local_col0] += xv * wv_b0;
          acc_b[local_col1] += xv * wv_b1;
        }
      } else {
#pragma unroll
        for (int local_col = 0; local_col < kTileCols; ++local_col) {
          const int col = col_base + local_col;
          const float wv_a =
              t16_q4_weight(tile_a, k_in_block, col);
          const float wv_b =
              t16_q4_weight(tile_b, k_in_block, col);
          acc_a[local_col] += xv * wv_a;
          acc_b[local_col] += xv * wv_b;
        }
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }

  __shared__ float xchg_a[4 * kTileCols];
  __shared__ float xchg_b[4 * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg_a[sub_id * kTileCols + local_col] = acc_a[local_col];
      xchg_b[sub_id * kTileCols + local_col] = acc_b[local_col];
    }
  }
  __syncthreads();

  if constexpr (parallel_tail) {
    if (tid < kTileCols) {
      const int local_col = tid;
      scalar_t* __restrict__ out_row_a =
          out_a + row * kOutFeatures + out_group * kTileCols;
      scalar_t* __restrict__ out_row_b = fuse_silu
          ? nullptr
          : out_b + row * kOutFeatures + out_group * kTileCols;
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total_a += xchg_a[g * kTileCols + local_col];
        total_b += xchg_b[g * kTileCols + local_col];
      }
      if constexpr (fuse_silu) {
        // Match the separate dual-GEMV -> BF16 store/load -> SiLU chain:
        // round gate and up independently before evaluating the same
        // g * sigmoid(g) * u expression and materializing BF16.
        const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
        const float up = round_trip_scalar_to_float<scalar_t>(total_b);
        const float sig = 1.0f / (1.0f + expf(-gate));
        out_row_a[local_col] =
            float_to_scalar<scalar_t>(gate * sig * up);
      } else {
        out_row_a[local_col] = float_to_scalar<scalar_t>(total_a);
        out_row_b[local_col] = float_to_scalar<scalar_t>(total_b);
      }
    }
  } else if (tid == 0) {
    scalar_t* __restrict__ out_row_a =
        out_a + row * kOutFeatures + out_group * kTileCols;
    scalar_t* __restrict__ out_row_b =
        out_b + row * kOutFeatures + out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total_a += xchg_a[g * kTileCols + local_col];
        total_b += xchg_b[g * kTileCols + local_col];
      }
      out_row_a[local_col] = float_to_scalar<scalar_t>(total_a);
      out_row_b[local_col] = float_to_scalar<scalar_t>(total_b);
    }
  }
}

// Quality-gated c=1 screen: keep the retained tile8/route ownership, but let
// each thread own one adjacent K pair and accumulate it with v_dot2_f32_f16.
// BF16 activations and dequantized Q4 values are converted to FP16 in
// registers; the output remains the production BF16-boundary SiLU result.
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_natural_tile8_halfdot_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    uint16_t* __restrict__ out,
    int64_t num_experts) {
  constexpr int kRows = 10;
  constexpr int kInFeatures = 3072;
  constexpr int kOutFeatures = 1024;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= kRows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  const int k0 = tid * 2;
  const int k1 = k0 + 1;
  const int sb = k0 >> 5;
  const int lane0 = k0 & 31;
  const int lane1 = lane0 + 1;
  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile_a =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_a, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const uint8_t* __restrict__ tile_b =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_b, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const int64_t k_base = blk * QK_K;
    const half2_t xv = {
        static_cast<half_t>(bf16_bits_to_float(x[k_base + k0])),
        static_cast<half_t>(bf16_bits_to_float(x[k_base + k1])),
    };
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const int col1 = col0 + 1;
      const int q_index0 =
          Q4_T16_Q_OFFSET +
          (sb * Q4_SUBBLOCK + lane0) * (T16_COLS / 2) +
          (col0 >> 1);
      const int q_index1 =
          Q4_T16_Q_OFFSET +
          (sb * Q4_SUBBLOCK + lane1) * (T16_COLS / 2) +
          (col0 >> 1);
      const uint8_t qa0 = tile_a[q_index0];
      const uint8_t qa1 = tile_a[q_index1];
      const uint8_t qb0 = tile_b[q_index0];
      const uint8_t qb1 = tile_b[q_index1];
      const uint32_t d_a = *reinterpret_cast<const uint32_t*>(
          tile_a + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin_a = *reinterpret_cast<const uint32_t*>(
          tile_a + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t scale_a = *reinterpret_cast<const uint16_t*>(
          tile_a + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t min_a = *reinterpret_cast<const uint16_t*>(
          tile_a + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      const uint32_t d_b = *reinterpret_cast<const uint32_t*>(
          tile_b + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin_b = *reinterpret_cast<const uint32_t*>(
          tile_b + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t scale_b = *reinterpret_cast<const uint16_t*>(
          tile_b + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t min_b = *reinterpret_cast<const uint16_t*>(
          tile_b + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);

      const float scale_a0 =
          fp16_bits_to_float(static_cast<uint16_t>(d_a)) *
          static_cast<float>(scale_a & 0xFFu);
      const float scale_a1 =
          fp16_bits_to_float(static_cast<uint16_t>(d_a >> 16)) *
          static_cast<float>(scale_a >> 8);
      const float min_a0 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a)) *
          static_cast<float>(min_a & 0xFFu);
      const float min_a1 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a >> 16)) *
          static_cast<float>(min_a >> 8);
      const float scale_b0 =
          fp16_bits_to_float(static_cast<uint16_t>(d_b)) *
          static_cast<float>(scale_b & 0xFFu);
      const float scale_b1 =
          fp16_bits_to_float(static_cast<uint16_t>(d_b >> 16)) *
          static_cast<float>(scale_b >> 8);
      const float min_b0 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b)) *
          static_cast<float>(min_b & 0xFFu);
      const float min_b1 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b >> 16)) *
          static_cast<float>(min_b >> 8);

      const half2_t wa0 = {
          static_cast<half_t>(
              static_cast<float>(qa0 & 0x0Fu) * scale_a0 - min_a0),
          static_cast<half_t>(
              static_cast<float>(qa1 & 0x0Fu) * scale_a0 - min_a0),
      };
      const half2_t wa1 = {
          static_cast<half_t>(
              static_cast<float>(qa0 >> 4) * scale_a1 - min_a1),
          static_cast<half_t>(
              static_cast<float>(qa1 >> 4) * scale_a1 - min_a1),
      };
      const half2_t wb0 = {
          static_cast<half_t>(
              static_cast<float>(qb0 & 0x0Fu) * scale_b0 - min_b0),
          static_cast<half_t>(
              static_cast<float>(qb1 & 0x0Fu) * scale_b0 - min_b0),
      };
      const half2_t wb1 = {
          static_cast<half_t>(
              static_cast<float>(qb0 >> 4) * scale_b1 - min_b1),
          static_cast<half_t>(
              static_cast<float>(qb1 >> 4) * scale_b1 - min_b1),
      };
      acc_a[local_col0] =
          __builtin_amdgcn_fdot2(xv, wa0, acc_a[local_col0], false);
      acc_a[local_col1] =
          __builtin_amdgcn_fdot2(xv, wa1, acc_a[local_col1], false);
      acc_b[local_col0] =
          __builtin_amdgcn_fdot2(xv, wb0, acc_b[local_col0], false);
      acc_b[local_col1] =
          __builtin_amdgcn_fdot2(xv, wb1, acc_b[local_col1], false);
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }

  __shared__ float xchg_a[4 * kTileCols];
  __shared__ float xchg_b[4 * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg_a[sub_id * kTileCols + local_col] = acc_a[local_col];
      xchg_b[sub_id * kTileCols + local_col] = acc_b[local_col];
    }
  }
  __syncthreads();

  if (tid < kTileCols) {
    const int local_col = tid;
    float total_a = 0.0f;
    float total_b = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      total_a += xchg_a[g * kTileCols + local_col];
      total_b += xchg_b[g * kTileCols + local_col];
    }
    const float gate = bf16_bits_to_float(float_to_bf16_bits(total_a));
    const float up = bf16_bits_to_float(float_to_bf16_bits(total_b));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[row * kOutFeatures + out_group * kTileCols + local_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

// Interleaved-resident sibling of the quality-gated FP16 dot2 screen above.
// The byte-neutral dual layout keeps corresponding gate/up metadata and
// nibbles adjacent; arithmetic and adjacent-K ownership are otherwise
// identical, so this is the candidate that can replace the production c=1
// interleaved consumer without changing resident bytes.
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_interleaved_natural_tile8_halfdot_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_dual,
    uint16_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts) {
  constexpr int kTopK = 10;
  constexpr int kInFeatures = 3072;
  constexpr int kOutFeatures = 1024;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t source_row = row / kTopK;
  if (source_row >= x_rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t out_tile = out_group >> 1;
  const int half = static_cast<int>(out_group & 1);
  const int col_base = half * kTileCols;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  const int k0 = tid * 2;
  const int k1 = k0 + 1;
  const int sb = k0 >> 5;
  const int lane0 = k0 & 31;
  const int lane1 = lane0 + 1;
  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile =
        expert_tile_ptr<Q4_T16_DUAL_BLOCK_BYTES>(
            tiles_dual, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const int64_t x_base =
        source_row * kInFeatures + blk * QK_K;
    const half2_t xv = {
        static_cast<half_t>(bf16_bits_to_float(x[x_base + k0])),
        static_cast<half_t>(bf16_bits_to_float(x[x_base + k1])),
    };
    const uint64_t q_dual0 = *reinterpret_cast<const uint64_t*>(
        tile + Q4_T16_DUAL_Q_OFFSET +
        ((sb * Q4_SUBBLOCK + lane0) * 2 + half) * 8);
    const uint64_t q_dual1 = *reinterpret_cast<const uint64_t*>(
        tile + Q4_T16_DUAL_Q_OFFSET +
        ((sb * Q4_SUBBLOCK + lane1) * 2 + half) * 8);
    const uint32_t q_a0 = static_cast<uint32_t>(q_dual0);
    const uint32_t q_b0 = static_cast<uint32_t>(q_dual0 >> 32);
    const uint32_t q_a1 = static_cast<uint32_t>(q_dual1);
    const uint32_t q_b1 = static_cast<uint32_t>(q_dual1 >> 32);
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const int col_pair = col0 >> 1;
      const uint8_t qa0 =
          static_cast<uint8_t>(q_a0 >> (local_pair * 8));
      const uint8_t qa1 =
          static_cast<uint8_t>(q_a1 >> (local_pair * 8));
      const uint8_t qb0 =
          static_cast<uint8_t>(q_b0 >> (local_pair * 8));
      const uint8_t qb1 =
          static_cast<uint8_t>(q_b1 >> (local_pair * 8));
      const uint64_t d_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_D_OFFSET + col_pair * 8);
      const uint64_t dmin_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_DMIN_OFFSET + col_pair * 8);
      const uint32_t scale_dual = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DUAL_SCALE_OFFSET +
          (sb * (T16_COLS / 2) + col_pair) * 4);
      const uint32_t min_dual = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DUAL_MIN_OFFSET +
          (sb * (T16_COLS / 2) + col_pair) * 4);
      const uint32_t d_a = static_cast<uint32_t>(d_dual);
      const uint32_t d_b = static_cast<uint32_t>(d_dual >> 32);
      const uint32_t dmin_a = static_cast<uint32_t>(dmin_dual);
      const uint32_t dmin_b = static_cast<uint32_t>(dmin_dual >> 32);
      const uint16_t scale_a = static_cast<uint16_t>(scale_dual);
      const uint16_t scale_b = static_cast<uint16_t>(scale_dual >> 16);
      const uint16_t min_a = static_cast<uint16_t>(min_dual);
      const uint16_t min_b = static_cast<uint16_t>(min_dual >> 16);

      const float scale_a0 =
          fp16_bits_to_float(static_cast<uint16_t>(d_a)) *
          static_cast<float>(scale_a & 0xFFu);
      const float scale_a1 =
          fp16_bits_to_float(static_cast<uint16_t>(d_a >> 16)) *
          static_cast<float>(scale_a >> 8);
      const float min_a0 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a)) *
          static_cast<float>(min_a & 0xFFu);
      const float min_a1 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a >> 16)) *
          static_cast<float>(min_a >> 8);
      const float scale_b0 =
          fp16_bits_to_float(static_cast<uint16_t>(d_b)) *
          static_cast<float>(scale_b & 0xFFu);
      const float scale_b1 =
          fp16_bits_to_float(static_cast<uint16_t>(d_b >> 16)) *
          static_cast<float>(scale_b >> 8);
      const float min_b0 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b)) *
          static_cast<float>(min_b & 0xFFu);
      const float min_b1 =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b >> 16)) *
          static_cast<float>(min_b >> 8);

      const half2_t wa0 = {
          static_cast<half_t>(
              static_cast<float>(qa0 & 0x0Fu) * scale_a0 - min_a0),
          static_cast<half_t>(
              static_cast<float>(qa1 & 0x0Fu) * scale_a0 - min_a0),
      };
      const half2_t wa1 = {
          static_cast<half_t>(
              static_cast<float>(qa0 >> 4) * scale_a1 - min_a1),
          static_cast<half_t>(
              static_cast<float>(qa1 >> 4) * scale_a1 - min_a1),
      };
      const half2_t wb0 = {
          static_cast<half_t>(
              static_cast<float>(qb0 & 0x0Fu) * scale_b0 - min_b0),
          static_cast<half_t>(
              static_cast<float>(qb1 & 0x0Fu) * scale_b0 - min_b0),
      };
      const half2_t wb1 = {
          static_cast<half_t>(
              static_cast<float>(qb0 >> 4) * scale_b1 - min_b1),
          static_cast<half_t>(
              static_cast<float>(qb1 >> 4) * scale_b1 - min_b1),
      };
      acc_a[local_col0] =
          __builtin_amdgcn_fdot2(xv, wa0, acc_a[local_col0], false);
      acc_a[local_col1] =
          __builtin_amdgcn_fdot2(xv, wa1, acc_a[local_col1], false);
      acc_b[local_col0] =
          __builtin_amdgcn_fdot2(xv, wb0, acc_b[local_col0], false);
      acc_b[local_col1] =
          __builtin_amdgcn_fdot2(xv, wb1, acc_b[local_col1], false);
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }

  __shared__ float xchg_a[4 * kTileCols];
  __shared__ float xchg_b[4 * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg_a[sub_id * kTileCols + local_col] = acc_a[local_col];
      xchg_b[sub_id * kTileCols + local_col] = acc_b[local_col];
    }
  }
  __syncthreads();

  if (tid < kTileCols) {
    const int local_col = tid;
    float total_a = 0.0f;
    float total_b = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      total_a += xchg_a[g * kTileCols + local_col];
      total_b += xchg_b[g * kTileCols + local_col];
    }
    const float gate = bf16_bits_to_float(float_to_bf16_bits(total_a));
    const float up = bf16_bits_to_float(float_to_bf16_bits(total_b));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[row * kOutFeatures + out_group * kTileCols + local_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

// Exact dual-interleaved T16 screen. Gate/up occupy the same total bytes as
// two ordinary T16 matrices, but corresponding Q/metadata vectors are
// adjacent so one vector transaction can feed both accumulator sets.
template <typename scalar_t>
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_interleaved_natural_tile8_silu_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_dual,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts) {
  constexpr int kTopK = 10;
  constexpr int kInFeatures = 3072;
  constexpr int kOutFeatures = 1024;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t source_row = row / kTopK;
  if (source_row >= x_rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t out_tile = out_group >> 1;
  const int half = static_cast<int>(out_group & 1);
  const int col_base = half * kTileCols;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile =
        expert_tile_ptr<Q4_T16_DUAL_BLOCK_BYTES>(
            tiles_dual, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const float xv =
          scalar_to_float<scalar_t>(
              x[source_row * kInFeatures + k_base + k_in_block]);
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const uint64_t q_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_Q_OFFSET +
          ((sb * Q4_SUBBLOCK + lane32) * 2 + half) * 8);
      const uint32_t q_a = static_cast<uint32_t>(q_dual);
      const uint32_t q_b = static_cast<uint32_t>(q_dual >> 32);
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int col_pair = col0 >> 1;
        const uint8_t packed_a =
            static_cast<uint8_t>(q_a >> (local_pair * 8));
        const uint8_t packed_b =
            static_cast<uint8_t>(q_b >> (local_pair * 8));
        const uint64_t d_dual = *reinterpret_cast<const uint64_t*>(
            tile + Q4_T16_DUAL_D_OFFSET + col_pair * 8);
        const uint64_t dmin_dual = *reinterpret_cast<const uint64_t*>(
            tile + Q4_T16_DUAL_DMIN_OFFSET + col_pair * 8);
        const uint32_t scale_dual = *reinterpret_cast<const uint32_t*>(
            tile + Q4_T16_DUAL_SCALE_OFFSET +
            (sb * (T16_COLS / 2) + col_pair) * 4);
        const uint32_t min_dual = *reinterpret_cast<const uint32_t*>(
            tile + Q4_T16_DUAL_MIN_OFFSET +
            (sb * (T16_COLS / 2) + col_pair) * 4);
        const uint32_t d_a = static_cast<uint32_t>(d_dual);
        const uint32_t d_b = static_cast<uint32_t>(d_dual >> 32);
        const uint32_t dmin_a = static_cast<uint32_t>(dmin_dual);
        const uint32_t dmin_b = static_cast<uint32_t>(dmin_dual >> 32);
        const uint16_t scale_a = static_cast<uint16_t>(scale_dual);
        const uint16_t scale_b = static_cast<uint16_t>(scale_dual >> 16);
        const uint16_t min_a = static_cast<uint16_t>(min_dual);
        const uint16_t min_b = static_cast<uint16_t>(min_dual >> 16);
        const float d_a0 =
            fp16_bits_to_float(static_cast<uint16_t>(d_a));
        const float d_a1 =
            fp16_bits_to_float(static_cast<uint16_t>(d_a >> 16));
        const float dmin_a0 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_a));
        const float dmin_a1 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_a >> 16));
        const float d_b0 =
            fp16_bits_to_float(static_cast<uint16_t>(d_b));
        const float d_b1 =
            fp16_bits_to_float(static_cast<uint16_t>(d_b >> 16));
        const float dmin_b0 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_b));
        const float dmin_b1 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_b >> 16));
        const float wv_a0 =
            d_a0 * static_cast<float>(scale_a & 0xFFu) *
                static_cast<float>(packed_a & 0x0Fu) -
            dmin_a0 * static_cast<float>(min_a & 0xFFu);
        const float wv_a1 =
            d_a1 * static_cast<float>(scale_a >> 8) *
                static_cast<float>(packed_a >> 4) -
            dmin_a1 * static_cast<float>(min_a >> 8);
        const float wv_b0 =
            d_b0 * static_cast<float>(scale_b & 0xFFu) *
                static_cast<float>(packed_b & 0x0Fu) -
            dmin_b0 * static_cast<float>(min_b & 0xFFu);
        const float wv_b1 =
            d_b1 * static_cast<float>(scale_b >> 8) *
                static_cast<float>(packed_b >> 4) -
            dmin_b1 * static_cast<float>(min_b >> 8);
        acc_a[local_col0] += xv * wv_a0;
        acc_a[local_col1] += xv * wv_a1;
        acc_b[local_col0] += xv * wv_b0;
        acc_b[local_col1] += xv * wv_b1;
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }

  __shared__ float xchg_a[4 * kTileCols];
  __shared__ float xchg_b[4 * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg_a[sub_id * kTileCols + local_col] = acc_a[local_col];
      xchg_b[sub_id * kTileCols + local_col] = acc_b[local_col];
    }
  }
  __syncthreads();

  if (tid < kTileCols) {
    const int local_col = tid;
    scalar_t* __restrict__ out_row =
        out + row * kOutFeatures + out_group * kTileCols;
    float total_a = 0.0f;
    float total_b = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      total_a += xchg_a[g * kTileCols + local_col];
      total_b += xchg_b[g * kTileCols + local_col];
    }
    const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
    const float up = round_trip_scalar_to_float<scalar_t>(total_b);
    const float sig = 1.0f / (1.0f + expf(-gate));
    out_row[local_col] = float_to_scalar<scalar_t>(gate * sig * up);
  }
}

// Decode-only dense/shared Q4T16 owner. Match the retained pack8 local32
// owner exactly: each lane owns eight contiguous K values per 256-wide
// superblock, every output uses the same FMA sequence and wave32 tree, and
// gate/up are independently rounded to BF16 before the fused SiLU product.
template <typename scalar_t>
__global__ __launch_bounds__(32, 4)
void q4_k_t16_dense_dual_local32_silu_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  const int k_lane_base = tid * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a =
        tiles_a + (out_tile * blocks_per_row + blk) *
                      Q4_T16_BLOCK_BYTES;
    const uint8_t* __restrict__ tile_b =
        tiles_b + (out_tile * blocks_per_row + blk) *
                      Q4_T16_BLOCK_BYTES;

    float scale_a[kTileCols];
    float min_a[kTileCols];
    float scale_b[kTileCols];
    float min_b[kTileCols];
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d_a = *reinterpret_cast<const uint32_t*>(
          tile_a + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin_a = *reinterpret_cast<const uint32_t*>(
          tile_a + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale_a =
          *reinterpret_cast<const uint16_t*>(
              tile_a + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min_a =
          *reinterpret_cast<const uint16_t*>(
              tile_a + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      const uint32_t d_b = *reinterpret_cast<const uint32_t*>(
          tile_b + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin_b = *reinterpret_cast<const uint32_t*>(
          tile_b + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale_b =
          *reinterpret_cast<const uint16_t*>(
              tile_b + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min_b =
          *reinterpret_cast<const uint16_t*>(
              tile_b + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale_a[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d_a)) *
          static_cast<float>(coeff_scale_a & 0xFFu);
      scale_a[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d_a >> 16)) *
          static_cast<float>(coeff_scale_a >> 8);
      min_a[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a)) *
          static_cast<float>(coeff_min_a & 0xFFu);
      min_a[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a >> 16)) *
          static_cast<float>(coeff_min_a >> 8);
      scale_b[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d_b)) *
          static_cast<float>(coeff_scale_b & 0xFFu);
      scale_b[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d_b >> 16)) *
          static_cast<float>(coeff_scale_b >> 8);
      min_b[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b)) *
          static_cast<float>(coeff_min_b & 0xFFu);
      min_b[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b >> 16)) *
          static_cast<float>(coeff_min_b >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv =
          scalar_to_float<scalar_t>(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed_a = tile_a[q_index];
        const uint8_t packed_b = tile_b[q_index];
        acc_a[local_col0] = fmaf(
            xv,
            static_cast<float>(packed_a & 0x0Fu) *
                    scale_a[local_col0] -
                min_a[local_col0],
            acc_a[local_col0]);
        acc_a[local_col1] = fmaf(
            xv,
            static_cast<float>(packed_a >> 4) *
                    scale_a[local_col1] -
                min_a[local_col1],
            acc_a[local_col1]);
        acc_b[local_col0] = fmaf(
            xv,
            static_cast<float>(packed_b & 0x0Fu) *
                    scale_b[local_col0] -
                min_b[local_col0],
            acc_b[local_col0]);
        acc_b[local_col1] = fmaf(
            xv,
            static_cast<float>(packed_b >> 4) *
                    scale_b[local_col1] -
                min_b[local_col1],
            acc_b[local_col1]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
  __shared__ float xchg[kTileCols];
  uint16_t gate[kTileCols];
  if (tid == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg[local_col] = acc_a[local_col];
      gate[local_col] =
          float_to_scalar<scalar_t>(0.0f + xchg[local_col]);
    }
  }
  __syncthreads();

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }
  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      const float gate_f =
          scalar_to_float<scalar_t>(gate[local_col]);
      const float up_f = round_trip_scalar_to_float<scalar_t>(
          0.0f + acc_b[local_col]);
      const float sig = 1.0f / (1.0f + expf(-gate_f));
      out[out_base + local_col] =
          float_to_scalar<scalar_t>(gate_f * sig * up_f);
    }
  }
}

// Decode-only dense/shared Q4T16 single-output owner. Match the retained
// pack8 local32 owner exactly: each lane owns eight contiguous K values per
// 256-wide superblock, every output uses the same FMA sequence and wave32
// reduction tree, and the final scalar is rounded directly to BF16.
template <typename scalar_t>
__global__ __launch_bounds__(32, 8)
void q4_k_t16_dense_single_local32_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;

  float acc[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc[local_col] = 0.0f;
  }

  const int k_lane_base = tid * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q4_T16_BLOCK_BYTES;

    float scale[kTileCols];
    float min_value[kTileCols];
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv =
          scalar_to_float<scalar_t>(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        acc[local_col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) *
                    scale[local_col0] -
                min_value[local_col0],
            acc[local_col0]);
        acc[local_col1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) *
                    scale[local_col1] -
                min_value[local_col1],
            acc[local_col1]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }
  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      out[out_base + local_col] =
          float_to_scalar<scalar_t>(0.0f + acc[local_col]);
    }
  }
}

// Exact serial-c1 sibling of the local32 Q4 owner. The dot and wave reduction
// are intentionally duplicated so the unfused producer's code generation stays
// frozen; only the two rounded-BF16 residual operations are folded into its
// lane-0 store.
template <typename scalar_t>
__global__ __launch_bounds__(32, 8)
void q4_k_t16_dense_single_local32_residual_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;

  float acc[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc[local_col] = 0.0f;
  }

  const int k_lane_base = tid * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q4_T16_BLOCK_BYTES;

    float scale[kTileCols];
    float min_value[kTileCols];
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv =
          scalar_to_float<scalar_t>(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        acc[local_col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) *
                    scale[local_col0] -
                min_value[local_col0],
            acc[local_col0]);
        acc[local_col1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) *
                    scale[local_col1] -
                min_value[local_col1],
            acc[local_col1]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }
  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      const int64_t idx = out_base + local_col;
      const scalar_t projected =
          float_to_scalar<scalar_t>(0.0f + acc[local_col]);
      out[idx] = float_to_scalar<scalar_t>(
          scalar_to_float<scalar_t>(residual[idx]) +
          scalar_to_float<scalar_t>(projected));
    }
  }
}

// Exact compact-T16 small-B rowtile. The local32 arithmetic stays identical
// to q4_k_t16_dense_single_local32_gemv_kernel, but each decoded weight is
// reused across verifier rows 2-4 while every row retains its own ordered FMA
// stream and wave32 reduction tree.
template <typename scalar_t, int ROW_TILE, int kTileCols>
__device__ __forceinline__ void q4_k_t16_dense_rowtile_accumulate(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    int lane,
    int64_t in_features,
    int64_t out_group,
    float (&acc)[ROW_TILE][kTileCols]) {
  static_assert(
      kTileCols == 4 || kTileCols == 8,
      "dense Q4T16 rowtile must own four or eight columns");
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      acc[row][local_col] = 0.0f;
    }
  }

  const int k_lane_base = lane * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q4_T16_BLOCK_BYTES;

    float scale[kTileCols];
    float min_value[kTileCols];
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      float weight[kTileCols];
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        weight[local_col0] =
            static_cast<float>(packed & 0x0Fu) *
                    scale[local_col0] -
                min_value[local_col0];
        weight[local_col1] =
            static_cast<float>(packed >> 4) *
                    scale[local_col1] -
                min_value[local_col1];
      }
#pragma unroll
      for (int row = 0; row < ROW_TILE; ++row) {
        const float xv = scalar_to_float<scalar_t>(
            x[static_cast<int64_t>(row) * in_features +
              blk * QK_K + k_in_block]);
#pragma unroll
        for (int local_col = 0;
             local_col < kTileCols;
             ++local_col) {
          acc[row][local_col] = fmaf(
              xv, weight[local_col], acc[row][local_col]);
        }
      }
    }
  }
}

// Qmicro sibling of the compact-T16 rowtile accumulator. Each K owner loads
// the four packed scale/min records for its eight output columns directly.
// This deliberately trades redundant cacheable metadata reads for no wave
// exchange or barrier. Weight reuse, per-row FMA order, and the wave32
// reduction boundary stay identical to the standard owner.
template <typename scalar_t, int ROW_TILE, int kTileCols>
__device__ __forceinline__ void q4_k_qmicro_t16_dense_rowtile_accumulate(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    int lane,
    int64_t in_features,
    int64_t out_group,
    float (&acc)[ROW_TILE][kTileCols]) {
  static_assert(
      kTileCols == 8,
      "dense qmicro Q4T16 rowtile must own eight columns");
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      acc[row][local_col] = 0.0f;
    }
  }

  const int k_lane_base = lane * 8;
  const int sb = k_lane_base >> 5;
  constexpr int META_QUARTETS = T16_COLS / 4;
  constexpr int MIN_RECORD_BASE = Q4_SUBBLOCKS * META_QUARTETS;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q4_QMICRO_T16_BLOCK_BYTES;
    uint32_t scale_word[2];
    uint32_t min_word[2];
#pragma unroll
    for (int quartet = 0; quartet < 2; ++quartet) {
      const int record =
          sb * META_QUARTETS + (col_base >> 2) + quartet;
      scale_word[quartet] = qmicro_unpack_record(
          tile + Q4_QMICRO_T16_META_OFFSET + record * 3);
      min_word[quartet] = qmicro_unpack_record(
          tile + Q4_QMICRO_T16_META_OFFSET +
              (MIN_RECORD_BASE + record) * 3);
    }

    float scale[kTileCols];
    float min_value[kTileCols];
#pragma unroll
    for (int local_pair = 0; local_pair < kTileCols / 2; ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_QMICRO_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_QMICRO_T16_DMIN_OFFSET + col0 * 2);
      const int quartet = local_col0 >> 2;
      const int shift = (local_col0 & 3) * 8;
      const uint16_t coeff_scale = static_cast<uint16_t>(
          (scale_word[quartet] >> shift) & 0xFFFFu);
      const uint16_t coeff_min = static_cast<uint16_t>(
          (min_word[quartet] >> shift) & 0xFFFFu);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      float weight[kTileCols];
#pragma unroll
      for (int local_pair = 0; local_pair < kTileCols / 2; ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_QMICRO_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        weight[local_col0] =
            static_cast<float>(packed & 0x0Fu) * scale[local_col0] -
            min_value[local_col0];
        weight[local_col1] =
            static_cast<float>(packed >> 4) * scale[local_col1] -
            min_value[local_col1];
      }
#pragma unroll
      for (int row = 0; row < ROW_TILE; ++row) {
        const float xv = scalar_to_float<scalar_t>(
            x[static_cast<int64_t>(row) * in_features +
              blk * QK_K + k_in_block]);
#pragma unroll
        for (int local_col = 0; local_col < kTileCols; ++local_col) {
          acc[row][local_col] = fmaf(
              xv, weight[local_col], acc[row][local_col]);
        }
      }
    }
  }
}

template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(32, 1)
void q4_k_qmicro_t16_dense_rowtile_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int lane = threadIdx.x;
  float acc[ROW_TILE][kTileCols];
  q4_k_qmicro_t16_dense_rowtile_accumulate<
      scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        out[static_cast<int64_t>(row) * out_features +
            out_base + local_col] =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
      }
    }
  }
}

template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(32, 1)
void q4_k_t16_dense_rowtile_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int lane = threadIdx.x;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        out[static_cast<int64_t>(row) * out_features +
            out_base + local_col] =
            float_to_scalar<scalar_t>(
                0.0f + acc[row][local_col]);
      }
    }
  }
}

// Exact two-wave geometry: independent waves own adjacent 8-column groups in
// one 16-column block. Each wave executes the established helper and reduction
// unchanged; there is no cross-wave arithmetic.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(64, 1)
void q4_k_t16_dense_rowtile16_w2_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t out_group = blockIdx.x * 2 + wave;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        out[static_cast<int64_t>(row) * out_features +
            out_base + local_col] =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
      }
    }
  }
}

// Exact grouped rows6 geometry: grid.y owns independent contiguous row chunks
// while every WG64 block executes the retained two-wave arithmetic unchanged.
// This consolidates physical verifier launch overhead without merging waves or
// changing any row's FMA/reduction/store order.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(64, 1)
void q4_k_t16_dense_rowtile16_w2_grouped_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t out_group = blockIdx.x * 2 + wave;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * ROW_TILE;
  const scalar_t* __restrict__ chunk_x = x + row_base * in_features;
  scalar_t* __restrict__ chunk_out = out + row_base * out_features;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      chunk_x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        chunk_out[static_cast<int64_t>(row) * out_features +
                  out_base + local_col] =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
      }
    }
  }
}

// Exact compact-Q4 FFN-down plus rounded-BF16 residual owner. The producer
// keeps the established dot/reduction and first BF16 boundary, then consumes
// BF16 residual bits and performs the same FP32 add/final BF16 rounding as the
// standalone primitive.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(32, 1)
void q4_k_t16_dense_rowtile_residual_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int lane = threadIdx.x;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        const int64_t idx =
            static_cast<int64_t>(row) * out_features +
            out_base + local_col;
        const scalar_t projected =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
        out[idx] = float_to_scalar<scalar_t>(
            scalar_to_float<scalar_t>(residual[idx]) +
            scalar_to_float<scalar_t>(projected));
      }
    }
  }
}

// Exact dense Q5T16 small-B rowtile. The 128-thread arithmetic partitions,
// per-thread K sequence, wave32 tree, and ordered wave-0..3 reduction match the
// one-expert direct producer. Four- or eight-column groups reuse decoded
// weights across rows 2-8 without changing any row's accumulation order.
template <typename scalar_t, int ROW_TILE, int TILE_COLS>
__global__ __launch_bounds__(128, 2)
void q5_k_t16_dense_rowtile_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  static_assert(TILE_COLS == 4 || TILE_COLS == 8,
                "Q5 rowtile columns must be 4 or 8");
  constexpr int kTileCols = TILE_COLS;
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = in_features / QK_K;
  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      acc[row][local_col] = 0.0f;
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q5_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      float weight[kTileCols];
      const int quant_row = sb * Q4_SUBBLOCK + lane32;
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int col1 = col0 + 1;
        const uint32_t d_pair =
            *reinterpret_cast<const uint32_t*>(
                tile + Q5_T16_D_OFFSET + col0 * 2);
        const uint32_t dmin_pair =
            *reinterpret_cast<const uint32_t*>(
                tile + Q5_T16_DMIN_OFFSET + col0 * 2);
        const uint16_t scale_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q5_T16_SCALE_OFFSET +
                sb * T16_COLS + col0);
        const uint16_t min_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q5_T16_MIN_OFFSET +
                sb * T16_COLS + col0);
        const uint8_t low_packed =
            tile[Q5_T16_QL_OFFSET +
                 quant_row * (T16_COLS / 2) +
                 (col0 >> 1)];
        const uint8_t high_packed =
            tile[Q5_T16_QH_OFFSET +
                 quant_row * (T16_COLS / 8) +
                 (col0 >> 3)];
        const uint8_t q0 =
            (low_packed & 0x0Fu) |
            (((high_packed >> (col0 & 7)) & 0x01u) << 4);
        const uint8_t q1 =
            (low_packed >> 4) |
            (((high_packed >> (col1 & 7)) & 0x01u) << 4);
        const float scale0 =
            fp16_bits_to_float(static_cast<uint16_t>(d_pair)) *
            static_cast<float>(scale_pair & 0xFFu);
        const float scale1 =
            fp16_bits_to_float(static_cast<uint16_t>(d_pair >> 16)) *
            static_cast<float>(scale_pair >> 8);
        const float min0 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_pair)) *
            static_cast<float>(min_pair & 0xFFu);
        const float min1 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_pair >> 16)) *
            static_cast<float>(min_pair >> 8);
        weight[local_col0] =
            scale0 * static_cast<float>(q0) - min0;
        weight[local_col1] =
            scale1 * static_cast<float>(q1) - min1;
      }
#pragma unroll
      for (int row = 0; row < ROW_TILE; ++row) {
        const float xv = scalar_to_float<scalar_t>(
            x[static_cast<int64_t>(row) * in_features +
              k_base + k_in_block]);
#pragma unroll
        for (int local_col = 0;
             local_col < kTileCols;
             ++local_col) {
          acc[row][local_col] += xv * weight[local_col];
        }
      }
    }
  }

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }

  __shared__ float xchg[ROW_TILE][4][kTileCols];
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        xchg[row][wave][local_col] = acc[row][local_col];
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        float total = 0.0f;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
          total += xchg[row][group][local_col];
        }
        out[static_cast<int64_t>(row) * out_features +
            out_base + local_col] =
            float_to_scalar<scalar_t>(total);
      }
    }
  }
}

// Exact grouped rows6 geometry for dense Q5T16. Grid.y owns independent
// contiguous row chunks while every WG128 block executes the retained
// four-wave/4-column arithmetic unchanged. This changes only launch ownership;
// each row keeps the same FMA, wave reduction, ordered wave-0..3 sum, and store.
template <typename scalar_t, int ROW_TILE, int TILE_COLS>
__global__ __launch_bounds__(128, 2)
void q5_k_t16_dense_rowtile_grouped_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  static_assert(TILE_COLS == 4 || TILE_COLS == 8,
                "Q5 grouped rowtile columns must be 4 or 8");
  constexpr int kTileCols = TILE_COLS;
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * ROW_TILE;
  const scalar_t* __restrict__ chunk_x = x + row_base * in_features;
  scalar_t* __restrict__ chunk_out = out + row_base * out_features;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = in_features / QK_K;
  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      acc[row][local_col] = 0.0f;
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) *
                    Q5_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      float weight[kTileCols];
      const int quant_row = sb * Q4_SUBBLOCK + lane32;
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int col1 = col0 + 1;
        const uint32_t d_pair =
            *reinterpret_cast<const uint32_t*>(
                tile + Q5_T16_D_OFFSET + col0 * 2);
        const uint32_t dmin_pair =
            *reinterpret_cast<const uint32_t*>(
                tile + Q5_T16_DMIN_OFFSET + col0 * 2);
        const uint16_t scale_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q5_T16_SCALE_OFFSET +
                sb * T16_COLS + col0);
        const uint16_t min_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q5_T16_MIN_OFFSET +
                sb * T16_COLS + col0);
        const uint8_t low_packed =
            tile[Q5_T16_QL_OFFSET +
                 quant_row * (T16_COLS / 2) +
                 (col0 >> 1)];
        const uint8_t high_packed =
            tile[Q5_T16_QH_OFFSET +
                 quant_row * (T16_COLS / 8) +
                 (col0 >> 3)];
        const uint8_t q0 =
            (low_packed & 0x0Fu) |
            (((high_packed >> (col0 & 7)) & 0x01u) << 4);
        const uint8_t q1 =
            (low_packed >> 4) |
            (((high_packed >> (col1 & 7)) & 0x01u) << 4);
        const float scale0 =
            fp16_bits_to_float(static_cast<uint16_t>(d_pair)) *
            static_cast<float>(scale_pair & 0xFFu);
        const float scale1 =
            fp16_bits_to_float(static_cast<uint16_t>(d_pair >> 16)) *
            static_cast<float>(scale_pair >> 8);
        const float min0 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_pair)) *
            static_cast<float>(min_pair & 0xFFu);
        const float min1 =
            fp16_bits_to_float(static_cast<uint16_t>(dmin_pair >> 16)) *
            static_cast<float>(min_pair >> 8);
        weight[local_col0] =
            scale0 * static_cast<float>(q0) - min0;
        weight[local_col1] =
            scale1 * static_cast<float>(q1) - min1;
      }
#pragma unroll
      for (int row = 0; row < ROW_TILE; ++row) {
        const float xv = scalar_to_float<scalar_t>(
            chunk_x[static_cast<int64_t>(row) * in_features +
                    k_base + k_in_block]);
#pragma unroll
        for (int local_col = 0;
             local_col < kTileCols;
             ++local_col) {
          acc[row][local_col] += xv * weight[local_col];
        }
      }
    }
  }

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }

  __shared__ float xchg[ROW_TILE][4][kTileCols];
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        xchg[row][wave][local_col] = acc[row][local_col];
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        float total = 0.0f;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
          total += xchg[row][group][local_col];
        }
        chunk_out[static_cast<int64_t>(row) * out_features +
                  out_base + local_col] =
            float_to_scalar<scalar_t>(total);
      }
    }
  }
}

// Two-wave compact-T16 FFN sibling. Each wave owns one matrix, reuses its
// decoded weights across rows, and rounds gate/up independently to BF16 in LDS
// before the unchanged SiLU product. This is the compact-layout analogue of
// the retained pack8 dual rowtile.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(64, 1)
void q4_k_t16_dense_dual_rowtile_silu_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_group = blockIdx.x;
  const uint8_t* __restrict__ tiles = wave == 0 ? tiles_a : tiles_b;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }

  __shared__ uint16_t projected[2][ROW_TILE][kTileCols];
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        projected[wave][row][local_col] =
            float_to_scalar<scalar_t>(
                0.0f + acc[row][local_col]);
      }
    }
  }
  __syncthreads();

  if (tid < ROW_TILE * kTileCols) {
    const int row = tid / kTileCols;
    const int local_col = tid % kTileCols;
    const float gate = scalar_to_float<scalar_t>(
        projected[0][row][local_col]);
    const float up = scalar_to_float<scalar_t>(
        projected[1][row][local_col]);
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features +
        out_group * kTileCols + local_col] =
        float_to_scalar<scalar_t>(gate * sig * up);
  }
}

// Sole-qmicro rows2-4 gate/up owner. Each wave retains one matrix and the
// standard rowtile's output/FMA/reduction sequence while consuming only the
// compressed resident metadata.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(64, 1)
void q4_k_qmicro_t16_dense_dual_rowtile_silu_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_group = blockIdx.x;
  const uint8_t* __restrict__ tiles = wave == 0 ? tiles_a : tiles_b;
  float acc[ROW_TILE][kTileCols];
  q4_k_qmicro_t16_dense_rowtile_accumulate<
      scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }

  __shared__ uint16_t projected[2][4][kTileCols];
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        projected[wave][row][local_col] =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
      }
    }
  }
  __syncthreads();

  if (tid < ROW_TILE * kTileCols) {
    const int row = tid / kTileCols;
    const int local_col = tid % kTileCols;
    const float gate = scalar_to_float<scalar_t>(
        projected[0][row][local_col]);
    const float up = scalar_to_float<scalar_t>(
        projected[1][row][local_col]);
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features +
        out_group * kTileCols + local_col] =
        float_to_scalar<scalar_t>(gate * sig * up);
  }
}

// Four-column pressure owner for measured narrow/selective dense Q4T16
// verifier rows. It preserves the production tile8 K/FMA/reduction/store
// order for every output while halving each block's accumulator set.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(32, 1)
void q4_k_t16_dense_rowtile_col4_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 4;
  const int64_t out_group = blockIdx.x;
  const int lane = threadIdx.x;
  float acc[ROW_TILE][kTileCols];
  q4_k_t16_dense_rowtile_accumulate<scalar_t, ROW_TILE, kTileCols>(
      x, tiles, lane, in_features, out_group, acc);

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      GGUF_T16_WAVE_REDUCE(acc[row][local_col]);
    }
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        out[static_cast<int64_t>(row) * out_features +
            out_base + local_col] =
            float_to_scalar<scalar_t>(0.0f + acc[row][local_col]);
      }
    }
  }
}

// Decode-only dense/shared dual-interleaved Q4T16 owner. This preserves the
// retained local32 arithmetic exactly while replacing the two same-size T16
// streams with one byte-neutral paired stream.
template <typename scalar_t, int kTileCols>
__global__ __launch_bounds__(32, 4)
void q4_k_t16_dense_dual_interleaved_local32_silu_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_dual,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  static_assert(
      kTileCols == 2 || kTileCols == 4 || kTileCols == 8,
      "dense dual-interleaved tile must be 2, 4, or 8 columns");
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t out_group = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int group_in_tile =
      static_cast<int>(out_group % kGroupsPerTile);
  const int col_base = group_in_tile * kTileCols;
  const int half = col_base >> 3;
  const int q_pair_base = (col_base & 7) >> 1;

  float acc_a[kTileCols];
  float acc_b[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc_a[local_col] = 0.0f;
    acc_b[local_col] = 0.0f;
  }

  const int k_lane_base = tid * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles_dual + (out_tile * blocks_per_row + blk) *
                         Q4_T16_DUAL_BLOCK_BYTES;

    float scale_a[kTileCols];
    float min_a[kTileCols];
    float scale_b[kTileCols];
    float min_b[kTileCols];
#pragma unroll
    for (int local_pair = 0;
         local_pair < kTileCols / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col_pair = (col_base + local_col0) >> 1;
      const uint64_t d_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_D_OFFSET + col_pair * 8);
      const uint64_t dmin_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_DMIN_OFFSET + col_pair * 8);
      const uint32_t coeff_scale_dual =
          *reinterpret_cast<const uint32_t*>(
              tile + Q4_T16_DUAL_SCALE_OFFSET +
              (sb * (T16_COLS / 2) + col_pair) * 4);
      const uint32_t coeff_min_dual =
          *reinterpret_cast<const uint32_t*>(
              tile + Q4_T16_DUAL_MIN_OFFSET +
              (sb * (T16_COLS / 2) + col_pair) * 4);
      const uint32_t d_a = static_cast<uint32_t>(d_dual);
      const uint32_t d_b = static_cast<uint32_t>(d_dual >> 32);
      const uint32_t dmin_a = static_cast<uint32_t>(dmin_dual);
      const uint32_t dmin_b = static_cast<uint32_t>(dmin_dual >> 32);
      const uint16_t coeff_scale_a =
          static_cast<uint16_t>(coeff_scale_dual);
      const uint16_t coeff_scale_b =
          static_cast<uint16_t>(coeff_scale_dual >> 16);
      const uint16_t coeff_min_a =
          static_cast<uint16_t>(coeff_min_dual);
      const uint16_t coeff_min_b =
          static_cast<uint16_t>(coeff_min_dual >> 16);
      scale_a[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d_a)) *
          static_cast<float>(coeff_scale_a & 0xFFu);
      scale_a[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d_a >> 16)) *
          static_cast<float>(coeff_scale_a >> 8);
      min_a[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a)) *
          static_cast<float>(coeff_min_a & 0xFFu);
      min_a[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_a >> 16)) *
          static_cast<float>(coeff_min_a >> 8);
      scale_b[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d_b)) *
          static_cast<float>(coeff_scale_b & 0xFFu);
      scale_b[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d_b >> 16)) *
          static_cast<float>(coeff_scale_b >> 8);
      min_b[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b)) *
          static_cast<float>(coeff_min_b & 0xFFu);
      min_b[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin_b >> 16)) *
          static_cast<float>(coeff_min_b >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv =
          scalar_to_float<scalar_t>(x[blk * QK_K + k_in_block]);
      const uint64_t q_dual = *reinterpret_cast<const uint64_t*>(
          tile + Q4_T16_DUAL_Q_OFFSET +
          ((sb * Q4_SUBBLOCK + lane32) * 2 + half) * 8);
      const uint32_t q_a = static_cast<uint32_t>(q_dual);
      const uint32_t q_b = static_cast<uint32_t>(q_dual >> 32);
#pragma unroll
      for (int local_pair = 0;
           local_pair < kTileCols / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const uint8_t packed_a =
            static_cast<uint8_t>(
                q_a >> ((q_pair_base + local_pair) * 8));
        const uint8_t packed_b =
            static_cast<uint8_t>(
                q_b >> ((q_pair_base + local_pair) * 8));
        acc_a[local_col0] = fmaf(
            xv,
            static_cast<float>(packed_a & 0x0Fu) *
                    scale_a[local_col0] -
                min_a[local_col0],
            acc_a[local_col0]);
        acc_a[local_col1] = fmaf(
            xv,
            static_cast<float>(packed_a >> 4) *
                    scale_a[local_col1] -
                min_a[local_col1],
            acc_a[local_col1]);
        acc_b[local_col0] = fmaf(
            xv,
            static_cast<float>(packed_b & 0x0Fu) *
                    scale_b[local_col0] -
                min_b[local_col0],
            acc_b[local_col0]);
        acc_b[local_col1] = fmaf(
            xv,
            static_cast<float>(packed_b >> 4) *
                    scale_b[local_col1] -
                min_b[local_col1],
            acc_b[local_col1]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_a[local_col]);
  }
  uint16_t gate[kTileCols];
  if (tid == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      gate[local_col] =
          float_to_scalar<scalar_t>(0.0f + acc_a[local_col]);
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc_b[local_col]);
  }
  if (tid == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      const float gate_f =
          scalar_to_float<scalar_t>(gate[local_col]);
      const float up_f = round_trip_scalar_to_float<scalar_t>(
          0.0f + acc_b[local_col]);
      const float sig = 1.0f / (1.0f + expf(-gate_f));
      out[out_base + local_col] =
          float_to_scalar<scalar_t>(gate_f * sig * up_f);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  // Pair consecutive occurrences of the same dynamic expert ID.  Physical C8
  // has at most 64 selected lanes, so two wave32 ballots build the full match
  // mask in one coalesced read and one barrier.  Odd occurrences are completed
  // by their preceding anchor block and return before touching output.
  __shared__ uint32_t expert_match_words[2];
  const bool lane_matches = tid < rows && selected[tid] == expert_id;
  const uint32_t wave_matches = static_cast<uint32_t>(__ballot(lane_matches));
  const int wave_id = tid >> 5;
  if ((tid & 31) == 0 && wave_id < 2) {
    expert_match_words[wave_id] = wave_matches;
  }
  __syncthreads();
  const uint64_t expert_matches = static_cast<uint64_t>(expert_match_words[0]) |
                                  (static_cast<uint64_t>(expert_match_words[1]) << 32);
  const uint64_t prior_mask = row == 0 ? 0 : ((uint64_t{1} << row) - 1);
  const int prior_matches = __popcll(expert_matches & prior_mask);
  if ((prior_matches & 1) != 0) return;
  const uint64_t later_mask = row == 63 ? 0 : (~uint64_t{0} << (row + 1));
  const uint64_t later_matches = expert_matches & later_mask;
  const int64_t paired_row = later_matches == 0
                                 ? -1
                                 : (__ffsll(static_cast<unsigned long long>(later_matches)) - 1);

  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t paired_x_row_idx = paired_row >= 0 ? paired_row / lanes_per_x_row : -1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;
  const scalar_t* __restrict__ paired_x_row =
      paired_row >= 0 ? x + paired_x_row_idx * in_features : nullptr;

  float acc0[T16_COLS];
  float acc1[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  if (paired_row < 0) {
    // Preserve the production dual kernel unchanged for unpaired lanes.
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
      const int64_t k_base = blk * QK_K;
      for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
        const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
          const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
          acc0[col] += xv * wv_a;
          acc1[col] += xv * wv_b;
        }
      }
    }

#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);

    __shared__ float xchg0[4 * T16_COLS];
    __shared__ float xchg1[4 * T16_COLS];
    const int sub_id = tid >> 5;
    const int sub_lane = tid & 31;
    if (sub_lane == 0) {
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        xchg0[sub_id * T16_COLS + col] = acc0[col];
        xchg1[sub_id * T16_COLS + col] = acc1[col];
      }
    }
    __syncthreads();

    if (tid == 0) {
      scalar_t* __restrict__ out_row_a = out_a + row * out_features + out_tile * T16_COLS;
      scalar_t* __restrict__ out_row_b = out_b + row * out_features + out_tile * T16_COLS;
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        float total_a = 0.0f;
        float total_b = 0.0f;
        for (int g = 0; g < 4; ++g) {
          total_a += xchg0[g * T16_COLS + col];
          total_b += xchg1[g * T16_COLS + col];
        }
        out_row_a[col] = float_to_scalar<scalar_t>(total_a);
        out_row_b[col] = float_to_scalar<scalar_t>(total_b);
      }
    }
    return;
  }

  // Reuse gate weights across both input rows while retaining the production
  // 128-thread accumulation partition and reduction order for each output.
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      const float xv1 = scalar_to_float<scalar_t>(paired_x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv = t16_q4_weight(tile, k_in_block, col);
        acc0[col] += xv0 * wv;
        acc1[col] += xv1 * wv;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);

  __shared__ float xchg0[4 * T16_COLS];
  __shared__ float xchg1[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row0 = out_a + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row1 = out_a + paired_row * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        total1 += xchg1[g * T16_COLS + col];
      }
      out_row0[col] = float_to_scalar<scalar_t>(total0);
      out_row1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  // Reuse up weights in the same way.  Processing gate and up sequentially
  // holds the live accumulator count equal to the production dual kernel.
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      const float xv1 = scalar_to_float<scalar_t>(paired_x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv = t16_q4_weight(tile, k_in_block, col);
        acc0[col] += xv0 * wv;
        acc1[col] += xv1 * wv;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);

  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row0 = out_b + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row1 = out_b + paired_row * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        total1 += xchg1[g * T16_COLS + col];
      }
      out_row0[col] = float_to_scalar<scalar_t>(total0);
      out_row1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
}

template <typename scalar_t, bool fuse_silu>
__global__ __launch_bounds__(128, 2)
void q4_k_qmicro_t16_selected_dual_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  __shared__ uint32_t metadata_words[2 * Q4_QMICRO_T16_META_BYTES / 4];
  uint8_t* __restrict__ metadata_a = reinterpret_cast<uint8_t*>(metadata_words);
  uint8_t* __restrict__ metadata_b = metadata_a + Q4_QMICRO_T16_META_BYTES;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_QMICRO_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_QMICRO_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int record = tid & 63;
    const bool load_b = tid >= 64;
    const uint8_t* __restrict__ source =
        (load_b ? tile_b : tile_a) + Q4_QMICRO_T16_META_OFFSET + record * 3;
    uint32_t* __restrict__ destination = reinterpret_cast<uint32_t*>(
        load_b ? metadata_b : metadata_a);
    destination[record] = qmicro_unpack_record(source);
    __syncthreads();

    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        acc_a[col] += xv * q4_qmicro_t16_weight(
            tile_a, metadata_a, k_in_block, col);
        acc_b[col] += xv * q4_qmicro_t16_weight(
            tile_b, metadata_b, k_in_block, col);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[4 * T16_COLS];
  __shared__ float xchg_b[4 * T16_COLS];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[wave * T16_COLS + col] = acc_a[col];
      xchg_b[wave * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row_a =
        out_a + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row_b = fuse_silu
        ? nullptr
        : out_b + row * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      if constexpr (fuse_silu) {
        const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
        const float up = round_trip_scalar_to_float<scalar_t>(total_b);
        const float sig = 1.0f / (1.0f + expf(-gate));
        out_row_a[col] = float_to_scalar<scalar_t>(gate * sig * up);
      } else {
        out_row_a[col] = float_to_scalar<scalar_t>(total_a);
        out_row_b[col] = float_to_scalar<scalar_t>(total_b);
      }
    }
  }
}

template <typename scalar_t, int tile_cols>
__global__ __launch_bounds__(128, 4)
void q5_k_qmicro_t16_selected_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features) {
  static_assert(tile_cols == 8, "Q5 qmicro direct tile must own eight columns");
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int groups_per_tile = T16_COLS / tile_cols;
  const int64_t out_tile = out_group / groups_per_tile;
  const int col_base = static_cast<int>(out_group % groups_per_tile) * tile_cols;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc[tile_cols];
#pragma unroll
  for (int col = 0; col < tile_cols; ++col) acc[col] = 0.0f;

  __shared__ uint32_t metadata_words[Q5_QMICRO_T16_META_BYTES / 4];
  uint8_t* __restrict__ metadata = reinterpret_cast<uint8_t*>(metadata_words);
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q5_QMICRO_T16_BLOCK_BYTES>(
        tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    if (tid < 64) {
      metadata_words[tid] = qmicro_unpack_record(
          tile + Q5_QMICRO_T16_META_OFFSET + tid * 3);
    }
    __syncthreads();

    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int local_col = 0; local_col < tile_cols; ++local_col) {
        const int col = col_base + local_col;
        const float d = fp16_bytes_to_float(
            tile + Q5_QMICRO_T16_D_OFFSET + col * 2);
        const float dmin = fp16_bytes_to_float(
            tile + Q5_QMICRO_T16_DMIN_OFFSET + col * 2);
        const float scale = d * static_cast<float>(
            metadata[sb * T16_COLS + col]);
        const float minv = dmin * static_cast<float>(
            metadata[Q4_SUBBLOCKS * T16_COLS + sb * T16_COLS + col]);
        const float q = static_cast<float>(
            q5_qmicro_t16_quant(tile, sb, lane32, col));
        acc[local_col] += xv * (scale * q - minv);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int col = 0; col < tile_cols; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);
  __shared__ float xchg[4 * tile_cols];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < tile_cols; ++col) {
      xchg[wave * tile_cols + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row =
        out + row * out_features + out_tile * T16_COLS + col_base;
#pragma unroll
    for (int col = 0; col < tile_cols; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) total += xchg[g * tile_cols + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 1) void q4_k_t16_selected_dual_silu_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
        const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
        acc_a[col] += xv * wv_a;
        acc_b[col] += xv * wv_b;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      // Preserve the split-kernel reduction/rounding contract: the legacy
      // direct path writes BF16 gate/up first, then the SiLU kernel reads those
      // rounded values.  Round-trip through scalar_t before applying SiLU so
      // P9.E2 full-model logits remain aligned with the existing reference.
      const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
      const float up = round_trip_scalar_to_float<scalar_t>(total_b);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out_row[col] = float_to_scalar<scalar_t>(gate * sig * up);
    }
  }
}

template <typename scalar_t, bool fuse_silu>
__global__ __launch_bounds__(128, 4) void q4_k_t16_selected_dual_q8_1_dp4a_direct_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t q8_blocks_per_row) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const gguf_q8_1_block* __restrict__ xq_row = xq + x_row_idx * q8_blocks_per_row;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  const int64_t groups4 = in_features >> 2;
  for (int64_t group = tid; group < groups4; group += blockDim.x) {
    const int64_t k4 = group << 2;
    const int64_t blk = k4 / QK_K;
    const int within = static_cast<int>(k4 - blk * QK_K);
    const int sb = within >> 5;
    const int lane4 = within & 31;
    const int64_t q8_index = blk * Q4_SUBBLOCKS + sb;
    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = t16_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);

#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      const float d_a = fp16_bytes_to_float(tile_a + Q4_T16_D_OFFSET + col * 2);
      const float dmin_a = fp16_bytes_to_float(tile_a + Q4_T16_DMIN_OFFSET + col * 2);
      const float scale_a = static_cast<float>(tile_a[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float min_a = static_cast<float>(tile_a[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const float d_b = fp16_bytes_to_float(tile_b + Q4_T16_D_OFFSET + col * 2);
      const float dmin_b = fp16_bytes_to_float(tile_b + Q4_T16_DMIN_OFFSET + col * 2);
      const float scale_b = static_cast<float>(tile_b[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float min_b = static_cast<float>(tile_b[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const int dot_a = t16_dp4a_u8_s8(t16_q4_pack4(tile_a, sb, lane4, col), x_pack, 0);
      const int dot_b = t16_dp4a_u8_s8(t16_q4_pack4(tile_b, sb, lane4, col), x_pack, 0);
      acc_a[col] += xd * (d_a * scale_a * static_cast<float>(dot_a) -
                          dmin_a * min_a * static_cast<float>(q8_sum));
      acc_b[col] += xd * (d_b * scale_b * static_cast<float>(dot_b) -
                          dmin_b * min_b * static_cast<float>(q8_sum));
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row_a = out_a + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row_b = fuse_silu ? nullptr : (out_b + row * out_features + out_tile * T16_COLS);
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      if constexpr (fuse_silu) {
        const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
        const float up = round_trip_scalar_to_float<scalar_t>(total_b);
        const float sig = 1.0f / (1.0f + expf(-gate));
        out_row_a[col] = float_to_scalar<scalar_t>(gate * sig * up);
      } else {
        out_row_a[col] = float_to_scalar<scalar_t>(total_a);
        out_row_b[col] = float_to_scalar<scalar_t>(total_b);
      }
    }
  }
}

// Accuracy-repaired Q4T16 c1 path: replay primary and residual Q8_1
// activation planes against each weight payload while it is already live.
// No resident weight sidecar is introduced; exact BF16 GEMV remains fallback.
template <typename scalar_t, bool dense_single_expert = false>
__global__ __launch_bounds__(128, 4)
void q4_k_t16_selected_dual_q8_1x2_dp4a_silu_direct_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t q8_blocks_per_row) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  int64_t expert_id;
  if constexpr (dense_single_expert) {
    expert_id = 0;
  } else {
    expert_id = selected[row];
  }
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx =
      (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const int64_t plane_stride = x_rows * q8_blocks_per_row;
  const gguf_q8_1_block* __restrict__ xq_row0 =
      xq + x_row_idx * q8_blocks_per_row;
  const gguf_q8_1_block* __restrict__ xq_row1 =
      xq + plane_stride + x_row_idx * q8_blocks_per_row;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  const int64_t groups4 = in_features >> 2;
  for (int64_t group = tid; group < groups4; group += blockDim.x) {
    const int64_t k4 = group << 2;
    const int64_t blk = k4 / QK_K;
    const int within = static_cast<int>(k4 - blk * QK_K);
    const int sb = within >> 5;
    const int lane4 = within & 31;
    const int64_t q8_index = blk * Q4_SUBBLOCKS + sb;
    const gguf_q8_1_block* __restrict__ xb0 = xq_row0 + q8_index;
    const gguf_q8_1_block* __restrict__ xb1 = xq_row1 + q8_index;
    const uint32_t x_pack0 =
        *reinterpret_cast<const uint32_t*>(xb0->qs + lane4);
    const uint32_t x_pack1 =
        *reinterpret_cast<const uint32_t*>(xb1->qs + lane4);
    const int q8_sum0 = t16_dp4a_u8_s8(0x01010101u, x_pack0, 0);
    const int q8_sum1 = t16_dp4a_u8_s8(0x01010101u, x_pack1, 0);
    const float xd0 = static_cast<float>(xb0->d);
    const float xd1 = static_cast<float>(xb1->d);

    const uint8_t* __restrict__ tile_a =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_a, expert_id, out_tile, blk, out_tiles,
            blocks_per_row);
    const uint8_t* __restrict__ tile_b =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            tiles_b, expert_id, out_tile, blk, out_tiles,
            blocks_per_row);

#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      const float d_a =
          fp16_bytes_to_float(tile_a + Q4_T16_D_OFFSET + col * 2);
      const float dmin_a =
          fp16_bytes_to_float(tile_a + Q4_T16_DMIN_OFFSET + col * 2);
      const float scale_a = static_cast<float>(
          tile_a[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float min_a = static_cast<float>(
          tile_a[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const float d_b =
          fp16_bytes_to_float(tile_b + Q4_T16_D_OFFSET + col * 2);
      const float dmin_b =
          fp16_bytes_to_float(tile_b + Q4_T16_DMIN_OFFSET + col * 2);
      const float scale_b = static_cast<float>(
          tile_b[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float min_b = static_cast<float>(
          tile_b[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const uint32_t weight_pack_a =
          t16_q4_pack4(tile_a, sb, lane4, col);
      const uint32_t weight_pack_b =
          t16_q4_pack4(tile_b, sb, lane4, col);
      const int dot_a0 =
          t16_dp4a_u8_s8(weight_pack_a, x_pack0, 0);
      const int dot_a1 =
          t16_dp4a_u8_s8(weight_pack_a, x_pack1, 0);
      const int dot_b0 =
          t16_dp4a_u8_s8(weight_pack_b, x_pack0, 0);
      const int dot_b1 =
          t16_dp4a_u8_s8(weight_pack_b, x_pack1, 0);
      const float weight_scale_a = d_a * scale_a;
      const float weight_min_a = dmin_a * min_a;
      const float weight_scale_b = d_b * scale_b;
      const float weight_min_b = dmin_b * min_b;
      acc_a[col] += xd0 *
          (weight_scale_a * static_cast<float>(dot_a0) -
           weight_min_a * static_cast<float>(q8_sum0));
      acc_a[col] += xd1 *
          (weight_scale_a * static_cast<float>(dot_a1) -
           weight_min_a * static_cast<float>(q8_sum1));
      acc_b[col] += xd0 *
          (weight_scale_b * static_cast<float>(dot_b0) -
           weight_min_b * static_cast<float>(q8_sum0));
      acc_b[col] += xd1 *
          (weight_scale_b * static_cast<float>(dot_b1) -
           weight_min_b * static_cast<float>(q8_sum1));
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc_a[col]);
  }
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc_b[col]);
  }

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row =
        out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
      const float up = round_trip_scalar_to_float<scalar_t>(total_b);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out_row[col] = float_to_scalar<scalar_t>(gate * sig * up);
    }
  }
}

// Exact dense-c1 residual-Q8_1x2 sibling with independent gate/up wave pairs.
// Waves 0-1 preserve the retained 64-thread gate K partition while waves 2-3
// preserve the same up partition. Each thread therefore owns one 16-column
// accumulator plane instead of both planes; only the small activation reads are
// repeated, while weight traffic, FMA order, reduction order, and output work
// remain unchanged.
template <typename scalar_t>
__global__ __launch_bounds__(128, 2)
void q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t q8_blocks_per_row) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_lane = tid & 31;
  const bool up_plane = wave_id >= 2;
  const int k_wave = wave_id & 1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const gguf_q8_1_block* __restrict__ xq_row0 = xq;
  const gguf_q8_1_block* __restrict__ xq_row1 = xq + q8_blocks_per_row;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  const int64_t groups4 = in_features >> 2;
  for (int64_t group = k_wave * 32 + wave_lane;
       group < groups4;
       group += 64) {
    const int64_t k4 = group << 2;
    const int64_t blk = k4 / QK_K;
    const int within = static_cast<int>(k4 - blk * QK_K);
    const int sb = within >> 5;
    const int lane4 = within & 31;
    const int64_t q8_index = blk * Q4_SUBBLOCKS + sb;
    const gguf_q8_1_block* __restrict__ xb0 = xq_row0 + q8_index;
    const gguf_q8_1_block* __restrict__ xb1 = xq_row1 + q8_index;
    const uint32_t x_pack0 =
        *reinterpret_cast<const uint32_t*>(xb0->qs + lane4);
    const uint32_t x_pack1 =
        *reinterpret_cast<const uint32_t*>(xb1->qs + lane4);
    const int q8_sum0 = t16_dp4a_u8_s8(0x01010101u, x_pack0, 0);
    const int q8_sum1 = t16_dp4a_u8_s8(0x01010101u, x_pack1, 0);
    const float xd0 = static_cast<float>(xb0->d);
    const float xd1 = static_cast<float>(xb1->d);

    const uint8_t* __restrict__ tile =
        expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
            up_plane ? tiles_b : tiles_a,
            0,
            out_tile,
            blk,
            out_tiles,
            blocks_per_row);

#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      const float d =
          fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col * 2);
      const float dmin =
          fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col * 2);
      const float scale = static_cast<float>(
          tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float minv = static_cast<float>(
          tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const uint32_t weight_pack = t16_q4_pack4(tile, sb, lane4, col);
      const int dot0 = t16_dp4a_u8_s8(weight_pack, x_pack0, 0);
      const int dot1 = t16_dp4a_u8_s8(weight_pack, x_pack1, 0);
      const float weight_scale = d * scale;
      const float weight_min = dmin * minv;
      acc[col] += xd0 *
          (weight_scale * static_cast<float>(dot0) -
           weight_min * static_cast<float>(q8_sum0));
      acc[col] += xd1 *
          (weight_scale * static_cast<float>(dot1) -
           weight_min * static_cast<float>(q8_sum1));
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc[col]);
  }

  __shared__ float xchg_a[2 * T16_COLS];
  __shared__ float xchg_b[2 * T16_COLS];
  if (wave_lane == 0) {
    float* __restrict__ xchg = up_plane ? xchg_b : xchg_a;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[k_wave * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
      const float up = round_trip_scalar_to_float<scalar_t>(total_b);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out_row[col] = float_to_scalar<scalar_t>(gate * sig * up);
    }
  }
}

// Qmicro-payload screen for the retained dense-c1 split-weight dataflow.
// Each eight-lane K32 group loads the eight compressed scale/min records it
// needs and exchanges unpacked quartets with wave shuffles. This avoids LDS and
// per-K256 barriers while preserving quant values, per-thread K ownership, FMA
// order, wave reduction, and the standard-T16 owner's BF16 output boundary.
template <typename scalar_t>
__global__ __launch_bounds__(128, 2)
void q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t q8_blocks_per_row) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_lane = tid & 31;
  const bool up_plane = wave_id >= 2;
  const int k_wave = wave_id & 1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const gguf_q8_1_block* __restrict__ xq_row0 = xq;
  const gguf_q8_1_block* __restrict__ xq_row1 = xq + q8_blocks_per_row;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        expert_tile_ptr<Q4_QMICRO_T16_BLOCK_BYTES>(
            up_plane ? tiles_b : tiles_a,
            0,
            out_tile,
            blk,
            out_tiles,
            blocks_per_row);
    const int group_in_block = k_wave * 32 + wave_lane;
    const int sb = group_in_block >> 3;
    const int lane_in_group = wave_lane & 7;
    const int lane4 = lane_in_group << 2;
    constexpr int META_QUARTETS = T16_COLS / 4;
    constexpr int MIN_RECORD_BASE = Q4_SUBBLOCKS * META_QUARTETS;
    const int record =
        (lane_in_group >= 4 ? MIN_RECORD_BASE : 0) +
        sb * META_QUARTETS + (lane_in_group & 3);
    const uint32_t lane_metadata = qmicro_unpack_record(
        tile + Q4_QMICRO_T16_META_OFFSET + record * 3);

    const int64_t q8_index = blk * Q4_SUBBLOCKS + sb;
    const gguf_q8_1_block* __restrict__ xb0 = xq_row0 + q8_index;
    const gguf_q8_1_block* __restrict__ xb1 = xq_row1 + q8_index;
    const uint32_t x_pack0 =
        *reinterpret_cast<const uint32_t*>(xb0->qs + lane4);
    const uint32_t x_pack1 =
        *reinterpret_cast<const uint32_t*>(xb1->qs + lane4);
    const int q8_sum0 = t16_dp4a_u8_s8(0x01010101u, x_pack0, 0);
    const int q8_sum1 = t16_dp4a_u8_s8(0x01010101u, x_pack1, 0);
    const float xd0 = static_cast<float>(xb0->d);
    const float xd1 = static_cast<float>(xb1->d);
    const int group_lane_base = wave_lane & ~7;

#pragma unroll
    for (int quartet = 0; quartet < META_QUARTETS; ++quartet) {
      const uint32_t scale_word =
          __shfl(lane_metadata, group_lane_base + quartet, 32);
      const uint32_t min_word =
          __shfl(lane_metadata, group_lane_base + 4 + quartet, 32);
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        const int col = quartet * 4 + col4;
        const float d = fp16_bytes_to_float(
            tile + Q4_QMICRO_T16_D_OFFSET + col * 2);
        const float dmin = fp16_bytes_to_float(
            tile + Q4_QMICRO_T16_DMIN_OFFSET + col * 2);
        const float scale = static_cast<float>(
            (scale_word >> (8 * col4)) & 0xFFu);
        const float minv = static_cast<float>(
            (min_word >> (8 * col4)) & 0xFFu);
        const uint32_t weight_pack =
            q4_qmicro_t16_pack4(tile, sb, lane4, col);
        const int dot0 = t16_dp4a_u8_s8(weight_pack, x_pack0, 0);
        const int dot1 = t16_dp4a_u8_s8(weight_pack, x_pack1, 0);
        const float weight_scale = d * scale;
        const float weight_min = dmin * minv;
        acc[col] += xd0 *
            (weight_scale * static_cast<float>(dot0) -
             weight_min * static_cast<float>(q8_sum0));
        acc[col] += xd1 *
            (weight_scale * static_cast<float>(dot1) -
             weight_min * static_cast<float>(q8_sum1));
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc[col]);
  }

  __shared__ float xchg_a[2 * T16_COLS];
  __shared__ float xchg_b[2 * T16_COLS];
  if (wave_lane == 0) {
    float* __restrict__ xchg = up_plane ? xchg_b : xchg_a;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[k_wave * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
      const float up = round_trip_scalar_to_float<scalar_t>(total_b);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out_row[col] = float_to_scalar<scalar_t>(gate * sig * up);
    }
  }
}

// Exact native rows2-8 sibling of the retained qmicro Q8_1x2 c1 owner.
// Four waves preserve c1's gate/up and K-half ownership. Each block owns eight
// output columns and carries independent row accumulators, so one qmicro weight
// traversal serves all rows without changing any row's dp4a/FMA/reduction tree.
// ROW_TILE is instantiated for 2..8 so any small decode batch (c<=8) shares one
// weight traversal; c>8 chunks into <=8-row groups at the dispatch site.
template <typename scalar_t, int ROW_TILE>
__global__ __launch_bounds__(128, 1)
void q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t q8_blocks_per_row) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_lane = tid & 31;
  const bool up_plane = wave_id >= 2;
  const int k_wave = wave_id & 1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const int64_t q8_plane_stride = ROW_TILE * q8_blocks_per_row;

  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int col = 0; col < kTileCols; ++col) {
      acc[row][col] = 0.0f;
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        expert_tile_ptr<Q4_QMICRO_T16_BLOCK_BYTES>(
            up_plane ? tiles_b : tiles_a,
            0,
            out_tile,
            blk,
            out_tiles,
            blocks_per_row);
    const int group_in_block = k_wave * 32 + wave_lane;
    const int sb = group_in_block >> 3;
    const int lane_in_group = wave_lane & 7;
    const int lane4 = lane_in_group << 2;
    constexpr int META_QUARTETS = T16_COLS / 4;
    constexpr int MIN_RECORD_BASE = Q4_SUBBLOCKS * META_QUARTETS;
    const int record =
        (lane_in_group >= 4 ? MIN_RECORD_BASE : 0) +
        sb * META_QUARTETS + (lane_in_group & 3);
    const uint32_t lane_metadata = qmicro_unpack_record(
        tile + Q4_QMICRO_T16_META_OFFSET + record * 3);
    const int64_t q8_index = blk * Q4_SUBBLOCKS + sb;

    uint32_t x_pack0[ROW_TILE];
    uint32_t x_pack1[ROW_TILE];
    int q8_sum0[ROW_TILE];
    int q8_sum1[ROW_TILE];
    float xd0[ROW_TILE];
    float xd1[ROW_TILE];
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
      const int64_t q8_row_offset = row * q8_blocks_per_row;
      const gguf_q8_1_block* __restrict__ xb0 =
          xq + q8_row_offset + q8_index;
      const gguf_q8_1_block* __restrict__ xb1 =
          xq + q8_plane_stride + q8_row_offset + q8_index;
      x_pack0[row] = *reinterpret_cast<const uint32_t*>(xb0->qs + lane4);
      x_pack1[row] = *reinterpret_cast<const uint32_t*>(xb1->qs + lane4);
      q8_sum0[row] = t16_dp4a_u8_s8(0x01010101u, x_pack0[row], 0);
      q8_sum1[row] = t16_dp4a_u8_s8(0x01010101u, x_pack1[row], 0);
      xd0[row] = static_cast<float>(xb0->d);
      xd1[row] = static_cast<float>(xb1->d);
    }

    const int group_lane_base = wave_lane & ~7;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      const int col = col_base + local_col;
      const int quartet = col >> 2;
      const int col4 = col & 3;
      const uint32_t scale_word =
          __shfl(lane_metadata, group_lane_base + quartet, 32);
      const uint32_t min_word =
          __shfl(lane_metadata, group_lane_base + 4 + quartet, 32);
      const float d = fp16_bytes_to_float(
          tile + Q4_QMICRO_T16_D_OFFSET + col * 2);
      const float dmin = fp16_bytes_to_float(
          tile + Q4_QMICRO_T16_DMIN_OFFSET + col * 2);
      const float scale = static_cast<float>(
          (scale_word >> (8 * col4)) & 0xFFu);
      const float minv = static_cast<float>(
          (min_word >> (8 * col4)) & 0xFFu);
      const uint32_t weight_pack =
          q4_qmicro_t16_pack4(tile, sb, lane4, col);
      const float weight_scale = d * scale;
      const float weight_min = dmin * minv;
#pragma unroll
      for (int row = 0; row < ROW_TILE; ++row) {
        const int dot0 = t16_dp4a_u8_s8(weight_pack, x_pack0[row], 0);
        const int dot1 = t16_dp4a_u8_s8(weight_pack, x_pack1[row], 0);
        acc[row][local_col] += xd0[row] *
            (weight_scale * static_cast<float>(dot0) -
             weight_min * static_cast<float>(q8_sum0[row]));
        acc[row][local_col] += xd1[row] *
            (weight_scale * static_cast<float>(dot1) -
             weight_min * static_cast<float>(q8_sum1[row]));
      }
    }
  }

#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
    for (int col = 0; col < kTileCols; ++col) {
      GGUF_T16_WAVE_REDUCE(acc[row][col]);
    }
  }

  __shared__ float xchg[2][2][ROW_TILE][kTileCols];
  if (wave_lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
#pragma unroll
      for (int col = 0; col < kTileCols; ++col) {
        xchg[up_plane ? 1 : 0][k_wave][row][col] = acc[row][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_TILE * kTileCols) {
    const int row = tid / kTileCols;
    const int local_col = tid % kTileCols;
    float total_a = 0.0f;
    float total_b = 0.0f;
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      total_a += xchg[0][group][row][local_col];
      total_b += xchg[1][group][row][local_col];
    }
    const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
    const float up = round_trip_scalar_to_float<scalar_t>(total_b);
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features +
        out_group * kTileCols + local_col] =
        float_to_scalar<scalar_t>(gate * sig * up);
  }
}

template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false,
    bool dense_single_expert = false>
__global__ __launch_bounds__(128, 2) void qk_t16_selected_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  static_assert(!qmicro_planar || qmicro,
                "planar qmicro requires qmicro layout");
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  int64_t expert_id;
  if constexpr (dense_single_expert) {
    expert_id = 0;
  } else {
    expert_id = selected[row];
  }
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      if constexpr (qtype == 4) {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv = t16_q4_weight(tile, k_in_block, col);
          acc[col] += xv * wv;
        }
      } else if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc[col] += xv * (scale * q - minv);
        }
      } else {
        const int g16 = k_in_block >> 4;
        const int8_t* __restrict__ scales =
            reinterpret_cast<const int8_t*>(
                tile + Q6_T16_SCALE_OFFSET);
        if constexpr (qmicro) {
          const int subblock = k_in_block >> 5;
          const int pack = (k_in_block & 31) >> 2;
          const int k4 = k_in_block & 3;
#pragma unroll
          for (int col_quartet = 0;
               col_quartet < T16_COLS / 4;
               ++col_quartet) {
            const int record_index =
                (subblock * (T16_COLS / 4) + col_quartet) * 8 +
                pack;
            const uint8_t* __restrict__ record =
                tile + Q6_T16_QMICRO_OFFSET +
                record_index * Q6_T16_QMICRO_RECORD_BYTES;
            uint16_t low_pair;
            if constexpr (qmicro_planar) {
              low_pair =
                  static_cast<uint16_t>(record[k4]) |
                  (static_cast<uint16_t>(record[4 + k4]) << 8);
            } else {
              low_pair =
                  reinterpret_cast<const uint16_t*>(record)[k4];
            }
            const uint8_t high_packed = record[8 + k4];
#pragma unroll
            for (int col4 = 0; col4 < 4; ++col4) {
              const int col = col_quartet * 4 + col4;
              const uint8_t low_byte =
                  static_cast<uint8_t>(
                      low_pair >> (8 * (col4 >> 1)));
              const uint8_t low =
                  (col4 & 1) ? (low_byte >> 4)
                             : (low_byte & 0x0Fu);
              const uint8_t high =
                  (high_packed >> (2 * col4)) & 0x03u;
              const int q =
                  static_cast<int>(low | (high << 4)) - 32;
              const float d = fp16_bytes_to_float(
                  tile + Q6_T16_D_OFFSET + col * 2);
              const float scale =
                  d * static_cast<float>(
                          scales[g16 * T16_COLS + col]);
              acc[col] += xv * (scale * static_cast<float>(q));
            }
          }
        } else {
#pragma unroll
          for (int col = 0; col < T16_COLS; ++col) {
            const float d = fp16_bytes_to_float(
                tile + Q6_T16_D_OFFSET + col * 2);
            const float scale =
                d * static_cast<float>(
                        scales[g16 * T16_COLS + col]);
            const float q = static_cast<float>(
                t16_q6_quant(tile, k_in_block, col));
            acc[col] += xv * (scale * q);
          }
        }
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
    scalar_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}

// Exact dense Q5T16 c=1 output-ownership screen. Each workgroup owns one
// eight-column half of a resident 16-column tile while preserving production
// K ownership, FMA order, wave32 tree, serial wave-0..3 reduction, and BF16
// store for every output.
template <typename scalar_t>
__global__ __launch_bounds__(128, 4)
void q5_k_t16_dense_tile8_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc[local_col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q5_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x[k_base + k_in_block]);
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        const int col = col_base + local_col;
        const float d = fp16_bytes_to_float(
            tile + Q5_T16_D_OFFSET + col * 2);
        const float dmin = fp16_bytes_to_float(
            tile + Q5_T16_DMIN_OFFSET + col * 2);
        const float scale = d * static_cast<float>(
            tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
        const float minv = dmin * static_cast<float>(
            tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
        const float q = static_cast<float>(
            t16_q5_quant(tile, sb, lane32, col));
        acc[local_col] += xv * (scale * q - minv);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }

  __shared__ float xchg[4 * kTileCols];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg[wave * kTileCols + local_col] = acc[local_col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_group_row =
        out + out_tile * T16_COLS + col_base;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * kTileCols + local_col];
      }
      out_group_row[local_col] = float_to_scalar<scalar_t>(total);
    }
  }
}

// Exact Qwen c=1/top8 selected-down ownership screen. Split each resident
// 16-column Q5/Q6 T16 tile into independent four- or eight-column workgroups
// while retaining the production local128 K ownership, FMA order, four-wave
// shuffle tree, serial wave-0..3 reduction, and BF16 store per output column.
template <typename scalar_t, int qtype, int tile_cols>
__global__ __launch_bounds__(128, 4)
void qk_t16_selected_qwen_tile_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t num_experts) {
  static_assert(qtype == 5 || qtype == 6, "qtype must be 5 or 6");
  static_assert(tile_cols == 4 || tile_cols == 8, "tile_cols must be 4 or 8");
  constexpr int kRows = 8;
  constexpr int kInFeatures = 512;
  constexpr int kOutFeatures = 2048;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  constexpr int kTileCols = tile_cols;
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int64_t out_group = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= kRows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;

  float acc[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc[local_col] = 0.0f;
  }

  for (int64_t blk = 0; blk < kBlocksPerRow; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(
          x[row * kInFeatures + k_base + k_in_block]);
      if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int local_col = 0; local_col < kTileCols; ++local_col) {
          const int col = col_base + local_col;
          const float d = fp16_bytes_to_float(
              tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(
              tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(
              tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(
              tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(
              t16_q5_quant(tile, sb, lane32, col));
          acc[local_col] += xv * (scale * q - minv);
        }
      } else {
        const int g16 = k_in_block >> 4;
        const int8_t* __restrict__ scales =
            reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
#pragma unroll
        for (int local_col = 0; local_col < kTileCols; ++local_col) {
          const int col = col_base + local_col;
          const float d = fp16_bytes_to_float(
              tile + Q6_T16_D_OFFSET + col * 2);
          const float scale = d * static_cast<float>(
              scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(
              t16_q6_quant(tile, k_in_block, col));
          acc[local_col] += xv * (scale * q);
        }
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }

  __shared__ float xchg[4 * kTileCols];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      xchg[wave * kTileCols + local_col] = acc[local_col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row =
        out + row * kOutFeatures + out_tile * T16_COLS + col_base;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * kTileCols + local_col];
      }
      out_row[local_col] = float_to_scalar<scalar_t>(total);
    }
  }
}

// Exact Laguna c=1 down specialization: ten selected expert lanes,
// K1024 -> N3072. Q4 and planar-qmicro Q6 instantiate the same per-thread K
// ownership, arithmetic, wave tree, wave-0..3 reduction, and BF16 boundary as
// qk_t16_selected_direct_gemv_kernel.
template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false,
    bool parallel_tail = false,
    bool weighted_tail = false,
    bool pair_coeff_payload = false>
__global__ __launch_bounds__(128, 4)
void qk_t16_selected_natural_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    const float* __restrict__ routing_weights,
    scalar_t* __restrict__ routed_out,
    int32_t* __restrict__ completion_counter,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 6,
                "natural Laguna down supports Q4 or Q6");
  static_assert(!qmicro_planar || qmicro,
                "planar qmicro requires qmicro layout");
  static_assert(!weighted_tail || parallel_tail,
                "weighted natural tail requires parallel output stores");
  static_assert(!pair_coeff_payload || qtype == 4,
                "paired coefficients apply only to natural Q4");
  constexpr int kRows = 10;
  constexpr int kInFeatures = 1024;
  constexpr int kOutFeatures = 3072;
  constexpr int kBlocksPerRow = kInFeatures / QK_K;
  constexpr int kOutTiles = kOutFeatures / T16_COLS;
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= kRows) return;
  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  if constexpr (!weighted_tail) {
    if (!valid_expert) return;
  }
  const scalar_t* __restrict__ x_row = x + row * kInFeatures;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < kBlocksPerRow && valid_expert; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(
          tiles, expert_id, out_tile, blk, kOutTiles, kBlocksPerRow);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const float xv =
          scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      if constexpr (qtype == 4) {
        if constexpr (pair_coeff_payload) {
          const int sb = k_in_block >> 5;
          const int lane32 = k_in_block & 31;
#pragma unroll
          for (int pair = 0; pair < T16_COLS / 2; ++pair) {
            const int col0 = pair * 2;
            const int col1 = col0 + 1;
            const uint8_t packed =
                tile[Q4_T16_Q_OFFSET +
                     (sb * Q4_SUBBLOCK + lane32) *
                         (T16_COLS / 2) +
                     pair];
            const uint32_t d = *reinterpret_cast<const uint32_t*>(
                tile + Q4_T16_D_OFFSET + col0 * 2);
            const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
                tile + Q4_T16_DMIN_OFFSET + col0 * 2);
            const uint16_t scale = *reinterpret_cast<const uint16_t*>(
                tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
            const uint16_t minv = *reinterpret_cast<const uint16_t*>(
                tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
            const float wv0 =
                fp16_bits_to_float(static_cast<uint16_t>(d)) *
                    static_cast<float>(scale & 0xFFu) *
                    static_cast<float>(packed & 0x0Fu) -
                fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
                    static_cast<float>(minv & 0xFFu);
            const float wv1 =
                fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
                    static_cast<float>(scale >> 8) *
                    static_cast<float>(packed >> 4) -
                fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
                    static_cast<float>(minv >> 8);
            acc[col0] += xv * wv0;
            acc[col1] += xv * wv1;
          }
        } else {
#pragma unroll
          for (int col = 0; col < T16_COLS; ++col) {
            const float wv =
                t16_q4_weight(tile, k_in_block, col);
            acc[col] += xv * wv;
          }
        }
      } else {
        const int g16 = k_in_block >> 4;
        const int8_t* __restrict__ scales =
            reinterpret_cast<const int8_t*>(
                tile + Q6_T16_SCALE_OFFSET);
        const int subblock = k_in_block >> 5;
        const int pack = (k_in_block & 31) >> 2;
        const int k4 = k_in_block & 3;
#pragma unroll
        for (int col_quartet = 0;
             col_quartet < T16_COLS / 4;
             ++col_quartet) {
          const int record_index =
              (subblock * (T16_COLS / 4) + col_quartet) * 8 +
              pack;
          const uint8_t* __restrict__ record =
              tile + Q6_T16_QMICRO_OFFSET +
              record_index * Q6_T16_QMICRO_RECORD_BYTES;
          uint16_t low_pair;
          if constexpr (qmicro_planar) {
            low_pair =
                static_cast<uint16_t>(record[k4]) |
                (static_cast<uint16_t>(record[4 + k4]) << 8);
          } else {
            low_pair =
                reinterpret_cast<const uint16_t*>(record)[k4];
          }
          const uint8_t high_packed = record[8 + k4];
#pragma unroll
          for (int col4 = 0; col4 < 4; ++col4) {
            const int col = col_quartet * 4 + col4;
            const uint8_t low_byte =
                static_cast<uint8_t>(
                    low_pair >> (8 * (col4 >> 1)));
            const uint8_t low =
                (col4 & 1) ? (low_byte >> 4)
                           : (low_byte & 0x0Fu);
            const uint8_t high =
                (high_packed >> (2 * col4)) & 0x03u;
            const int q =
                static_cast<int>(low | (high << 4)) - 32;
            const float d = fp16_bytes_to_float(
                tile + Q6_T16_D_OFFSET + col * 2);
            const float scale =
                d * static_cast<float>(
                        scales[g16 * T16_COLS + col]);
            acc[col] += xv * (scale * static_cast<float>(q));
          }
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc[col]);
  }

  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[sub_id * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if constexpr (parallel_tail) {
    if (tid < T16_COLS) {
      const int col = tid;
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      scalar_t* __restrict__ out_row =
          out + row * kOutFeatures + out_tile * T16_COLS;
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  } else if (tid == 0) {
    scalar_t* __restrict__ out_row =
        out + row * kOutFeatures + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }

  if constexpr (weighted_tail) {
    __syncthreads();
    __shared__ int last_producer;
    if (tid == 0) {
      __threadfence();
      const int prior = atomicAdd(completion_counter + out_tile, 1);
      last_producer = prior + 1 == kRows ? 1 : 0;
    }
    __syncthreads();
    if (last_producer == 0) {
      return;
    }
    __threadfence();

    if (tid < T16_COLS) {
      const int col = out_tile * T16_COLS + tid;
      float routed = 0.0f;
#pragma unroll
      for (int route = 0; route < kRows; ++route) {
        routed = fmaf(
            scalar_to_float<scalar_t>(
                out[route * kOutFeatures + col]),
            routing_weights[route],
            routed);
      }
      routed_out[col] = float_to_scalar<scalar_t>(routed);
    }
    __syncthreads();
    if (tid == 0) {
      atomicExch(completion_counter + out_tile, 0);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void q5_k_t16_selected_pairreuse_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  __shared__ uint32_t expert_match_words[2];
  const bool lane_matches = tid < rows && selected[tid] == expert_id;
  const uint32_t wave_matches = static_cast<uint32_t>(__ballot(lane_matches));
  const int wave_id = tid >> 5;
  if ((tid & 31) == 0 && wave_id < 2) {
    expert_match_words[wave_id] = wave_matches;
  }
  __syncthreads();
  const uint64_t expert_matches = static_cast<uint64_t>(expert_match_words[0]) |
                                  (static_cast<uint64_t>(expert_match_words[1]) << 32);
  const uint64_t prior_mask = row == 0 ? 0 : ((uint64_t{1} << row) - 1);
  const int prior_matches = __popcll(expert_matches & prior_mask);
  if ((prior_matches & 1) != 0) return;
  const uint64_t later_mask = row == 63 ? 0 : (~uint64_t{0} << (row + 1));
  const uint64_t later_matches = expert_matches & later_mask;
  const int64_t paired_row = later_matches == 0
                                 ? -1
                                 : (__ffsll(static_cast<unsigned long long>(later_matches)) - 1);

  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t paired_x_row_idx = paired_row >= 0 ? paired_row / lanes_per_x_row : -1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;
  const scalar_t* __restrict__ paired_x_row =
      paired_row >= 0 ? x + paired_x_row_idx * in_features : nullptr;

  float acc0[T16_COLS];
  float acc1[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(
        tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      if (paired_row >= 0) {
        const float xv1 = scalar_to_float<scalar_t>(paired_x_row[k_base + k_in_block]);
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc0[col] += xv0 * (scale * q - minv);
          acc1[col] += xv1 * (scale * q - minv);
        }
      } else {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc0[col] += xv0 * (scale * q - minv);
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
  if (paired_row >= 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);
  }

  __shared__ float xchg0[4 * T16_COLS];
  __shared__ float xchg1[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      if (paired_row >= 0) xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row0 = out + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row1 =
        paired_row >= 0 ? out + paired_row * out_features + out_tile * T16_COLS : nullptr;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        if (paired_row >= 0) total1 += xchg1[g * T16_COLS + col];
      }
      out_row0[col] = float_to_scalar<scalar_t>(total0);
      if (paired_row >= 0) out_row1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 2) void q6_k_t16_selected_pairreuse_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  __shared__ uint32_t expert_match_words[2];
  const bool lane_matches = tid < rows && selected[tid] == expert_id;
  const uint32_t wave_matches = static_cast<uint32_t>(__ballot(lane_matches));
  const int wave_id = tid >> 5;
  if ((tid & 31) == 0 && wave_id < 2) {
    expert_match_words[wave_id] = wave_matches;
  }
  __syncthreads();
  const uint64_t expert_matches = static_cast<uint64_t>(expert_match_words[0]) |
                                  (static_cast<uint64_t>(expert_match_words[1]) << 32);
  const uint64_t prior_mask = row == 0 ? 0 : ((uint64_t{1} << row) - 1);
  const int prior_matches = __popcll(expert_matches & prior_mask);
  if ((prior_matches & 1) != 0) return;
  const uint64_t later_mask = row == 63 ? 0 : (~uint64_t{0} << (row + 1));
  const uint64_t later_matches = expert_matches & later_mask;
  const int64_t paired_row = later_matches == 0
                                 ? -1
                                 : (__ffsll(static_cast<unsigned long long>(later_matches)) - 1);

  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t paired_x_row_idx = paired_row >= 0 ? paired_row / lanes_per_x_row : -1;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;
  const scalar_t* __restrict__ paired_x_row =
      paired_row >= 0 ? x + paired_x_row_idx * in_features : nullptr;

  float acc0[T16_COLS];
  float acc1[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc0[col] = 0.0f;
    acc1[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(
        tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv0 = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      const int g16 = k_in_block >> 4;
      if (paired_row >= 0) {
        const float xv1 = scalar_to_float<scalar_t>(paired_x_row[k_base + k_in_block]);
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales =
              reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          acc0[col] += xv0 * (scale * q);
          acc1[col] += xv1 * (scale * q);
        }
      } else {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales =
              reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          acc0[col] += xv0 * (scale * q);
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc0[col]);
  if (paired_row >= 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc1[col]);
  }

  __shared__ float xchg0[4 * T16_COLS];
  __shared__ float xchg1[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg0[sub_id * T16_COLS + col] = acc0[col];
      if (paired_row >= 0) xchg1[sub_id * T16_COLS + col] = acc1[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row0 = out + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row1 =
        paired_row >= 0 ? out + paired_row * out_features + out_tile * T16_COLS : nullptr;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total0 = 0.0f;
      float total1 = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total0 += xchg0[g * T16_COLS + col];
        if (paired_row >= 0) total1 += xchg1[g * T16_COLS + col];
      }
      out_row0[col] = float_to_scalar<scalar_t>(total0);
      if (paired_row >= 0) out_row1[col] = float_to_scalar<scalar_t>(total1);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(128, 4) void qk_t16_selected_q8_1_dp4a_direct_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t q8_blocks_per_row) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const gguf_q8_1_block* __restrict__ xq_row = xq + x_row_idx * q8_blocks_per_row;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  const int64_t groups4 = in_features >> 2;
  for (int64_t group = tid; group < groups4; group += blockDim.x) {
    const int64_t k4 = group << 2;
    const int64_t blk = k4 / QK_K;
    const int within = static_cast<int>(k4 - blk * QK_K);
    const int lane4 = within & 31;
    const int q8_index = static_cast<int>(blk * Q4_SUBBLOCKS + (within >> 5));
    const gguf_q8_1_block* __restrict__ xb = xq_row + q8_index;
    const uint32_t x_pack = *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
    const int q8_sum = t16_dp4a_u8_s8(0x01010101u, x_pack, 0);
    const float xd = static_cast<float>(xb->d);

    const uint8_t* __restrict__ tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(
        tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int sb = within >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
      const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
      const float scale = static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
      const float minv = static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
      const int dot = t16_dp4a_u8_s8(t16_q5_pack4(tile, sb, lane4, col), x_pack, 0);
      acc[col] += xd * (d * scale * static_cast<float>(dot) -
                        dmin * minv * static_cast<float>(q8_sum));
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
    scalar_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}

template <typename scalar_t>
int launch_q4_dual(const void* x,
                   const void* expert_start_compact,
                   const void* tiles_a,
                   const void* tiles_b,
                   void* out,
                   int64_t compact_rows,
                   int64_t in_features,
                   int64_t out_features_a,
                   int64_t out_features_b,
                   int64_t num_experts,
                   hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles_a || !tiles_b || !out || compact_rows <= 0 ||
      in_features <= 0 || out_features_a <= 0 || out_features_b <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features_a % T16_COLS) != 0 ||
      (out_features_b % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS),
                  static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(expert_start_compact),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out), compact_rows, in_features,
                     out_features_a, out_features_b, num_experts);
  return hipGetLastError();
}

template <typename scalar_t, int qtype>
int launch_qk_single(const void* x,
                     const void* expert_start_compact,
                     const void* tiles,
                     void* out,
                     int64_t compact_rows,
                     int64_t in_features,
                     int64_t out_features,
                     int64_t num_experts,
                     hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles || !out || compact_rows <= 0 || in_features <= 0 ||
      out_features <= 0 || num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL((qk_t16_selected_gemv_kernel<scalar_t, qtype>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(expert_start_compact),
                     reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
                     compact_rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_pairreuse_compact(const void* x,
                                     const void* expert_start_compact,
                                     const void* tiles_a,
                                     const void* tiles_b,
                                     void* out,
                                     int64_t compact_rows,
                                     int64_t in_features,
                                     int64_t out_features_a,
                                     int64_t out_features_b,
                                     int64_t num_experts,
                                     hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles_a || !tiles_b || !out ||
      compact_rows <= 0 || in_features <= 0 || out_features_a <= 0 ||
      out_features_b <= 0 || num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS),
      static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_pairreuse_compact_gemv_kernel<scalar_t>),
      grid, block, 0, stream, reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(expert_start_compact),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b), reinterpret_cast<scalar_t*>(out),
      compact_rows, in_features, out_features_a, out_features_b, num_experts);
  return hipGetLastError();
}

template <typename scalar_t, int qtype>
int launch_qk_pairreuse_compact(const void* x,
                                const void* expert_start_compact,
                                const void* tiles,
                                void* out,
                                int64_t compact_rows,
                                int64_t in_features,
                                int64_t out_features,
                                int64_t num_experts,
                                hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles || !out || compact_rows <= 0 ||
      in_features <= 0 || out_features <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL(
      (qk_t16_selected_pairreuse_compact_gemv_kernel<scalar_t, qtype>),
      grid, block, 0, stream, reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(expert_start_compact),
      reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
      compact_rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_grouped_smallm(const void* x,
                                  const void* expert_start_compact,
                                  const void* active_experts,
                                  const void* active_count,
                                  const void* tiles_a,
                                  const void* tiles_b,
                                  void* out_a,
                                  void* out_b,
                                  int64_t compact_rows,
                                  int64_t in_features,
                                  int64_t out_features,
                                  int64_t num_experts,
                                  hipStream_t stream) {
  if (!x || !expert_start_compact || !active_experts || !active_count ||
      !tiles_a || !tiles_b || !out_a || !out_b || compact_rows <= 0 ||
      in_features <= 0 || out_features <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const int64_t total_tasks =
      (out_features / GROUPED_SMALLM_COLS) * num_experts;
  const int64_t grid_blocks =
      total_tasks < GROUPED_SMALLM_PERSISTENT_BLOCKS
          ? total_tasks
          : GROUPED_SMALLM_PERSISTENT_BLOCKS;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(grid_blocks));
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_grouped_smallm_kernel<scalar_t>), grid, block, 0,
      stream, reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(expert_start_compact),
      reinterpret_cast<const int64_t*>(active_experts),
      reinterpret_cast<const int64_t*>(active_count),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out_a), reinterpret_cast<scalar_t*>(out_b),
      compact_rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false>
int launch_qk_grouped_smallm(const void* x,
                             const void* expert_start_compact,
                             const void* active_experts,
                             const void* active_count,
                             const void* tiles,
                             void* out,
                             int64_t compact_rows,
                             int64_t in_features,
                             int64_t out_features,
                             int64_t num_experts,
                             hipStream_t stream) {
  if (!x || !expert_start_compact || !active_experts || !active_count ||
      !tiles || !out || compact_rows <= 0 || in_features <= 0 ||
      out_features <= 0 || num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const int64_t total_tasks =
      (out_features / GROUPED_SMALLM_COLS) * num_experts;
  const int64_t grid_blocks =
      total_tasks < GROUPED_SMALLM_PERSISTENT_BLOCKS
          ? total_tasks
          : GROUPED_SMALLM_PERSISTENT_BLOCKS;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(grid_blocks));
  hipLaunchKernelGGL(
      (qk_t16_selected_grouped_smallm_kernel<
          scalar_t, qtype, qmicro, qmicro_planar>), grid,
      block, 0, stream, reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(expert_start_compact),
      reinterpret_cast<const int64_t*>(active_experts),
      reinterpret_cast<const int64_t*>(active_count),
      reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
      compact_rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_direct(const void* x,
                          const void* selected,
                          const void* tiles_a,
                          const void* tiles_b,
                          void* out_a,
                          void* out_b,
                          int64_t x_rows,
                          int64_t rows,
                          int64_t in_features,
                          int64_t out_features,
                          int64_t num_experts,
                          hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b || x_rows <= 0 || rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 || (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  // Match the legacy raw-GGUF selected dual GEMV reduction topology (128
  // threads / 4 waves) so replacement-layout Q4 gate/up decode remains
  // bitwise-aligned with the P9.E2 row-GEMV reference path.
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_direct_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a), reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out_a), reinterpret_cast<scalar_t*>(out_b),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_natural(const void* x,
                           const void* selected,
                           const void* tiles_a,
                           const void* tiles_b,
                           void* out_a,
                           void* out_b,
                           int64_t x_rows,
                           int64_t rows,
                           int64_t in_features,
                           int64_t out_features,
                           int64_t num_experts,
                           hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / T16_COLS, 10);
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_natural_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out_a),
      reinterpret_cast<scalar_t*>(out_b),
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_natural_tile8(const void* x,
                                 const void* selected,
                                 const void* tiles_a,
                                 const void* tiles_b,
                                 void* out_a,
                                 void* out_b,
                                 int64_t x_rows,
                                 int64_t rows,
                                 int64_t in_features,
                                 int64_t out_features,
                                 int64_t num_experts,
                                 hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, 10);
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_natural_tile8_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out_a),
      reinterpret_cast<scalar_t*>(out_b),
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_natural_tile8_parallel(const void* x,
                                          const void* selected,
                                          const void* tiles_a,
                                          const void* tiles_b,
                                          void* out_a,
                                          void* out_b,
                                          int64_t x_rows,
                                          int64_t rows,
                                          int64_t in_features,
                                          int64_t out_features,
                                          int64_t num_experts,
                                          hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, 10);
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_natural_tile8_gemv_kernel<scalar_t, true>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out_a),
      reinterpret_cast<scalar_t*>(out_b),
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_natural_tile8_parallel_silu_pairq(
    const void* x,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, 10);
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_natural_tile8_gemv_kernel<
          scalar_t, true, true, true>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      nullptr,
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_natural_tile8_parallel_silu_paircoeff(
    const void* x,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, 10);
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_natural_tile8_gemv_kernel<
          scalar_t, true, true, true, true>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      nullptr,
      num_experts);
  return hipGetLastError();
}

int launch_q4_dual_natural_tile8_parallel_silu_halfdot(
    const void* x,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out ||
      x_rows != 1 || rows != 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, 10);
  hipLaunchKernelGGL(
      q4_k_t16_selected_dual_natural_tile8_halfdot_silu_gemv_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<uint16_t*>(out),
      num_experts);
  return hipGetLastError();
}

int launch_q4_dual_interleaved_natural_tile8_parallel_silu_halfdot(
    const void* x,
    const void* selected,
    const void* tiles_dual,
    const void*,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_dual || !out ||
      x_rows <= 0 || rows != x_rows * 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, static_cast<uint32_t>(rows));
  hipLaunchKernelGGL(
      q4_k_t16_selected_dual_interleaved_natural_tile8_halfdot_silu_gemv_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_dual),
      reinterpret_cast<uint16_t*>(out),
      x_rows,
      rows,
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_interleaved_natural_tile8_parallel_silu(
    const void* x,
    const void* selected,
    const void* tiles_dual,
    const void*,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_dual || !out ||
      x_rows <= 0 || rows != x_rows * 10 || in_features != 3072 ||
      out_features != 1024 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(1024 / 8, static_cast<uint32_t>(rows));
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_interleaved_natural_tile8_silu_gemv_kernel<
          scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_dual),
      reinterpret_cast<scalar_t*>(out),
      x_rows,
      rows,
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_dual_local32_silu(
    const void* x,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_dual_local32_silu_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_single_local32(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_single_local32_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_single_local32_residual(
    const void* x,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !residual || !out || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_single_local32_residual_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<const scalar_t*>(residual),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q5_dense_direct(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (qk_t16_selected_direct_gemv_kernel<
          scalar_t, 5, false, false, true>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      nullptr,
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      rows,
      rows,
      in_features,
      out_features,
      1);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q5_dense_tile8(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  hipLaunchKernelGGL(
      (q5_k_t16_dense_tile8_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename scalar_t, int TILE_COLS>
int launch_q5_dense_rowtile(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  static_assert(TILE_COLS == 4 || TILE_COLS == 8,
                "Q5 rowtile columns must be 4 or 8");
  if (!x || !tiles || !out || rows < 2 || rows > 8 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / TILE_COLS));
#define Q5_T16_DENSE_ROWTILE_CASE(N)                                        \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q5_k_t16_dense_rowtile_gemv_kernel<scalar_t, N, TILE_COLS>),        \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q5_T16_DENSE_ROWTILE_CASE(2);
    Q5_T16_DENSE_ROWTILE_CASE(3);
    Q5_T16_DENSE_ROWTILE_CASE(4);
    Q5_T16_DENSE_ROWTILE_CASE(5);
    Q5_T16_DENSE_ROWTILE_CASE(6);
    Q5_T16_DENSE_ROWTILE_CASE(7);
    Q5_T16_DENSE_ROWTILE_CASE(8);
    default:
      return hipErrorInvalidValue;
  }
#undef Q5_T16_DENSE_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q5_dense_rowtile_grouped_rows6(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 12 || (rows % 6) != 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  constexpr int kTileCols = 4;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / kTileCols),
      static_cast<unsigned int>(rows / 6));
  hipLaunchKernelGGL(
      (q5_k_t16_dense_rowtile_grouped_gemv_kernel<scalar_t, 6, kTileCols>),
      grid, block, 0, stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out), in_features, out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_rowtile(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 2 || rows > 8 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_T16_DENSE_ROWTILE_CASE(N)                                        \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_t16_dense_rowtile_gemv_kernel<scalar_t, N>),                   \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_T16_DENSE_ROWTILE_CASE(2);
    Q4_T16_DENSE_ROWTILE_CASE(3);
    Q4_T16_DENSE_ROWTILE_CASE(4);
    Q4_T16_DENSE_ROWTILE_CASE(5);
    Q4_T16_DENSE_ROWTILE_CASE(6);
    Q4_T16_DENSE_ROWTILE_CASE(7);
    Q4_T16_DENSE_ROWTILE_CASE(8);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_T16_DENSE_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_rowtile16_w2(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out ||
      (rows != 2 && rows != 3 && rows != 4 && rows != 6 && rows != 8) ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(64);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
#define Q4_T16_DENSE_ROWTILE16_W2_CASE(N)                                   \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_t16_dense_rowtile16_w2_gemv_kernel<scalar_t, N>),              \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_T16_DENSE_ROWTILE16_W2_CASE(2);
    Q4_T16_DENSE_ROWTILE16_W2_CASE(3);
    Q4_T16_DENSE_ROWTILE16_W2_CASE(4);
    Q4_T16_DENSE_ROWTILE16_W2_CASE(6);
    Q4_T16_DENSE_ROWTILE16_W2_CASE(8);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_T16_DENSE_ROWTILE16_W2_CASE
  return hipGetLastError();
}

template <typename scalar_t, int ROW_TILE>
int launch_q4_dense_rowtile16_w2_grouped(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 2 * ROW_TILE ||
      (rows % ROW_TILE) != 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>(rows / ROW_TILE));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_rowtile16_w2_grouped_gemv_kernel<scalar_t, ROW_TILE>),
      grid, block, 0, stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out), in_features, out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_qmicro_dense_rowtile(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 1 || rows > 4 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_QMICRO_T16_DENSE_ROWTILE_CASE(N)                                 \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_qmicro_t16_dense_rowtile_gemv_kernel<scalar_t, N>),            \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_QMICRO_T16_DENSE_ROWTILE_CASE(1);
    Q4_QMICRO_T16_DENSE_ROWTILE_CASE(2);
    Q4_QMICRO_T16_DENSE_ROWTILE_CASE(3);
    Q4_QMICRO_T16_DENSE_ROWTILE_CASE(4);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_QMICRO_T16_DENSE_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_rowtile_residual(
    const void* x,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !residual || !out || rows < 2 || rows > 4 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_T16_DENSE_ROWTILE_RESIDUAL_CASE(N)                               \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_t16_dense_rowtile_residual_gemv_kernel<scalar_t, N>),          \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<const scalar_t*>(residual),                         \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_T16_DENSE_ROWTILE_RESIDUAL_CASE(2);
    Q4_T16_DENSE_ROWTILE_RESIDUAL_CASE(3);
    Q4_T16_DENSE_ROWTILE_RESIDUAL_CASE(4);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_T16_DENSE_ROWTILE_RESIDUAL_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_dual_rowtile_silu(
    const void* x,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out || rows < 2 || rows > 8 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(64);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_T16_DENSE_DUAL_ROWTILE_CASE(N)                                   \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_t16_dense_dual_rowtile_silu_gemv_kernel<scalar_t, N>),         \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles_a),                           \
        reinterpret_cast<const uint8_t*>(tiles_b),                           \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(2);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(3);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(4);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(5);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(6);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(7);
    Q4_T16_DENSE_DUAL_ROWTILE_CASE(8);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_T16_DENSE_DUAL_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_qmicro_dense_dual_rowtile_silu(
    const void* x,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out || rows < 2 || rows > 4 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(64);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_QMICRO_T16_DENSE_DUAL_ROWTILE_CASE(N)                            \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_qmicro_t16_dense_dual_rowtile_silu_gemv_kernel<scalar_t, N>),  \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles_a),                           \
        reinterpret_cast<const uint8_t*>(tiles_b),                           \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_QMICRO_T16_DENSE_DUAL_ROWTILE_CASE(2);
    Q4_QMICRO_T16_DENSE_DUAL_ROWTILE_CASE(3);
    Q4_QMICRO_T16_DENSE_DUAL_ROWTILE_CASE(4);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_QMICRO_T16_DENSE_DUAL_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dense_rowtile_col4(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 1 || rows > 8 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(out_features / 4));
#define Q4_T16_DENSE_ROWTILE_COL4_CASE(N)                                   \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_t16_dense_rowtile_col4_gemv_kernel<scalar_t, N>),              \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const scalar_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<scalar_t*>(out), in_features, out_features);        \
    break
  switch (rows) {
    Q4_T16_DENSE_ROWTILE_COL4_CASE(1);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(2);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(3);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(4);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(5);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(6);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(7);
    Q4_T16_DENSE_ROWTILE_COL4_CASE(8);
    default:
      return hipErrorInvalidValue;
  }
#undef Q4_T16_DENSE_ROWTILE_COL4_CASE
  return hipGetLastError();
}

template <typename scalar_t, int kTileCols>
int launch_q4_dense_dual_interleaved_local32_silu(
    const void* x,
    const void* tiles_dual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles_dual || !out || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(out_features / kTileCols));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_dual_interleaved_local32_silu_gemv_kernel<
          scalar_t, kTileCols>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles_dual),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_pairreuse_direct(const void* x,
                                    const void* selected,
                                    const void* tiles_a,
                                    const void* tiles_b,
                                    void* out_a,
                                    void* out_b,
                                    int64_t x_rows,
                                    int64_t rows,
                                    int64_t in_features,
                                    int64_t out_features,
                                    int64_t num_experts,
                                    hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b || x_rows <= 0 || rows <= 0 ||
      rows > 64 || num_experts <= 0 || in_features <= 0 || out_features <= 0 ||
      (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel<scalar_t>),
                     grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out_a),
                     reinterpret_cast<scalar_t*>(out_b),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t, bool fuse_silu>
int launch_q4_qmicro_dual_direct(
    const void* x,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out_a,
    void* out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a ||
      (!fuse_silu && !out_b) || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || in_features <= 0 || out_features <= 0 ||
      num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (q4_k_qmicro_t16_selected_dual_direct_gemv_kernel<scalar_t, fuse_silu>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out_a),
      reinterpret_cast<scalar_t*>(out_b),
      x_rows,
      rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q5_qmicro_direct(
    const void* x,
    const void* selected,
    const void* tiles,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || in_features <= 0 || out_features <= 0 ||
      num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int tile_cols = 8;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / tile_cols),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (q5_k_qmicro_t16_selected_gemv_kernel<scalar_t, tile_cols>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q4_dual_silu_direct(const void* x,
                                const void* selected,
                                const void* tiles_a,
                                const void* tiles_b,
                                void* out,
                                int64_t x_rows,
                                int64_t rows,
                                int64_t in_features,
                                int64_t out_features,
                                int64_t num_experts,
                                hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || in_features <= 0 || out_features <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_silu_direct_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a), reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out), x_rows, rows, in_features, out_features,
                     num_experts);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool fuse_silu>
int launch_q4_dual_q8_1_dp4a_direct(const void* xq,
                                    const void* selected,
                                    const void* tiles_a,
                                    const void* tiles_b,
                                    void* out_a,
                                    void* out_b,
                                    int64_t x_rows,
                                    int64_t rows,
                                    int64_t in_features,
                                    int64_t out_features,
                                    int64_t num_experts,
                                    hipStream_t stream) {
  if (!xq || !selected || !tiles_a || !tiles_b || !out_a || (!fuse_silu && !out_b) ||
      x_rows <= 0 || rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      out_features <= 0 || (rows % x_rows) != 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(selected_t16_dp4a_threads());
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_q8_1_dp4a_direct_gemv_kernel<scalar_t, fuse_silu>),
                     grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out_a),
                     reinterpret_cast<scalar_t*>(out_b),
                     x_rows, rows, in_features, out_features, num_experts, q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}


template <typename scalar_t>
int launch_q4_dual_q8_1x2_dp4a_silu_direct(
    const void* xq,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!xq || !selected || !tiles_a || !tiles_b || !out ||
      x_rows <= 0 || rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 ||
      (rows % x_rows) != 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(selected_t16_dp4a_threads());
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_q8_1x2_dp4a_silu_direct_gemv_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      x_rows,
      rows,
      in_features,
      out_features,
      num_experts,
      q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q4_dense_dual_q8_1x2_dp4a_silu(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!xq || !tiles_a || !tiles_b || !out || rows != 1 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(selected_t16_dp4a_threads());
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      (q4_k_t16_selected_dual_q8_1x2_dp4a_silu_direct_gemv_kernel<
          scalar_t, true>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      nullptr,
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      1,
      1,
      in_features,
      out_features,
      1,
      q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q4_dense_dual_q8_1x2_split_weight_dp4a_silu(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!xq || !tiles_a || !tiles_b || !out || rows != 1 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      (q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_kernel<
          scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features,
      q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q4_qmicro_dense_dual_q8_1x2_split_weight_dp4a_silu(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!xq || !tiles_a || !tiles_b || !out || rows != 1 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      (q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_kernel<
          scalar_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const gguf_q8_1_block*>(xq),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<scalar_t*>(out),
      in_features,
      out_features,
      q8_blocks_per_row);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q4_qmicro_dense_dual_q8_1x2_rowtile8_dp4a_silu(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!xq || !tiles_a || !tiles_b || !out || rows < 2 || rows > 8 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q4_QMICRO_Q8X2_ROWTILE8_CASE(N)                                     \
  case N:                                                                    \
    hipLaunchKernelGGL(                                                       \
        (q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_gemv_kernel<  \
            scalar_t, N>),                                                    \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const gguf_q8_1_block*>(xq),                        \
        reinterpret_cast<const uint8_t*>(tiles_a),                           \
        reinterpret_cast<const uint8_t*>(tiles_b),                           \
        reinterpret_cast<scalar_t*>(out),                                    \
        in_features, out_features, q8_blocks_per_row);                       \
    break
  switch (rows) {
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(2);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(3);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(4);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(5);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(6);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(7);
    Q4_QMICRO_Q8X2_ROWTILE8_CASE(8);
    default:
      return static_cast<int>(hipErrorInvalidValue);
  }
#undef Q4_QMICRO_Q8X2_ROWTILE8_CASE
  return static_cast<int>(hipGetLastError());
}

template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false>
int launch_qk_direct(const void* x,
                     const void* selected,
                     const void* tiles,
                     void* out,
                     int64_t x_rows,
                     int64_t rows,
                     int64_t in_features,
                     int64_t out_features,
                     int64_t num_experts,
                     hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((qk_t16_selected_direct_gemv_kernel<
                         scalar_t, qtype, qmicro, qmicro_planar>),
                     grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t, int qtype, int tile_cols>
int launch_qk_qwen_tile(
    const void* x,
    const void* selected,
    const void* tiles,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows != 8 || rows != 8 ||
      in_features != 512 || out_features != 2048 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  static_assert(tile_cols == 4 || tile_cols == 8, "tile_cols must be 4 or 8");
  const dim3 block(128);
  const dim3 grid(2048 / tile_cols, 8);
  hipLaunchKernelGGL(
      (qk_t16_selected_qwen_tile_gemv_kernel<scalar_t, qtype, tile_cols>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      num_experts);
  return hipGetLastError();
}

template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false,
    bool parallel_tail = false>
int launch_qk_natural(const void* x,
                      const void* selected,
                      const void* tiles,
                      void* out,
                      int64_t x_rows,
                      int64_t rows,
                      int64_t in_features,
                      int64_t out_features,
                      int64_t num_experts,
                      hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows != 10 || rows != 10 ||
      in_features != 1024 || out_features != 3072 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(3072 / T16_COLS, 10);
  hipLaunchKernelGGL(
      (qk_t16_selected_natural_gemv_kernel<
          scalar_t, qtype, qmicro, qmicro_planar, parallel_tail>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      nullptr,
      nullptr,
      nullptr,
      num_experts);
  return hipGetLastError();
}

template <
    typename scalar_t,
    int qtype,
    bool qmicro = false,
    bool qmicro_planar = false,
    bool pair_coeff_payload = false>
int launch_qk_natural_parallel_weighted(
    const void* x,
    const void* selected,
    const void* tiles,
    void* out,
    const void* routing_weights,
    void* routed_out,
    void* completion_counter,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!x || !selected || !tiles || !out || !routing_weights || !routed_out ||
      !completion_counter || x_rows != 10 || rows != 10 ||
      in_features != 1024 || out_features != 3072 || num_experts <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(3072 / T16_COLS, 10);
  hipLaunchKernelGGL(
      (qk_t16_selected_natural_gemv_kernel<
          scalar_t,
          qtype,
          qmicro,
          qmicro_planar,
          true,
          true,
          pair_coeff_payload>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const scalar_t*>(x),
      reinterpret_cast<const int64_t*>(selected),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<scalar_t*>(out),
      reinterpret_cast<const float*>(routing_weights),
      reinterpret_cast<scalar_t*>(routed_out),
      reinterpret_cast<int32_t*>(completion_counter),
      num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q5_pairreuse_direct(const void* x,
                               const void* selected,
                               const void* tiles,
                               void* out,
                               int64_t x_rows,
                               int64_t rows,
                               int64_t in_features,
                               int64_t out_features,
                               int64_t num_experts,
                               hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 || rows > 64 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 ||
      (rows % x_rows) != 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q5_k_t16_selected_pairreuse_direct_gemv_kernel<scalar_t>),
                     grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<scalar_t*>(out),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q6_pairreuse_direct(const void* x,
                               const void* selected,
                               const void* tiles,
                               void* out,
                               int64_t x_rows,
                               int64_t rows,
                               int64_t in_features,
                               int64_t out_features,
                               int64_t num_experts,
                               hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 || rows > 64 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 ||
      (rows % x_rows) != 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q6_k_t16_selected_pairreuse_direct_gemv_kernel<scalar_t>),
                     grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<scalar_t*>(out),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_qk_q8_1_dp4a_direct(const void* xq,
                               const void* selected,
                               const void* tiles,
                               void* out,
                               int64_t x_rows,
                               int64_t rows,
                               int64_t in_features,
                               int64_t out_features,
                               int64_t num_experts,
                               hipStream_t stream) {
  if (!xq || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const dim3 block(selected_t16_q5_dp4a_threads());
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((qk_t16_selected_q8_1_dp4a_direct_gemv_kernel<scalar_t>),
                     grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
                     x_rows, rows, in_features, out_features, num_experts, q8_blocks_per_row);
  return hipGetLastError();
}

}  // namespace


extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_direct<uint16_t>(x, selected, tiles_a, tiles_b, out_a, out_b,
                                         x_rows, rows, in_features, out_features,
                                         num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out_a, void* out_b, int64_t x_rows,
    int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_natural<uint16_t>(
      x, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out_a, void* out_b, int64_t x_rows,
    int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_natural_tile8<uint16_t>(
      x, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out_a, void* out_b, int64_t x_rows,
    int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_natural_tile8_parallel<uint16_t>(
      x, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_pairq_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_natural_tile8_parallel_silu_pairq<uint16_t>(
      x, selected, tiles_a, tiles_b, out, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_natural_tile8_parallel_silu_paircoeff<uint16_t>(
      x, selected, tiles_a, tiles_b, out, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_natural_tile8_parallel_silu_halfdot(
      x, selected, tiles_a, tiles_b, out, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_dual,
    const void* tiles_unused, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_interleaved_natural_tile8_parallel_silu_halfdot(
      x, selected, tiles_dual, tiles_unused, out, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_dual,
    const void* tiles_unused, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_interleaved_natural_tile8_parallel_silu<uint16_t>(
      x, selected, tiles_dual, tiles_unused, out, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_dual_local32_silu_gemv_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_dual_local32_silu<uint16_t>(
      x, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_single_local32_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_single_local32<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_qmicro_t16_dense_single_local32_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  if (rows != 1) return hipErrorInvalidValue;
  return launch_q4_qmicro_dense_rowtile<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_single_local32_gemv_bf16_residual_bf16_out(
    const void* x, const void* tiles, const void* residual, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_single_local32_residual<uint16_t>(
      x, tiles, residual, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_dual_q8_1x2_dp4a_silu_gemv_bf16_bf16_out(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_dual_q8_1x2_dp4a_silu<uint16_t>(
      xq, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_bf16_bf16_out(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_dual_q8_1x2_split_weight_dp4a_silu<uint16_t>(
      xq, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_split_weight_dp4a_silu_gemv_bf16_bf16_out(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_qmicro_dense_dual_q8_1x2_split_weight_dp4a_silu<
      uint16_t>(
      xq, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_gemv_bf16_bf16_out(
    const void* xq,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_qmicro_dense_dual_q8_1x2_rowtile8_dp4a_silu<uint16_t>(
      xq, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile16_w2_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile16_w2<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows6_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile16_w2_grouped<uint16_t, 6>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows8_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile16_w2_grouped<uint16_t, 8>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_qmicro_t16_dense_rowtile_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  if (rows < 2 || rows > 4) return hipErrorInvalidValue;
  return launch_q4_qmicro_dense_rowtile<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile_gemv_bf16_residual_bf16_out(
    const void* x, const void* tiles, const void* residual, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile_residual<uint16_t>(
      x, tiles, residual, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q5_dense_direct<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_gemv_decode_tile8_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q5_dense_tile8<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q5_dense_rowtile<uint16_t, 4>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q5_dense_rowtile<uint16_t, 8>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_gemv_rowtile_grouped_rows6_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q5_dense_rowtile_grouped_rows6<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_dual_rowtile_silu_gemv_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_dual_rowtile_silu<uint16_t>(
      x, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_qmicro_t16_dense_dual_rowtile_silu_gemv_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_qmicro_dense_dual_rowtile_silu<uint16_t>(
      x, tiles_a, tiles_b, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_rowtile_col4_gemv_bf16_bf16_out(
    const void* x, const void* tiles, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_rowtile_col4<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_gemv_bf16_bf16_out(
    const void* x, const void* tiles_dual, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dense_dual_interleaved_local32_silu<uint16_t, 2>(
      x, tiles_dual, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_pairreuse_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_pairreuse_direct<uint16_t>(
      x, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_direct<half_t>(x, selected, tiles_a, tiles_b, out_a, out_b,
                                       x_rows, rows, in_features, out_features,
                                       num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_qmicro_t16_selected_dual_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out_a, void* out_b, int64_t x_rows,
    int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q4_qmicro_dual_direct<uint16_t, false>(
      x, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a,
    const void* tiles_b, void* out, int64_t x_rows, int64_t rows,
    int64_t num_experts, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q4_qmicro_dual_direct<uint16_t, true>(
      x, selected, tiles_a, tiles_b, out, nullptr, x_rows, rows,
      in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_silu_direct<uint16_t>(x, selected, tiles_a, tiles_b, out,
                                               x_rows, rows, in_features,
                                               out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_q8_1_dp4a_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_q8_1_dp4a_direct<uint16_t, false>(
      xq, selected, tiles_a, tiles_b, out_a, out_b, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_q8_1_dp4a_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_q8_1_dp4a_direct<uint16_t, true>(
      xq, selected, tiles_a, tiles_b, out, nullptr, x_rows, rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_q8_1x2_dp4a_bf16_bf16_out(
    const void* xq,
    const void* selected,
    const void* tiles_a,
    const void* tiles_b,
    void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_dual_q8_1x2_dp4a_silu_direct<uint16_t>(
      xq,
      selected,
      tiles_a,
      tiles_b,
      out,
      x_rows,
      rows,
      in_features,
      out_features,
      num_experts,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 4>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_natural_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural<uint16_t, 4>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_natural_parallel_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural<uint16_t, 4, false, false, true>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_natural_parallel_weighted_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    const void* routing_weights, void* routed_out, void* completion_counter,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural_parallel_weighted<uint16_t, 4>(
      x, selected, tiles, out, routing_weights, routed_out,
      completion_counter, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_natural_parallel_paircoeff_weighted_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    const void* routing_weights, void* routed_out, void* completion_counter,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural_parallel_weighted<uint16_t, 4, false, false, true>(
      x, selected, tiles, out, routing_weights, routed_out,
      completion_counter, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 4>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 5>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int
hipengine_gguf_q5_k_qmicro_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q5_qmicro_direct<uint16_t>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_qwen_tile<uint16_t, 5, 8>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_pairreuse_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q5_pairreuse_direct<uint16_t>(x, selected, tiles, out, x_rows,
                                              rows, in_features, out_features,
                                              num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_q8_1_dp4a_bf16_bf16_out(
    const void* xq, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_q8_1_dp4a_direct<uint16_t>(xq, selected, tiles, out,
                                             x_rows, rows, in_features,
                                             out_features, num_experts,
                                             stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 5>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 6>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int hipengine_gguf_q6_k_t16_qmicro_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 6, true>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 6, true, true>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural<uint16_t, 6, true, true>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural<uint16_t, 6, true, true, true>(
      x, selected, tiles, out, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_selected_natural_parallel_weighted_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    const void* routing_weights, void* routed_out, void* completion_counter,
    int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_qk_natural_parallel_weighted<uint16_t, 6, true, true>(
      x, selected, tiles, out, routing_weights, routed_out,
      completion_counter, x_rows, rows, in_features, out_features,
      num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_pairreuse_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q6_pairreuse_direct<uint16_t>(x, selected, tiles, out, x_rows,
                                              rows, in_features, out_features,
                                              num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 6>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_grouped_smallm_bf16_bf16_out(
    const void* x, const void* expert_start_compact,
    const void* active_experts, const void* active_count, const void* tiles_a,
    const void* tiles_b, void* out_a, void* out_b, int64_t compact_rows,
    int64_t in_features, int64_t out_features, int64_t num_experts,
    hipStream_t stream) {
  return launch_q4_dual_grouped_smallm<uint16_t>(
      x, expert_start_compact, active_experts, active_count, tiles_a, tiles_b,
      out_a, out_b, compact_rows, in_features, out_features, num_experts,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_grouped_smallm_bf16_bf16_out(
    const void* x, const void* expert_start_compact,
    const void* active_experts, const void* active_count, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features,
    int64_t out_features, int64_t num_experts, hipStream_t stream) {
  return launch_qk_grouped_smallm<uint16_t, 4>(
      x, expert_start_compact, active_experts, active_count, tiles, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_grouped_smallm_bf16_bf16_out(
    const void* x, const void* expert_start_compact,
    const void* active_experts, const void* active_count, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features,
    int64_t out_features, int64_t num_experts, hipStream_t stream) {
  return launch_qk_grouped_smallm<uint16_t, 6>(
      x, expert_start_compact, active_experts, active_count, tiles, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_qmicro_selected_grouped_smallm_bf16_bf16_out(
    const void* x, const void* expert_start_compact,
    const void* active_experts, const void* active_count, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features,
    int64_t out_features, int64_t num_experts, hipStream_t stream) {
  return launch_qk_grouped_smallm<uint16_t, 6, true>(
      x, expert_start_compact, active_experts, active_count, tiles, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_selected_grouped_smallm_bf16_bf16_out(
    const void* x, const void* expert_start_compact,
    const void* active_experts, const void* active_count, const void* tiles,
    void* out, int64_t compact_rows, int64_t in_features,
    int64_t out_features, int64_t num_experts, hipStream_t stream) {
  return launch_qk_grouped_smallm<uint16_t, 6, true, true>(
      x, expert_start_compact, active_experts, active_count, tiles, out,
      compact_rows, in_features, out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles_a, const void* tiles_b,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features_a,
    int64_t out_features_b, int64_t num_experts, hipStream_t stream) {
  return launch_q4_dual<uint16_t>(x, expert_start_compact, tiles_a, tiles_b, out,
                                  compact_rows, in_features, out_features_a,
                                  out_features_b, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles_a, const void* tiles_b,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features_a,
    int64_t out_features_b, int64_t num_experts, hipStream_t stream) {
  return launch_q4_dual<half_t>(x, expert_start_compact, tiles_a, tiles_b, out,
                                compact_rows, in_features, out_features_a,
                                out_features_b, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_pairreuse_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles_a,
    const void* tiles_b, void* out, int64_t compact_rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t num_experts,
    hipStream_t stream) {
  return launch_q4_dual_pairreuse_compact<uint16_t>(
      x, expert_start_compact, tiles_a, tiles_b, out, compact_rows, in_features,
      out_features_a, out_features_b, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 4>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 4>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_pairreuse_compact<uint16_t, 4>(
      x, expert_start_compact, tiles, out, compact_rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 5>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 5>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_pairreuse_compact<uint16_t, 5>(
      x, expert_start_compact, tiles, out, compact_rows, in_features,
      out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 6>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 6>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_pairreuse_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_pairreuse_compact<uint16_t, 6>(
      x, expert_start_compact, tiles, out, compact_rows, in_features,
      out_features, num_experts, stream);
}
