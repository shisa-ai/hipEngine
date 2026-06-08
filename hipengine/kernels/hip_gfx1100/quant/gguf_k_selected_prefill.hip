// Grouped/selected MoE WMMA prefill for raw GGUF Q5_K/Q6_K down experts.
//
// P8.5 implementation: consumes the same compact-MoE scheduler ABI used by
// P8.4 selected Q4_K gate+up and the PARO AWQ compact WMMA path:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   expert_start_wmma    [num_experts + 1]
//   tile_expert          [wmma_total_rows / 16]
//   qweight              raw GGUF rank-3 expert weights [E, out_features, row_bytes]
//   out                  [compact_rows, out_features]
//
// No new compact-MoE ABI and no resident weight sidecar/repack. Q5_K/Q6_K raw
// GGUF bytes stay resident and are dequantized into half WMMA operands in the
// inner K loop.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS
#define HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS 2
#endif

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int QK_K = 256;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int K_TILE = 16;
constexpr int K_TILES_PER_BLOCK = QK_K / K_TILE;  // == 16

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

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

// Q4_K/Q5_K shared packed 6-bit scale/min helpers. Must match gguf_k_gemv.hip.
__device__ inline uint8_t gguf_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline float q5_k_weight_in_block(const uint8_t* block, int within) {
  const int subblock = within >> 5;
  const int lane32 = within & 31;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane32];
  const uint8_t ql = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane32] >> subblock) & 0x01u;
  const uint8_t q = ql | (high << 4);
  const float scale = static_cast<float>(gguf_k_scale(scales, subblock));
  const float minv = static_cast<float>(gguf_k_min(scales, subblock));
  return d * scale * static_cast<float>(q) - dmin * minv;
}

__device__ inline float q6_k_weight_in_block(const uint8_t* block, int within) {
  const int group32 = within >> 5;
  const int lane = within & 31;
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const float d = fp16_bytes_to_float(block + 208);

  const int base64 = (group32 >= 4) ? 64 : 0;
  const bool low_nibble = (group32 & 2) == 0;
  const int ql_group = group32 & 1;
  const uint8_t ql_byte = ql[base64 + ql_group * 32 + lane];
  const uint8_t low = low_nibble ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
  const int qh_base = (group32 >= 4) ? 32 : 0;
  const uint8_t high = (qh[qh_base + lane] >> (2 * (group32 & 3))) & 0x03u;
  const int q = static_cast<int>(low | (high << 4)) - 32;
  const int scale = static_cast<int>(scales[within >> 4]);
  return d * static_cast<float>(scale * q);
}

template <int qtype>
__device__ inline float dequant_k_weight_in_block(const uint8_t* block, int within) {
  if constexpr (qtype == 5) {
    return q5_k_weight_in_block(block, within);
  } else {
    return q6_k_weight_in_block(block, within);
  }
}

template <int qtype>
__device__ inline int qtype_row_bytes(int64_t in_features) {
  if constexpr (qtype == 5) {
    return static_cast<int>((in_features / QK_K) * Q5_K_BLOCK_BYTES);
  } else {
    return static_cast<int>((in_features / QK_K) * Q6_K_BLOCK_BYTES);
  }
}

template <int qtype>
__device__ inline int qtype_block_bytes() {
  if constexpr (qtype == 5) {
    return Q5_K_BLOCK_BYTES;
  } else {
    return Q6_K_BLOCK_BYTES;
  }
}

template <typename scalar_t>
__device__ inline half16_t load_compact_row_half16(
    const scalar_t* __restrict__ x,
    int64_t compact_row,
    int64_t in_features,
    int64_t k_base,
    bool valid_row) {
  half16_t out;
  if (valid_row) {
    const scalar_t* __restrict__ row = x + compact_row * in_features + k_base;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = static_cast<half_t>(scalar_to_float<scalar_t>(row[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = static_cast<half_t>(0.0f);
  }
  return out;
}

template <>
__device__ inline half16_t load_compact_row_half16<half_t>(
    const half_t* __restrict__ x,
    int64_t compact_row,
    int64_t in_features,
    int64_t k_base,
    bool valid_row) {
  if (!valid_row) {
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = static_cast<half_t>(0.0f);
    return out;
  }
  return *reinterpret_cast<const half16_t*>(x + compact_row * in_features + k_base);
}

// One wave32 block computes a 16x16 tile of a compact selected-MoE down
// projection output [compact_rows, out_features].
template <typename scalar_t, int qtype>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_k_selected_wmma_prefill_compact_legacy_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  if (col_start >= out_features) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int lane16 = tid & 15;
  const int64_t out_col = col_start + lane16;
  const int64_t local_row = tile_row_in_expert + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < out_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = qtype_row_bytes<qtype>(in_features);
  const int64_t expert_bytes = out_features * row_bytes;
  const int block_bytes = qtype_block_bytes<qtype>();
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + out_col * row_bytes) : expert_base;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr = weight_row + blk * block_bytes;
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
      const int k_off = kt * K_TILE;
      const int64_t k_base = g_k_base + k_off;
      const half16_t a_reg = load_compact_row_half16<scalar_t>(
          x, compact_row, in_features, k_base, valid_row);
      half16_t b_reg;
      if (valid_col) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          b_reg[kk] = static_cast<half_t>(
              dequant_k_weight_in_block<qtype>(block_ptr, k_off + kk));
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }
  }

  const int half_id = tid >> 4;
  if (out_col < out_features) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features + out_col] = float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}


// One wave32 block computes an AWQ_TILE_M x AWQ_TILE_N tile bundle of a
// compact selected-MoE down projection output [compact_rows, out_features].
// P9.C1 exposes the same TM/TN search space as the dense Q8_0 WMMA prefill
// kernel while preserving the compact scheduler ABI.
template <typename scalar_t, int qtype, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_k_selected_wmma_prefill_compact_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const int col_tile = blockIdx.x;
  const int row_tile_group = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * AWQ_TILE_M;
  if (col_start >= out_features) return;

  const int64_t row_tile_base = static_cast<int64_t>(row_tile_group) * TN;
  const int64_t tile_count = wmma_total_rows / 16;

  int64_t expert_id[TN];
  int64_t compact_base[TN];
  int64_t expert_count[TN];
  int64_t tile_row_in_expert[TN];
  bool valid_expert[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int64_t row_tile = row_tile_base + tn;
    if (row_tile < tile_count) {
      expert_id[tn] = tile_expert[row_tile];
      valid_expert[tn] = (expert_id[tn] >= 0 && expert_id[tn] < num_experts);
    } else {
      expert_id[tn] = -1;
      valid_expert[tn] = false;
    }
    if (valid_expert[tn]) {
      compact_base[tn] = expert_start_compact[expert_id[tn]];
      expert_count[tn] = expert_start_compact[expert_id[tn] + 1] - compact_base[tn];
      const int64_t wmma_base = expert_start_wmma[expert_id[tn]];
      const int64_t row_start_in_wmma = (row_tile_base + tn) * 16;
      tile_row_in_expert[tn] = row_start_in_wmma - wmma_base;
    } else {
      compact_base[tn] = 0;
      expert_count[tn] = 0;
      tile_row_in_expert[tn] = 0;
    }
  }

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = qtype_row_bytes<qtype>(in_features);
  const int64_t expert_bytes = out_features * row_bytes;
  const int block_bytes = qtype_block_bytes<qtype>();

  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
      const int k_off = kt * K_TILE;
      const int64_t k_base = g_k_base + k_off;

      half16_t a_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const int64_t local_row = tile_row_in_expert[tn] + lane16;
        const bool valid_row = valid_expert[tn] && local_row < expert_count[tn];
        const int64_t compact_row = valid_row ? (compact_base[tn] + local_row) : 0;
        a_reg[tn] = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
      }

#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const int64_t out_col = col_start + tm * 16 + lane16;
        const bool col_in_bounds = out_col < out_features;
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          const bool valid_col = col_in_bounds && valid_expert[tn];
          const uint8_t* __restrict__ expert_base =
              valid_col ? (qweight + expert_id[tn] * expert_bytes) : qweight;
          const uint8_t* __restrict__ block_ptr =
              valid_col ? (expert_base + out_col * row_bytes + blk * block_bytes) : qweight;
          half16_t b_reg;
          if (valid_col) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              b_reg[kk] = static_cast<half_t>(
                  dequant_k_weight_in_block<qtype>(block_ptr, k_off + kk));
            }
          } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
          }
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg[tn], b_reg, acc[tm][tn]);
        }
      }
    }
  }

#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    if (!valid_expert[tn]) continue;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const int64_t out_col = col_start + tm * 16 + lane16;
      if (out_col >= out_features) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert[tn] + half_id + j * 2;
        if (local_out_row < expert_count[tn]) {
          const int64_t compact_out_row = compact_base[tn] + local_out_row;
          if (compact_out_row < compact_rows) {
            out[compact_out_row * out_features + out_col] = float_to_scalar<scalar_t>(acc[tm][tn][j]);
          }
        }
      }
    }
  }
}


bool valid_selected_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n);

// P9.C7 Q5_K optimized legacy-shape kernel.
//
// Same compact ABI and 16x16 tile shape as the retained P8.5 Q5 down kernel,
// but hoists Q5_K d/dmin and packed scale/min decode once per 32-value
// subblock instead of recomputing it inside q5_k_weight_in_block() for every kk.
template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q5_k_selected_wmma_prefill_compact_opt_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  if (col_start >= out_features) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int lane16 = tid & 15;
  const int64_t out_col = col_start + lane16;
  const int64_t local_row = tile_row_in_expert + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < out_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + out_col * row_bytes) : expert_base;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr = weight_row + blk * Q5_K_BLOCK_BYTES;
    const float d = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
    const float dmin = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
    const uint8_t* __restrict__ scales = block_ptr + 4;
    const uint8_t* __restrict__ qh = block_ptr + 16;
    const uint8_t* __restrict__ qs = block_ptr + 48;
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
      const float scale_f = valid_col ? d * static_cast<float>(gguf_k_scale(scales, sb)) : 0.0f;
      const float min_f = valid_col ? dmin * static_cast<float>(gguf_k_min(scales, sb)) : 0.0f;
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        const int k_off = sb * 32 + kt * 16;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
        half16_t b_reg;
        if (valid_col) {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * 16 + kk;
            const uint8_t packed = qs[(sb >> 1) * 32 + lane32];
            const uint8_t ql = (sb & 1) ? (packed >> 4) : (packed & 0x0Fu);
            const uint8_t high = (qh[lane32] >> sb) & 0x01u;
            const uint8_t q = ql | (high << 4);
            b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
          }
        } else {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
      }
    }
  }

  const int half_id = tid >> 4;
  if (out_col < out_features) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features + out_col] = float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

template <typename scalar_t>
int launch_q5_selected_opt(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_common(
          x, expert_start_compact, expert_start_wmma, tile_expert, qweight, out,
          compact_rows, in_features, out_features, num_experts, wmma_total_rows, 16, 16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gguf_q5_k_selected_wmma_prefill_compact_opt_kernel<scalar_t>),
      dim3(
          static_cast<unsigned int>((out_features + 15) / 16),
          static_cast<unsigned int>(wmma_total_rows / 16)),
      dim3(32),
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

bool valid_selected_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n) {
  return x != nullptr && expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && qweight != nullptr && out != nullptr &&
      compact_rows > 0 && in_features > 0 && out_features > 0 && num_experts > 0 &&
      wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 && (in_features % QK_K) == 0 &&
      (tile_m == 16 || tile_m == 32 || tile_m == 64) &&
      (tile_n == 16 || tile_n == 32);
}

template <typename scalar_t, int qtype>
int launch_selected(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (!valid_selected_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight,
          out,
          compact_rows,
          in_features,
          out_features,
          num_experts,
          wmma_total_rows,
          tile_m,
          tile_n)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_tiles = wmma_total_rows / 16;
  const int64_t tile_n_tiles = tile_n / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((row_tiles + tile_n_tiles - 1) / tile_n_tiles));

#define HIPENGINE_LAUNCH_K_SELECTED(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_k_selected_wmma_prefill_compact_kernel<scalar_t, qtype, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      expert_start_compact, \
      expert_start_wmma, \
      tile_expert, \
      qweight, \
      out, \
      compact_rows, \
      in_features, \
      out_features, \
      num_experts, \
      wmma_total_rows); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) {
    hipLaunchKernelGGL(
        (gguf_k_selected_wmma_prefill_compact_legacy_kernel<scalar_t, qtype>),
        grid,
        block,
        0,
        stream,
        x,
        expert_start_compact,
        expert_start_wmma,
        tile_expert,
        qweight,
        out,
        compact_rows,
        in_features,
        out_features,
        num_experts);
    return static_cast<int>(hipGetLastError());
  }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_K_SELECTED(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_K_SELECTED(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_K_SELECTED(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_K_SELECTED(64, 16); }
  HIPENGINE_LAUNCH_K_SELECTED(64, 32);
#undef HIPENGINE_LAUNCH_K_SELECTED
}

}  // namespace

#define HIPENGINE_GGUF_K_SELECTED_WMMA_EXPORT(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_selected_wmma_prefill_compact_bf16_bf16_out( \
    const uint16_t* x, \
    const int64_t* expert_start_compact, \
    const int64_t* expert_start_wmma, \
    const int64_t* tile_expert, \
    const uint8_t* qweight, \
    uint16_t* out, \
    int64_t compact_rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t num_experts, \
    int64_t wmma_total_rows, \
    int64_t tile_m, \
    int64_t tile_n, \
    hipStream_t stream) { \
  return launch_selected<uint16_t, QTYPE>( \
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight, out, \
      compact_rows, in_features, out_features, num_experts, wmma_total_rows, tile_m, tile_n, stream); \
} \
extern "C" int hipengine_##NAME##_selected_wmma_prefill_compact_fp16_fp16_out( \
    const half_t* x, \
    const int64_t* expert_start_compact, \
    const int64_t* expert_start_wmma, \
    const int64_t* tile_expert, \
    const uint8_t* qweight, \
    half_t* out, \
    int64_t compact_rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t num_experts, \
    int64_t wmma_total_rows, \
    int64_t tile_m, \
    int64_t tile_n, \
    hipStream_t stream) { \
  return launch_selected<half_t, QTYPE>( \
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight, out, \
      compact_rows, in_features, out_features, num_experts, wmma_total_rows, tile_m, tile_n, stream); \
}

HIPENGINE_GGUF_K_SELECTED_WMMA_EXPORT(gguf_q5_k, 5)
HIPENGINE_GGUF_K_SELECTED_WMMA_EXPORT(gguf_q6_k, 6)

extern "C" int hipengine_gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_q5_selected_opt<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q5_k_selected_wmma_prefill_compact_opt_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_q5_selected_opt<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}
