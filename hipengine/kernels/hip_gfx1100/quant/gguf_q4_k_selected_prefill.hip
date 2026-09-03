// Grouped/selected MoE WMMA prefill for raw GGUF Q4_K gate+up experts.
//
// P8.4 implementation note: this file mirrors
// gemm_awq_selected_dual_pack8_wmma_compact_kernel from
// hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip. It consumes the
// exact compact-MoE ABI produced by qwen35_moe_group_count/prefix,
// qwen35_moe_group_scatter_gather, and qwen35_moe_wmma_tile_map:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   expert_start_wmma    [num_experts + 1]
//   tile_expert          [wmma_total_rows / 16]
//
// The only algorithmic change from the AWQ/PARO template is the inner K-loop
// weight dequant: AWQ's qweight/qzeros/scales triple is replaced by raw GGUF
// block_q4_K bytes in rank-3 expert tensors [E, out_features, row_bytes].
// There is no new scheduler ABI and no resident weight sidecar/repack.

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
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_SCALE_BYTES = 12;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + Q4_K_SCALE_BYTES;
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;          // == 8
constexpr int K_TILE = 16;                                    // WMMA K
constexpr int K_TILES_PER_SUBBLOCK = Q4_K_SUBBLOCK / K_TILE;  // == 2

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

__device__ inline uint8_t gguf_q4_k_quant(
    const uint8_t* qs,
    int subblock,
    int lane32) {
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * 32 + lane32];
  return (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline float gguf_q4_k_weight(
    const uint8_t* __restrict__ row,
    int64_t column) {
  const int64_t block_index = column / QK_K;
  const int within = static_cast<int>(column % QK_K);
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + block_index * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t q = gguf_q4_k_quant(block + Q4_K_QS_OFFSET, subblock, lane);
  return d * static_cast<float>(gguf_q4_k_scale(scales, subblock))
      * static_cast<float>(q)
      - dmin * static_cast<float>(gguf_q4_k_min(scales, subblock));
}

struct q4_k_weight_pair {
  float first;
  float second;
};

__device__ inline q4_k_weight_pair gguf_q4_k_weight_pair128(
    const uint8_t* __restrict__ row,
    int64_t block_index,
    int lane128,
    const float* __restrict__ d_scales,
    const float* __restrict__ dmin_mins) {
  const int subblock = lane128 >> 5;
  const int lane = lane128 & 31;
  const uint8_t* qs = row + block_index * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
  const int paired_subblock = subblock + 4;
  const uint8_t q0 = gguf_q4_k_quant(qs, subblock, lane);
  const uint8_t q1 = gguf_q4_k_quant(qs, paired_subblock, lane);
  const int meta_base = static_cast<int>(block_index) * Q4_K_SUBBLOCKS;
  return {
      d_scales[meta_base + subblock] * static_cast<float>(q0)
          - dmin_mins[meta_base + subblock],
      d_scales[meta_base + paired_subblock] * static_cast<float>(q1)
          - dmin_mins[meta_base + paired_subblock],
  };
}

__device__ inline float reduce_block_sum_strict(float value, float* wave_sums) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  if (lane == 0) wave_sums[wave] = value;
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int current = 0; current < waves; ++current) total += wave_sums[current];
  }
  __syncthreads();
  return total;
}

// Exact grouped counterpart of the selected dual scalar prefill kernel. The
// column assignment and wave/block reduction tree are unchanged; only Q4_K
// weight dequantization is shared across up to eight rows of one expert.
template <int ROW_BATCH, int OUT_BATCH, bool PAIR_BLOCK_WEIGHTS = false>
__global__ __launch_bounds__(128, 2)
void gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ output_a,
    uint16_t* __restrict__ output_b,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features) {
  __shared__ float wave_sums[8];
  constexpr int MAX_Q4_BLOCKS = 16;
  __shared__ float d_scales_a[MAX_Q4_BLOCKS * Q4_K_SUBBLOCKS];
  __shared__ float dmin_mins_a[MAX_Q4_BLOCKS * Q4_K_SUBBLOCKS];
  __shared__ float d_scales_b[MAX_Q4_BLOCKS * Q4_K_SUBBLOCKS];
  __shared__ float dmin_mins_b[MAX_Q4_BLOCKS * Q4_K_SUBBLOCKS];
  const int64_t q4_blocks = in_features / QK_K;
  if (q4_blocks > MAX_Q4_BLOCKS) return;
  const int64_t out_base = blockIdx.x * OUT_BATCH;
  if (out_base >= out_features) return;
  const int64_t row_bytes = (in_features / QK_K) * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * row_bytes;
  // Launching gridDim.y == num_experts preserves the original one-expert CTA.
  // A smaller fixed worker grid iterates experts without changing any output's
  // column assignment or strict 128-thread reduction tree.
  for (int64_t expert = blockIdx.y; expert < num_experts; expert += gridDim.y) {
    const int64_t start = expert_start[expert];
    const int64_t end = expert_start[expert + 1];
    if (start < 0 || end < start || end > compact_rows || start == end) continue;
#pragma unroll
    for (int out_offset = 0; out_offset < OUT_BATCH; ++out_offset) {
      const int64_t out = out_base + out_offset;
      if (out >= out_features) break;
      const uint8_t* weight_a = qweight_a + expert * expert_bytes + out * row_bytes;
      const uint8_t* weight_b = qweight_b + expert * expert_bytes + out * row_bytes;
      if (PAIR_BLOCK_WEIGHTS) {
        const int64_t metadata_count = q4_blocks * Q4_K_SUBBLOCKS;
        for (int64_t metadata = threadIdx.x; metadata < metadata_count;
             metadata += blockDim.x) {
          const int64_t block_index = metadata / Q4_K_SUBBLOCKS;
          const int subblock = static_cast<int>(metadata % Q4_K_SUBBLOCKS);
          const uint8_t* block_a = weight_a + block_index * Q4_K_BLOCK_BYTES;
          const uint8_t* block_b = weight_b + block_index * Q4_K_BLOCK_BYTES;
          const float d_a = fp16_bytes_to_float(block_a);
          const float dmin_a = fp16_bytes_to_float(block_a + 2);
          const float d_b = fp16_bytes_to_float(block_b);
          const float dmin_b = fp16_bytes_to_float(block_b + 2);
          d_scales_a[metadata] = d_a * static_cast<float>(
              gguf_q4_k_scale(block_a + 4, subblock));
          dmin_mins_a[metadata] = dmin_a * static_cast<float>(
              gguf_q4_k_min(block_a + 4, subblock));
          d_scales_b[metadata] = d_b * static_cast<float>(
              gguf_q4_k_scale(block_b + 4, subblock));
          dmin_mins_b[metadata] = dmin_b * static_cast<float>(
              gguf_q4_k_min(block_b + 4, subblock));
        }
        __syncthreads();
      }
      for (int64_t base = start; base < end; base += ROW_BATCH) {
        float acc_a[ROW_BATCH] = {};
        float acc_b[ROW_BATCH] = {};
        if (PAIR_BLOCK_WEIGHTS) {
          for (int64_t block_index = 0; block_index < in_features / QK_K;
               ++block_index) {
            const int64_t column0 = block_index * QK_K + threadIdx.x;
            const int64_t column1 = column0 + 128;
            const q4_k_weight_pair wa = gguf_q4_k_weight_pair128(
                weight_a, block_index, threadIdx.x, d_scales_a, dmin_mins_a);
            const q4_k_weight_pair wb = gguf_q4_k_weight_pair128(
                weight_b, block_index, threadIdx.x, d_scales_b, dmin_mins_b);
#pragma unroll
            for (int row = 0; row < ROW_BATCH; ++row) {
              const int64_t compact_row = base + row;
              if (compact_row < end) {
                const float value0 = bf16_bits_to_float(
                    input[compact_row * in_features + column0]);
                acc_a[row] += value0 * wa.first;
                acc_b[row] += value0 * wb.first;
                const float value1 = bf16_bits_to_float(
                    input[compact_row * in_features + column1]);
                acc_a[row] += value1 * wa.second;
                acc_b[row] += value1 * wb.second;
              }
            }
          }
        } else {
          for (int64_t column = threadIdx.x; column < in_features;
               column += blockDim.x) {
            const float wa = gguf_q4_k_weight(weight_a, column);
            const float wb = gguf_q4_k_weight(weight_b, column);
#pragma unroll
            for (int row = 0; row < ROW_BATCH; ++row) {
              const int64_t compact_row = base + row;
              if (compact_row < end) {
                const float value = bf16_bits_to_float(
                    input[compact_row * in_features + column]);
                acc_a[row] += value * wa;
                acc_b[row] += value * wb;
              }
            }
          }
        }
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          const float total = reduce_block_sum_strict(acc_a[row], wave_sums);
          if (threadIdx.x == 0 && base + row < end) {
            output_a[(base + row) * out_features + out] = float_to_bf16_bits(total);
          }
          __syncthreads();
        }
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          const float total = reduce_block_sum_strict(acc_b[row], wave_sums);
          if (threadIdx.x == 0 && base + row < end) {
            output_b[(base + row) * out_features + out] = float_to_bf16_bits(total);
          }
          __syncthreads();
        }
      }
    }
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

// One wave32 block computes a 16x16 tile of the compact selected-MoE output.
// The output is row-major [compact_rows, out_features_a + out_features_b],
// matching the PARO compact dual WMMA kernel. Gate columns occupy the first
// out_features_a columns; up columns occupy the second out_features_b columns.
template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_legacy_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = col_start >= out_features_a;
  const int64_t local_col_start = use_b ? (col_start - out_features_a) : col_start;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int t_col = tid & 15;
  const int t_row_in_tile = tid & 15;
  const int64_t out_col = local_col_start + t_col;
  const int64_t local_row = tile_row_in_expert + t_row_in_tile;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < local_out_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + out_col * row_bytes) : expert_base;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
    const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
    const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
    const uint8_t* __restrict__ scales = block_ptr + 4;
    const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
    const int64_t g_k_base = blk * QK_K;

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = valid_col ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb)) : 0.0f;
      const float min_f = valid_col ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb)) : 0.0f;
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
        half16_t b_reg;
        if (valid_col) {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
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
  const int64_t global_out_col = col_start + t_col;
  if (global_out_col < out_features_total) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}


// One wave32 block computes an AWQ_TILE_M x AWQ_TILE_N tile bundle of the
// compact selected-MoE output. Each bundle is made from 16x16 WMMA tiles. The
// output is row-major [compact_rows, out_features_a + out_features_b], matching
// the PARO compact dual WMMA kernel. Gate columns occupy the first
// out_features_a columns; up columns occupy the second out_features_b columns.
//
// P9.C1 turns the P8.4 one-16x16-tile block into a tunable multi-tile block.
// Larger TILE_M amortises compact activation loads across 2-4 output columns;
// TILE_N=32 can combine two neighbouring WMMA row tiles but still handles the
// case where those row tiles belong to different experts by resolving expert
// metadata per TN tile.
template <typename scalar_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
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
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

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
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

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
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
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
          const int64_t global_col_base = col_start + tm * 16;
          const bool use_b = global_col_base >= out_features_a;
          const int64_t local_col_base = use_b ? (global_col_base - out_features_a) : global_col_base;
          const int64_t local_out_features = use_b ? out_features_b : out_features_a;
          const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
          const int64_t local_col = local_col_base + lane16;
          const bool col_in_bounds = (global_col_base + lane16) < out_features_total &&
              local_col < local_out_features;

#pragma unroll
          for (int tn = 0; tn < TN; ++tn) {
            const bool valid_col = col_in_bounds && valid_expert[tn];
            const int64_t expert_bytes = local_out_features * row_bytes;
            const uint8_t* __restrict__ expert_base =
                valid_col ? (qweight + expert_id[tn] * expert_bytes) : qweight;
            const uint8_t* __restrict__ block_ptr =
                valid_col ? (expert_base + local_col * row_bytes + blk * Q4_K_BLOCK_BYTES) : qweight;
            const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
            const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
            const uint8_t* __restrict__ scales = block_ptr + 4;
            const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
            const float scale_f = valid_col ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb)) : 0.0f;
            const float min_f = valid_col ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb)) : 0.0f;
            half16_t b_reg;
            if (valid_col) {
#pragma unroll
              for (int kk = 0; kk < 16; ++kk) {
                const int lane32 = kt * K_TILE + kk;
                const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
                b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
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
  }

#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    if (!valid_expert[tn]) continue;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const int64_t global_out_col = col_start + tm * 16 + lane16;
      if (global_out_col >= out_features_total) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert[tn] + half_id + j * 2;
        if (local_out_row < expert_count[tn]) {
          const int64_t compact_out_row = compact_base[tn] + local_out_row;
          if (compact_out_row < compact_rows) {
            out[compact_out_row * out_features_total + global_out_col] =
                float_to_scalar<scalar_t>(acc[tm][tn][j]);
          }
        }
      }
    }
  }
}


bool valid_selected_dual_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n);


// P9.C5 prototype: Q4_K side metadata (predecoded scale/min) 32x16 kernel.
//
// Sidecar layout is fp16 row-major:
//   [num_experts, out_features, blocks_per_row, Q4_K_SUBBLOCKS, 2]
// where the last dimension is (scale_f, min_f). Raw qweight is still used for
// q nibbles, preserving GGUF Q4_K math while avoiding per-WMMA-tile scale/min
// bitfield decode and d/dmin half conversion.
template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta32_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    const half_t* __restrict__ sidemeta_a,
    const half_t* __restrict__ sidemeta_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 32;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

  float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const int64_t local_row = tile_row_in_expert + lane16;
        const bool valid_row = local_row < expert_count;
        const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);

#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
          const int64_t global_col_base = col_start + tm * 16;
          const bool use_b = global_col_base >= out_features_a;
          const int64_t local_col_base = use_b ? (global_col_base - out_features_a) : global_col_base;
          const int64_t local_out_features = use_b ? out_features_b : out_features_a;
          const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
          const half_t* __restrict__ sidemeta = use_b ? sidemeta_b : sidemeta_a;
          const int64_t local_col = local_col_base + lane16;
          const bool valid_col = (global_col_base + lane16) < out_features_total &&
              local_col < local_out_features;
          const int64_t expert_bytes = local_out_features * row_bytes;
          const uint8_t* __restrict__ expert_base =
              valid_col ? (qweight + expert_id * expert_bytes) : qweight;
          const uint8_t* __restrict__ block_ptr =
              valid_col ? (expert_base + local_col * row_bytes + blk * Q4_K_BLOCK_BYTES) : qweight;
          const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
          const int64_t meta_idx =
              (((expert_id * local_out_features + local_col) * blocks_per_row + blk) *
                   Q4_K_SUBBLOCKS + sb) * 2;
          const float scale_f = valid_col ? static_cast<float>(sidemeta[meta_idx]) : 0.0f;
          const float min_f = valid_col ? static_cast<float>(sidemeta[meta_idx + 1]) : 0.0f;
          half16_t b_reg;
          if (valid_col) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              const int lane32 = kt * K_TILE + kk;
              const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
              b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
            }
          } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
          }
          if (tm == 0) {
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc0);
          } else {
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc1);
          }
        }
      }
    }
  }

#pragma unroll
  for (int tm = 0; tm < 2; ++tm) {
    const int64_t global_out_col = col_start + tm * 16 + lane16;
    if (global_out_col >= out_features_total) continue;
    const float8_t acc = (tm == 0) ? acc0 : acc1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

template <typename scalar_t>
int launch_selected_dual_sidemeta32(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    const half_t* sidemeta_a,
    const half_t* sidemeta_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_common(
          x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a, qweight_b, out,
          compact_rows, in_features, out_features_a, out_features_b, num_experts,
          wmma_total_rows, 32, 16) || sidemeta_a == nullptr || sidemeta_b == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta32_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      sidemeta_a,
      sidemeta_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

// P9.C4 prototype: hot/full-tile hybrid for Q4_K selected dual.
//
// The full-tile kernel processes only row tiles from hot experts whose 16 rows
// are all real rows (not WMMA padding). It keeps the retained Q4 32x16 column
// grouping but removes per-row validity checks in the hot path. A companion
// tail/cold kernel computes every row tile not handled by the hot kernel using
// the legacy 16x16 implementation style. Together they preserve the compact
// selected-MoE ABI and produce the same output as the normal selected dual
// kernel without duplicate stores.
template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile32_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t hot_threshold) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 32;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  if (expert_count < hot_threshold) return;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  if (tile_row_in_expert < 0 || tile_row_in_expert + 15 >= expert_count) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

  float8_t acc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float8_t acc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const int64_t compact_row = compact_base + tile_row_in_expert + lane16;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, true);

#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
          const int64_t global_col_base = col_start + tm * 16;
          const bool use_b = global_col_base >= out_features_a;
          const int64_t local_col_base = use_b ? (global_col_base - out_features_a) : global_col_base;
          const int64_t local_out_features = use_b ? out_features_b : out_features_a;
          const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
          const int64_t local_col = local_col_base + lane16;
          const bool valid_col = (global_col_base + lane16) < out_features_total &&
              local_col < local_out_features;
          const int64_t expert_bytes = local_out_features * row_bytes;
          const uint8_t* __restrict__ expert_base =
              valid_col ? (qweight + expert_id * expert_bytes) : qweight;
          const uint8_t* __restrict__ block_ptr =
              valid_col ? (expert_base + local_col * row_bytes + blk * Q4_K_BLOCK_BYTES) : qweight;
          const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
          const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
          const uint8_t* __restrict__ scales = block_ptr + 4;
          const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
          const float scale_f = valid_col ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb)) : 0.0f;
          const float min_f = valid_col ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb)) : 0.0f;
          half16_t b_reg;
          if (valid_col) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              const int lane32 = kt * K_TILE + kk;
              const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
              b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
            }
          } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
          }
          if (tm == 0) {
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc0);
          } else {
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc1);
          }
        }
      }
    }
  }

#pragma unroll
  for (int tm = 0; tm < 2; ++tm) {
    const int64_t global_out_col = col_start + tm * 16 + lane16;
    if (global_out_col >= out_features_total) continue;
    const float8_t acc = (tm == 0) ? acc0 : acc1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      const int64_t compact_out_row = compact_base + local_out_row;
      out[compact_out_row * out_features_total + global_out_col] =
          float_to_scalar<scalar_t>(acc[j]);
    }
  }
}

template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_tailcold16_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t hot_threshold) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  if (expert_count >= hot_threshold && tile_row_in_expert >= 0 &&
      tile_row_in_expert + 15 < expert_count) {
    return;
  }

  const bool use_b = col_start >= out_features_a;
  const int64_t local_col_start = use_b ? (col_start - out_features_a) : col_start;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int t_col = tid & 15;
  const int t_row_in_tile = tid & 15;
  const int64_t out_col = local_col_start + t_col;
  const int64_t local_row = tile_row_in_expert + t_row_in_tile;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < local_out_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + out_col * row_bytes) : expert_base;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
    const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
    const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
    const uint8_t* __restrict__ scales = block_ptr + 4;
    const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
    const int64_t g_k_base = blk * QK_K;

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = valid_col ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb)) : 0.0f;
      const float min_f = valid_col ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb)) : 0.0f;
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
        half16_t b_reg;
        if (valid_col) {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
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
  const int64_t global_out_col = col_start + t_col;
  if (global_out_col < out_features_total) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}


template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_selected_dual_wmma_prefill_compact_tail_by_expert16_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int col_tile = blockIdx.x;
  const int64_t expert_id = static_cast<int64_t>(blockIdx.y);
  const int tid = threadIdx.x;
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t residual = expert_count & 15;
  if (expert_count <= 0 || residual == 0) return;

  const int64_t tile_row_in_expert = expert_count - residual;
  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const bool use_b = col_start >= out_features_a;
  const int64_t local_col_start = use_b ? (col_start - out_features_a) : col_start;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int t_col = tid & 15;
  const int t_row_in_tile = tid & 15;
  const int64_t out_col = local_col_start + t_col;
  const int64_t local_row = tile_row_in_expert + t_row_in_tile;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < local_out_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + out_col * row_bytes) : expert_base;

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
    const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
    const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
    const uint8_t* __restrict__ scales = block_ptr + 4;
    const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
    const int64_t g_k_base = blk * QK_K;

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = valid_col ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb)) : 0.0f;
      const float min_f = valid_col ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb)) : 0.0f;
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
        half16_t b_reg;
        if (valid_col) {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
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
  const int64_t global_out_col = col_start + t_col;
  if (global_out_col < out_features_total) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

template <typename scalar_t>
int launch_selected_dual_hot_fulltile32_tailcold16(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t hot_threshold,
    hipStream_t stream) {
  if (!valid_selected_dual_common(
          x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a, qweight_b, out,
          compact_rows, in_features, out_features_a, out_features_b, num_experts,
          wmma_total_rows, 32, 16) || hot_threshold <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid_hot(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  const dim3 grid_tail(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  const dim3 grid_tail_by_expert(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(num_experts));
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile32_kernel<scalar_t>),
      grid_hot,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      hot_threshold);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  if (hot_threshold <= 1) {
    hipLaunchKernelGGL(
        (gguf_q4_k_selected_dual_wmma_prefill_compact_tail_by_expert16_kernel<scalar_t>),
        grid_tail_by_expert,
        block,
        0,
        stream,
        x,
        expert_start_compact,
        expert_start_wmma,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts);
  } else {
    hipLaunchKernelGGL(
        (gguf_q4_k_selected_dual_wmma_prefill_compact_tailcold16_kernel<scalar_t>),
        grid_tail,
        block,
        0,
        stream,
        x,
        expert_start_compact,
        expert_start_wmma,
        tile_expert,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        hot_threshold);
  }
  return static_cast<int>(hipGetLastError());
}

bool valid_selected_dual_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n) {
  return x != nullptr && expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && qweight_a != nullptr && qweight_b != nullptr && out != nullptr &&
      compact_rows > 0 && in_features > 0 && out_features_a > 0 && out_features_b > 0 &&
      (out_features_a % 16) == 0 && (out_features_b % 16) == 0 &&
      num_experts > 0 && wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 &&
      (in_features % QK_K) == 0 &&
      (tile_m == 16 || tile_m == 32 || tile_m == 64) &&
      (tile_n == 16 || tile_n == 32);
}

template <typename scalar_t>
int launch_selected_dual(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (!valid_selected_dual_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows,
          tile_m,
          tile_n)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const int64_t tile_n_tiles = tile_n / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + tile_m - 1) / tile_m),
      static_cast<unsigned int>((row_tiles + tile_n_tiles - 1) / tile_n_tiles));

#define HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q4_k_selected_dual_wmma_prefill_compact_kernel<scalar_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      expert_start_compact, \
      expert_start_wmma, \
      tile_expert, \
      qweight_a, \
      qweight_b, \
      out, \
      compact_rows, \
      in_features, \
      out_features_a, \
      out_features_b, \
      num_experts, \
      wmma_total_rows); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) {
    hipLaunchKernelGGL(
        (gguf_q4_k_selected_dual_wmma_prefill_compact_legacy_kernel<scalar_t>),
        grid,
        block,
        0,
        stream,
        x,
        expert_start_compact,
        expert_start_wmma,
        tile_expert,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts);
    return static_cast<int>(hipGetLastError());
  }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(64, 16); }
  HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL(64, 32);
#undef HIPENGINE_LAUNCH_Q4K_SELECTED_DUAL
}

// ---------------------------------------------------------------------------
// iu8-WMMA selected dual gate/up prefill (llama.cpp mul_mat_q arithmetic
// class; lineage ggml-org/llama.cpp@17252c769 ggml/src/ggml-cuda/mmq.cuh,
// mma.cuh RDNA3 int path). Activations and Q4_K weights are requantized per
// 32-k subblock to signed int8, multiplied through the RDNA3.5 iu8 wave-MMA
// (empirically pinned ABI: one call computes D[32m][8n] += sum_{k<16}
// B[m][k]*A[n][k]; second-operand thread m holds row m's k bytes, first
// operand uses even lanes {0..14} and odd lanes {17..31} with per-row-half
// column replication, identity byte->k on both sides) and rescaled in f32 at
// every subblock boundary. 128-thread blocks stage the 16-row activation
// tile in LDS; each warp owns 32 output columns (one per lane).
// ---------------------------------------------------------------------------
typedef int __attribute__((vector_size(4 * sizeof(int)))) iu8_int32x4;
typedef int __attribute__((vector_size(8 * sizeof(int)))) iu8_int32x8;

constexpr int IU8_LDS_ROW_STRIDE = QK_K + 16;  // bank-conflict padding

template <typename scalar_t>
__launch_bounds__(128, 2)
__global__ void gguf_q4_k_selected_dual_wmma_iu8_prefill_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  constexpr int IU8_ACT_PLANES = 3;  // residual quantization planes (ds4 class)
  __shared__ int8_t lds_q8a[IU8_ACT_PLANES * 16 * IU8_LDS_ROW_STRIDE];
  __shared__ float lds_da[IU8_ACT_PLANES * 16 * Q4_K_SUBBLOCKS];
  __shared__ int lds_sumq8a[IU8_ACT_PLANES * 16 * Q4_K_SUBBLOCKS];
  __shared__ int64_t meta[4];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * 128;
  const int64_t row_tile = blockIdx.y;

  if (tid == 0) {
    const int64_t expert = tile_expert[row_tile];
    const bool valid = expert >= 0 && expert < num_experts;
    meta[0] = valid ? expert : -1;
    meta[1] = valid ? expert_start_compact[expert] : 0;
    meta[2] = valid ? (expert_start_compact[expert + 1] - meta[1]) : 0;
    meta[3] = valid ? (row_tile * 16 - expert_start_wmma[expert]) : 0;
  }
  __syncthreads();
  const int64_t expert_id = meta[0];
  if (expert_id < 0) return;
  const int64_t compact_base = meta[1];
  const int64_t expert_count = meta[2];
  const int64_t tile_row_in_expert = meta[3];

  const int64_t global_col = col_base + warp * 32 + lane;
  const bool col_valid = global_col < out_features_total;
  const bool use_b = col_valid && global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* qweight = use_b ? qweight_b : qweight_a;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const bool wcol_valid = col_valid && local_col < local_out_features;
  const uint8_t* block_base = wcol_valid
      ? (qweight + expert_id * (local_out_features * row_bytes) +
         local_col * row_bytes)
      : nullptr;

  float sum[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) sum[r] = 0.0f;

  // A-operand lane -> activation row (hardware permutation).
  int a_row = -1;
  if ((lane & 1) == 0 && lane < 16) {
    a_row = lane >> 1;
  } else if ((lane & 1) == 1 && lane >= 17) {
    a_row = (lane - 17) >> 1;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    // Cooperative activation staging: thread -> (actrow = tid/8, sub = tid%8).
    {
      const int actrow = tid >> 3;
      const int sub = tid & 7;
      const int64_t local_row = tile_row_in_expert + actrow;
      const bool row_valid = local_row < expert_count;
      const int64_t compact_row = row_valid ? (compact_base + local_row) : 0;
      const scalar_t* row_ptr =
          x + compact_row * in_features + blk * QK_K + sub * Q4_K_SUBBLOCK;
      float vals[32];
      float amax = 0.0f;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const float v = row_valid ? scalar_to_float(row_ptr[l]) : 0.0f;
        vals[l] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      float work[32];
#pragma unroll
      for (int l = 0; l < 32; ++l) work[l] = vals[l];
#pragma unroll
      for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
        float plane_max = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) plane_max = fmaxf(plane_max, fabsf(work[l]));
        const float da = plane_max > 0.0f ? plane_max / 127.0f : 0.0f;
        lds_da[plane * 16 * Q4_K_SUBBLOCKS + actrow * Q4_K_SUBBLOCKS + sub] = da;
        int8_t* dst = lds_q8a + plane * 16 * IU8_LDS_ROW_STRIDE +
                      actrow * IU8_LDS_ROW_STRIDE + sub * Q4_K_SUBBLOCK;
        int qsum = 0;
        if (da > 0.0f) {
          const float inv = 1.0f / da;
#pragma unroll
          for (int l = 0; l < 32; ++l) {
            int q = __float2int_rn(work[l] * inv);
            q = max(-127, min(127, q));
            dst[l] = static_cast<int8_t>(q);
            qsum += q;
            work[l] = fmaf(-static_cast<float>(q), da, work[l]);
          }
        } else {
#pragma unroll
          for (int l = 0; l < 32; ++l) dst[l] = 0;
        }
        lds_sumq8a[plane * 16 * Q4_K_SUBBLOCKS + actrow * Q4_K_SUBBLOCKS + sub] = qsum;
      }
    }
    __syncthreads();

    const uint8_t* block_ptr = block_base + blk * Q4_K_BLOCK_BYTES;
    // Hoist the per-block metadata decode out of the sub loop: d/dmin and
    // all eight subblock scale/min values are read once per K block.
    float d_f = 0.0f, dmin_f = 0.0f;
    float sub_d[Q4_K_SUBBLOCKS];
    float sub_m[Q4_K_SUBBLOCKS];
    {
      if (wcol_valid) {
        d_f = fp16_bytes_to_float(block_ptr);
        dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* scales = block_ptr + 4;
#pragma unroll
        for (int sub = 0; sub < Q4_K_SUBBLOCKS; ++sub) {
          sub_d[sub] = d_f * static_cast<float>(gguf_q4_k_scale(scales, sub));
          sub_m[sub] = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sub));
        }
      } else {
#pragma unroll
        for (int sub = 0; sub < Q4_K_SUBBLOCKS; ++sub) {
          sub_d[sub] = 0.0f;
          sub_m[sub] = 0.0f;
        }
      }
    }
#pragma unroll 1
    for (int sub = 0; sub < Q4_K_SUBBLOCKS; ++sub) {
      // Weight bytes are the raw Q4_K nibble values q (0..15) exactly; the
      // min-offset term is reconstructed per subblock from the staged
      // activation q8 sums (llama mul_mat_q ds-trick), so the weight path has
      // no requantization error at all.
      int8_t wq8[32];
      if (wcol_valid) {
        const uint8_t* qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
          wq8[l] = static_cast<int8_t>(gguf_q4_k_quant(qs, sub, l));
        }
      } else {
#pragma unroll
        for (int l = 0; l < 32; ++l) wq8[l] = 0;
      }
      const float dsub = sub_d[sub];
      const float msub = sub_m[sub];

#pragma unroll
      for (int g = 0; g < 2; ++g) {
        iu8_int32x8 acc[IU8_ACT_PLANES];
#pragma unroll
        for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
          acc[plane] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
#pragma unroll
        for (int khalf = 0; khalf < 2; ++khalf) {
          signed char bbytes[16];
#pragma unroll
          for (int j = 0; j < 16; ++j) bbytes[j] = wq8[khalf * 16 + j];
          iu8_int32x4 b_frag;
          {
            int32_t words[4];
            memcpy(words, bbytes, 16);
            b_frag = (iu8_int32x4){words[0], words[1], words[2], words[3]};
          }
#pragma unroll
          for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
            signed char abytes[16];
            if (a_row >= 0) {
              const int8_t* src = lds_q8a + plane * 16 * IU8_LDS_ROW_STRIDE +
                                  (g * 8 + a_row) * IU8_LDS_ROW_STRIDE +
                                  sub * Q4_K_SUBBLOCK + khalf * 16;
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = src[j];
            } else {
#pragma unroll
              for (int j = 0; j < 16; ++j) abytes[j] = 0;
            }
            iu8_int32x4 a_frag;
            {
              int32_t words[4];
              memcpy(words, abytes, 16);
              a_frag = (iu8_int32x4){words[0], words[1], words[2], words[3]};
            }
            acc[plane] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_frag, true, b_frag, acc[plane], true);
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int r = g * 8 + i;
          float main_sum = 0.0f;
          float act_sum = 0.0f;
#pragma unroll
          for (int plane = 0; plane < IU8_ACT_PLANES; ++plane) {
            const float da = lds_da[plane * 16 * Q4_K_SUBBLOCKS + r * Q4_K_SUBBLOCKS + sub];
            const int asumb = lds_sumq8a[plane * 16 * Q4_K_SUBBLOCKS + r * Q4_K_SUBBLOCKS + sub];
            const int* cr = reinterpret_cast<const int*>(&acc[plane]);
            main_sum += da * static_cast<float>(cr[i]);
            act_sum += da * static_cast<float>(asumb);
          }
          sum[r] += dsub * main_sum - msub * act_sum;
        }
      }
    }
    __syncthreads();
  }

  if (!col_valid) return;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const int64_t local_row = tile_row_in_expert + r;
    if (local_row < expert_count) {
      const int64_t compact_row = compact_base + local_row;
      if (compact_row < compact_rows) {
        out[compact_row * out_features_total + global_col] =
            float_to_scalar<scalar_t>(sum[r]);
      }
    }
  }
}

template <typename scalar_t>
static int launch_selected_dual_wmma_iu8(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_common(
          x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a,
          qweight_b, out, compact_rows, in_features, out_features_a,
          out_features_b, num_experts, wmma_total_rows, 64, 16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if ((out_features_a % 128) != 0 || (out_features_b % 128) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 127) / 128),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_wmma_iu8_prefill_kernel<scalar_t>),
      grid, block, 0, stream,
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a,
      qweight_b, out, compact_rows, in_features, out_features_a,
      out_features_b, num_experts, wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_selected_dual_grouped_rowbatch8_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* qweight_a,
    const void* qweight_b,
    void* output_a,
    void* output_b,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !qweight_a || !qweight_b || !output_a ||
      !output_b || compact_rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % QK_K) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int row_batch = 8;
  constexpr int threads = 128;
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<row_batch, 1>),
      dim3(static_cast<uint32_t>(out_features), static_cast<uint32_t>(num_experts)),
      dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output_a), static_cast<uint16_t*>(output_b),
      compact_rows, num_experts, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_selected_dual_grouped_rowbatch8_out4_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* qweight_a,
    const void* qweight_b,
    void* output_a,
    void* output_b,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !qweight_a || !qweight_b || !output_a ||
      !output_b || compact_rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % QK_K) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int row_batch = 8;
  constexpr int out_batch = 4;
  constexpr int threads = 128;
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<row_batch, out_batch>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(num_experts)),
      dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output_a), static_cast<uint16_t*>(output_b),
      compact_rows, num_experts, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* qweight_a,
    const void* qweight_b,
    void* output_a,
    void* output_b,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !qweight_a || !qweight_b || !output_a ||
      !output_b || compact_rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % QK_K) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int row_batch = 8;
  constexpr int out_batch = 4;
  constexpr int expert_workers = 64;
  constexpr int threads = 128;
  const int64_t worker_count = num_experts < expert_workers ? num_experts : expert_workers;
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<row_batch, out_batch, true>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(worker_count)),
      dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output_a), static_cast<uint16_t*>(output_b),
      compact_rows, num_experts, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* qweight_a,
    const void* qweight_b,
    void* output_a,
    void* output_b,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  // PF-3 M1 candidate: identical template, grid, and 128-thread strict
  // reduction tree as the production expertgrid64 owner, instantiated with
  // PAIR_BLOCK_WEIGHTS=false. The per-thread column order (column = tid,
  // stride 128) and the direct gguf_q4_k_weight fp32 dequant match the
  // owner's PAIR=true math bit-for-bit; only the shared-metadata predecode
  // phase and its __syncthreads barrier are removed from the schedule.
  if (!input || !expert_start || !qweight_a || !qweight_b || !output_a ||
      !output_b || compact_rows <= 0 || num_experts <= 0 || in_features <= 0 ||
      (in_features % QK_K) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int row_batch = 8;
  constexpr int out_batch = 4;
  constexpr int expert_workers = 64;
  constexpr int threads = 128;
  const int64_t worker_count = num_experts < expert_workers ? num_experts : expert_workers;
  hipLaunchKernelGGL(
      (gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<row_batch, out_batch, false>),
      dim3(static_cast<uint32_t>((out_features + out_batch - 1) / out_batch),
           static_cast<uint32_t>(worker_count)),
      dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(qweight_a),
      static_cast<const uint8_t*>(qweight_b),
      static_cast<uint16_t*>(output_a), static_cast<uint16_t*>(output_b),
      compact_rows, num_experts, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_selected_dual<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      tile_m,
      tile_n,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_selected_dual<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      tile_m,
      tile_n,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t hot_threshold,
    hipStream_t stream) {
  return launch_selected_dual_hot_fulltile32_tailcold16<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      hot_threshold,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    int64_t hot_threshold,
    hipStream_t stream) {
  return launch_selected_dual_hot_fulltile32_tailcold16<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      hot_threshold,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    const half_t* sidemeta_a,
    const half_t* sidemeta_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_sidemeta32<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      sidemeta_a,
      sidemeta_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    const half_t* sidemeta_a,
    const half_t* sidemeta_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_sidemeta32<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      sidemeta_a,
      sidemeta_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_wmma_iu8<uint16_t>(
      x, expert_start_compact, expert_start_wmma, tile_expert, qweight_a,
      qweight_b, out, compact_rows, in_features, out_features_a,
      out_features_b, num_experts, wmma_total_rows, stream);
}
