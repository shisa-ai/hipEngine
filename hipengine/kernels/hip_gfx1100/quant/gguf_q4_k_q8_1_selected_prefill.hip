// Diagnostic selected-MoE Q4_K x prequantized-Q8_1 prefill prototype.
//
// This standalone kernel is a same-shape experiment for the llama.cpp MMQ
// prefill hypothesis.  It consumes raw GGUF Q4_K gate/up expert weights and a
// prequantized activation tensor with one Q8_1-style scale and sum per 32-K
// block.  It is intentionally not wired into model dispatch; use the matching
// Python wrapper/microbench to compare against the resident Q4T16 selected-WMMA
// prefill path.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q8_1_BLOCK = 32;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_SCALE_BYTES = 12;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + Q4_K_SCALE_BYTES;
constexpr int Q4_K_QS_BYTES = QK_K / 2;
constexpr int Q4_K_TILE16_COLS = 16;
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int Q8_1_MMQ_BLOCK = 4 * Q8_1_BLOCK;        // == 128

struct block_q8_1_mmq_ds4 {
  // Matches llama.cpp's DS4 block_q8_1_mmq layout for Q4_K/Q5_K MMQ:
  // four (scale, sum) half2 pairs followed by 128 int8 quants.
  uint16_t ds4[8];
  int8_t qs[Q8_1_MMQ_BLOCK];
};
static_assert(sizeof(block_q8_1_mmq_ds4) == 4 * Q8_1_BLOCK + 8 * sizeof(uint16_t),
              "unexpected DS4 Q8_1 MMQ block size");

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

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_fp16_bits(float value) {
  union {
    half_t f16;
    uint16_t u16;
  } out;
  out.f16 = static_cast<half_t>(value);
  return out.u16;
}

__device__ inline float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  return fp16_bits_to_float(bits);
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

__device__ inline int pack_i8x4(const int8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline int pack_u8x4(const uint8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(ptr[i]) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline int pack_q4_k_u8x4(
    const uint8_t* qs,
    int subblock,
    int lane32_base) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(gguf_q4_k_quant(qs, subblock, lane32_base + i)) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline float wave32_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float wave32_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return value;
}

__launch_bounds__(128, 1)
__global__ void gguf_q8_1_mmq_ds4_pack_bf16_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_ds4* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t q8_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int ds4_idx = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  if (row >= rows || q8_block >= blocks_per_row) return;

  const int64_t in_idx = row * hidden + q8_block * Q8_1_MMQ_BLOCK + tid;
  const float value = bf16_bits_to_float(x[in_idx]);
  const float sum = wave32_sum(value);
  const float max_abs = wave32_max(fabsf(value));

  __shared__ float scale_shared[4];
  __shared__ float sum_shared[4];
  if (lane == 0) {
    scale_shared[ds4_idx] = max_abs / 127.0f;
    sum_shared[ds4_idx] = sum;
  }
  __syncthreads();

  const float scale = scale_shared[ds4_idx];
  int q = 0;
  if (scale > 0.0f) {
    const float qf = fminf(127.0f, fmaxf(-127.0f, rintf(value / scale)));
    q = static_cast<int>(qf);
  }

  block_q8_1_mmq_ds4* __restrict__ out_block = out + row * blocks_per_row + q8_block;
  out_block->qs[tid] = static_cast<int8_t>(q);
  if (lane == 0) {
    out_block->ds4[2 * ds4_idx + 0] = float_to_fp16_bits(scale_shared[ds4_idx]);
    out_block->ds4[2 * ds4_idx + 1] = float_to_fp16_bits(sum_shared[ds4_idx]);
  }
}

bool valid_q8_1_mmq_ds4_pack(
    const void* x,
    const void* out,
    int64_t rows,
    int64_t hidden) {
  return x != nullptr && out != nullptr && rows > 0 && hidden > 0 &&
      (hidden % Q8_1_MMQ_BLOCK) == 0;
}

bool valid_q8_1_selected_dual_common(
    const void* x_qs,
    const void* x_d,
    const void* x_sum,
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
    int64_t wmma_total_rows) {
  return x_qs != nullptr && x_d != nullptr && x_sum != nullptr &&
      expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && qweight_a != nullptr && qweight_b != nullptr &&
      out != nullptr && compact_rows > 0 && in_features > 0 &&
      out_features_a > 0 && out_features_b > 0 && num_experts > 0 &&
      wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 &&
      (in_features % QK_K) == 0 && (out_features_a % 16) == 0 &&
      (out_features_b % 16) == 0;
}

// One 256-thread block computes a scalar 16-row x 16-column output tile.  This
// deliberately simple mapping is a correctness/performance baseline for the
// integer-dot direction; it is not yet a llama.cpp-style tiled MMQ kernel.
__launch_bounds__(256, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_prefill_compact32_kernel(
    const int8_t* __restrict__ x_qs,
    const float* __restrict__ x_d,
    const float* __restrict__ x_sum,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_lane = tid >> 4;
  const int col_lane = tid & 15;

  const int64_t global_col = static_cast<int64_t>(col_tile) * 16 + col_lane;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  const int64_t local_row = tile_row_in_expert + row_lane;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + local_col * row_bytes) : expert_base;

  float acc = 0.0f;
  if (valid_row && valid_col && compact_row < compact_rows) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block_ptr);
      const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        const int64_t q8_block = compact_row * q8_blocks_per_row + blk * Q4_K_SUBBLOCKS + sb;
        const int8_t* __restrict__ xq = x_qs + q8_block * Q8_1_BLOCK;
        const float xd = x_d[q8_block];
        const float xsum = x_sum[q8_block];
        int dot = 0;
#pragma unroll
        for (int lane32 = 0; lane32 < Q8_1_BLOCK; ++lane32) {
          const int q4 = static_cast<int>(gguf_q4_k_quant(qs, sb, lane32));
          dot += q4 * static_cast<int>(xq[lane32]);
        }
        const float scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        const float min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
        acc += scale_f * xd * static_cast<float>(dot) - min_f * xsum;
      }
    }
    out[compact_row * out_features_total + global_col] = float_to_bf16_bits(acc);
  }
}

__launch_bounds__(256, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_lane = tid >> 4;
  const int col_lane = tid & 15;

  const int64_t global_col = static_cast<int64_t>(col_tile) * 16 + col_lane;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  const int64_t local_row = tile_row_in_expert + row_lane;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + local_col * row_bytes) : expert_base;

  float acc = 0.0f;
  if (valid_row && valid_col && compact_row < compact_rows) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block_ptr);
      const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        const int64_t q8_block = compact_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
        const int ds4_idx = sb & 3;
        const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
        const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK;
        const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
        const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
        int dot = 0;
#pragma unroll
        for (int lane32 = 0; lane32 < Q8_1_BLOCK; ++lane32) {
          const int q4 = static_cast<int>(gguf_q4_k_quant(qs, sb, lane32));
          dot += q4 * static_cast<int>(xq[lane32]);
        }
        const float scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        const float min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
        acc += scale_f * xd * static_cast<float>(dot) - min_f * xsum;
      }
    }
    out[compact_row * out_features_total + global_col] = float_to_bf16_bits(acc);
  }
}

__launch_bounds__(32, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
          };
        }
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
          };
        }
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(128, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 64 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
          };
        }
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ q4_a,
    const float* __restrict__ scale_a,
    const float* __restrict__ min_a,
    const uint8_t* __restrict__ q4_b,
    const float* __restrict__ scale_b,
    const float* __restrict__ min_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  if (local_col >= local_out_features) return;
  const uint8_t* __restrict__ q4 = use_b ? q4_b : q4_a;
  const float* __restrict__ scale = use_b ? scale_b : scale_a;
  const float* __restrict__ minv = use_b ? min_b : min_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t out_tiles = local_out_features / Q4_K_TILE16_COLS;
  const int64_t local_tile = local_col / Q4_K_TILE16_COLS;
  const int64_t col_in_tile = local_col & (Q4_K_TILE16_COLS - 1);

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const int64_t preview_index =
          (((((expert_id * out_tiles + local_tile) * Q4_K_TILE16_COLS + col_in_tile) *
             blocks_per_row + blk) * Q4_K_SUBBLOCKS + sb));
      const uint8_t* __restrict__ q4_vec = q4 + preview_index * Q4_K_SUBBLOCK;
      const float scale_f = scale[preview_index];
      const float min_f = minv[preview_index];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_u8x4(q4_vec + lane32_base + 0),
            pack_u8x4(q4_vec + lane32_base + 4),
            pack_u8x4(q4_vec + lane32_base + 8),
            pack_u8x4(q4_vec + lane32_base + 12),
        };
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  __shared__ uint8_t sh_qs[2][16][Q4_K_QS_BYTES];
  __shared__ float sh_scale[2][16][Q4_K_SUBBLOCKS];
  __shared__ float sh_min[2][16][Q4_K_SUBBLOCKS];

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t expert_id_raw = tile_expert[row_tile];
  const bool valid_expert = expert_id_raw >= 0 && expert_id_raw < num_experts;
  const int64_t expert_id = valid_expert ? expert_id_raw : 0;

  const int64_t compact_base = valid_expert ? expert_start_compact[expert_id] : 0;
  const int64_t expert_count = valid_expert
      ? (expert_start_compact[expert_id + 1] - compact_base)
      : 0;
  const int64_t wmma_base = valid_expert ? expert_start_wmma[expert_id] : 0;
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool in_output_range = global_col < out_features_total;
  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = valid_expert && in_output_range && local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = valid_expert && local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (half_id == 0) {
      const uint8_t* __restrict__ block_ptr = valid_col
          ? (weight_row + blk * Q4_K_BLOCK_BYTES)
          : weight_row;
      const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int i = 0; i < Q4_K_QS_BYTES; ++i) {
        sh_qs[wave_id][lane16][i] = valid_col ? qs[i] : 0;
      }
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        sh_scale[wave_id][lane16][sb] = valid_col
            ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb))
            : 0.0f;
        sh_min[wave_id][lane16][sb] = valid_col
            ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb))
            : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = sh_scale[wave_id][lane16][sb];
      const float min_f = sh_min[wave_id][lane16][sb];
      const uint8_t* __restrict__ qs_tile = &sh_qs[wave_id][lane16][0];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 0),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 4),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 8),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 12),
        };
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  __shared__ uint8_t sh_q4[2][16][Q4_K_SUBBLOCKS][Q4_K_SUBBLOCK];
  __shared__ float sh_scale[2][16][Q4_K_SUBBLOCKS];
  __shared__ float sh_min[2][16][Q4_K_SUBBLOCKS];

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t expert_id_raw = tile_expert[row_tile];
  const bool valid_expert = expert_id_raw >= 0 && expert_id_raw < num_experts;
  const int64_t expert_id = valid_expert ? expert_id_raw : 0;

  const int64_t compact_base = valid_expert ? expert_start_compact[expert_id] : 0;
  const int64_t expert_count = valid_expert
      ? (expert_start_compact[expert_id + 1] - compact_base)
      : 0;
  const int64_t wmma_base = valid_expert ? expert_start_wmma[expert_id] : 0;
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool in_output_range = global_col < out_features_total;
  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = valid_expert && in_output_range && local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = valid_expert && local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (half_id == 0) {
      const uint8_t* __restrict__ block_ptr = valid_col
          ? (weight_row + blk * Q4_K_BLOCK_BYTES)
          : weight_row;
      const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        sh_scale[wave_id][lane16][sb] = valid_col
            ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb))
            : 0.0f;
        sh_min[wave_id][lane16][sb] = valid_col
            ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb))
            : 0.0f;
#pragma unroll
        for (int lane32 = 0; lane32 < Q4_K_SUBBLOCK; ++lane32) {
          sh_q4[wave_id][lane16][sb][lane32] = valid_col
              ? gguf_q4_k_quant(qs, sb, lane32)
              : 0;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = sh_scale[wave_id][lane16][sb];
      const float min_f = sh_min[wave_id][lane16][sb];
      const uint8_t* __restrict__ q4_tile = &sh_q4[wave_id][lane16][sb][0];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_u8x4(q4_tile + lane32_base + 0),
            pack_u8x4(q4_tile + lane32_base + 4),
            pack_u8x4(q4_tile + lane32_base + 8),
            pack_u8x4(q4_tile + lane32_base + 12),
        };
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(32, 1)
__global__ void gguf_q4_k_q8_1_wmma_i8_probe_16x16_kernel(
    const int8_t* __restrict__ a_rows,
    const uint8_t* __restrict__ b_cols,
    int32_t* __restrict__ out) {
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  int32x4_t a_vec = {
      pack_i8x4(a_rows + lane16 * 16 + 0),
      pack_i8x4(a_rows + lane16 * 16 + 4),
      pack_i8x4(a_rows + lane16 * 16 + 8),
      pack_i8x4(a_rows + lane16 * 16 + 12),
  };
  int32x4_t b_vec = {
      pack_u8x4(b_cols + lane16 * 16 + 0),
      pack_u8x4(b_cols + lane16 * 16 + 4),
      pack_u8x4(b_cols + lane16 * 16 + 8),
      pack_u8x4(b_cols + lane16 * 16 + 12),
  };
  int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
  acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
      true, a_vec, true, b_vec, acc, false);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int row = half_id + j * 2;
    out[row * 16 + lane16] = acc[j];
  }
}

int launch_wmma_i8_probe_16x16(
    const int8_t* a_rows,
    const uint8_t* b_cols,
    int32_t* out,
    hipStream_t stream) {
  if (a_rows == nullptr || b_cols == nullptr || out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gguf_q4_k_q8_1_wmma_i8_probe_16x16_kernel,
      dim3(1),
      dim3(32),
      0,
      stream,
      a_rows,
      b_cols,
      out);
  return static_cast<int>(hipGetLastError());
}

int launch_q8_1_mmq_ds4_pack_bf16(
    const uint16_t* x,
    block_q8_1_mmq_ds4* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (!valid_q8_1_mmq_ds4_pack(x, out, rows, hidden)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  hipLaunchKernelGGL(
      gguf_q8_1_mmq_ds4_pack_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks_per_row), static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x,
      out,
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_compact32(
    const int8_t* x_qs,
    const float* x_d,
    const float* x_sum,
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
  if (!valid_q8_1_selected_dual_common(
          x_qs,
          x_d,
          x_sum,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_qs,
      x_d,
      x_sum,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma64_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 63) / 64),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_preview_wmma32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* q4_a,
    const float* scale_a,
    const float* min_a,
    const uint8_t* q4_b,
    const float* scale_b,
    const float* min_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          q4_a,
          q4_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows) || scale_a == nullptr || min_a == nullptr ||
      scale_b == nullptr || min_b == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      q4_a,
      scale_a,
      min_a,
      q4_b,
      scale_b,
      min_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_ldspack_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_lds_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_compact32(
    const block_q8_1_mmq_ds4* x_q8,
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
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
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
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
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
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_q8_1_wmma_i8_probe_16x16(
    const int8_t* a_rows,
    const uint8_t* b_cols,
    int32_t* out,
    hipStream_t stream) {
  return launch_wmma_i8_probe_16x16(a_rows, b_cols, out, stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_pack_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_pack_bf16(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
    const int8_t* x_qs,
    const float* x_d,
    const float* x_sum,
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
  return launch_selected_dual_q8_1_compact32(
      x_qs,
      x_d,
      x_sum,
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_wmma_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_wmma32_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_wmma64_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* q4_a,
    const float* scale_a,
    const float* min_a,
    const uint8_t* q4_b,
    const float* scale_b,
    const float* min_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_preview_wmma32_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      q4_a,
      scale_a,
      min_a,
      q4_b,
      scale_b,
      min_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_wmma32_ldspack_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_wmma32_lds_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
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
  return launch_selected_dual_q8_1_ds4_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
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
      stream);
}
