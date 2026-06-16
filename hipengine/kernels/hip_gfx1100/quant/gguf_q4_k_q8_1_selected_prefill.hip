// Diagnostic selected-MoE Q4_K x prequantized-Q8_1 prefill prototype.
//
// This standalone kernel is a same-shape experiment for the llama.cpp MMQ
// prefill hypothesis.  It consumes raw GGUF Q4_K gate/up expert weights and a
// prequantized activation tensor with one Q8_1-style scale and sum per 32-K
// block.  It is intentionally not wired into model dispatch; use the matching
// Python wrapper/microbench to compare against the resident Q4T16 selected-WMMA
// prefill path.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q8_1_BLOCK = 32;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_SCALE_BYTES = 12;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + Q4_K_SCALE_BYTES;
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
