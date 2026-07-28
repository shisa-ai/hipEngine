// Expert-major scalar and compact-WMMA prefill for raw GGUF IQ experts.
//
// Scheduling follows qwen-kernel@52e240f9's grouped IQ3/IQ4 shaders, but the
// ABI is hipEngine's existing count/prefix/scatter-gather/tile-map contract.
// The included direct source supplies the pinned IQ tables and exact decode
// helpers; it also keeps the direct selected kernels in this code object as the
// required unfused/fallback lineage.

#include "gguf_iq_gemv.hip"

namespace {

typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

struct IQ2Segment {
  float magnitude[8];
  float scale;
};

struct IQ2Pair16 {
  float magnitude[16];
  float scale;
};

struct IQ3Segment {
  float magnitude[8];
  float scale;
};

struct IQ4Segment {
  float value[32];
  float scale;
};

__device__ inline IQ2Segment load_iq2_segment(
    const uint8_t* __restrict__ weight_row,
    int group8_linear) {
  IQ2Segment segment;
  const int block_idx = group8_linear >> 5;
  const int group8 = group8_linear & 31;
  const int group32 = group8 >> 2;
  const int local8 = group8 & 3;
  const uint8_t* __restrict__ block =
      weight_row + static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
  const uint16_t q = load_u16_le(block + 2 + 2 * (4 * group32 + local8));
  const uint8_t scale_byte = block[66 + group32];
  const uint32_t scale =
      (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
  const uint32_t signs = iq3_sign_byte(static_cast<uint32_t>(q >> 9));
  const uint16_t grid = IQ2_XS_GRID_PACKED[q & 511U];
  segment.scale = fp16_bytes_to_float(block) *
                  (0.5f + static_cast<float>(scale)) * 0.25f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t code = (static_cast<uint32_t>(grid) >> (2 * j)) & 3U;
    segment.magnitude[j] =
        iq2_xs_signed_magnitude(code, (signs >> j) & 1U);
  }
  return segment;
}

__device__ inline float dot_iq2_segment(
    const IQ2Segment& segment,
    const uint16_t* __restrict__ x) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sum += segment.magnitude[j] * bf16_bits_to_float(x[j]);
  }
  return segment.scale * sum;
}

// Decode the same adjacent two-group unit as production's local64 pair16
// selected kernel, but keep the decoded magnitudes live across routed rows.
__device__ inline IQ2Pair16 load_iq2_pair16(
    const uint8_t* __restrict__ weight_row,
    int pair16) {
  IQ2Pair16 segment;
  const int block_idx = pair16 >> 4;
  const int group8_even = (pair16 & 15) * 2;
  const int group32 = group8_even >> 2;
  const int local8 = group8_even & 3;
  const uint8_t* __restrict__ block =
      weight_row + static_cast<int64_t>(block_idx) * IQ2_XS_BLOCK_BYTES;
  const uint32_t packed_q =
      load_u32_le(block + 2 + 2 * (4 * group32 + local8));
  const uint16_t q0 = static_cast<uint16_t>(packed_q);
  const uint16_t q1 = static_cast<uint16_t>(packed_q >> 16);
  const uint8_t scale_byte = block[66 + group32];
  const uint32_t scale =
      (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
  const uint32_t signs0 = iq3_sign_byte(static_cast<uint32_t>(q0 >> 9));
  const uint32_t signs1 = iq3_sign_byte(static_cast<uint32_t>(q1 >> 9));
  const uint16_t grid0 = IQ2_XS_GRID_PACKED[q0 & 511U];
  const uint16_t grid1 = IQ2_XS_GRID_PACKED[q1 & 511U];
  segment.scale = fp16_bytes_to_float(block) *
                  (0.5f + static_cast<float>(scale)) * 0.25f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t code0 = (static_cast<uint32_t>(grid0) >> (2 * j)) & 3U;
    const uint32_t code1 = (static_cast<uint32_t>(grid1) >> (2 * j)) & 3U;
    segment.magnitude[j] =
        iq2_xs_signed_magnitude(code0, (signs0 >> j) & 1U);
    segment.magnitude[8 + j] =
        iq2_xs_signed_magnitude(code1, (signs1 >> j) & 1U);
  }
  return segment;
}

__device__ inline float dot_iq2_pair16(
    const IQ2Pair16& segment,
    const uint16_t* __restrict__ x) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    sum0 += segment.magnitude[j] * bf16_bits_to_float(x[j]);
    sum1 += segment.magnitude[8 + j] * bf16_bits_to_float(x[8 + j]);
  }
  return segment.scale * sum0 + segment.scale * sum1;
}

__device__ inline IQ3Segment load_iq3_segment(
    const uint8_t* __restrict__ weight_row,
    int group8_linear) {
  IQ3Segment segment;
  const int block_idx = group8_linear >> 5;
  const int group8 = group8_linear & 31;
  const int group32 = group8 >> 2;
  const int local8 = group8 & 3;
  const uint8_t* __restrict__ block =
      weight_row + static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
  const uint8_t* __restrict__ qs = block + 2;
  const uint32_t aux = load_u32_le(qs + 64 + 4 * group32);
  const uint32_t signs = iq3_sign_byte((aux >> (7 * local8)) & 127U);
  const uint32_t grid1 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8]];
  const uint32_t grid2 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8 + 1]];
  segment.scale =
      fp16_bytes_to_float(block) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float magnitude1 = static_cast<float>((grid1 >> (8 * j)) & 255U);
    const float magnitude2 = static_cast<float>((grid2 >> (8 * j)) & 255U);
    segment.magnitude[j] =
        (signs & (1U << j)) != 0U ? -magnitude1 : magnitude1;
    segment.magnitude[j + 4] =
        (signs & (1U << (j + 4))) != 0U ? -magnitude2 : magnitude2;
  }
  return segment;
}

__device__ inline float dot_iq3_segment(
    const IQ3Segment& segment,
    const uint16_t* __restrict__ x) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sum += segment.magnitude[j] * bf16_bits_to_float(x[j]);
    sum += segment.magnitude[j + 4] * bf16_bits_to_float(x[j + 4]);
  }
  return segment.scale * sum;
}

__device__ inline IQ4Segment load_iq4_segment(
    const uint8_t* __restrict__ weight_row,
    int subblock_linear) {
  IQ4Segment segment;
  const int block_idx = subblock_linear >> 3;
  const int subblock = subblock_linear & 7;
  const uint8_t* __restrict__ block =
      weight_row + static_cast<int64_t>(block_idx) * IQ4_XS_BLOCK_BYTES;
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scale_l_byte = block[4 + (subblock >> 1)];
  const uint32_t scale_l =
      (static_cast<uint32_t>(scale_l_byte) >> (4 * (subblock & 1))) & 0x0FU;
  const uint32_t scale_h =
      (static_cast<uint32_t>(scales_h) >> (2 * subblock)) & 0x03U;
  const int scale = static_cast<int>(scale_l | (scale_h << 4)) - 32;
  segment.scale = fp16_bytes_to_float(block) * static_cast<float>(scale);
  const uint8_t* __restrict__ qs = block + 8 + subblock * 16;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint8_t packed = qs[j];
    segment.value[j] = static_cast<float>(IQ4_NL_VALUES[packed & 0x0FU]);
    segment.value[j + 16] = static_cast<float>(IQ4_NL_VALUES[packed >> 4]);
  }
  return segment;
}

__device__ inline float dot_iq4_segment(
    const IQ4Segment& segment,
    const uint16_t* __restrict__ x) {
  float sum_lo = 0.0f;
  float sum_hi = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    sum_lo += segment.value[j] * bf16_bits_to_float(x[j]);
    sum_hi += segment.value[j + 16] * bf16_bits_to_float(x[j + 16]);
  }
  return segment.scale * (sum_lo + sum_hi);
}

template <int ROW_BATCH>
__device__ inline void reduce_block_batched(
    float (&value)[ROW_BATCH],
    float* __restrict__ wave_sums) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value[row] += __shfl_down(value[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      wave_sums[row * 8 + wave] = value[row];
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value[row] = 0.0f;
      for (int wave_idx = 0; wave_idx < waves; ++wave_idx) {
        value[row] += wave_sums[row * 8 + wave_idx];
      }
    }
  }
}

template <int ROW_BATCH>
__device__ inline void reduce_block_pairs_batched(
    float (&value_a)[ROW_BATCH],
    float (&value_b)[ROW_BATCH],
    float* __restrict__ wave_sums_a,
    float* __restrict__ wave_sums_b) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value_a[row] += __shfl_down(value_a[row], offset);
      value_b[row] += __shfl_down(value_b[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      wave_sums_a[row * 8 + wave] = value_a[row];
      wave_sums_b[row * 8 + wave] = value_b[row];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value_a[row] = 0.0f;
      value_b[row] = 0.0f;
#pragma unroll
      for (int wave_idx = 0; wave_idx < 8; ++wave_idx) {
        value_a[row] += wave_sums_a[row * 8 + wave_idx];
        value_b[row] += wave_sums_b[row * 8 + wave_idx];
      }
    }
  }
}

template <int ROW_BATCH>
__device__ inline void reduce_local64_pairs_batched(
    float (&value_a)[ROW_BATCH],
    float (&value_b)[ROW_BATCH],
    float* __restrict__ wave_sums_a,
    float* __restrict__ wave_sums_b) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value_a[row] += __shfl_down(value_a[row], offset);
      value_b[row] += __shfl_down(value_b[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      wave_sums_a[row * 2 + wave] = value_a[row];
      wave_sums_b[row * 2 + wave] = value_b[row];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      value_a[row] = 0.0f;
      value_b[row] = 0.0f;
      value_a[row] += wave_sums_a[row * 2];
      value_b[row] += wave_sums_b[row * 2];
      value_a[row] += wave_sums_a[row * 2 + 1];
      value_b[row] += wave_sums_b[row * 2 + 1];
    }
  }
}

// IQ2_XS uses the same 8-value/local256 mapping and exact reduction boundary
// as selected decode. At Laguna K=3072, each lane processes at most two groups.
__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_grouped_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int groups8 = in_features / 8;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  const int64_t out_stride = static_cast<int64_t>(2) * out_features;

  if (groups8 > blockDim.x) {
    for (int64_t row = begin; row < end; ++row) {
      float gate_acc = 0.0f;
      float up_acc = 0.0f;
      for (int group = threadIdx.x; group < groups8; group += blockDim.x) {
        const IQ2Segment gate_segment =
            load_iq2_segment(gate_weight + weight_offset, group);
        const IQ2Segment up_segment =
            load_iq2_segment(up_weight + weight_offset, group);
        const uint16_t* __restrict__ x_segment =
            x + row * in_features + group * 8;
        gate_acc += dot_iq2_segment(gate_segment, x_segment);
        up_acc += dot_iq2_segment(up_segment, x_segment);
      }
      reduce_block_pair(gate_acc, up_acc);
      if (threadIdx.x == 0) {
        out[row * out_stride + out_col] = float_to_bf16_bits(gate_acc);
        out[row * out_stride + out_features + out_col] =
            float_to_bf16_bits(up_acc);
      }
      __syncthreads();
    }
    return;
  }

  const bool active = threadIdx.x < groups8;
  IQ2Segment gate_segment{};
  IQ2Segment up_segment{};
  int x_offset = 0;
  if (active) {
    const int group = threadIdx.x;
    gate_segment = load_iq2_segment(gate_weight + weight_offset, group);
    up_segment = load_iq2_segment(up_weight + weight_offset, group);
    x_offset = group * 8;
  }
  for (int64_t row = begin; row < end; ++row) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (active) {
      const uint16_t* __restrict__ x_segment =
          x + row * in_features + x_offset;
      gate_acc = dot_iq2_segment(gate_segment, x_segment);
      up_acc = dot_iq2_segment(up_segment, x_segment);
    }
    reduce_block_pair(gate_acc, up_acc);
    if (threadIdx.x == 0) {
      out[row * out_stride + out_col] = float_to_bf16_bits(gate_acc);
      out[row * out_stride + out_features + out_col] =
          float_to_bf16_bits(up_acc);
    }
    __syncthreads();
  }
}

template <int ROW_BATCH, bool FUSE_SILU>
__device__ __forceinline__ void iq2_xs_grouped_rowbatch_body(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int64_t begin,
    int64_t end,
    int in_features,
    int out_features,
    int out_col,
    int64_t weight_offset,
    float* __restrict__ wave_sums_gate,
    float* __restrict__ wave_sums_up) {
  const int groups8 = in_features / 8;
  const int64_t out_stride =
      static_cast<int64_t>(FUSE_SILU ? 1 : 2) * out_features;
  for (int64_t row_base = begin; row_base < end; row_base += ROW_BATCH) {
    float gate_acc[ROW_BATCH] = {};
    float up_acc[ROW_BATCH] = {};
    for (int group = threadIdx.x; group < groups8; group += blockDim.x) {
      const IQ2Segment gate_segment =
          load_iq2_segment(gate_weight + weight_offset, group);
      const IQ2Segment up_segment =
          load_iq2_segment(up_weight + weight_offset, group);
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          const uint16_t* __restrict__ x_segment =
              x + row * in_features + group * 8;
          gate_acc[row_idx] += dot_iq2_segment(gate_segment, x_segment);
          up_acc[row_idx] += dot_iq2_segment(up_segment, x_segment);
        }
      }
    }
    reduce_block_pairs_batched(
        gate_acc, up_acc, wave_sums_gate, wave_sums_up);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          if constexpr (FUSE_SILU) {
            const float gate = bf16_bits_to_float(
                float_to_bf16_bits(gate_acc[row_idx]));
            const float up = bf16_bits_to_float(
                float_to_bf16_bits(up_acc[row_idx]));
            const float sig = 1.0f / (1.0f + expf(-gate));
            out[row * out_stride + out_col] =
                float_to_bf16_bits(gate * sig * up);
          } else {
            out[row * out_stride + out_col] =
                float_to_bf16_bits(gate_acc[row_idx]);
            out[row * out_stride + out_features + out_col] =
                float_to_bf16_bits(up_acc[row_idx]);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int ROW_BATCH, bool FUSE_SILU>
__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  __shared__ float wave_sums_gate[ROW_BATCH * 8];
  __shared__ float wave_sums_up[ROW_BATCH * 8];
  iq2_xs_grouped_rowbatch_body<ROW_BATCH, FUSE_SILU>(
      x,
      gate_weight,
      up_weight,
      out,
      begin,
      end,
      in_features,
      out_features,
      out_col,
      weight_offset,
      wave_sums_gate,
      wave_sums_up);
}

// Pair16/local64 WPF-2b candidate. One block owns one output column so the
// screen isolates routed-row reuse from output tiling. Every row preserves the
// production pair16 loop, two-wave shuffle tree, serial wave sum, BF16 gate/up
// projection boundaries, and fused SiLU boundary.
template <int ROW_BATCH>
__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int pairs16 = in_features / 16;
  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row = gate_weight + weight_offset;
  const uint8_t* __restrict__ up_row = up_weight + weight_offset;
  __shared__ float wave_sums_gate[ROW_BATCH * 2];
  __shared__ float wave_sums_up[ROW_BATCH * 2];

  for (int64_t row_base = begin; row_base < end; row_base += ROW_BATCH) {
    float gate_acc[ROW_BATCH] = {};
    float up_acc[ROW_BATCH] = {};
    for (int pair16 = threadIdx.x; pair16 < pairs16;
         pair16 += blockDim.x) {
      const IQ2Pair16 gate_segment = load_iq2_pair16(gate_row, pair16);
      const IQ2Pair16 up_segment = load_iq2_pair16(up_row, pair16);
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          const uint16_t* __restrict__ x_pair =
              x + row * in_features + pair16 * 16;
          gate_acc[row_idx] += dot_iq2_pair16(gate_segment, x_pair);
          up_acc[row_idx] += dot_iq2_pair16(up_segment, x_pair);
        }
      }
    }
    reduce_local64_pairs_batched(
        gate_acc, up_acc, wave_sums_gate, wave_sums_up);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          const float gate =
              bf16_bits_to_float(float_to_bf16_bits(gate_acc[row_idx]));
          const float up =
              bf16_bits_to_float(float_to_bf16_bits(up_acc[row_idx]));
          const float sig = 1.0f / (1.0f + expf(-gate));
          out[row * out_features + out_col] =
              float_to_bf16_bits(gate * sig * up);
        }
      }
    }
    __syncthreads();
  }
}

// Every block sees one expert's row count, so this decision is wave-uniform and
// adapts mixed routing without copying expert counts back to the host.
__launch_bounds__(256, 2)
__global__ void gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int row_bytes = (in_features / QK_K) * IQ2_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  __shared__ float wave_sums_gate[4 * 8];
  __shared__ float wave_sums_up[4 * 8];
  const int64_t rows = end - begin;
  if (rows == 1) {
    iq2_xs_grouped_rowbatch_body<1, false>(
        x, gate_weight, up_weight, out, begin, end, in_features, out_features,
        out_col, weight_offset, wave_sums_gate, wave_sums_up);
  } else if (rows == 2) {
    iq2_xs_grouped_rowbatch_body<2, false>(
        x, gate_weight, up_weight, out, begin, end, in_features, out_features,
        out_col, weight_offset, wave_sums_gate, wave_sums_up);
  } else {
    iq2_xs_grouped_rowbatch_body<4, false>(
        x, gate_weight, up_weight, out, begin, end, in_features, out_features,
        out_col, weight_offset, wave_sums_gate, wave_sums_up);
  }
}

// One block owns one (expert, output-column) pair. Every participating lane
// dequantizes its weight segment once, retains it in registers, and reuses it
// across every compact row routed to that expert.
__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_dual_grouped_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int groups8 = in_features / 8;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  const int64_t out_stride = static_cast<int64_t>(2) * out_features;

  // K=3072 has 384 eight-value groups. Preserve the direct selected kernel's
  // per-thread group order without retaining two gate/up segments in VGPRs.
  if (groups8 > blockDim.x) {
    for (int64_t row = begin; row < end; ++row) {
      float gate_acc = 0.0f;
      float up_acc = 0.0f;
      for (int group = threadIdx.x; group < groups8; group += blockDim.x) {
        const IQ3Segment gate_segment =
            load_iq3_segment(gate_weight + weight_offset, group);
        const IQ3Segment up_segment =
            load_iq3_segment(up_weight + weight_offset, group);
        const uint16_t* __restrict__ x_segment =
            x + row * in_features + group * 8;
        gate_acc += dot_iq3_segment(gate_segment, x_segment);
        up_acc += dot_iq3_segment(up_segment, x_segment);
      }
      reduce_block_pair(gate_acc, up_acc);
      if (threadIdx.x == 0) {
        out[row * out_stride + out_col] = float_to_bf16_bits(gate_acc);
        out[row * out_stride + out_features + out_col] =
            float_to_bf16_bits(up_acc);
      }
      __syncthreads();
    }
    return;
  }

  const bool active = threadIdx.x < groups8;
  IQ3Segment gate_segment{};
  IQ3Segment up_segment{};
  int x_offset = 0;
  if (active) {
    const int group = threadIdx.x;
    gate_segment = load_iq3_segment(gate_weight + weight_offset, group);
    up_segment = load_iq3_segment(up_weight + weight_offset, group);
    x_offset = group * 8;
  }

  for (int64_t row = begin; row < end; ++row) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (active) {
      const uint16_t* __restrict__ x_segment =
          x + row * in_features + x_offset;
      gate_acc = dot_iq3_segment(gate_segment, x_segment);
      up_acc = dot_iq3_segment(up_segment, x_segment);
    }
    reduce_block_pair(gate_acc, up_acc);
    if (threadIdx.x == 0) {
      out[row * out_stride + out_col] = float_to_bf16_bits(gate_acc);
      out[row * out_stride + out_features + out_col] =
          float_to_bf16_bits(up_acc);
    }
    __syncthreads();
  }
}

// Batch independent compact rows through one pair of block barriers. Each row
// keeps the exact RT1 lane dot, wave32 shuffle tree, and serial wave-0..7
// accumulation; only the synchronization schedule is interleaved.
template <int ROW_BATCH, bool FUSE_SILU>
__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int groups8 = in_features / 8;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  __shared__ float wave_sums_gate[ROW_BATCH * 8];
  __shared__ float wave_sums_up[ROW_BATCH * 8];
  const int64_t out_stride =
      static_cast<int64_t>(FUSE_SILU ? 1 : 2) * out_features;

  if (groups8 > blockDim.x) {
    for (int64_t row_base = begin; row_base < end; row_base += ROW_BATCH) {
      float gate_acc[ROW_BATCH] = {};
      float up_acc[ROW_BATCH] = {};
      for (int group = threadIdx.x; group < groups8; group += blockDim.x) {
        const IQ3Segment gate_segment =
            load_iq3_segment(gate_weight + weight_offset, group);
        const IQ3Segment up_segment =
            load_iq3_segment(up_weight + weight_offset, group);
#pragma unroll
        for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
          const int64_t row = row_base + row_idx;
          if (row < end) {
            const uint16_t* __restrict__ x_segment =
                x + row * in_features + group * 8;
            gate_acc[row_idx] += dot_iq3_segment(gate_segment, x_segment);
            up_acc[row_idx] += dot_iq3_segment(up_segment, x_segment);
          }
        }
      }
      reduce_block_pairs_batched(
          gate_acc, up_acc, wave_sums_gate, wave_sums_up);
      if (threadIdx.x == 0) {
#pragma unroll
        for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
          const int64_t row = row_base + row_idx;
          if (row < end) {
            if constexpr (FUSE_SILU) {
              const float gate = bf16_bits_to_float(
                  float_to_bf16_bits(gate_acc[row_idx]));
              const float up = bf16_bits_to_float(
                  float_to_bf16_bits(up_acc[row_idx]));
              const float sig = 1.0f / (1.0f + expf(-gate));
              out[row * out_stride + out_col] =
                  float_to_bf16_bits(gate * sig * up);
            } else {
              out[row * out_stride + out_col] =
                  float_to_bf16_bits(gate_acc[row_idx]);
              out[row * out_stride + out_features + out_col] =
                  float_to_bf16_bits(up_acc[row_idx]);
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  const bool active = threadIdx.x < groups8;
  IQ3Segment gate_segment{};
  IQ3Segment up_segment{};
  int x_offset = 0;
  if (active) {
    const int group = threadIdx.x;
    gate_segment = load_iq3_segment(gate_weight + weight_offset, group);
    up_segment = load_iq3_segment(up_weight + weight_offset, group);
    x_offset = group * 8;
  }

  for (int64_t row_base = begin; row_base < end; row_base += ROW_BATCH) {
    float gate_acc[ROW_BATCH] = {};
    float up_acc[ROW_BATCH] = {};
#pragma unroll
    for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
      const int64_t row = row_base + row_idx;
      if (row < end && active) {
        const uint16_t* __restrict__ x_segment =
            x + row * in_features + x_offset;
        gate_acc[row_idx] = dot_iq3_segment(gate_segment, x_segment);
        up_acc[row_idx] = dot_iq3_segment(up_segment, x_segment);
      }
    }
    reduce_block_pairs_batched(
        gate_acc, up_acc, wave_sums_gate, wave_sums_up);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          if constexpr (FUSE_SILU) {
            const float gate = bf16_bits_to_float(
                float_to_bf16_bits(gate_acc[row_idx]));
            const float up = bf16_bits_to_float(
                float_to_bf16_bits(up_acc[row_idx]));
            const float sig = 1.0f / (1.0f + expf(-gate));
            out[row * out_stride + out_col] =
                float_to_bf16_bits(gate * sig * up);
          } else {
            out[row * out_stride + out_col] =
                float_to_bf16_bits(gate_acc[row_idx]);
            out[row * out_stride + out_features + out_col] =
                float_to_bf16_bits(up_acc[row_idx]);
          }
        }
      }
    }
    __syncthreads();
  }
}

// One block owns one IQ3 (expert, output-column) down row and reuses its
// decoded weight segments across compact routed rows. Row batching preserves
// each selected-single dot and reduction boundary while amortizing barriers.
template <int ROW_BATCH>
__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int groups8 = in_features / 8;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  __shared__ float wave_sums[ROW_BATCH * 8];

  for (int64_t row_base = begin; row_base < end; row_base += ROW_BATCH) {
    float acc[ROW_BATCH] = {};
    for (int group = threadIdx.x; group < groups8; group += blockDim.x) {
      const IQ3Segment segment =
          load_iq3_segment(qweight + weight_offset, group);
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          acc[row_idx] += dot_iq3_segment(
              segment, x + row * in_features + group * 8);
        }
      }
    }
    reduce_block_batched(acc, wave_sums);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int row_idx = 0; row_idx < ROW_BATCH; ++row_idx) {
        const int64_t row = row_base + row_idx;
        if (row < end) {
          out[row * out_features + out_col] =
              float_to_bf16_bits(acc[row_idx]);
        }
      }
    }
    __syncthreads();
  }
}

__launch_bounds__(128, 2)
__global__ void gguf_iq4_xs_selected_dual_grouped_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int subblocks = in_features / 32;
  const bool active = threadIdx.x < subblocks;
  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  IQ4Segment gate_segment{};
  IQ4Segment up_segment{};
  int x_offset = 0;
  if (active) {
    const int subblock = threadIdx.x;
    gate_segment = load_iq4_segment(gate_weight + weight_offset, subblock);
    up_segment = load_iq4_segment(up_weight + weight_offset, subblock);
    x_offset = subblock * 32;
  }

  const int64_t out_stride = static_cast<int64_t>(2) * out_features;
  for (int64_t row = begin; row < end; ++row) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (active) {
      const uint16_t* __restrict__ x_segment =
          x + row * in_features + x_offset;
      gate_acc = dot_iq4_segment(gate_segment, x_segment);
      up_acc = dot_iq4_segment(up_segment, x_segment);
    }
    reduce_block_pair(gate_acc, up_acc);
    if (threadIdx.x == 0) {
      out[row * out_stride + out_col] = float_to_bf16_bits(gate_acc);
      out[row * out_stride + out_features + out_col] =
          float_to_bf16_bits(up_acc);
    }
    __syncthreads();
  }
}

__launch_bounds__(128, 2)
__global__ void gguf_iq4_xs_selected_grouped_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int in_features,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  const int subblocks = in_features / 32;
  const bool active = threadIdx.x < subblocks;
  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  IQ4Segment segment{};
  int x_offset = 0;
  if (active) {
    const int subblock = threadIdx.x;
    segment = load_iq4_segment(weight_row, subblock);
    x_offset = subblock * 32;
  }

  for (int64_t row = begin; row < end; ++row) {
    float acc = 0.0f;
    if (active) {
      acc = dot_iq4_segment(segment, x + row * in_features + x_offset);
    }
    const float total = reduce_block_sum(acc);
    if (threadIdx.x == 0) {
      out[row * out_features + out_col] = float_to_bf16_bits(total);
    }
    __syncthreads();
  }
}

// The production IQ4 down shape has K=512, so only lanes 0..15 own
// subblocks. The local128 fallback therefore reduces one populated wave plus
// three all-zero waves. This one-wave sibling keeps the populated wave's dot
// and shuffle association exactly while removing 96 idle lanes and all
// cross-wave barriers.
__launch_bounds__(32, 8)
__global__ void gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int compact_rows,
    int out_features,
    int num_experts) {
  const int out_col = blockIdx.x;
  const int expert = blockIdx.y;
  if (out_col >= out_features || expert >= num_experts) return;
  const int64_t begin = expert_start_compact[expert];
  const int64_t end = expert_start_compact[expert + 1];
  if (begin >= end) return;

  constexpr int in_features = 512;
  constexpr int subblocks = in_features / 32;
  const bool active = threadIdx.x < subblocks;
  constexpr int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(expert) * expert_bytes +
      static_cast<int64_t>(out_col) * row_bytes;
  IQ4Segment segment{};
  int x_offset = 0;
  if (active) {
    const int subblock = threadIdx.x;
    segment = load_iq4_segment(weight_row, subblock);
    x_offset = subblock * 32;
  }

  for (int64_t row = begin; row < end; ++row) {
    float acc = 0.0f;
    if (active) {
      acc = dot_iq4_segment(segment, x + row * in_features + x_offset);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down(acc, offset);
    }
    if (threadIdx.x == 0) {
      // Match reduce_block_sum's wave-0 accumulation into a +0.0 total. The
      // three omitted waves are identically +0.0 for this K=512 shape.
      float total = 0.0f;
      total += acc;
      total += 0.0f;
      total += 0.0f;
      total += 0.0f;
      out[row * out_features + out_col] = float_to_bf16_bits(total);
    }
  }
}

__device__ inline half16_t load_compact_bf16_half16(
    const uint16_t* __restrict__ x,
    int64_t row,
    int64_t in_features,
    int64_t k_base,
    bool valid) {
  half16_t values;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = valid
        ? static_cast<half_t>(bf16_bits_to_float(x[row * in_features + k_base + i]))
        : static_cast<half_t>(0.0f);
  }
  return values;
}

__device__ inline half16_t load_iq2_weight_half16(
    const uint8_t* __restrict__ block,
    int within_base,
    bool valid) {
  half16_t values;
  if (!valid) {
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = static_cast<half_t>(0.0f);
    return values;
  }
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const int group8 = (within_base >> 3) + pair;
    const int group32 = group8 >> 2;
    const int local8 = group8 & 3;
    const uint16_t q = load_u16_le(block + 2 + 2 * (4 * group32 + local8));
    const uint8_t scale_byte = block[66 + group32];
    const uint32_t scale =
        (static_cast<uint32_t>(scale_byte) >> (4 * (local8 >> 1))) & 0x0FU;
    const uint32_t signs = iq3_sign_byte(static_cast<uint32_t>(q >> 9));
    const uint16_t grid = IQ2_XS_GRID_PACKED[q & 511U];
    const float dequant_scale = fp16_bytes_to_float(block) *
                                (0.5f + static_cast<float>(scale)) * 0.25f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t code = (static_cast<uint32_t>(grid) >> (2 * j)) & 3U;
      values[pair * 8 + j] = static_cast<half_t>(
          dequant_scale * iq2_xs_signed_magnitude(code, (signs >> j) & 1U));
    }
  }
  return values;
}

__device__ inline half16_t load_iq3_weight_half16(
    const uint8_t* __restrict__ block,
    int within_base,
    bool valid) {
  half16_t values;
  if (!valid) {
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = static_cast<half_t>(0.0f);
    return values;
  }
  const uint8_t* __restrict__ qs = block + 2;
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const int group8 = (within_base >> 3) + pair;
    const int group32 = group8 >> 2;
    const int local8 = group8 & 3;
    const uint32_t aux = load_u32_le(qs + 64 + 4 * group32);
    const uint32_t signs = iq3_sign_byte((aux >> (7 * local8)) & 127U);
    const uint32_t grid1 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8]];
    const uint32_t grid2 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8 + 1]];
    const float scale =
        fp16_bytes_to_float(block) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float magnitude1 = static_cast<float>((grid1 >> (8 * j)) & 255U);
      const float magnitude2 = static_cast<float>((grid2 >> (8 * j)) & 255U);
      values[pair * 8 + j] = static_cast<half_t>(
          scale * ((signs & (1U << j)) != 0U ? -magnitude1 : magnitude1));
      values[pair * 8 + j + 4] = static_cast<half_t>(
          scale * ((signs & (1U << (j + 4))) != 0U ? -magnitude2 : magnitude2));
    }
  }
  return values;
}

__device__ inline half16_t load_iq4_weight_half16(
    const uint8_t* __restrict__ block,
    int within_base,
    bool valid) {
  half16_t values;
  if (!valid) {
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = static_cast<half_t>(0.0f);
    return values;
  }
  const int subblock = within_base >> 5;
  const bool upper = (within_base & 16) != 0;
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scale_l_byte = block[4 + (subblock >> 1)];
  const uint32_t scale_l =
      (static_cast<uint32_t>(scale_l_byte) >> (4 * (subblock & 1))) & 0x0FU;
  const uint32_t scale_h =
      (static_cast<uint32_t>(scales_h) >> (2 * subblock)) & 0x03U;
  const int scale = static_cast<int>(scale_l | (scale_h << 4)) - 32;
  const float dequant_scale = fp16_bytes_to_float(block) * static_cast<float>(scale);
  const uint8_t* __restrict__ qs = block + 8 + subblock * 16;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint8_t packed = qs[j];
    const int code = upper ? (packed >> 4) : (packed & 0x0F);
    values[j] = static_cast<half_t>(
        dequant_scale * static_cast<float>(IQ4_NL_VALUES[code]));
  }
  return values;
}

template <int QTYPE>
__device__ inline int iq_block_bytes() {
  return QTYPE == 2 ? IQ2_XS_BLOCK_BYTES
                    : (QTYPE == 3 ? IQ3_XXS_BLOCK_BYTES : IQ4_XS_BLOCK_BYTES);
}

template <int QTYPE>
__device__ inline half16_t load_iq_weight_half16(
    const uint8_t* __restrict__ block,
    int within_base,
    bool valid) {
  if constexpr (QTYPE == 2) {
    return load_iq2_weight_half16(block, within_base, valid);
  } else if constexpr (QTYPE == 3) {
    return load_iq3_weight_half16(block, within_base, valid);
  } else {
    return load_iq4_weight_half16(block, within_base, valid);
  }
}

// A wave computes one 16-column by 16-row tile. BF16 activations and raw IQ
// weights are converted to F16 WMMA fragments in registers; no resident weight
// sidecar is introduced.
template <int QTYPE>
__launch_bounds__(32, 2)
__global__ void gguf_iq_selected_dual_wmma_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t row_tile = blockIdx.y;
  const int64_t expert = tile_expert[row_tile];
  if (expert < 0 || expert >= num_experts) return;
  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t compact_base = expert_start_compact[expert];
  const int64_t expert_count = expert_start_compact[expert + 1] - compact_base;
  const int64_t local_row_base = row_tile * 16 - expert_start_wmma[expert];
  const int64_t local_row = local_row_base + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? compact_base + local_row : 0;

  const int64_t out_total = 2 * out_features;
  const int64_t global_col = static_cast<int64_t>(blockIdx.x) * 16 + lane16;
  const bool use_up = global_col >= out_features;
  const int64_t local_col = use_up ? global_col - out_features : global_col;
  const bool valid_col = global_col < out_total && local_col < out_features;
  const uint8_t* __restrict__ qweight = use_up ? up_weight : gate_weight;
  const int64_t blocks_per_row = in_features / QK_K;
  const int block_bytes = iq_block_bytes<QTYPE>();
  const int64_t row_bytes = blocks_per_row * block_bytes;
  const int64_t expert_bytes = out_features * row_bytes;
  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block = valid_col
        ? qweight + expert * expert_bytes + local_col * row_bytes +
            block_idx * block_bytes
        : qweight;
#pragma unroll 1
    for (int kt = 0; kt < 16; ++kt) {
      const int within_base = kt * 16;
      const int64_t k_base = block_idx * QK_K + within_base;
      const half16_t a_reg = load_compact_bf16_half16(
          x, compact_row, in_features, k_base, valid_row);
      const half16_t b_reg =
          load_iq_weight_half16<QTYPE>(block, within_base, valid_col);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }
  }

  if (global_col < out_total) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t row_in_expert = local_row_base + half_id + j * 2;
      if (row_in_expert < expert_count) {
        const int64_t out_row = compact_base + row_in_expert;
        if (out_row < compact_rows) {
          out[out_row * out_total + global_col] = float_to_bf16_bits(acc[j]);
        }
      }
    }
  }
}

__launch_bounds__(32, 2)
__global__ void gguf_iq4_xs_selected_wmma_prefill_compact_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t row_tile = blockIdx.y;
  const int64_t expert = tile_expert[row_tile];
  if (expert < 0 || expert >= num_experts) return;
  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t compact_base = expert_start_compact[expert];
  const int64_t expert_count = expert_start_compact[expert + 1] - compact_base;
  const int64_t local_row_base = row_tile * 16 - expert_start_wmma[expert];
  const int64_t local_row = local_row_base + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? compact_base + local_row : 0;

  const int64_t out_col = static_cast<int64_t>(blockIdx.x) * 16 + lane16;
  const bool valid_col = out_col < out_features;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = out_features * row_bytes;
  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const uint8_t* __restrict__ block = valid_col
        ? qweight + expert * expert_bytes + out_col * row_bytes +
            block_idx * IQ4_XS_BLOCK_BYTES
        : qweight;
#pragma unroll 1
    for (int kt = 0; kt < 16; ++kt) {
      const int within_base = kt * 16;
      const int64_t k_base = block_idx * QK_K + within_base;
      const half16_t a_reg = load_compact_bf16_half16(
          x, compact_row, in_features, k_base, valid_row);
      const half16_t b_reg = load_iq4_weight_half16(block, within_base, valid_col);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
    }
  }

  if (out_col < out_features) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t row_in_expert = local_row_base + half_id + j * 2;
      if (row_in_expert < expert_count) {
        const int64_t out_row = compact_base + row_in_expert;
        if (out_row < compact_rows) {
          out[out_row * out_features + out_col] = float_to_bf16_bits(acc[j]);
        }
      }
    }
  }
}

inline bool valid_common(
    const void* x,
    const void* expert_start_compact,
    const void* qweight,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  return x != nullptr && expert_start_compact != nullptr && qweight != nullptr &&
      out != nullptr && compact_rows > 0 && in_features > 0 &&
      (in_features % QK_K) == 0 && in_features <= 3072 && out_features > 0 &&
      num_experts > 0;
}

template <typename Kernel>
int launch_grouped_dual(
    Kernel kernel,
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int threads,
    hipStream_t stream) {
  if (!valid_common(
          x, expert_start_compact, gate_weight, out, compact_rows, in_features,
          out_features, num_experts) || up_weight == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  kernel<<<dim3(out_features, num_experts), dim3(threads), 0, stream>>>(
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      static_cast<int>(compact_rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(num_experts));
  return static_cast<int>(hipGetLastError());
}

template <typename Kernel>
int launch_grouped_single(
    Kernel kernel,
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int threads,
    hipStream_t stream) {
  if (!valid_common(
          x, expert_start_compact, qweight, out, compact_rows, in_features,
          out_features, num_experts)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  kernel<<<dim3(out_features, num_experts), dim3(threads), 0, stream>>>(
      x,
      expert_start_compact,
      qweight,
      out,
      static_cast<int>(compact_rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(num_experts));
  return static_cast<int>(hipGetLastError());
}

template <int QTYPE>
int launch_wmma_dual(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_common(
          x, expert_start_compact, gate_weight, out, compact_rows, in_features,
          out_features, num_experts) || expert_start_wmma == nullptr ||
      tile_expert == nullptr || up_weight == nullptr ||
      (out_features % 16) != 0 || wmma_total_rows <= 0 ||
      (wmma_total_rows % 16) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((2 * out_features + 15) / 16),
      static_cast<unsigned int>(wmma_total_rows / 16));
  hipLaunchKernelGGL(
      (gguf_iq_selected_dual_wmma_prefill_compact_kernel<QTYPE>),
      grid,
      dim3(32),
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      gguf_iq2_xs_selected_dual_grouped_prefill_compact_kernel,
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch_kernel<4, false>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch_kernel<8, false>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch_kernel<4, true>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch_kernel<8, true>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_kernel<4>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      64,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_kernel<8>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      64,
      stream);
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      gguf_iq2_xs_selected_dual_grouped_prefill_compact_adaptive_kernel,
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      gguf_iq3_xxs_selected_dual_grouped_prefill_compact_kernel,
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch_kernel<4, false>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch_kernel<8, false>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch_kernel<4, true>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_silu_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      (gguf_iq3_xxs_selected_dual_grouped_prefill_compact_rowbatch_kernel<8, true>),
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_single(
      (gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch_kernel<1>),
      x,
      expert_start_compact,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      in_features == 1024 ? 128 : 256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_single(
      (gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch_kernel<4>),
      x,
      expert_start_compact,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      in_features == 1024 ? 128 : 256,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_single(
      (gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch_kernel<8>),
      x,
      expert_start_compact,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      in_features == 1024 ? 128 : 256,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  return launch_grouped_dual(
      gguf_iq4_xs_selected_dual_grouped_prefill_compact_kernel,
      x,
      expert_start_compact,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      128,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_grouped_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!valid_common(
          x, expert_start_compact, qweight, out, compact_rows, in_features,
          out_features, num_experts)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gguf_iq4_xs_selected_grouped_prefill_compact_kernel,
      dim3(out_features, num_experts),
      dim3(128),
      0,
      stream,
      x,
      expert_start_compact,
      qweight,
      out,
      static_cast<int>(compact_rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(num_experts));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (!valid_common(
          x, expert_start_compact, qweight, out, compact_rows, in_features,
          out_features, num_experts) || in_features != 512) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_kernel,
      dim3(out_features, num_experts),
      dim3(32),
      0,
      stream,
      x,
      expert_start_compact,
      qweight,
      out,
      static_cast<int>(compact_rows),
      static_cast<int>(out_features),
      static_cast<int>(num_experts));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_wmma_dual<2>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_wmma_dual<3>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_wmma_dual<4>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      gate_weight,
      up_weight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_wmma_prefill_compact_bf16_bf16_out(
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
  if (!valid_common(
          x, expert_start_compact, qweight, out, compact_rows, in_features,
          out_features, num_experts) || expert_start_wmma == nullptr ||
      tile_expert == nullptr || (out_features % 16) != 0 ||
      wmma_total_rows <= 0 || (wmma_total_rows % 16) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((out_features + 15) / 16),
      static_cast<unsigned int>(wmma_total_rows / 16));
  hipLaunchKernelGGL(
      gguf_iq4_xs_selected_wmma_prefill_compact_kernel,
      grid,
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
