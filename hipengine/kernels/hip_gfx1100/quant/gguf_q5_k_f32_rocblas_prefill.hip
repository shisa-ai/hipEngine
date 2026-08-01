// Exact-value raw Q5_K/Q6_K to F32 expansion for transient ordered prefill.
//
// The dequantized plane is caller-owned projection-local scratch. Q5 preserves
// its raw scale/min expression and Q6 forms the signed integer scale*quant
// product before the single F32 super-scale multiply. Ordered consumers retain
// the exact coltile association; no persistent weight sidecar exists.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q5_K_THREADS = 64;
constexpr int Q6_K_THREADS = 64;

__device__ __forceinline__ float fp16_bytes_to_float(const uint8_t* ptr) {
  union {
    uint16_t u16;
    half_t f16;
  } value;
  value.u16 = static_cast<uint16_t>(ptr[0]) |
              (static_cast<uint16_t>(ptr[1]) << 8);
  return static_cast<float>(value.f16);
}

__device__ __forceinline__ uint8_t q5_k_scale(
    const uint8_t* __restrict__ scales,
    int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int index = subblock - 4;
  return (scales[8 + index] & 0x0Fu) |
         ((scales[index] >> 2) & 0x30u);
}

__device__ __forceinline__ uint8_t q5_k_min(
    const uint8_t* __restrict__ scales,
    int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int index = subblock - 4;
  return (scales[8 + index] >> 4) |
         ((scales[4 + index] >> 2) & 0x30u);
}

__device__ __forceinline__ float q5_k_value(
    const uint8_t* __restrict__ block,
    int within) {
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* __restrict__ scales = block + 4;
  const uint8_t* __restrict__ qh = block + 16;
  const uint8_t* __restrict__ qs = block + 48;
  const uint8_t packed = qs[(subblock >> 1) * 32 + lane];
  const uint8_t low =
      (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const uint8_t high = (qh[lane] >> subblock) & 0x01u;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const float scale = static_cast<float>(q5_k_scale(scales, subblock));
  const float min_value = static_cast<float>(q5_k_min(scales, subblock));
  const float quant = static_cast<float>(low | (high << 4));
  return d * scale * quant - dmin * min_value;
}

__device__ __forceinline__ float q6_k_exact_value(
    float d,
    int8_t scale,
    int quant) {
  // Match the retained raw-Q6 coltile expression exactly: the signed integer
  // product is formed first, converted once to F32, then multiplied by d.
  return d * static_cast<float>(static_cast<int>(scale) * quant);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u32;
    float f32;
  } value;
  value.u32 = static_cast<uint32_t>(bits) << 16;
  return value.f32;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } bits;
  bits.f32 = value;
  const uint32_t lsb = (bits.u32 >> 16) & 1u;
  bits.u32 += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits.u32 >> 16);
}

template <typename out_t>
__device__ __forceinline__ void store_output(
    out_t* __restrict__ out,
    int64_t index,
    float value) {
  out[index] = static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ void store_output<uint16_t>(
    uint16_t* __restrict__ out,
    int64_t index,
    float value) {
  out[index] = float_to_bf16_bits(value);
}

__device__ __forceinline__ void expand_q5_block(
    const uint8_t* __restrict__ block,
    float* __restrict__ out,
    int tid) {
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int within = tid + item * Q5_K_THREADS;
    out[within] = q5_k_value(block, within);
  }
}

__global__ void gguf_q5_k_dequantize_f32_exact_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out) {
  const int64_t block_id = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const uint8_t* __restrict__ block =
      qweight + block_id * Q5_K_BLOCK_BYTES;
  float* __restrict__ dst = out + block_id * QK_K;
  expand_q5_block(block, dst, tid);
}

// H5X writes the same exact expanded values in output-tile/K/column order.
// Every K row is 16-byte aligned because each admitted COL_TILE is a multiple
// of four; matching consumers can therefore load adjacent columns as float4.
template <int COL_TILE>
__launch_bounds__(QK_K)
__global__ void gguf_q5_k_dequantize_f32_exact_tile_k_col_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out,
    int in_features) {
  static_assert(COL_TILE % 4 == 0, "tile-K-col rows must be float4 aligned");
  // Linear output ownership makes each wave store contiguous tile-K-col
  // elements. The source block/within mapping still evaluates q5_k_value once
  // per output value with the exact row-major producer expression.
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * QK_K + threadIdx.x;
  const int64_t tile_elements =
      static_cast<int64_t>(in_features) * COL_TILE;
  const int out_tile = static_cast<int>(element / tile_elements);
  const int64_t tile_offset = element - out_tile * tile_elements;
  const int k = static_cast<int>(tile_offset / COL_TILE);
  const int tile_col =
      static_cast<int>(tile_offset - static_cast<int64_t>(k) * COL_TILE);
  const int output_col = out_tile * COL_TILE + tile_col;
  const int64_t blocks_per_output = in_features / QK_K;
  const int64_t block_id =
      static_cast<int64_t>(output_col) * blocks_per_output + k / QK_K;
  const int within = k & (QK_K - 1);
  const uint8_t* __restrict__ block =
      qweight + block_id * Q5_K_BLOCK_BYTES;
  out[element] = q5_k_value(block, within);
}

template <int ROW_BATCH>
__launch_bounds__(256)
__global__ void gguf_bf16_activation_pack_tile_k_row_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ out,
    int rows,
    int in_features) {
  constexpr int PADDED_ROWS = ((ROW_BATCH + 7) / 8) * 8;
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements =
      static_cast<int64_t>(row_groups) * in_features * PADDED_ROWS;
  if (element >= elements) return;

  const int row_slot = static_cast<int>(element % PADDED_ROWS);
  const int64_t record = element / PADDED_ROWS;
  const int k = static_cast<int>(record % in_features);
  const int row_group = static_cast<int>(record / in_features);
  const int row = row_group * ROW_BATCH + row_slot;
  out[element] = row_slot < ROW_BATCH && row < rows
      ? input[static_cast<int64_t>(row) * in_features + k]
      : 0;
}

__global__ void gguf_q6_k_dequantize_f32_exact_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ out) {
  const int64_t block_id = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int ip = tid >> 5;
  const int lane = tid & 31;
  const int scale_index = 8 * ip + (lane >> 4);
  const uint8_t* __restrict__ block =
      qweight + block_id * Q6_K_BLOCK_BYTES;
  const uint8_t* __restrict__ ql = block + 64 * ip + lane;
  const uint8_t qh = block[128 + 32 * ip + lane];
  const int8_t* __restrict__ scales =
      reinterpret_cast<const int8_t*>(block + 192) + scale_index;
  const float d = fp16_bytes_to_float(block + 208);
  float* __restrict__ dst = out + block_id * QK_K + 128 * ip + lane;

  dst[0] = q6_k_exact_value(
      d,
      scales[0],
      static_cast<int>((ql[0] & 0x0Fu) | (((qh >> 0) & 3u) << 4)) - 32);
  dst[32] = q6_k_exact_value(
      d,
      scales[2],
      static_cast<int>((ql[32] & 0x0Fu) | (((qh >> 2) & 3u) << 4)) - 32);
  dst[64] = q6_k_exact_value(
      d,
      scales[4],
      static_cast<int>((ql[0] >> 4) | (((qh >> 4) & 3u) << 4)) - 32);
  dst[96] = q6_k_exact_value(
      d,
      scales[6],
      static_cast<int>((ql[32] >> 4) | (((qh >> 6) & 3u) << 4)) - 32);
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
__launch_bounds__(128, (COL_TILE * ROW_BATCH >= 64 ? 1 : 2))
__global__ void gguf_q5_k_f32_weight_ordered_coltile_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32 ||
          COL_TILE * ROW_BATCH == 48 ||
          COL_TILE * ROW_BATCH == 64 ||
          COL_TILE * ROW_BATCH == 80 ||
          COL_TILE * ROW_BATCH == 96,
      "ordered F32 Q5 screens only constant-32/48/64/80/96-accumulator output tiles");
  const int out_col_base = static_cast<int>(blockIdx.x) * COL_TILE;
  const int row_base = static_cast<int>(blockIdx.y) * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        weight + static_cast<int64_t>(out_col_base + col) * in_features;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = weight_rows[col][k];
    }
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const float input_value = bf16_bits_to_float(
            input[static_cast<int64_t>(row) * in_features + k]);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// H5L preserves the ordered arithmetic body but makes row-group ownership the
// fast linear workgroup dimension. Consecutive workgroups therefore revisit
// one compact F32 weight tile before advancing to the next output tile.
template <typename out_t, int COL_TILE, int ROW_BATCH>
__launch_bounds__(128, (COL_TILE * ROW_BATCH >= 64 ? 1 : 2))
__global__ void gguf_q5_k_f32_weight_ordered_weight_major_coltile_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32 ||
          COL_TILE * ROW_BATCH == 48 ||
          COL_TILE * ROW_BATCH == 64 ||
          COL_TILE * ROW_BATCH == 80 ||
          COL_TILE * ROW_BATCH == 96,
      "weight-major ordered F32 Q5 screens only retained accumulator tiles");
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int workgroup = static_cast<int>(blockIdx.x);
  const int row_group = workgroup % row_groups;
  const int out_tile = workgroup / row_groups;
  const int out_col_base = out_tile * COL_TILE;
  const int row_base = row_group * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        weight + static_cast<int64_t>(out_col_base + col) * in_features;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = weight_rows[col][k];
    }
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const float input_value = bf16_bits_to_float(
            input[static_cast<int64_t>(row) * in_features + k]);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// H5X preserves H5L's local128/four-wave arithmetic and workgroup order, but
// reads a tile-K-col F32 plane through aligned four-column records.
template <typename out_t, int COL_TILE, int ROW_BATCH>
__launch_bounds__(128, (COL_TILE * ROW_BATCH >= 64 ? 1 : 2))
__global__ void gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32 ||
          COL_TILE * ROW_BATCH == 64 ||
          COL_TILE * ROW_BATCH == 80 ||
          COL_TILE * ROW_BATCH == 96,
      "tile-K-col Q5 uses only retained H5L accumulator tiles");
  static_assert(COL_TILE % 4 == 0, "tile-K-col rows must be float4 aligned");
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int workgroup = static_cast<int>(blockIdx.x);
  const int row_group = workgroup % row_groups;
  const int out_tile = workgroup / row_groups;
  const int out_col_base = out_tile * COL_TILE;
  const int row_base = row_group * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* __restrict__ weight_tile =
      weight + static_cast<int64_t>(out_tile) * in_features * COL_TILE;
  float acc[ROW_BATCH][COL_TILE] = {};
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
    const float4* __restrict__ records = reinterpret_cast<const float4*>(
        weight_tile + static_cast<int64_t>(k) * COL_TILE);
#pragma unroll
    for (int record_index = 0; record_index < COL_TILE / 4; ++record_index) {
      const float4 record = records[record_index];
      weights[record_index * 4 + 0] = record.x;
      weights[record_index * 4 + 1] = record.y;
      weights[record_index * 4 + 2] = record.z;
      weights[record_index * 4 + 3] = record.w;
    }
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const float input_value = bf16_bits_to_float(
            input[static_cast<int64_t>(row) * in_features + k]);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// H5Y preserves the selected H5X/H5L weight plane and arithmetic while
// reading one aligned BF16 row record from [row_group][k][row_slot].
// Specialization keeps the intended physical load width explicit: b64 for
// rowbatch4, b64+u16 for rowbatch5, b128 for rowbatch8, b128+b32 for
// rowbatch10, and b128+b64 for rowbatch12.
template <typename out_t, int COL_TILE, int ROW_BATCH, bool TILE_K_COL>
__launch_bounds__(128, (COL_TILE * ROW_BATCH >= 64 ? 1 : 2))
__global__ void gguf_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      COL_TILE * ROW_BATCH == 32 ||
          COL_TILE * ROW_BATCH == 64 ||
          COL_TILE * ROW_BATCH == 80 ||
          COL_TILE * ROW_BATCH == 96,
      "activation-tile-K-row Q5 uses only production accumulator tiles");
  static_assert(
      !TILE_K_COL || COL_TILE % 4 == 0,
      "tile-K-col weight rows must be float4 aligned");
  constexpr int PADDED_ROWS = ((ROW_BATCH + 7) / 8) * 8;
  constexpr int ACTIVATION_WORDS = (ROW_BATCH + 1) / 2;
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int workgroup = static_cast<int>(blockIdx.x);
  const int row_group = workgroup % row_groups;
  const int out_tile = workgroup / row_groups;
  const int out_col_base = out_tile * COL_TILE;
  const int row_base = row_group * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* __restrict__ weight_tile = nullptr;
  const float* weight_rows[COL_TILE];
  if constexpr (TILE_K_COL) {
    weight_tile =
        weight + static_cast<int64_t>(out_tile) * in_features * COL_TILE;
  } else {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weight_rows[col] =
          weight + static_cast<int64_t>(out_col_base + col) * in_features;
    }
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  const uint16_t* __restrict__ activation_group =
      input + static_cast<int64_t>(row_group) * in_features * PADDED_ROWS;
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
    if constexpr (TILE_K_COL) {
      const float4* __restrict__ records = reinterpret_cast<const float4*>(
          weight_tile + static_cast<int64_t>(k) * COL_TILE);
#pragma unroll
      for (int record_index = 0; record_index < COL_TILE / 4; ++record_index) {
        const float4 record = records[record_index];
        weights[record_index * 4 + 0] = record.x;
        weights[record_index * 4 + 1] = record.y;
        weights[record_index * 4 + 2] = record.z;
        weights[record_index * 4 + 3] = record.w;
      }
    } else {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        weights[col] = weight_rows[col][k];
      }
    }

    const uint16_t* __restrict__ activation_record =
        activation_group + static_cast<int64_t>(k) * PADDED_ROWS;
    uint32_t activation_words[ACTIVATION_WORDS];
    if constexpr (ROW_BATCH <= 5) {
      const uint2 first4 =
          *reinterpret_cast<const uint2*>(activation_record);
      activation_words[0] = first4.x;
      activation_words[1] = first4.y;
      if constexpr (ROW_BATCH == 5) {
        activation_words[2] = activation_record[4];
      }
    } else {
      const uint4 first8 =
          *reinterpret_cast<const uint4*>(activation_record);
      activation_words[0] = first8.x;
      activation_words[1] = first8.y;
      activation_words[2] = first8.z;
      activation_words[3] = first8.w;
      if constexpr (ROW_BATCH == 10) {
        activation_words[4] =
            *reinterpret_cast<const uint32_t*>(activation_record + 8);
      } else if constexpr (ROW_BATCH == 12) {
        const uint2 next4 =
            *reinterpret_cast<const uint2*>(activation_record + 8);
        activation_words[4] = next4.x;
        activation_words[5] = next4.y;
      }
    }

#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const uint16_t input_bits = static_cast<uint16_t>(
            activation_words[row_index >> 1] >> ((row_index & 1) * 16));
        const float input_value = bf16_bits_to_float(input_bits);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// H6E transfers H5Y's exact BF16 activation-row plane to the three H5W Q6
// roles while retaining their row-major F32 weight plane and four-wave math.
// Keeping a separate kernel symbol makes physical load/resource qualification
// independent from the retained Q5 owner.
template <typename out_t, int ROW_BATCH>
__launch_bounds__(128, 1)
__global__ void gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      ROW_BATCH == 4 || ROW_BATCH == 5,
      "H6E Q6 uses only the admitted rowbatch4/5 roles");
  constexpr int COL_TILE = 16;
  constexpr int PADDED_ROWS = 8;
  constexpr int ACTIVATION_WORDS = (ROW_BATCH + 1) / 2;
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int workgroup = static_cast<int>(blockIdx.x);
  const int row_group = workgroup % row_groups;
  const int out_tile = workgroup / row_groups;
  const int out_col_base = out_tile * COL_TILE;
  const int row_base = row_group * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        weight + static_cast<int64_t>(out_col_base + col) * in_features;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  const uint16_t* __restrict__ activation_group =
      input + static_cast<int64_t>(row_group) * in_features * PADDED_ROWS;
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = weight_rows[col][k];
    }

    const uint16_t* __restrict__ activation_record =
        activation_group + static_cast<int64_t>(k) * PADDED_ROWS;
    uint32_t activation_words[ACTIVATION_WORDS];
    const uint2 first4 = *reinterpret_cast<const uint2*>(activation_record);
    activation_words[0] = first4.x;
    activation_words[1] = first4.y;
    if constexpr (ROW_BATCH == 5) {
      activation_words[2] = activation_record[4];
    }

#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const uint16_t input_bits = static_cast<uint16_t>(
            activation_words[row_index >> 1] >> ((row_index & 1) * 16));
        const float input_value = bf16_bits_to_float(input_bits);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

// H6U preserves H6E's four-wave compute and exact reduction association while
// replacing only the five wave-peer transfers with permlanex16 plus DPP adds.
__device__ inline float h6u_permlanex16_f32(float value) {
  union FloatBits {
    float value;
    uint32_t bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_permlanex16(
      data.bits, data.bits, 0x76543210U, 0xFEDCBA98U, false, false);
  return data.value;
}

template <int DPP_CTRL>
__device__ inline float h6u_dpp_move_f32(float value) {
  union FloatBits {
    float value;
    int bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_mov_dpp(
      data.bits, DPP_CTRL, 0xF, 0xF, true);
  return data.value;
}

__device__ inline float h6u_dpp_add_row_shl1_f32(float value) {
  float result;
  asm volatile(
      "v_add_f32_dpp %0, %1, %1 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1"
      : "=v"(result)
      : "v"(value));
  return result;
}

template <int ROW_BATCH, int COL_TILE>
__device__ inline void h6u_reduce_wave_accumulators_dpp(
    float (&acc)[ROW_BATCH][COL_TILE]) {
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h6u_permlanex16_f32(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h6u_dpp_move_f32<0x108>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h6u_dpp_move_f32<0x104>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] += h6u_dpp_move_f32<0x102>(acc[row][col]);
    }
  }
#pragma unroll
  for (int row = 0; row < ROW_BATCH; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      acc[row][col] = h6u_dpp_add_row_shl1_f32(acc[row][col]);
    }
  }
}

template <typename out_t, int ROW_BATCH>
__launch_bounds__(128, 1)
__global__ void gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_kernel(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  static_assert(
      ROW_BATCH == 4 || ROW_BATCH == 5,
      "H6U Q6 uses only the admitted rowbatch4/5 roles");
  constexpr int COL_TILE = 16;
  constexpr int PADDED_ROWS = 8;
  constexpr int ACTIVATION_WORDS = (ROW_BATCH + 1) / 2;
  const int row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int workgroup = static_cast<int>(blockIdx.x);
  const int row_group = workgroup % row_groups;
  const int out_tile = workgroup / row_groups;
  const int out_col_base = out_tile * COL_TILE;
  const int row_base = row_group * ROW_BATCH;
  const int tid = static_cast<int>(threadIdx.x);
  if (out_col_base >= out_features || row_base >= rows) return;

  const float* weight_rows[COL_TILE];
#pragma unroll
  for (int col = 0; col < COL_TILE; ++col) {
    weight_rows[col] =
        weight + static_cast<int64_t>(out_col_base + col) * in_features;
  }

  float acc[ROW_BATCH][COL_TILE] = {};
  const uint16_t* __restrict__ activation_group =
      input + static_cast<int64_t>(row_group) * in_features * PADDED_ROWS;
  for (int k = tid; k < in_features; k += blockDim.x) {
    float weights[COL_TILE];
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      weights[col] = weight_rows[col][k];
    }

    const uint16_t* __restrict__ activation_record =
        activation_group + static_cast<int64_t>(k) * PADDED_ROWS;
    uint32_t activation_words[ACTIVATION_WORDS];
    const uint2 first4 = *reinterpret_cast<const uint2*>(activation_record);
    activation_words[0] = first4.x;
    activation_words[1] = first4.y;
    if constexpr (ROW_BATCH == 5) {
      activation_words[2] = activation_record[4];
    }

#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const uint16_t input_bits = static_cast<uint16_t>(
            activation_words[row_index >> 1] >> ((row_index & 1) * 16));
        const float input_value = bf16_bits_to_float(input_bits);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
  }

  h6u_reduce_wave_accumulators_dpp<ROW_BATCH, COL_TILE>(acc);
  __shared__ float wave_sums[4][ROW_BATCH][COL_TILE];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        wave_sums[wave][row_index][col] = acc[row_index][col];
      }
    }
  }
  __syncthreads();

  if (tid < ROW_BATCH * COL_TILE) {
    const int row_index = tid / COL_TILE;
    const int col = tid - row_index * COL_TILE;
    const int row = row_base + row_index;
    if (row < rows) {
      float total = 0.0f;
#pragma unroll
      for (int wave_index = 0; wave_index < 4; ++wave_index) {
        total += wave_sums[wave_index][row_index][col];
      }
      store_output<out_t>(
          out,
          static_cast<int64_t>(row) * out_features + out_col_base + col,
          total);
    }
  }
}

__global__ void gguf_q5_k_dequantize_bf16_to_f32_exact_fused_kernel(
    const uint8_t* __restrict__ qweight,
    float* __restrict__ weight_out,
    const uint16_t* __restrict__ input,
    float* __restrict__ input_out,
    int64_t weight_blocks,
    int64_t input_count) {
  const int64_t task = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (task >= weight_blocks) {
    const int64_t element =
        (task - weight_blocks) * Q5_K_THREADS + tid;
    if (element < input_count) {
      input_out[element] = bf16_bits_to_float(input[element]);
    }
    return;
  }

  const uint8_t* __restrict__ block =
      qweight + task * Q5_K_BLOCK_BYTES;
  float* __restrict__ dst = weight_out + task * QK_K;
  expand_q5_block(block, dst, tid);
}

bool valid_shape(int64_t in_features, int64_t out_features) {
  return in_features > 0 && in_features % QK_K == 0 && out_features > 0;
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
int launch_q5_k_f32_weight_ordered(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 ||
      !valid_shape(in_features, out_features) ||
      out_features % COL_TILE != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>(out_features / COL_TILE),
      static_cast<unsigned int>((rows + ROW_BATCH - 1) / ROW_BATCH));
  hipLaunchKernelGGL(
      (gguf_q5_k_f32_weight_ordered_coltile_kernel<
          out_t,
          COL_TILE,
          ROW_BATCH>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
int launch_q5_k_f32_weight_ordered_weight_major(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 ||
      !valid_shape(in_features, out_features) ||
      out_features % COL_TILE != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t output_tiles = out_features / COL_TILE;
  const dim3 grid(static_cast<unsigned int>(row_groups * output_tiles));
  hipLaunchKernelGGL(
      (gguf_q5_k_f32_weight_ordered_weight_major_coltile_kernel<
          out_t,
          COL_TILE,
          ROW_BATCH>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <int COL_TILE>
int launch_q5_k_dequantize_f32_exact_tile_k_col(
    const uint8_t* qweight,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !out || !valid_shape(in_features, out_features) ||
      out_features % COL_TILE != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_blocks = out_features * (in_features / QK_K);
  hipLaunchKernelGGL(
      (gguf_q5_k_dequantize_f32_exact_tile_k_col_kernel<COL_TILE>),
      dim3(static_cast<unsigned int>(total_blocks)),
      dim3(QK_K),
      0,
      stream,
      qweight,
      out,
      static_cast<int>(in_features));
  return static_cast<int>(hipGetLastError());
}

template <int ROW_BATCH>
int launch_bf16_activation_pack_tile_k_row(
    const uint16_t* input,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    hipStream_t stream) {
  if (!input || !out || rows <= 0 || in_features <= 0 ||
      in_features % QK_K != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int PADDED_ROWS = ((ROW_BATCH + 7) / 8) * 8;
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t elements = row_groups * in_features * PADDED_ROWS;
  const int64_t blocks = (elements + 255) / 256;
  hipLaunchKernelGGL(
      (gguf_bf16_activation_pack_tile_k_row_kernel<ROW_BATCH>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      input,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int COL_TILE, int ROW_BATCH, bool TILE_K_COL>
int launch_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 ||
      !valid_shape(in_features, out_features) ||
      out_features % COL_TILE != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t output_tiles = out_features / COL_TILE;
  const dim3 grid(static_cast<unsigned int>(row_groups * output_tiles));
  hipLaunchKernelGGL(
      (gguf_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row_kernel<
          out_t,
          COL_TILE,
          ROW_BATCH,
          TILE_K_COL>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int ROW_BATCH, int EXACT_K, int EXACT_N>
int launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 || in_features != EXACT_K ||
      out_features != EXACT_N) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int COL_TILE = 16;
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t output_tiles = out_features / COL_TILE;
  const dim3 grid(static_cast<unsigned int>(row_groups * output_tiles));
  hipLaunchKernelGGL(
      (gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_kernel<
          out_t,
          ROW_BATCH>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int ROW_BATCH, int EXACT_K, int EXACT_N>
int launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 || in_features != EXACT_K ||
      out_features != EXACT_N) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int COL_TILE = 16;
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t output_tiles = out_features / COL_TILE;
  const dim3 grid(static_cast<unsigned int>(row_groups * output_tiles));
  hipLaunchKernelGGL(
      (gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_kernel<
          out_t,
          ROW_BATCH>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename out_t, int COL_TILE, int ROW_BATCH>
int launch_q5_k_f32_weight_ordered_weight_major_tile_k_col(
    const uint16_t* input,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !weight || !out || rows <= 0 ||
      !valid_shape(in_features, out_features) ||
      out_features % COL_TILE != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_groups = (rows + ROW_BATCH - 1) / ROW_BATCH;
  const int64_t output_tiles = out_features / COL_TILE;
  const dim3 grid(static_cast<unsigned int>(row_groups * output_tiles));
  hipLaunchKernelGGL(
      (gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_kernel<
          out_t,
          COL_TILE,
          ROW_BATCH>),
      grid,
      dim3(128),
      0,
      stream,
      input,
      weight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q5_k_dequantize_f32_exact(
    const uint8_t* qweight,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !out || !valid_shape(in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_blocks = out_features * (in_features / QK_K);
  hipLaunchKernelGGL(
      gguf_q5_k_dequantize_f32_exact_kernel,
      dim3(static_cast<unsigned int>(total_blocks)),
      dim3(Q5_K_THREADS),
      0,
      stream,
      qweight,
      out);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q6_k_dequantize_f32_exact(
    const uint8_t* qweight,
    float* out,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!qweight || !out || !valid_shape(in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total_blocks = out_features * (in_features / QK_K);
  hipLaunchKernelGGL(
      gguf_q6_k_dequantize_f32_exact_kernel,
      dim3(static_cast<unsigned int>(total_blocks)),
      dim3(Q6_K_THREADS),
      0,
      stream,
      qweight,
      out);
  return static_cast<int>(hipGetLastError());
}

#define HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT(COL_TILE, ROW_BATCH, OUTPUT_DTYPE) \
extern "C" int hipengine_gguf_q5_k_dequantize_f32_exact_tile_k_col_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_##OUTPUT_DTYPE##_out( \
    const uint8_t* qweight, float* out, int64_t in_features, \
    int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_dequantize_f32_exact_tile_k_col<COL_TILE>( \
      qweight, out, in_features, out_features, stream); \
}

HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT(8, 4, bf16)
HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT(16, 5, bf16)
HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT(16, 5, f32)
HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT(8, 10, f32)

#undef HIPENGINE_Q5_TILE_K_COL_PRODUCER_EXPORT

#define HIPENGINE_Q5_F32_ORDERED_EXPORTS(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_bf16_out( \
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered<uint16_t, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
} \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* input, const float* weight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered<float, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

HIPENGINE_Q5_F32_ORDERED_EXPORTS(4, 8)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(8, 4)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(4, 16)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(8, 8)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(16, 4)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(12, 4)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(8, 10)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(16, 5)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(8, 12)
HIPENGINE_Q5_F32_ORDERED_EXPORTS(12, 8)

#undef HIPENGINE_Q5_F32_ORDERED_EXPORTS

#define HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_bf16_out( \
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major<uint16_t, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

#define HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_F32_EXPORT(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* input, const float* weight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major<float, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(8, 4)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(8, 12)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(16, 5)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(12, 8)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT(16, 4)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_F32_EXPORT(16, 5)
HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_F32_EXPORT(8, 10)

#undef HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_BF16_EXPORT
#undef HIPENGINE_Q5_F32_ORDERED_WEIGHT_MAJOR_F32_EXPORT

#define HIPENGINE_Q5_TILE_K_COL_BF16_EXPORT(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_bf16_out( \
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major_tile_k_col< \
      uint16_t, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

#define HIPENGINE_Q5_TILE_K_COL_F32_EXPORT(COL_TILE, ROW_BATCH) \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* input, const float* weight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major_tile_k_col< \
      float, COL_TILE, ROW_BATCH>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

HIPENGINE_Q5_TILE_K_COL_BF16_EXPORT(8, 4)
HIPENGINE_Q5_TILE_K_COL_BF16_EXPORT(16, 5)
HIPENGINE_Q5_TILE_K_COL_F32_EXPORT(16, 5)
HIPENGINE_Q5_TILE_K_COL_F32_EXPORT(8, 10)

#undef HIPENGINE_Q5_TILE_K_COL_BF16_EXPORT
#undef HIPENGINE_Q5_TILE_K_COL_F32_EXPORT

#define HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT( \
    COL_TILE, ROW_BATCH, OUTPUT_DTYPE, WEIGHT_LAYOUT, TILE_K_COL) \
extern "C" int hipengine_gguf_bf16_activation_pack_tile_k_row_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_##OUTPUT_DTYPE##_out( \
    const uint16_t* input, uint16_t* out, int64_t rows, \
    int64_t in_features, hipStream_t stream) { \
  return launch_bf16_activation_pack_tile_k_row<ROW_BATCH>( \
      input, out, rows, in_features, stream); \
} \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_##WEIGHT_LAYOUT##_activation_tile_k_row_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_##OUTPUT_DTYPE##_out( \
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row< \
      uint16_t, COL_TILE, ROW_BATCH, TILE_K_COL>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

#define HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_F32_EXPORT( \
    COL_TILE, ROW_BATCH, WEIGHT_LAYOUT, TILE_K_COL) \
extern "C" int hipengine_gguf_bf16_activation_pack_tile_k_row_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* input, uint16_t* out, int64_t rows, \
    int64_t in_features, hipStream_t stream) { \
  return launch_bf16_activation_pack_tile_k_row<ROW_BATCH>( \
      input, out, rows, in_features, stream); \
} \
extern "C" int hipengine_gguf_q5_k_f32_weight_ordered_weight_major_##WEIGHT_LAYOUT##_activation_tile_k_row_coltile##COL_TILE##_rowbatch##ROW_BATCH##_bf16_f32_out( \
    const uint16_t* input, const float* weight, float* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row< \
      float, COL_TILE, ROW_BATCH, TILE_K_COL>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT(8, 4, bf16, tile_k_col, true)
HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT(8, 12, bf16, row_major, false)
HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT(16, 5, bf16, tile_k_col, true)
HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT(12, 8, bf16, row_major, false)
HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_F32_EXPORT(16, 5, tile_k_col, true)
HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_F32_EXPORT(8, 10, tile_k_col, true)

#undef HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_BF16_EXPORT
#undef HIPENGINE_Q5_ACTIVATION_TILE_K_ROW_F32_EXPORT

#define HIPENGINE_Q6_ACTIVATION_TILE_K_ROW_EXPORT( \
    ROW_BATCH, OUTPUT_DTYPE, OUTPUT_TYPE, EXACT_K, EXACT_N) \
extern "C" int hipengine_gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_coltile16_rowbatch##ROW_BATCH##_bf16_##OUTPUT_DTYPE##_out( \
    const uint16_t* input, const float* weight, OUTPUT_TYPE* out, int64_t rows, \
    int64_t in_features, int64_t out_features, hipStream_t stream) { \
  return launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row< \
      OUTPUT_TYPE, ROW_BATCH, EXACT_K, EXACT_N>( \
      input, weight, out, rows, in_features, out_features, stream); \
}

HIPENGINE_Q6_ACTIVATION_TILE_K_ROW_EXPORT(5, bf16, uint16_t, 3072, 1024)
HIPENGINE_Q6_ACTIVATION_TILE_K_ROW_EXPORT(4, bf16, uint16_t, 1024, 3072)
HIPENGINE_Q6_ACTIVATION_TILE_K_ROW_EXPORT(5, f32, float, 3072, 1024)

#undef HIPENGINE_Q6_ACTIVATION_TILE_K_ROW_EXPORT

extern "C" int hipengine_gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_coltile16_rowbatch5_bf16_bf16_out(
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction<
      uint16_t, 5, 3072, 1024>(
      input, weight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_coltile16_rowbatch4_bf16_bf16_out(
    const uint16_t* input, const float* weight, uint16_t* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction<
      uint16_t, 4, 1024, 3072>(
      input, weight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_coltile16_rowbatch5_bf16_f32_out(
    const uint16_t* input, const float* weight, float* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_k_f32_weight_ordered_weight_major_row_major_activation_tile_k_row_dpp_wave_reduction<
      float, 5, 3072, 1024>(
      input, weight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
    const uint8_t* qweight,
    float* weight_out,
    const uint16_t* input,
    float* input_out,
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
      (input_count + Q5_K_THREADS - 1) / Q5_K_THREADS;
  hipLaunchKernelGGL(
      gguf_q5_k_dequantize_bf16_to_f32_exact_fused_kernel,
      dim3(static_cast<unsigned int>(weight_blocks + input_blocks)),
      dim3(Q5_K_THREADS),
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
