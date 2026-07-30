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
