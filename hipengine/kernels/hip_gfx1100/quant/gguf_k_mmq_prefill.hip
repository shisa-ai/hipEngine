// Raw Q5_K/Q6_K x producer-row Q8_1 MMQ32 prefill for gfx1100.
//
// Source lineage: llama.cpp HIP 1ebf790cda38, ggml-cuda
// quantize.cu/mmq.cuh/vecdotq.cuh. One local128 workgroup stages and reuses a
// 32-column raw-weight tile and 32 producer rows for each K32 interval. The
// resident Q5_K/Q6_K bytes remain output-major GGUF; no weight sidecar exists.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q8_BLOCK = 128;
constexpr int Q8_GROUP = 32;
constexpr int Q5_K_BLOCK_BYTES = 176;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int MMQ_ROWS = 32;
constexpr int MMQ_COLS = 32;
constexpr int MMQ_THREADS = 128;
constexpr int SOURCE_MMQ_I = 128;
constexpr int SOURCE_MMQ_J = 128;
constexpr int SOURCE_MMQ_THREADS = 256;
constexpr int SOURCE_C8_MMQ_I = 64;
constexpr int SOURCE_C8_MMQ_J = 32;
constexpr int SOURCE_C8_MMQ_THREADS = 128;
constexpr int SOURCE_WEIGHT_STRIDE_WORDS = 76;
constexpr int SOURCE_C8_F32_WEIGHT_STRIDE_WORDS = 80;
// llama.cpp's gfx1100 Q5_K I128/J128/K256 specialization reserves this
// dynamic-LDS class. The live layouts below consume 57,344 bytes; retaining
// the complete 57,856-byte class keeps launch/resource attribution comparable.
constexpr int SOURCE_MMQ_SHARED_BYTES = 57856;
constexpr int SOURCE_C8_MMQ_SHARED_BYTES =
    SOURCE_C8_MMQ_I * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS * sizeof(uint32_t) +
    SOURCE_C8_MMQ_J * 160;
constexpr int SOURCE_C8_MMQ_J16_SHARED_BYTES =
    SOURCE_C8_MMQ_I * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS * sizeof(uint32_t) +
    16 * 160;

struct q8_1_ds4_kmajor_block {
  uint16_t ds4[8];
  int8_t qs[Q8_BLOCK];
};
static_assert(sizeof(q8_1_ds4_kmajor_block) == 144,
              "unexpected source DS4 Q8_1 block size");

struct q8_1_d4s4_f32_block {
  float d4[4];
  float s4[4];
  int8_t qs[Q8_BLOCK];
};
static_assert(sizeof(q8_1_d4s4_f32_block) == 160,
              "unexpected Q8_1 D4S4-F32 block size");

struct q8_1_mmq32_cache {
  uint32_t qs[8];
  float d;
  float sum;
  int quant_sum0;
  int quant_sum1;
};

struct q8_1_d8s8_f32_block {
  float d8[8];
  float s8[8];
  int8_t qs[Q8_BLOCK];
};
static_assert(sizeof(q8_1_d8s8_f32_block) == 192,
              "unexpected Q8_1 D8S8-F32 block size");

struct q8_1_d8r8s8_f32_block {
  float d8[8];
  float dr8[8];
  float s8[8];
  int8_t qs[Q8_BLOCK];
  int8_t rs[Q8_BLOCK];
};
static_assert(sizeof(q8_1_d8r8s8_f32_block) == 352,
              "unexpected Q8_1 D8R8S8-F32 block size");

struct q8_1_d8_mmq32_cache {
  uint32_t qs[8];
  float d0;
  float d1;
  float sum0;
  float sum1;
  int quant_sum0;
  int quant_sum1;
};

struct q8_1_d8r8_mmq32_cache {
  uint32_t qs[8];
  uint32_t rs[8];
  float d0;
  float d1;
  float dr0;
  float dr1;
  float sum0;
  float sum1;
  int quant_sum0;
  int quant_sum1;
  int residual_sum0;
  int residual_sum1;
};

struct k_mmq32_cache {
  uint32_t qs[8];
  float scale0;
  float scale1;
  float minimum;
};

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

__device__ inline uint16_t float_to_fp16_bits(float value) {
  union {
    half_t f16;
    uint16_t u16;
  } out;
  out.f16 = static_cast<half_t>(value);
  return out.u16;
}

__device__ inline float fp16_bits_to_float(uint16_t value) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = value;
  return static_cast<float>(in.f16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t value = static_cast<uint16_t>(ptr[0]) |
                         (static_cast<uint16_t>(ptr[1]) << 8);
  return fp16_bits_to_float(value);
}

__device__ inline uint32_t pack_i8x4(const int8_t* ptr) {
  return static_cast<uint32_t>(static_cast<uint8_t>(ptr[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[3])) << 24);
}

__device__ inline uint32_t pack_u8x4(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ inline uint8_t q45_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) return scales[subblock] & 0x3Fu;
  const int index = subblock - 4;
  return (scales[8 + index] & 0x0Fu) |
         ((scales[index] >> 2) & 0x30u);
}

__device__ inline uint8_t q45_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) return scales[4 + subblock] & 0x3Fu;
  const int index = subblock - 4;
  return (scales[8 + index] >> 4) |
         ((scales[4 + index] >> 2) & 0x30u);
}

__device__ inline uint32_t q5_pack4_unsigned(
    const uint8_t* block,
    int subblock,
    int lane4) {
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;
  uint32_t low = *reinterpret_cast<const uint32_t*>(
      qs + (subblock >> 1) * 32 + lane4);
  low = (subblock & 1) ? ((low >> 4) & 0x0F0F0F0Fu)
                       : (low & 0x0F0F0F0Fu);
  const uint32_t high_bits =
      *reinterpret_cast<const uint32_t*>(qh + lane4);
  const uint32_t high =
      ((high_bits >> subblock) & 0x01010101u) << 4;
  return low | high;
}

__device__ inline uint8_t q6_unsigned(
    const uint8_t* block,
    int group32,
    int lane) {
  const uint8_t* qh = block + 128;
  const int base64 = group32 >= 4 ? 64 : 0;
  const int ql_index = base64 + (group32 & 1) * 32 + lane;
  const uint8_t ql_byte = block[ql_index];
  const uint8_t low = (group32 & 2) == 0
      ? (ql_byte & 0x0Fu)
      : (ql_byte >> 4);
  const int qh_index = (group32 >= 4 ? 32 : 0) + lane;
  const uint8_t high =
      (qh[qh_index] >> (2 * (group32 & 3))) & 0x03u;
  return low | (high << 4);
}

__device__ inline uint32_t q6_pack4_unsigned(
    const uint8_t* block,
    int group32,
    int lane4) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    packed |= static_cast<uint32_t>(q6_unsigned(block, group32, lane4 + i))
              << (8 * i);
  }
  return packed;
}

__device__ inline int dp4a_u8_s8(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), c, false);
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

template <typename output_t>
__device__ inline void store_output(output_t* out, int64_t index, float value) {
  out[index] = static_cast<output_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(
    uint16_t* out,
    int64_t index,
    float value) {
  out[index] = float_to_bf16_bits(value);
}

__launch_bounds__(256, 2)
__global__ void q8_1_d4s4_f32_quantize_bf16_kernel(
    const uint16_t* __restrict__ x,
    q8_1_d4s4_f32_block* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 =
      (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] =
        bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
    sum += values[i];
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const int block = i0 / Q8_BLOCK;
  const int within = i0 & (Q8_BLOCK - 1);
  q8_1_d4s4_f32_block* dst =
      out + static_cast<int64_t>(row) * (hidden / Q8_BLOCK) + block;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d == 0.0f ? 0 : static_cast<int>(roundf(values[i] / d));
    q = max(-128, min(127, q));
    dst->qs[within + i] = static_cast<int8_t>(q);
  }
  if ((within & 31) == 0) {
    const int group = within >> 5;
    dst->d4[group] = d;
    dst->s4[group] = sum;
  }
}

__launch_bounds__(256, 2)
__global__ void q8_1_d8s8_f32_quantize_bf16_kernel(
    const uint16_t* __restrict__ x,
    q8_1_d8s8_f32_block* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 =
      (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] =
        bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
    sum += values[i];
  }
#pragma unroll
  for (int offset = 2; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const int block = i0 / Q8_BLOCK;
  const int within = i0 & (Q8_BLOCK - 1);
  q8_1_d8s8_f32_block* dst =
      out + static_cast<int64_t>(row) * (hidden / Q8_BLOCK) + block;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d == 0.0f ? 0 : static_cast<int>(roundf(values[i] / d));
    q = max(-128, min(127, q));
    dst->qs[within + i] = static_cast<int8_t>(q);
  }
  if ((within & 15) == 0) {
    const int group = within >> 4;
    dst->d8[group] = d;
    dst->s8[group] = sum;
  }
}

__launch_bounds__(256, 2)
__global__ void q8_1_d8r8s8_f32_quantize_bf16_kernel(
    const uint16_t* __restrict__ x,
    q8_1_d8r8s8_f32_block* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 =
      (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] =
        bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
    sum += values[i];
  }
#pragma unroll
  for (int offset = 2; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  int primary[4];
  float residual[4];
  float residual_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d == 0.0f ? 0 : static_cast<int>(roundf(values[i] / d));
    q = max(-128, min(127, q));
    primary[i] = q;
    volatile float reconstructed = static_cast<float>(q) * d;
    residual[i] = values[i] - reconstructed;
    residual_amax = fmaxf(residual_amax, fabsf(residual[i]));
  }
#pragma unroll
  for (int offset = 2; offset > 0; offset >>= 1) {
    residual_amax =
        fmaxf(residual_amax, __shfl_xor(residual_amax, offset, 32));
  }
  const float dr = residual_amax == 0.0f ? 0.0f : residual_amax / 127.0f;

  const int block = i0 / Q8_BLOCK;
  const int within = i0 & (Q8_BLOCK - 1);
  q8_1_d8r8s8_f32_block* dst =
      out + static_cast<int64_t>(row) * (hidden / Q8_BLOCK) + block;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int qr = dr == 0.0f ? 0 : static_cast<int>(roundf(residual[i] / dr));
    qr = max(-128, min(127, qr));
    dst->qs[within + i] = static_cast<int8_t>(primary[i]);
    dst->rs[within + i] = static_cast<int8_t>(qr);
  }
  if ((within & 15) == 0) {
    const int group = within >> 4;
    dst->d8[group] = d;
    dst->dr8[group] = dr;
    dst->s8[group] = sum;
  }
}

// Source-faithful llama.cpp DS4 producer. hipEngine's graph boundary is BF16,
// so values are expanded to F32 before applying the same 32-value scale/sum,
// round-away quantization, FP16 metadata, and K-major transpose.
__launch_bounds__(256, 2)
__global__ void q8_1_ds4_quantize_bf16_kmajor_kernel(
    const uint16_t* __restrict__ x,
    q8_1_ds4_kmajor_block* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 =
      (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] =
        bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
    sum = (sum + values[i]);
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }

  const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
  const float d = d_inv == 0.0f ? 0.0f : 1.0f / d_inv;
  const int k_block = i0 / Q8_BLOCK;
  const int iqs = i0 & (Q8_BLOCK - 1);
  q8_1_ds4_kmajor_block* dst =
      out + static_cast<int64_t>(k_block) * rows + row;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d_inv == 0.0f
        ? 0
        : static_cast<int>(roundf(values[i] * d_inv));
    q = max(-128, min(127, q));
    dst->qs[iqs + i] = static_cast<int8_t>(q);
  }
  if ((iqs & 31) == 0) {
    const int group = iqs >> 5;
    dst->ds4[2 * group + 0] = float_to_fp16_bits(d);
    dst->ds4[2 * group + 1] = float_to_fp16_bits(sum);
  }
}

// Precision-preserving source-layout producer. The quantized values and FP32
// metadata match q8_1_d4s4_f32_quantize_bf16_kernel, while only the record
// order changes from row-major to K-major for coalesced MMQ consumption.
__launch_bounds__(256, 2)
__global__ void q8_1_d4s4_f32_quantize_bf16_kmajor_kernel(
    const uint16_t* __restrict__ x,
    q8_1_d4s4_f32_block* __restrict__ out,
    int rows,
    int hidden) {
  const int row = static_cast<int>(blockIdx.x);
  const int i0 =
      (static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x)) * 4;
  if (row >= rows || i0 >= hidden) return;

  float values[4];
  float amax = 0.0f;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] =
        bf16_bits_to_float(x[static_cast<int64_t>(row) * hidden + i0 + i]);
    amax = fmaxf(amax, fabsf(values[i]));
    sum += values[i];
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    sum += __shfl_xor(sum, offset, 32);
  }

  const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
  const int k_block = i0 / Q8_BLOCK;
  const int within = i0 & (Q8_BLOCK - 1);
  q8_1_d4s4_f32_block* dst =
      out + static_cast<int64_t>(k_block) * rows + row;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = d == 0.0f ? 0 : static_cast<int>(roundf(values[i] / d));
    q = max(-128, min(127, q));
    dst->qs[within + i] = static_cast<int8_t>(q);
  }
  if ((within & 31) == 0) {
    const int group = within >> 5;
    dst->d4[group] = d;
    dst->s4[group] = sum;
  }
}

template <int qtype, typename output_t, int kMinBlocksPerProcessor>
__launch_bounds__(128, kMinBlocksPerProcessor)
__global__ void gguf_k_raw_mmq32_q8_1_d4s4_f32_kernel(
    const q8_1_d4s4_f32_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int rows,
    int hidden,
    int out_features) {
  static_assert(qtype == 5 || qtype == 6, "MMQ32 supports Q5_K/Q6_K");
  __shared__ k_mmq32_cache weight_tile[MMQ_COLS];
  __shared__ q8_1_mmq32_cache activation_tile[MMQ_ROWS];

  const int tid = static_cast<int>(threadIdx.x);
  const int output_col_base = static_cast<int>(blockIdx.x) * MMQ_COLS;
  const int input_row_base = static_cast<int>(blockIdx.y) * MMQ_ROWS;
  const int load_part = tid & 3;
  const int load_index = tid >> 2;
  const int owned_col = tid & 31;
  const int owned_row_residue = tid >> 5;
  const int blocks_per_weight_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / Q8_BLOCK;
  const int weight_row_bytes = blocks_per_weight_row *
      (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  const int k32_iters = hidden / Q8_GROUP;
  for (int k32 = 0; k32 < k32_iters; ++k32) {
    if (load_index < MMQ_COLS) {
      const int output_col = output_col_base + load_index;
      const bool valid_col = output_col < out_features;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      float scale0 = 0.0f;
      float scale1 = 0.0f;
      float minimum = 0.0f;
      if (valid_col) {
        const int weight_block = k32 >> 3;
        const int group32 = k32 & 7;
        const uint8_t* block =
            qweight + static_cast<int64_t>(output_col) * weight_row_bytes +
            static_cast<int64_t>(weight_block) *
                (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);
        const int lane8 = load_part * 8;
        if constexpr (qtype == 5) {
          q0 = q5_pack4_unsigned(block, group32, lane8);
          q1 = q5_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block);
            const float dmin = fp16_bytes_to_float(block + 2);
            const uint8_t* scales = block + 4;
            scale0 = d * static_cast<float>(q45_scale(scales, group32));
            minimum = dmin * static_cast<float>(q45_min(scales, group32));
          }
        } else {
          q0 = q6_pack4_unsigned(block, group32, lane8);
          q1 = q6_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block + 208);
            const int8_t* scales =
                reinterpret_cast<const int8_t*>(block + 192);
            scale0 = d * static_cast<float>(scales[2 * group32]);
            scale1 = d * static_cast<float>(scales[2 * group32 + 1]);
          }
        }
      }
      weight_tile[load_index].qs[2 * load_part] = q0;
      weight_tile[load_index].qs[2 * load_part + 1] = q1;
      if (load_part == 0) {
        weight_tile[load_index].scale0 = scale0;
        weight_tile[load_index].scale1 = scale1;
        weight_tile[load_index].minimum = minimum;
      }
    }

    if (load_index < MMQ_ROWS) {
      const int input_row = input_row_base + load_index;
      const bool valid_row = input_row < rows;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      float d = 0.0f;
      float sum = 0.0f;
      int quant_sum0 = 0;
      int quant_sum1 = 0;
      if (valid_row) {
        const q8_1_d4s4_f32_block& block =
            xq[static_cast<int64_t>(input_row) * q8_blocks_per_row + (k32 >> 2)];
        const int group = k32 & 3;
        const int8_t* group_qs = block.qs + group * Q8_GROUP;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(
            group_qs + load_part * 8);
        q0 = words[0];
        q1 = words[1];
        if (load_part == 0) {
          d = block.d4[group];
          sum = block.s4[group];
#pragma unroll
          for (int lane = 0; lane < 16; ++lane) {
            quant_sum0 += static_cast<int>(group_qs[lane]);
            quant_sum1 += static_cast<int>(group_qs[lane + 16]);
          }
        }
      }
      activation_tile[load_index].qs[2 * load_part] = q0;
      activation_tile[load_index].qs[2 * load_part + 1] = q1;
      if (load_part == 0) {
        activation_tile[load_index].d = d;
        activation_tile[load_index].sum = sum;
        activation_tile[load_index].quant_sum0 = quant_sum0;
        activation_tile[load_index].quant_sum1 = quant_sum1;
      }
    }
    __syncthreads();

    const k_mmq32_cache weight = weight_tile[owned_col];
#pragma unroll
    for (int row_group = 0; row_group < 8; ++row_group) {
      const int row_local = row_group * 4 + owned_row_residue;
      const int input_row = input_row_base + row_local;
      if (input_row >= rows) continue;
      const q8_1_mmq32_cache activation = activation_tile[row_local];
      if constexpr (qtype == 5) {
        int dot = 0;
#pragma unroll
        for (int word = 0; word < 8; ++word) {
          dot = dp4a_u8_s8(weight.qs[word], activation.qs[word], dot);
        }
        acc[row_group] +=
            activation.d * weight.scale0 * static_cast<float>(dot) -
            weight.minimum * activation.sum;
      } else {
        int dot0 = 0;
        int dot1 = 0;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
          dot0 = dp4a_u8_s8(
              weight.qs[word], activation.qs[word], dot0);
          dot1 = dp4a_u8_s8(
              weight.qs[word + 4], activation.qs[word + 4], dot1);
        }
        dot0 -= 32 * activation.quant_sum0;
        dot1 -= 32 * activation.quant_sum1;
        acc[row_group] += activation.d *
            (weight.scale0 * static_cast<float>(dot0) +
             weight.scale1 * static_cast<float>(dot1));
      }
    }
    __syncthreads();
  }

  const int output_col = output_col_base + owned_col;
  if (output_col >= out_features) return;
#pragma unroll
  for (int row_group = 0; row_group < 8; ++row_group) {
    const int row_local = row_group * 4 + owned_row_residue;
    const int input_row = input_row_base + row_local;
    if (input_row < rows) {
      store_output<output_t>(
          out,
          static_cast<int64_t>(input_row) * out_features + output_col,
          acc[row_group]);
    }
  }
}

template <int qtype, typename output_t>
__launch_bounds__(128, 2)
__global__ void gguf_k_raw_mmq32_q8_1_d8s8_f32_kernel(
    const q8_1_d8s8_f32_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int rows,
    int hidden,
    int out_features) {
  static_assert(qtype == 5 || qtype == 6, "MMQ32 supports Q5_K/Q6_K");
  __shared__ k_mmq32_cache weight_tile[MMQ_COLS];
  __shared__ q8_1_d8_mmq32_cache activation_tile[MMQ_ROWS];

  const int tid = static_cast<int>(threadIdx.x);
  const int output_col_base = static_cast<int>(blockIdx.x) * MMQ_COLS;
  const int input_row_base = static_cast<int>(blockIdx.y) * MMQ_ROWS;
  const int load_part = tid & 3;
  const int load_index = tid >> 2;
  const int owned_col = tid & 31;
  const int owned_row_residue = tid >> 5;
  const int blocks_per_weight_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / Q8_BLOCK;
  const int weight_row_bytes = blocks_per_weight_row *
      (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  const int k32_iters = hidden / Q8_GROUP;
  for (int k32 = 0; k32 < k32_iters; ++k32) {
    if (load_index < MMQ_COLS) {
      const int output_col = output_col_base + load_index;
      const bool valid_col = output_col < out_features;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      float scale0 = 0.0f;
      float scale1 = 0.0f;
      float minimum = 0.0f;
      if (valid_col) {
        const int weight_block = k32 >> 3;
        const int group32 = k32 & 7;
        const uint8_t* block =
            qweight + static_cast<int64_t>(output_col) * weight_row_bytes +
            static_cast<int64_t>(weight_block) *
                (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);
        const int lane8 = load_part * 8;
        if constexpr (qtype == 5) {
          q0 = q5_pack4_unsigned(block, group32, lane8);
          q1 = q5_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block);
            const float dmin = fp16_bytes_to_float(block + 2);
            const uint8_t* scales = block + 4;
            scale0 = d * static_cast<float>(q45_scale(scales, group32));
            minimum = dmin * static_cast<float>(q45_min(scales, group32));
          }
        } else {
          q0 = q6_pack4_unsigned(block, group32, lane8);
          q1 = q6_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block + 208);
            const int8_t* scales =
                reinterpret_cast<const int8_t*>(block + 192);
            scale0 = d * static_cast<float>(scales[2 * group32]);
            scale1 = d * static_cast<float>(scales[2 * group32 + 1]);
          }
        }
      }
      weight_tile[load_index].qs[2 * load_part] = q0;
      weight_tile[load_index].qs[2 * load_part + 1] = q1;
      if (load_part == 0) {
        weight_tile[load_index].scale0 = scale0;
        weight_tile[load_index].scale1 = scale1;
        weight_tile[load_index].minimum = minimum;
      }
    }

    if (load_index < MMQ_ROWS) {
      const int input_row = input_row_base + load_index;
      const bool valid_row = input_row < rows;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      float d0 = 0.0f;
      float d1 = 0.0f;
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      int quant_sum0 = 0;
      int quant_sum1 = 0;
      if (valid_row) {
        const q8_1_d8s8_f32_block& block =
            xq[static_cast<int64_t>(input_row) * q8_blocks_per_row + (k32 >> 2)];
        const int group = (k32 & 3) * 2;
        const int8_t* group_qs = block.qs + group * 16;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(
            group_qs + load_part * 8);
        q0 = words[0];
        q1 = words[1];
        if (load_part == 0) {
          d0 = block.d8[group];
          d1 = block.d8[group + 1];
          sum0 = block.s8[group];
          sum1 = block.s8[group + 1];
#pragma unroll
          for (int lane = 0; lane < 16; ++lane) {
            quant_sum0 += static_cast<int>(group_qs[lane]);
            quant_sum1 += static_cast<int>(group_qs[lane + 16]);
          }
        }
      }
      activation_tile[load_index].qs[2 * load_part] = q0;
      activation_tile[load_index].qs[2 * load_part + 1] = q1;
      if (load_part == 0) {
        activation_tile[load_index].d0 = d0;
        activation_tile[load_index].d1 = d1;
        activation_tile[load_index].sum0 = sum0;
        activation_tile[load_index].sum1 = sum1;
        activation_tile[load_index].quant_sum0 = quant_sum0;
        activation_tile[load_index].quant_sum1 = quant_sum1;
      }
    }
    __syncthreads();

    const k_mmq32_cache weight = weight_tile[owned_col];
#pragma unroll
    for (int row_group = 0; row_group < 8; ++row_group) {
      const int row_local = row_group * 4 + owned_row_residue;
      const int input_row = input_row_base + row_local;
      if (input_row >= rows) continue;
      const q8_1_d8_mmq32_cache activation = activation_tile[row_local];
      int dot0 = 0;
      int dot1 = 0;
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        dot0 = dp4a_u8_s8(weight.qs[word], activation.qs[word], dot0);
        dot1 = dp4a_u8_s8(
            weight.qs[word + 4], activation.qs[word + 4], dot1);
      }
      if constexpr (qtype == 5) {
        acc[row_group] += weight.scale0 *
            (activation.d0 * static_cast<float>(dot0) +
             activation.d1 * static_cast<float>(dot1)) -
            weight.minimum * (activation.sum0 + activation.sum1);
      } else {
        dot0 -= 32 * activation.quant_sum0;
        dot1 -= 32 * activation.quant_sum1;
        acc[row_group] +=
            activation.d0 * weight.scale0 * static_cast<float>(dot0) +
            activation.d1 * weight.scale1 * static_cast<float>(dot1);
      }
    }
    __syncthreads();
  }

  const int output_col = output_col_base + owned_col;
  if (output_col >= out_features) return;
#pragma unroll
  for (int row_group = 0; row_group < 8; ++row_group) {
    const int row_local = row_group * 4 + owned_row_residue;
    const int input_row = input_row_base + row_local;
    if (input_row < rows) {
      store_output<output_t>(
          out,
          static_cast<int64_t>(input_row) * out_features + output_col,
          acc[row_group]);
    }
  }
}

template <int qtype, typename output_t>
__launch_bounds__(128, 2)
__global__ void gguf_k_raw_mmq32_q8_1_d8r8s8_f32_kernel(
    const q8_1_d8r8s8_f32_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int rows,
    int hidden,
    int out_features) {
  static_assert(qtype == 5 || qtype == 6, "MMQ32 supports Q5_K/Q6_K");
  __shared__ k_mmq32_cache weight_tile[MMQ_COLS];
  __shared__ q8_1_d8r8_mmq32_cache activation_tile[MMQ_ROWS];

  const int tid = static_cast<int>(threadIdx.x);
  const int output_col_base = static_cast<int>(blockIdx.x) * MMQ_COLS;
  const int input_row_base = static_cast<int>(blockIdx.y) * MMQ_ROWS;
  const int load_part = tid & 3;
  const int load_index = tid >> 2;
  const int owned_col = tid & 31;
  const int owned_row_residue = tid >> 5;
  const int blocks_per_weight_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / Q8_BLOCK;
  const int weight_row_bytes = blocks_per_weight_row *
      (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  const int k32_iters = hidden / Q8_GROUP;
  for (int k32 = 0; k32 < k32_iters; ++k32) {
    if (load_index < MMQ_COLS) {
      const int output_col = output_col_base + load_index;
      const bool valid_col = output_col < out_features;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      float scale0 = 0.0f;
      float scale1 = 0.0f;
      float minimum = 0.0f;
      if (valid_col) {
        const int weight_block = k32 >> 3;
        const int group32 = k32 & 7;
        const uint8_t* block =
            qweight + static_cast<int64_t>(output_col) * weight_row_bytes +
            static_cast<int64_t>(weight_block) *
                (qtype == 5 ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES);
        const int lane8 = load_part * 8;
        if constexpr (qtype == 5) {
          q0 = q5_pack4_unsigned(block, group32, lane8);
          q1 = q5_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block);
            const float dmin = fp16_bytes_to_float(block + 2);
            const uint8_t* scales = block + 4;
            scale0 = d * static_cast<float>(q45_scale(scales, group32));
            minimum = dmin * static_cast<float>(q45_min(scales, group32));
          }
        } else {
          q0 = q6_pack4_unsigned(block, group32, lane8);
          q1 = q6_pack4_unsigned(block, group32, lane8 + 4);
          if (load_part == 0) {
            const float d = fp16_bytes_to_float(block + 208);
            const int8_t* scales =
                reinterpret_cast<const int8_t*>(block + 192);
            scale0 = d * static_cast<float>(scales[2 * group32]);
            scale1 = d * static_cast<float>(scales[2 * group32 + 1]);
          }
        }
      }
      weight_tile[load_index].qs[2 * load_part] = q0;
      weight_tile[load_index].qs[2 * load_part + 1] = q1;
      if (load_part == 0) {
        weight_tile[load_index].scale0 = scale0;
        weight_tile[load_index].scale1 = scale1;
        weight_tile[load_index].minimum = minimum;
      }
    }

    if (load_index < MMQ_ROWS) {
      const int input_row = input_row_base + load_index;
      const bool valid_row = input_row < rows;
      uint32_t q0 = 0;
      uint32_t q1 = 0;
      uint32_t r0 = 0;
      uint32_t r1 = 0;
      float d0 = 0.0f;
      float d1 = 0.0f;
      float dr0 = 0.0f;
      float dr1 = 0.0f;
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      int quant_sum0 = 0;
      int quant_sum1 = 0;
      int residual_sum0 = 0;
      int residual_sum1 = 0;
      if (valid_row) {
        const q8_1_d8r8s8_f32_block& block =
            xq[static_cast<int64_t>(input_row) * q8_blocks_per_row + (k32 >> 2)];
        const int group = (k32 & 3) * 2;
        const int8_t* group_qs = block.qs + group * 16;
        const int8_t* group_rs = block.rs + group * 16;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(
            group_qs + load_part * 8);
        const uint32_t* residual_words = reinterpret_cast<const uint32_t*>(
            group_rs + load_part * 8);
        q0 = words[0];
        q1 = words[1];
        r0 = residual_words[0];
        r1 = residual_words[1];
        if (load_part == 0) {
          d0 = block.d8[group];
          d1 = block.d8[group + 1];
          dr0 = block.dr8[group];
          dr1 = block.dr8[group + 1];
          sum0 = block.s8[group];
          sum1 = block.s8[group + 1];
#pragma unroll
          for (int lane = 0; lane < 16; ++lane) {
            quant_sum0 += static_cast<int>(group_qs[lane]);
            quant_sum1 += static_cast<int>(group_qs[lane + 16]);
            residual_sum0 += static_cast<int>(group_rs[lane]);
            residual_sum1 += static_cast<int>(group_rs[lane + 16]);
          }
        }
      }
      activation_tile[load_index].qs[2 * load_part] = q0;
      activation_tile[load_index].qs[2 * load_part + 1] = q1;
      activation_tile[load_index].rs[2 * load_part] = r0;
      activation_tile[load_index].rs[2 * load_part + 1] = r1;
      if (load_part == 0) {
        activation_tile[load_index].d0 = d0;
        activation_tile[load_index].d1 = d1;
        activation_tile[load_index].dr0 = dr0;
        activation_tile[load_index].dr1 = dr1;
        activation_tile[load_index].sum0 = sum0;
        activation_tile[load_index].sum1 = sum1;
        activation_tile[load_index].quant_sum0 = quant_sum0;
        activation_tile[load_index].quant_sum1 = quant_sum1;
        activation_tile[load_index].residual_sum0 = residual_sum0;
        activation_tile[load_index].residual_sum1 = residual_sum1;
      }
    }
    __syncthreads();

    const k_mmq32_cache weight = weight_tile[owned_col];
#pragma unroll
    for (int row_group = 0; row_group < 8; ++row_group) {
      const int row_local = row_group * 4 + owned_row_residue;
      const int input_row = input_row_base + row_local;
      if (input_row >= rows) continue;
      const q8_1_d8r8_mmq32_cache activation = activation_tile[row_local];
      int dot0 = 0;
      int dot1 = 0;
      int residual_dot0 = 0;
      int residual_dot1 = 0;
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        dot0 = dp4a_u8_s8(weight.qs[word], activation.qs[word], dot0);
        dot1 = dp4a_u8_s8(
            weight.qs[word + 4], activation.qs[word + 4], dot1);
        residual_dot0 = dp4a_u8_s8(
            weight.qs[word], activation.rs[word], residual_dot0);
        residual_dot1 = dp4a_u8_s8(
            weight.qs[word + 4], activation.rs[word + 4], residual_dot1);
      }
      if constexpr (qtype == 5) {
        acc[row_group] += weight.scale0 *
            (activation.d0 * static_cast<float>(dot0) +
             activation.dr0 * static_cast<float>(residual_dot0) +
             activation.d1 * static_cast<float>(dot1) +
             activation.dr1 * static_cast<float>(residual_dot1)) -
            weight.minimum * (activation.sum0 + activation.sum1);
      } else {
        dot0 -= 32 * activation.quant_sum0;
        dot1 -= 32 * activation.quant_sum1;
        residual_dot0 -= 32 * activation.residual_sum0;
        residual_dot1 -= 32 * activation.residual_sum1;
        acc[row_group] += weight.scale0 *
            (activation.d0 * static_cast<float>(dot0) +
             activation.dr0 * static_cast<float>(residual_dot0)) +
            weight.scale1 *
            (activation.d1 * static_cast<float>(dot1) +
             activation.dr1 * static_cast<float>(residual_dot1));
      }
    }
    __syncthreads();
  }

  const int output_col = output_col_base + owned_col;
  if (output_col >= out_features) return;
#pragma unroll
  for (int row_group = 0; row_group < 8; ++row_group) {
    const int row_local = row_group * 4 + owned_row_residue;
    const int input_row = input_row_base + row_local;
    if (input_row < rows) {
      store_output<output_t>(
          out,
          static_cast<int64_t>(input_row) * out_features + output_col,
          acc[row_group]);
    }
  }
}

// Eight wave32s own the 128 output columns of one I128/J128 tile. Raw Q5_K
// bytes are expanded once per K256 slab into the llama.cpp-shaped unsigned
// weight tile; one K-major DS4 activation slab is staged for each K128 half.
// Each wave computes its 16 output columns across all eight J16 row minitiles.
template <typename output_t, bool fallback>
__launch_bounds__(256, 2)
__global__ void gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_kernel(
    const q8_1_ds4_kmajor_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int rows,
    int hidden,
    int out_features) {
  typedef int i32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int i32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  extern __shared__ uint8_t shared_bytes[];
  uint32_t* weight_tile = reinterpret_cast<uint32_t*>(shared_bytes);
  q8_1_ds4_kmajor_block* activation =
      reinterpret_cast<q8_1_ds4_kmajor_block*>(
          weight_tile + SOURCE_MMQ_I * SOURCE_WEIGHT_STRIDE_WORDS);

  const int wave = static_cast<int>(threadIdx.y);
  const int wave_tid = static_cast<int>(threadIdx.x);
  const int tid = wave * 32 + wave_tid;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;
  const int output_base = static_cast<int>(blockIdx.x) * SOURCE_MMQ_I;
  const int row_base = static_cast<int>(blockIdx.y) * SOURCE_MMQ_J;
  const int weight_input_local = wave * 16 + lane16;
  const int weight_input_col = output_base + weight_input_local;
  const bool valid_weight_input = !fallback || weight_input_col < out_features;
  const int blocks_per_weight_row = hidden / QK_K;
  const int weight_row_bytes = blocks_per_weight_row * Q5_K_BLOCK_BYTES;

  float acc[8][8];
#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[rt][j] = 0.0f;
  }

  for (int weight_block = 0; weight_block < blocks_per_weight_row;
       ++weight_block) {
    // Four lane8 subgroups per wave expand four output rows concurrently.
    // Across four rounds the eight waves populate all I128 rows.
    const int load_row_in_wave = wave_tid >> 3;
    const int load_lane8 = wave_tid & 7;
#pragma unroll
    for (int load_round = 0; load_round < 4; ++load_round) {
      const int output_round =
          load_round * 32 + wave * 4 + load_row_in_wave;
      const int global_output = output_base + output_round;
      const bool valid = !fallback || global_output < out_features;
      const uint8_t* block = valid
          ? qweight + static_cast<int64_t>(global_output) * weight_row_bytes +
                static_cast<int64_t>(weight_block) * Q5_K_BLOCK_BYTES
          : nullptr;
#pragma unroll
      for (int sb = 0; sb < 8; ++sb) {
        uint32_t packed = 0;
        if (valid) packed = q5_pack4_unsigned(block, sb, load_lane8 * 4);
        weight_tile[
            output_round * SOURCE_WEIGHT_STRIDE_WORDS +
            sb * 8 + load_lane8] = packed;
      }
      float scale = 0.0f;
      float minimum = 0.0f;
      if (valid) {
        const float d = fp16_bytes_to_float(block);
        const float dmin = fp16_bytes_to_float(block + 2);
        scale = d * static_cast<float>(q45_scale(block + 4, load_lane8));
        minimum =
            -dmin * static_cast<float>(q45_min(block + 4, load_lane8));
      }
      uint16_t* row_dm = reinterpret_cast<uint16_t*>(
          weight_tile + output_round * SOURCE_WEIGHT_STRIDE_WORDS + 64);
      row_dm[2 * load_lane8 + 0] = float_to_fp16_bits(scale);
      row_dm[2 * load_lane8 + 1] = float_to_fp16_bits(minimum);
    }
    __syncthreads();

#pragma unroll
    for (int khalf = 0; khalf < 2; ++khalf) {
      const int activation_block = weight_block * 2 + khalf;
      uint32_t* activation_words = reinterpret_cast<uint32_t*>(activation);
      constexpr int words_per_record = sizeof(q8_1_ds4_kmajor_block) / 4;
      for (int index = tid;
           index < SOURCE_MMQ_J * words_per_record;
           index += SOURCE_MMQ_THREADS) {
        const int local_row = index / words_per_record;
        const int word = index - local_row * words_per_record;
        const int global_row = row_base + local_row;
        uint32_t value = 0;
        if (!fallback || global_row < rows) {
          const q8_1_ds4_kmajor_block* source =
              xq + static_cast<int64_t>(activation_block) * rows + global_row;
          value = reinterpret_cast<const uint32_t*>(source)[word];
        }
        activation_words[index] = value;
      }
      __syncthreads();

      for (int local_sb = 0; local_sb < 4; ++local_sb) {
        const int weight_sb = khalf * 4 + local_sb;
        i32x4_t weight0 = {0, 0, 0, 0};
        i32x4_t weight1 = {0, 0, 0, 0};
        if (valid_weight_input) {
          const uint8_t* weight_values = reinterpret_cast<const uint8_t*>(
              weight_tile +
              weight_input_local * SOURCE_WEIGHT_STRIDE_WORDS) +
              weight_sb * 32;
          weight0 = i32x4_t{
              static_cast<int>(pack_u8x4(weight_values + 0)),
              static_cast<int>(pack_u8x4(weight_values + 4)),
              static_cast<int>(pack_u8x4(weight_values + 8)),
              static_cast<int>(pack_u8x4(weight_values + 12)),
          };
          weight1 = i32x4_t{
              static_cast<int>(pack_u8x4(weight_values + 16)),
              static_cast<int>(pack_u8x4(weight_values + 20)),
              static_cast<int>(pack_u8x4(weight_values + 24)),
              static_cast<int>(pack_u8x4(weight_values + 28)),
          };
        }

        i32x8_t dots[8];
#pragma unroll
        for (int rt = 0; rt < 8; ++rt) {
          const int activation_row = rt * 16 + lane16;
          const q8_1_ds4_kmajor_block& xb = activation[activation_row];
          const int8_t* activation_values = xb.qs + local_sb * 32;
          const i32x4_t activation0 = i32x4_t{
              static_cast<int>(pack_i8x4(activation_values + 0)),
              static_cast<int>(pack_i8x4(activation_values + 4)),
              static_cast<int>(pack_i8x4(activation_values + 8)),
              static_cast<int>(pack_i8x4(activation_values + 12)),
          };
          const i32x4_t activation1 = i32x4_t{
              static_cast<int>(pack_i8x4(activation_values + 16)),
              static_cast<int>(pack_i8x4(activation_values + 20)),
              static_cast<int>(pack_i8x4(activation_values + 24)),
              static_cast<int>(pack_i8x4(activation_values + 28)),
          };
          dots[rt] = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
          dots[rt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight0, true, activation0, dots[rt], true);
          dots[rt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight1, true, activation1, dots[rt], true);
        }

#pragma unroll
        for (int rt = 0; rt < 8; ++rt) {
          const int activation_row = rt * 16 + lane16;
          const q8_1_ds4_kmajor_block& xb = activation[activation_row];
          const float activation_scale =
              fp16_bits_to_float(xb.ds4[2 * local_sb + 0]);
          const float activation_sum =
              fp16_bits_to_float(xb.ds4[2 * local_sb + 1]);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int output_local = wave * 16 + half_id + j * 2;
            if (!fallback ||
                (output_base + output_local < out_features &&
                 row_base + activation_row < rows)) {
              const uint16_t* row_dm =
                  reinterpret_cast<const uint16_t*>(
                      weight_tile +
                      output_local * SOURCE_WEIGHT_STRIDE_WORDS + 64);
              const float weight_scale =
                  fp16_bits_to_float(row_dm[2 * weight_sb + 0]);
              const float weight_minimum =
                  fp16_bits_to_float(row_dm[2 * weight_sb + 1]);
              acc[rt][j] +=
                  weight_scale * activation_scale *
                      static_cast<float>(dots[rt][j]) +
                  weight_minimum * activation_sum;
            }
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int rt = 0; rt < 8; ++rt) {
    const int local_row = rt * 16 + lane16;
    const int global_row = row_base + local_row;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int output_local = wave * 16 + half_id + j * 2;
      const int output_col = output_base + output_local;
      if (!fallback ||
          (global_row < rows && output_col < out_features)) {
        store_output<output_t>(
            out,
            static_cast<int64_t>(global_row) * out_features + output_col,
            acc[rt][j]);
      }
    }
  }
}

// C8 target-verifier tile matching llama.cpp's small-matrix gfx1100 MMQ
// geometry: four waves own 64 output columns and one or two J16 row minitiles.
// Unlike the rejected source-faithful DS4 route, this variant keeps FP32 D4S4
// metadata so source tiling does not add an independent precision downgrade.
template <typename output_t, bool fallback, int tile_rows>
__launch_bounds__(128, 4)
__global__ void gguf_q5_k_mmq_i64_j32_k256_q8_1_d4s4_f32_kernel(
    const q8_1_d4s4_f32_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    output_t* __restrict__ out,
    int rows,
    int hidden,
    int out_features) {
  typedef int i32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int i32x8_t __attribute__((__vector_size__(8 * sizeof(int))));
  static_assert(tile_rows == 16 || tile_rows == 32, "unsupported J tile");

  extern __shared__ uint8_t shared_bytes[];
  uint32_t* weight_tile = reinterpret_cast<uint32_t*>(shared_bytes);
  q8_1_d4s4_f32_block* activation =
      reinterpret_cast<q8_1_d4s4_f32_block*>(
          weight_tile +
          SOURCE_C8_MMQ_I * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS);

  const int wave = static_cast<int>(threadIdx.y);
  const int wave_tid = static_cast<int>(threadIdx.x);
  const int tid = wave * 32 + wave_tid;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;
  const int output_base = static_cast<int>(blockIdx.x) * SOURCE_C8_MMQ_I;
  const int row_base = static_cast<int>(blockIdx.y) * tile_rows;
  const int weight_input_local = wave * 16 + lane16;
  const int weight_input_col = output_base + weight_input_local;
  const bool valid_weight_input = !fallback || weight_input_col < out_features;
  const int blocks_per_weight_row = hidden / QK_K;
  const int weight_row_bytes = blocks_per_weight_row * Q5_K_BLOCK_BYTES;

  float acc[tile_rows / 16][8];
#pragma unroll
  for (int rt = 0; rt < tile_rows / 16; ++rt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[rt][j] = 0.0f;
  }

  for (int weight_block = 0; weight_block < blocks_per_weight_row;
       ++weight_block) {
    const int load_row_in_wave = wave_tid >> 3;
    const int load_lane8 = wave_tid & 7;
#pragma unroll
    for (int load_round = 0; load_round < 4; ++load_round) {
      const int output_round =
          load_round * 16 + wave * 4 + load_row_in_wave;
      const int global_output = output_base + output_round;
      const bool valid = !fallback || global_output < out_features;
      const uint8_t* block = valid
          ? qweight + static_cast<int64_t>(global_output) * weight_row_bytes +
                static_cast<int64_t>(weight_block) * Q5_K_BLOCK_BYTES
          : nullptr;
      const int lane4 = load_lane8 * 4;
      uint32_t high_bits = 0;
      if (valid) {
        high_bits = *reinterpret_cast<const uint32_t*>(block + 16 + lane4);
      }
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        uint32_t low = 0;
        if (valid) {
          low = *reinterpret_cast<const uint32_t*>(
              block + 48 + pair * 32 + lane4);
        }
        const int even_sb = pair * 2;
        const uint32_t even_high =
            ((high_bits >> even_sb) & 0x01010101u) << 4;
        const uint32_t odd_high =
            ((high_bits >> (even_sb + 1)) & 0x01010101u) << 4;
        weight_tile[
            output_round * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS +
            even_sb * 8 + load_lane8] =
                (low & 0x0F0F0F0Fu) | even_high;
        weight_tile[
            output_round * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS +
            (even_sb + 1) * 8 + load_lane8] =
                ((low >> 4) & 0x0F0F0F0Fu) | odd_high;
      }
      float scale = 0.0f;
      float minimum = 0.0f;
      if (valid) {
        const float d = fp16_bytes_to_float(block);
        const float dmin = fp16_bytes_to_float(block + 2);
        scale = d * static_cast<float>(q45_scale(block + 4, load_lane8));
        minimum =
            -dmin * static_cast<float>(q45_min(block + 4, load_lane8));
      }
      float* row_dm = reinterpret_cast<float*>(
          weight_tile +
          output_round * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS + 64);
      row_dm[2 * load_lane8 + 0] = scale;
      row_dm[2 * load_lane8 + 1] = minimum;
    }
    __syncthreads();

#pragma unroll
    for (int khalf = 0; khalf < 2; ++khalf) {
      const int activation_block = weight_block * 2 + khalf;
      uint32_t* activation_words = reinterpret_cast<uint32_t*>(activation);
      constexpr int words_per_record = sizeof(q8_1_d4s4_f32_block) / 4;
      for (int index = tid;
           index < tile_rows * words_per_record;
           index += SOURCE_C8_MMQ_THREADS) {
        const int local_row = index / words_per_record;
        const int word = index - local_row * words_per_record;
        const int global_row = row_base + local_row;
        uint32_t value = 0;
        if (!fallback || global_row < rows) {
          const q8_1_d4s4_f32_block* source =
              xq + static_cast<int64_t>(activation_block) * rows + global_row;
          value = reinterpret_cast<const uint32_t*>(source)[word];
        }
        activation_words[index] = value;
      }
      __syncthreads();

#pragma unroll 2
      for (int local_sb = 0; local_sb < 4; ++local_sb) {
        const int weight_sb = khalf * 4 + local_sb;
        i32x4_t weight0 = {0, 0, 0, 0};
        i32x4_t weight1 = {0, 0, 0, 0};
        if (valid_weight_input) {
          const uint8_t* weight_values = reinterpret_cast<const uint8_t*>(
              weight_tile +
              weight_input_local * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS) +
              weight_sb * 32;
          weight0 = i32x4_t{
              static_cast<int>(pack_u8x4(weight_values + 0)),
              static_cast<int>(pack_u8x4(weight_values + 4)),
              static_cast<int>(pack_u8x4(weight_values + 8)),
              static_cast<int>(pack_u8x4(weight_values + 12)),
          };
          weight1 = i32x4_t{
              static_cast<int>(pack_u8x4(weight_values + 16)),
              static_cast<int>(pack_u8x4(weight_values + 20)),
              static_cast<int>(pack_u8x4(weight_values + 24)),
              static_cast<int>(pack_u8x4(weight_values + 28)),
          };
        }

        i32x8_t dots[tile_rows / 16];
#pragma unroll
        for (int rt = 0; rt < tile_rows / 16; ++rt) {
          const int activation_row = rt * 16 + lane16;
          const q8_1_d4s4_f32_block& xb = activation[activation_row];
          const int8_t* activation_values = xb.qs + local_sb * 32;
          const i32x4_t activation0 = i32x4_t{
              static_cast<int>(pack_i8x4(activation_values + 0)),
              static_cast<int>(pack_i8x4(activation_values + 4)),
              static_cast<int>(pack_i8x4(activation_values + 8)),
              static_cast<int>(pack_i8x4(activation_values + 12)),
          };
          const i32x4_t activation1 = i32x4_t{
              static_cast<int>(pack_i8x4(activation_values + 16)),
              static_cast<int>(pack_i8x4(activation_values + 20)),
              static_cast<int>(pack_i8x4(activation_values + 24)),
              static_cast<int>(pack_i8x4(activation_values + 28)),
          };
          dots[rt] = i32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
          dots[rt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight0, true, activation0, dots[rt], true);
          dots[rt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, weight1, true, activation1, dots[rt], true);
        }

#pragma unroll
        for (int rt = 0; rt < tile_rows / 16; ++rt) {
          const int activation_row = rt * 16 + lane16;
          const q8_1_d4s4_f32_block& xb = activation[activation_row];
          const float activation_scale = xb.d4[local_sb];
          const float activation_sum = xb.s4[local_sb];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int output_local = wave * 16 + half_id + j * 2;
            if (!fallback ||
                (output_base + output_local < out_features &&
                 row_base + activation_row < rows)) {
              const float* row_dm = reinterpret_cast<const float*>(
                  weight_tile +
                  output_local * SOURCE_C8_F32_WEIGHT_STRIDE_WORDS + 64);
              const float weight_scale = row_dm[2 * weight_sb + 0];
              const float weight_minimum = row_dm[2 * weight_sb + 1];
              acc[rt][j] +=
                  weight_scale * activation_scale *
                      static_cast<float>(dots[rt][j]) +
                  weight_minimum * activation_sum;
            }
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int rt = 0; rt < tile_rows / 16; ++rt) {
    const int local_row = rt * 16 + lane16;
    const int global_row = row_base + local_row;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int output_local = wave * 16 + half_id + j * 2;
      const int output_col = output_base + output_local;
      if (!fallback ||
          (global_row < rows && output_col < out_features)) {
        store_output<output_t>(
            out,
            static_cast<int64_t>(global_row) * out_features + output_col,
            acc[rt][j]);
      }
    }
  }
}

int launch_quantize_ds4_kmajor(
    const uint16_t* x,
    q8_1_ds4_kmajor_block* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % Q8_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_1_ds4_quantize_bf16_kmajor_kernel,
      grid,
      block,
      0,
      stream,
      x,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

int launch_quantize_d4s4_f32_kmajor(
    const uint16_t* x,
    q8_1_d4s4_f32_block* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % Q8_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_1_d4s4_f32_quantize_bf16_kmajor_kernel,
      grid,
      block,
      0,
      stream,
      x,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

template <typename output_t>
int launch_q5_source_mmq(
    const q8_1_ds4_kmajor_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32, 8);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + SOURCE_MMQ_I - 1) / SOURCE_MMQ_I),
      static_cast<unsigned int>((rows + SOURCE_MMQ_J - 1) / SOURCE_MMQ_J));
  if ((rows % SOURCE_MMQ_J) == 0 &&
      (out_features % SOURCE_MMQ_I) == 0) {
    hipLaunchKernelGGL(
        (gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_kernel<output_t, false>),
        grid,
        block,
        SOURCE_MMQ_SHARED_BYTES,
        stream,
        xq,
        qweight,
        out,
        static_cast<int>(rows),
        static_cast<int>(hidden),
        static_cast<int>(out_features));
  } else {
    hipLaunchKernelGGL(
        (gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_kernel<output_t, true>),
        grid,
        block,
        SOURCE_MMQ_SHARED_BYTES,
        stream,
        xq,
        qweight,
        out,
        static_cast<int>(rows),
        static_cast<int>(hidden),
        static_cast<int>(out_features));
  }
  return static_cast<int>(hipGetLastError());
}

template <typename output_t, int tile_rows>
int launch_q5_source_c8_mmq_tile(
    const q8_1_d4s4_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  const dim3 block(32, 4);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + SOURCE_C8_MMQ_I - 1) / SOURCE_C8_MMQ_I),
      static_cast<unsigned int>((rows + tile_rows - 1) / tile_rows));
  constexpr int shared_bytes = tile_rows == 16
      ? SOURCE_C8_MMQ_J16_SHARED_BYTES
      : SOURCE_C8_MMQ_SHARED_BYTES;
  if ((rows % tile_rows) == 0 &&
      (out_features % SOURCE_C8_MMQ_I) == 0) {
    hipLaunchKernelGGL(
        (gguf_q5_k_mmq_i64_j32_k256_q8_1_d4s4_f32_kernel<
            output_t, false, tile_rows>),
        grid,
        block,
        shared_bytes,
        stream,
        xq,
        qweight,
        out,
        static_cast<int>(rows),
        static_cast<int>(hidden),
        static_cast<int>(out_features));
  } else {
    hipLaunchKernelGGL(
        (gguf_q5_k_mmq_i64_j32_k256_q8_1_d4s4_f32_kernel<
            output_t, true, tile_rows>),
        grid,
        block,
        shared_bytes,
        stream,
        xq,
        qweight,
        out,
        static_cast<int>(rows),
        static_cast<int>(hidden),
        static_cast<int>(out_features));
  }
  return static_cast<int>(hipGetLastError());
}

template <typename output_t>
int launch_q5_source_c8_mmq(
    const q8_1_d4s4_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // Retained policy after the 8-run L3 gate (2026-09-04): the J16 minitile
  // won the 48-weight leaf 1.20x with bit-identical outputs, but the e2e
  // default-versus-rollback gate could not confirm it (pooled -0.20%, first
  // position won 4/4 order pairs under ~0.6%/run host drift), so the literal
  // J32 policy remains the default. The forced i64_j16 launchers stay
  // registered for leaf/diagnostic use; see worklog entry for iteration 26.
  if (rows < SOURCE_C8_MMQ_J) {
    return launch_q5_source_c8_mmq_tile<output_t, 16>(
        xq, qweight, out, rows, hidden, out_features, stream);
  }
  return launch_q5_source_c8_mmq_tile<output_t, SOURCE_C8_MMQ_J>(
      xq, qweight, out, rows, hidden, out_features, stream);
}

int launch_quantize(
    const uint16_t* x,
    q8_1_d4s4_f32_block* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % Q8_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_1_d4s4_f32_quantize_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

int launch_quantize_d8(
    const uint16_t* x,
    q8_1_d8s8_f32_block* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % Q8_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_1_d8s8_f32_quantize_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

int launch_quantize_d8r8(
    const uint16_t* x,
    q8_1_d8r8s8_f32_block* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (x == nullptr || out == nullptr || rows <= 0 || hidden <= 0 ||
      (hidden % Q8_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>((hidden + 1023) / 1024));
  hipLaunchKernelGGL(
      q8_1_d8r8s8_f32_quantize_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden));
  return static_cast<int>(hipGetLastError());
}

template <int qtype, typename output_t>
int launch_mmq(
    const q8_1_d4s4_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(MMQ_THREADS);
  const dim3 grid(
      static_cast<unsigned int>((out_features + MMQ_COLS - 1) / MMQ_COLS),
      static_cast<unsigned int>((rows + MMQ_ROWS - 1) / MMQ_ROWS));
  hipLaunchKernelGGL(
      (gguf_k_raw_mmq32_q8_1_d4s4_f32_kernel<qtype, output_t, 2>),
      grid,
      block,
      0,
      stream,
      xq,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <int qtype, typename output_t, int kMinBlocksPerProcessor>
int launch_mmq_minblocks(
    const q8_1_d4s4_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(MMQ_THREADS);
  const dim3 grid(
      static_cast<unsigned int>((out_features + MMQ_COLS - 1) / MMQ_COLS),
      static_cast<unsigned int>((rows + MMQ_ROWS - 1) / MMQ_ROWS));
  hipLaunchKernelGGL(
      (gguf_k_raw_mmq32_q8_1_d4s4_f32_kernel<
          qtype,
          output_t,
          kMinBlocksPerProcessor>),
      grid,
      block,
      0,
      stream,
      xq,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <int qtype, typename output_t>
int launch_mmq_d8(
    const q8_1_d8s8_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(MMQ_THREADS);
  const dim3 grid(
      static_cast<unsigned int>((out_features + MMQ_COLS - 1) / MMQ_COLS),
      static_cast<unsigned int>((rows + MMQ_ROWS - 1) / MMQ_ROWS));
  hipLaunchKernelGGL(
      (gguf_k_raw_mmq32_q8_1_d8s8_f32_kernel<qtype, output_t>),
      grid,
      block,
      0,
      stream,
      xq,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <int qtype, typename output_t>
int launch_mmq_d8r8(
    const q8_1_d8r8s8_f32_block* xq,
    const uint8_t* qweight,
    output_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  if (xq == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || hidden <= 0 || (hidden % QK_K) != 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(MMQ_THREADS);
  const dim3 grid(
      static_cast<unsigned int>((out_features + MMQ_COLS - 1) / MMQ_COLS),
      static_cast<unsigned int>((rows + MMQ_ROWS - 1) / MMQ_ROWS));
  hipLaunchKernelGGL(
      (gguf_k_raw_mmq32_q8_1_d8r8s8_f32_kernel<qtype, output_t>),
      grid,
      block,
      0,
      stream,
      xq,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(hidden),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q8_1_ds4_quantize_bf16_kmajor(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize_ds4_kmajor(
      x,
      reinterpret_cast<q8_1_ds4_kmajor_block*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_d4s4_f32_quantize_bf16_kmajor(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize_d4s4_f32_kmajor(
      x,
      reinterpret_cast<q8_1_d4s4_f32_block*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int
hipengine_gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out(
    const void* xq,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_mmq<uint16_t>(
      reinterpret_cast<const q8_1_ds4_kmajor_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

extern "C" int
hipengine_gguf_q5_k_mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out(
    const void* xq,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_mmq<float>(
      reinterpret_cast<const q8_1_ds4_kmajor_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

extern "C" int
hipengine_gguf_q5_k_mmq_i64_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
    const void* xq,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_c8_mmq<uint16_t>(
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

extern "C" int
hipengine_gguf_q5_k_mmq_i64_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out(
    const void* xq,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_c8_mmq<float>(
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

// Forced J16 minitile launchers: the same d4s4_f32_kmajor producer layout and
// the same per-(row, output) accumulation order as the J32 owner, but each
// workgroup owns a 16-row minitile so R32 launches two grid.y blocks per
// output tile (160 workgroups instead of 80 on gfx1100 C8 ssm_out shapes).
// Candidate leaf for the C8-P2 Q5 pool; the J32 owner remains the registered
// default and strict rollback.
extern "C" int
hipengine_gguf_q5_k_mmq_i64_j16_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out(
    const void* xq,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_c8_mmq_tile<uint16_t, 16>(
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

extern "C" int
hipengine_gguf_q5_k_mmq_i64_j16_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out(
    const void* xq,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_source_c8_mmq_tile<float, 16>(
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq),
      qweight,
      out,
      rows,
      hidden,
      out_features,
      stream);
}

extern "C" int hipengine_gguf_q8_1_d4s4_f32_quantize_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize(
      x,
      reinterpret_cast<q8_1_d4s4_f32_block*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_d8s8_f32_quantize_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize_d8(
      x,
      reinterpret_cast<q8_1_d8s8_f32_block*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_d8r8s8_f32_quantize_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_quantize_d8r8(
      x,
      reinterpret_cast<q8_1_d8r8s8_f32_block*>(out),
      rows,
      hidden,
      stream);
}

#define HIPENGINE_GGUF_K_MMQ_EXPORTS(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d4s4_f32_bf16_bf16_out( \
    const void* xq, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq<QTYPE, uint16_t>( \
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
} \
extern "C" int hipengine_##NAME##_mmq32_mb4_q8_1_d4s4_f32_bf16_bf16_out( \
    const void* xq, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq_minblocks<QTYPE, uint16_t, 4>( \
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
} \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d4s4_f32_bf16_f32_out( \
    const void* xq, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq<QTYPE, float>( \
      reinterpret_cast<const q8_1_d4s4_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
}

HIPENGINE_GGUF_K_MMQ_EXPORTS(gguf_q5_k, 5)
HIPENGINE_GGUF_K_MMQ_EXPORTS(gguf_q6_k, 6)

#undef HIPENGINE_GGUF_K_MMQ_EXPORTS

#define HIPENGINE_GGUF_K_MMQ_D8_EXPORTS(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d8s8_f32_bf16_bf16_out( \
    const void* xq, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq_d8<QTYPE, uint16_t>( \
      reinterpret_cast<const q8_1_d8s8_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
} \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d8s8_f32_bf16_f32_out( \
    const void* xq, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq_d8<QTYPE, float>( \
      reinterpret_cast<const q8_1_d8s8_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
}

HIPENGINE_GGUF_K_MMQ_D8_EXPORTS(gguf_q5_k, 5)
HIPENGINE_GGUF_K_MMQ_D8_EXPORTS(gguf_q6_k, 6)

#undef HIPENGINE_GGUF_K_MMQ_D8_EXPORTS

#define HIPENGINE_GGUF_K_MMQ_D8R8_EXPORTS(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d8r8s8_f32_bf16_bf16_out( \
    const void* xq, const uint8_t* qweight, uint16_t* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq_d8r8<QTYPE, uint16_t>( \
      reinterpret_cast<const q8_1_d8r8s8_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
} \
extern "C" int hipengine_##NAME##_mmq32_q8_1_d8r8s8_f32_bf16_f32_out( \
    const void* xq, const uint8_t* qweight, float* out, int64_t rows, \
    int64_t hidden, int64_t out_features, hipStream_t stream) { \
  return launch_mmq_d8r8<QTYPE, float>( \
      reinterpret_cast<const q8_1_d8r8s8_f32_block*>(xq), qweight, out, \
      rows, hidden, out_features, stream); \
}

HIPENGINE_GGUF_K_MMQ_D8R8_EXPORTS(gguf_q5_k, 5)
HIPENGINE_GGUF_K_MMQ_D8R8_EXPORTS(gguf_q6_k, 6)

#undef HIPENGINE_GGUF_K_MMQ_D8R8_EXPORTS
