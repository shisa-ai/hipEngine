// Raw selected-MoE GEMV kernels for GGUF Q3_K NextN gate/up experts.
//
// The block decoder mirrors llama.cpp block_q3_K (110 bytes / 256 values):
// 32-byte high mask, 64-byte packed 2-bit quants, 12 packed 6-bit scales,
// and one fp16 super-scale.  Kernels consume raw device pointers and preserve
// the registered unfused BF16 projection and SiLU rounding boundaries.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q3_K_BLOCK_BYTES = 110;

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

__device__ inline uint32_t load_u32_le(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ inline int q3_k_scale(const uint8_t* scales, int scale_index) {
  const uint32_t s0 = load_u32_le(scales);
  const uint32_t s1 = load_u32_le(scales + 4);
  const uint32_t s2 = load_u32_le(scales + 8);
  constexpr uint32_t KMASK1 = 0x03030303U;
  constexpr uint32_t KMASK2 = 0x0F0F0F0FU;
  uint32_t packed;
  if (scale_index < 4) {
    packed = (s0 & KMASK2) | ((s2 & KMASK1) << 4);
  } else if (scale_index < 8) {
    packed = (s1 & KMASK2) | (((s2 >> 2) & KMASK1) << 4);
  } else if (scale_index < 12) {
    packed = ((s0 >> 4) & KMASK2) | (((s2 >> 4) & KMASK1) << 4);
  } else {
    packed = ((s1 >> 4) & KMASK2) | (((s2 >> 6) & KMASK1) << 4);
  }
  return static_cast<int>((packed >> (8 * (scale_index & 3))) & 0xFFU) - 32;
}

__device__ inline float q3_k_weight_in_block(
    const uint8_t* __restrict__ block,
    int within) {
  const uint8_t* __restrict__ hmask = block;
  const uint8_t* __restrict__ qs = block + 32;
  const uint8_t* __restrict__ scales = block + 96;
  const float d = fp16_bytes_to_float(block + 108);

  const int half = within >> 7;
  const int local128 = within & 127;
  const int group32 = local128 >> 5;
  const int segment16 = (local128 >> 4) & 1;
  const int lane16 = within & 15;
  const int q_index = half * 32 + segment16 * 16 + lane16;
  const int q = static_cast<int>((qs[q_index] >> (2 * group32)) & 0x03U);
  const int mask_bit = 1 << (half * 4 + group32);
  const int signed_q = q - ((hmask[segment16 * 16 + lane16] & mask_bit) ? 0 : 4);
  const int scale_index = half * 8 + group32 * 2 + segment16;
  return d * static_cast<float>(q3_k_scale(scales, scale_index) * signed_q);
}

__device__ inline float reduce_block_sum(float value) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) wave_sums[wave] = value;
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) total += wave_sums[i];
  }
  return total;
}

__device__ inline void reduce_block_pair(float& value_a, float& value_b) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value_a += __shfl_down(value_a, offset);
    value_b += __shfl_down(value_b, offset);
  }
  __shared__ float wave_sums_a[8];
  __shared__ float wave_sums_b[8];
  if (lane == 0) {
    wave_sums_a[wave] = value_a;
    wave_sums_b[wave] = value_b;
  }
  __syncthreads();
  if (tid == 0) {
    value_a = 0.0f;
    value_b = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      value_a += wave_sums_a[i];
      value_b += wave_sums_b[i];
    }
  }
}

__device__ inline int selected_x_row(int row, int x_rows, int rows) {
  return x_rows == 1 ? 0 : row / (rows / x_rows);
}

__launch_bounds__(256, 2)
__global__ void gguf_q3_k_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int blocks = in_features / QK_K;
  const int row_bytes = blocks * Q3_K_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr = x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
      const int block_idx = k >> 8;
      const int within = k & 255;
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * Q3_K_BLOCK_BYTES;
      acc = fmaf(bf16_bits_to_float(x_row_ptr[k]), q3_k_weight_in_block(block, within), acc);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] = float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_q3_k_selected_dual_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int blocks = in_features / QK_K;
  const int row_bytes = blocks * Q3_K_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t row_offset = expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row = valid_expert ? gate_weight + row_offset : gate_weight;
  const uint8_t* __restrict__ up_row = valid_expert ? up_weight + row_offset : up_weight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr = x + static_cast<int64_t>(x_row) * in_features;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  if (valid_expert) {
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
      const int block_idx = k >> 8;
      const int within = k & 255;
      const int64_t block_offset = static_cast<int64_t>(block_idx) * Q3_K_BLOCK_BYTES;
      const float xv = bf16_bits_to_float(x_row_ptr[k]);
      gate_acc = fmaf(xv, q3_k_weight_in_block(gate_row + block_offset, within), gate_acc);
      up_acc = fmaf(xv, q3_k_weight_in_block(up_row + block_offset, within), up_acc);
    }
  }
  reduce_block_pair(gate_acc, up_acc);
  if (threadIdx.x == 0) {
    const float gate = bf16_bits_to_float(float_to_bf16_bits(gate_acc));
    const float up = bf16_bits_to_float(float_to_bf16_bits(up_acc));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

inline bool valid_selected(
    const void* x,
    const void* selected,
    const void* qweight,
    const void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads) {
  return x != nullptr && selected != nullptr && qweight != nullptr && out != nullptr &&
      x_rows > 0 && rows > 0 && rows % x_rows == 0 && num_experts > 0 &&
      in_features > 0 && in_features % QK_K == 0 && out_features > 0 &&
      (threads == 64 || threads == 128 || threads == 256);
}

}  // namespace

extern "C" int hipengine_gguf_q3_k_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_selected(x, selected, qweight, out, x_rows, rows, num_experts,
                      in_features, out_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_q3_k_selected_gemv_kernel<<<
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)), 0, stream>>>(
      x, selected, qweight, out, static_cast<int>(x_rows), static_cast<int>(rows),
      static_cast<int>(num_experts), static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (up_weight == nullptr ||
      !valid_selected(x, selected, gate_weight, out, x_rows, rows, num_experts,
                      in_features, out_features, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  gguf_q3_k_selected_dual_silu_gemv_kernel<<<
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)), 0, stream>>>(
      x, selected, gate_weight, up_weight, out, static_cast<int>(x_rows),
      static_cast<int>(rows), static_cast<int>(num_experts),
      static_cast<int>(in_features), static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}
