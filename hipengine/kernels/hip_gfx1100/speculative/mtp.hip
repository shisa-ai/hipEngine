// Native MTP proposal helper kernels.
//
// These kernels implement provider-neutral pieces of the Qwen3.6/Qwen3.5 MTP
// head.  The first boundary normalizes the current token embedding and the
// target hidden row independently, concatenating the two BF16 streams so the
// existing dense BF16 GEMV kernel can apply mtp.fc.

#include <hip/hip_runtime.h>
#include <stdint.h>

using half_t = _Float16;

namespace {

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

__device__ inline float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) {
      total += shared[i];
    }
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

bool valid_shape(int64_t rows, int64_t hidden_size, int64_t vocab_size, int64_t threads) {
  return rows > 0 && hidden_size > 0 && vocab_size > 0 && (threads == 64 || threads == 128 || threads == 256);
}

__global__ void mtp_fuse_inputs_f16_bf16_kernel(
    const int64_t* __restrict__ token_ids,
    const half_t* __restrict__ embedding_f16,
    const uint16_t* __restrict__ target_hidden_bf16,
    const uint16_t* __restrict__ embed_norm_weight_bf16,
    const uint16_t* __restrict__ hidden_norm_weight_bf16,
    uint16_t* __restrict__ out_concat_bf16,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int stream = static_cast<int>(blockIdx.y);  // 0: token embedding, 1: target hidden
  if (row >= rows || stream < 0 || stream > 1) {
    return;
  }
  extern __shared__ float scratch[];

  const int64_t token = token_ids[row];
  const bool valid_token = token >= 0 && token < vocab_size;
  float sum_sq = 0.0f;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    if (stream == 0) {
      value = valid_token ? static_cast<float>(embedding_f16[token * hidden_size + col]) : 0.0f;
    } else {
      value = bf16_bits_to_float(target_hidden_bf16[row * hidden_size + col]);
    }
    sum_sq += value * value;
  }
  const float variance_sum = block_reduce_sum(sum_sq, scratch);
  const float inv_rms = rsqrtf(variance_sum / static_cast<float>(hidden_size) + eps);

  const uint16_t* weight = (stream == 0) ? embed_norm_weight_bf16 : hidden_norm_weight_bf16;
  const int64_t out_base = row * (2 * hidden_size) + static_cast<int64_t>(stream) * hidden_size;
  for (int64_t col = static_cast<int64_t>(threadIdx.x); col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    if (stream == 0) {
      value = valid_token ? static_cast<float>(embedding_f16[token * hidden_size + col]) : 0.0f;
    } else {
      value = bf16_bits_to_float(target_hidden_bf16[row * hidden_size + col]);
    }
    const float scale = 1.0f + bf16_bits_to_float(weight[col]);
    out_concat_bf16[out_base + col] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

}  // namespace

extern "C" int hipengine_mtp_fuse_inputs_f16_bf16(
    const int64_t* token_ids,
    const half_t* embedding_f16,
    const uint16_t* target_hidden_bf16,
    const uint16_t* embed_norm_weight_bf16,
    const uint16_t* hidden_norm_weight_bf16,
    uint16_t* out_concat_bf16,
    int64_t rows,
    int64_t hidden_size,
    int64_t vocab_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (!valid_shape(rows, hidden_size, vocab_size, threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      mtp_fuse_inputs_f16_bf16_kernel,
      dim3(static_cast<unsigned int>(rows), 2U),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(((threads + 31) / 32) * sizeof(float)),
      stream,
      token_ids,
      embedding_f16,
      target_hidden_bf16,
      embed_norm_weight_bf16,
      hidden_norm_weight_bf16,
      out_concat_bf16,
      rows,
      hidden_size,
      vocab_size,
      eps);
  return static_cast<int>(hipGetLastError());
}
