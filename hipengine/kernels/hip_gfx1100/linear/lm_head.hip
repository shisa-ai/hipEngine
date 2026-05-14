// Minimal GPU lm-head + argmax bring-up kernel for Qwen3.5/PARO E2E smokes.
// Hidden is BF16 bits, weight is checkpoint FP16 bits, logits are FP32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline bool better_pair(float value, int64_t index, float best_value, int64_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__global__ void lm_head_fp16_logits_kernel(
    const uint16_t* __restrict__ hidden_bf16,
    const uint16_t* __restrict__ weight_fp16,
    float* __restrict__ logits,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= vocab_size) {
    return;
  }
  const int tid = threadIdx.x;
  const __half* weight = reinterpret_cast<const __half*>(weight_fp16);
  const int64_t weight_base = row * hidden_size;
  float acc = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden_size / vec_stride) * vec_stride;
  for (int64_t k = static_cast<int64_t>(tid) * 8; k < main_end; k += vec_stride) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 0]), __half2float(weight[weight_base + k + 0]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 1]), __half2float(weight[weight_base + k + 1]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 2]), __half2float(weight[weight_base + k + 2]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 3]), __half2float(weight[weight_base + k + 3]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 4]), __half2float(weight[weight_base + k + 4]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 5]), __half2float(weight[weight_base + k + 5]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 6]), __half2float(weight[weight_base + k + 6]), acc);
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k + 7]), __half2float(weight[weight_base + k + 7]), acc);
  }
  for (int64_t k = main_end + tid; k < hidden_size; k += blockDim.x) {
    acc = fmaf(bf16_bits_to_float(hidden_bf16[k]), __half2float(weight[weight_base + k]), acc);
  }

  extern __shared__ float partial[];
  partial[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    logits[row] = partial[0];
  }
}

__global__ void argmax_stage1_kernel(
    const float* __restrict__ logits,
    float* __restrict__ block_values,
    int64_t* __restrict__ block_indices,
    int64_t vocab_size) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  const int64_t chunk = static_cast<int64_t>(blockDim.x) * 4;
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk;
  const int64_t end = min(begin + chunk, vocab_size);

  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = begin + tid; i < end; i += blockDim.x) {
    const float value = logits[i];
    if (better_pair(value, i, best_value, best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_values[blockIdx.x] = values[0];
    block_indices[blockIdx.x] = indices[0];
  }
}

__global__ void argmax_stage2_kernel(
    const float* __restrict__ block_values,
    const int64_t* __restrict__ block_indices,
    int64_t num_blocks,
    int64_t* __restrict__ out_index,
    float* __restrict__ out_value) {
  extern __shared__ unsigned char shared_raw[];
  float* values = reinterpret_cast<float*>(shared_raw);
  int64_t* indices = reinterpret_cast<int64_t*>(values + blockDim.x);
  const int tid = threadIdx.x;
  float best_value = -3.4028234663852886e+38F;
  int64_t best_index = INT64_MAX;
  for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
    const float value = block_values[i];
    const int64_t index = block_indices[i];
    if (better_pair(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && better_pair(values[tid + stride], indices[tid + stride], values[tid], indices[tid])) {
      values[tid] = values[tid + stride];
      indices[tid] = indices[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *out_index = indices[0];
    *out_value = values[0];
  }
}

bool valid_threads(int64_t threads) {
  return threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_lm_head_fp16_argmax_bf16(
    const uint16_t* hidden_bf16,
    const uint16_t* weight_fp16,
    float* logits,
    float* block_values,
    int64_t* block_indices,
    int64_t* out_index,
    float* out_value,
    int64_t hidden_size,
    int64_t vocab_size,
    int64_t threads,
    hipStream_t stream) {
  if (hidden_size <= 0 || vocab_size <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int t = static_cast<unsigned int>(threads);
  hipLaunchKernelGGL(
      lm_head_fp16_logits_kernel,
      dim3(static_cast<unsigned int>(vocab_size)),
      dim3(t),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_bf16,
      weight_fp16,
      logits,
      hidden_size,
      vocab_size);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  const int64_t stage1_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
  const size_t argmax_smem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int64_t));
  hipLaunchKernelGGL(
      argmax_stage1_kernel,
      dim3(static_cast<unsigned int>(stage1_blocks)),
      dim3(t),
      argmax_smem,
      stream,
      logits,
      block_values,
      block_indices,
      vocab_size);
  err = hipGetLastError();
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }

  hipLaunchKernelGGL(
      argmax_stage2_kernel,
      dim3(1),
      dim3(t),
      argmax_smem,
      stream,
      block_values,
      block_indices,
      stage1_blocks,
      out_index,
      out_value);
  return static_cast<int>(hipGetLastError());
}
