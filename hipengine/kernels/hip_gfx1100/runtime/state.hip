// Small graph-friendly runtime state helpers for HIPENGINE Qwen3.5/PARO.
// These kernels keep decode token/position state device-resident so a future
// HIP graph replay path does not depend on host-side memcpy between tokens.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void embedding_lookup_bf16_i64_kernel(
    const uint16_t* embedding,
    const int64_t* token_id,
    uint16_t* out,
    int64_t hidden_size,
    int64_t vocab_size) {
  const int64_t token = token_id[0];
  if (token < 0 || token >= vocab_size) {
    return;
  }
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= hidden_size) {
    return;
  }
  out[idx] = embedding[token * hidden_size + idx];
}

__global__ void set_i64_scalar_kernel(int64_t* out, int64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
  }
}

__global__ void set_decode_position_i64_kernel(
    int64_t* position,
    int64_t* context,
    int64_t value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    position[0] = value;
    context[0] = value + 1;
  }
}

__global__ void advance_decode_position_i64_kernel(int64_t* position, int64_t* context) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int64_t next = position[0] + 1;
    position[0] = next;
    context[0] = next + 1;
  }
}

bool valid_embedding_shape(int64_t hidden_size, int64_t vocab_size) {
  return hidden_size > 0 && vocab_size > 0;
}

}  // namespace

extern "C" int hipengine_embedding_lookup_bf16_i64(
    const uint16_t* embedding,
    const int64_t* token_id,
    uint16_t* out,
    int64_t hidden_size,
    int64_t vocab_size,
    hipStream_t stream) {
  if (!valid_embedding_shape(hidden_size, vocab_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((hidden_size + threads - 1) / threads);
  hipLaunchKernelGGL(
      embedding_lookup_bf16_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      embedding,
      token_id,
      out,
      hidden_size,
      vocab_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_set_i64_scalar(int64_t* out, int64_t value, hipStream_t stream) {
  hipLaunchKernelGGL(
      set_i64_scalar_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      out,
      value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_set_decode_position_i64(
    int64_t* position,
    int64_t* context,
    int64_t value,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      set_decode_position_i64_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      position,
      context,
      value);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_advance_decode_position_i64(
    int64_t* position,
    int64_t* context,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      advance_decode_position_i64_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      position,
      context);
  return static_cast<int>(hipGetLastError());
}
