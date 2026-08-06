// Moonshine decoder gated-SiLU FP16 activation primitive (CUDA sm_120a).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;

__global__ void moonshine_gated_silu_fp16_kernel(
    const half_t* __restrict__ fc1_output,
    half_t* __restrict__ output,
    int64_t rows,
    int64_t intermediate_size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = rows * intermediate_size;
  if (index >= elements) return;
  const int64_t row = index / intermediate_size;
  const int64_t column = index - row * intermediate_size;
  const int64_t input_offset = row * (2 * intermediate_size) + column;
  const float value = static_cast<float>(fc1_output[input_offset]);
  const float gate = static_cast<float>(
      fc1_output[input_offset + intermediate_size]);
  const float activated = gate / (1.0f + expf(-gate));
  output[index] = static_cast<half_t>(value * activated);
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_gated_silu_fp16(
    const half_t* fc1_output,
    half_t* output,
    int64_t rows,
    int64_t intermediate_size,
    int64_t threads,
    cudaStream_t stream) {
  if (rows <= 0 || intermediate_size <= 0 || threads <= 0 || threads > 256 ||
      (threads % 32) != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t elements = rows * intermediate_size;
  const int64_t blocks = (elements + threads - 1) / threads;
  moonshine_gated_silu_fp16_kernel<<<
      dim3(blocks), dim3(threads), 0, stream>>>(
      fc1_output, output, rows, intermediate_size);
  return cudaGetLastError();
}
