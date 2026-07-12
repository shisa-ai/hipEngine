// Torch-free device-side row gather for hipEngine runtime glue.
//
// Gathers whole rows from a row-major FP32 table using device-resident int32
// row indices.  This keeps a resident draft/embedding chain on-device: the row
// id produced by a prior kernel (e.g. top-k argmax) never has to round-trip to
// the host to select the next row.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void gather_f32_rows_by_i32id_kernel(
    const float* __restrict__ table,
    const int32_t* __restrict__ ids,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    int64_t vocab) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = rows * hidden;
  if (idx < count) {
    const int64_t row = idx / hidden;
    const int64_t col = idx - row * hidden;
    const int64_t id = static_cast<int64_t>(ids[row]);
    out[idx] = (id < 0 || id >= vocab) ? 0.0f : table[id * hidden + col];
  }
}

}  // namespace

extern "C" int hipengine_gather_f32_rows_by_i32id(
    const float* table,
    const int32_t* ids,
    float* out,
    int64_t rows,
    int64_t hidden,
    int64_t vocab,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || vocab <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t count = rows * hidden;
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(
      gather_f32_rows_by_i32id_kernel, dim3(blocks), dim3(threads), 0, stream,
      table, ids, out, rows, hidden, vocab);
  return static_cast<int>(hipGetLastError());
}
