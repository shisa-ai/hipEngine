// cuBLASLt long-bucket route epilogues for the Moonshine batch encoder.
//
// The C6/7.4 long-bucket screen measures cuBLASLt fp16 GEMM at 50-159x over
// the custom row-projection kernel at 1,248 rows, but cuBLASLt's output
// diverges at the FP32-reassociation (ULP) level.  To keep the retained
// FP16-rounding contract ("add bias/residual in FP32, round once to FP16"),
// the encoder's fc1/fc2 GEMMs run cuBLASLt with an FP32 C/D boundary and
// these two element-wise epilogue kernels finish the boundary exactly the way
// the custom ``moonshine_f16_projection_{bias,bias_residual}`` kernels do:
// only the GEMM dot-product reduction order differs.
//
//   out[r, c] = fp16(gemm_f32[r, c] + bias[c])                      (fc1)
//   out[r, c] = fp16(gemm_f32[r, c] + bias[c] + residual[r, c])     (fc2)
//
// The plain QKV / output projections have no bias and use cuBLASLt FP16
// output directly (single rounding, matching the custom kernel).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;

__device__ inline float moonshine_bias_value(const half_t* bias, int64_t column) {
  return static_cast<float>(bias[column]);
}

__global__ void moonshine_f16_bias_round_fp32_kernel(
    const float* __restrict__ gemm,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t elements,
    int64_t out_features) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const int64_t column = index % out_features;
    output[index] =
        static_cast<half_t>(gemm[index] + moonshine_bias_value(bias, column));
  }
}

__global__ void moonshine_f16_bias_residual_round_fp32_kernel(
    const float* __restrict__ gemm,
    const half_t* __restrict__ bias,
    const half_t* __restrict__ residual,
    half_t* __restrict__ output,
    int64_t elements,
    int64_t out_features) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const int64_t column = index % out_features;
    output[index] = static_cast<half_t>(
        gemm[index] + moonshine_bias_value(bias, column) +
        static_cast<float>(residual[index]));
  }
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_f16_bias_round_fp32(
    const float* gemm, const half_t* bias, half_t* output,
    int64_t rows, int64_t out_features, int64_t threads,
    cudaStream_t stream) {
  if (rows <= 0 || out_features <= 0) return 1;
  if (threads != 64 && threads != 128 && threads != 256) return 1;
  const int64_t elements = rows * out_features;
  const unsigned blocks = static_cast<unsigned>((elements + threads - 1) / threads);
  moonshine_f16_bias_round_fp32_kernel<<<blocks, static_cast<unsigned>(threads),
                                         0, stream>>>(
      gemm, bias, output, elements, out_features);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_f16_bias_residual_round_fp32(
    const float* gemm, const half_t* bias, const half_t* residual,
    half_t* output, int64_t rows, int64_t out_features, int64_t threads,
    cudaStream_t stream) {
  if (rows <= 0 || out_features <= 0) return 1;
  if (threads != 64 && threads != 128 && threads != 256) return 1;
  const int64_t elements = rows * out_features;
  const unsigned blocks = static_cast<unsigned>((elements + threads - 1) / threads);
  moonshine_f16_bias_residual_round_fp32_kernel<<<
      blocks, static_cast<unsigned>(threads), 0, stream>>>(
      gemm, bias, residual, output, elements, out_features);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
