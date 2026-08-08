// cuDNN long-bucket conv-route epilogues for the Moonshine batch encoder.
//
// The C6/7.4 conv screen measures cudnnConvolutionForward at 9.2-23.6x over
// the custom Moonshine conv kernels at 1,248 frames (convs ~11.9 -> ~0.7 ms),
// but cuDNN's output diverges from the exact custom kernel at the
// FP32-reassociation (ULP) level.  cudnnConvolutionForward only computes the
// convolution (fp16 in/out, FP32 accumulate); the element-wise activation is
// applied by these kernels, mirroring the retained rounding contract:
//
//   conv1:  out[i] = fp16(tanhf(fp32(in[i])))                  (element-wise)
//   conv2:  out[i] = fp16(gelu_f32(fp32(in[i]) + fp32(bias[ch])))  (in place)
//   conv3:  out[i] = fp16(gelu_f32(fp32(in[i]) + fp32(bias[ch])))
//           then transpose NCHW [plane, ch, pos] -> row-major
//           [plane, pos, ch] into the encoder's "hidden" layout.
//
// The only divergence from the exact custom route is that cuDNN rounds the
// conv accumulator to fp16 before the epilogue (and sums in a different
// order), so tanh/gelu run on the fp16 conv output rather than the fp32
// accumulator.  That is the accepted C8 re-derived numerical gate
// (max-abs <= 2^-3, rel-L2 <= 5e-3 at rows = 1248), not a boundary change.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using half_t = __half;

__device__ inline float moonshine_gelu_f32(float value) {
  const float kGeluConstant = 0.70710678118654752440f;  // 1 / sqrt(2)
  return 0.5f * value * (1.0f + erff(value * kGeluConstant));
}
__device__ inline half_t moonshine_gelu_half(half_t value) {
  return static_cast<half_t>(moonshine_gelu_f32(static_cast<float>(value)));
}

__global__ void moonshine_tanh_apply_fp16_kernel(
    const half_t* __restrict__ input,
    half_t* __restrict__ output,
    int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = static_cast<half_t>(tanhf(static_cast<float>(input[index])));
  }
}

// NCHW channel-major [planes, channels, length]; bias per channel.
__global__ void moonshine_bias_gelu_apply_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t elements,
    int64_t channels,
    int64_t length) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const int64_t channel = (index / length) % channels;
    const float value =
        static_cast<float>(input[index]) + static_cast<float>(bias[channel]);
    output[index] = moonshine_gelu_half(static_cast<half_t>(value));
  }
}

// Read NCHW channel-major [planes, channels, length] and write row-major
// [planes, length, channels] (the encoder "hidden" layout) with bias+gelu.
__global__ void moonshine_bias_gelu_apply_rowmajor_fp16_kernel(
    const half_t* __restrict__ input,
    const half_t* __restrict__ bias,
    half_t* __restrict__ output,
    int64_t planes,
    int64_t channels,
    int64_t length) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = planes * channels * length;
  if (index < elements) {
    const int64_t plane = index / (channels * length);
    const int64_t rem = index % (channels * length);
    const int64_t channel = rem / length;
    const int64_t position = rem % length;
    const float value =
        static_cast<float>(input[index]) + static_cast<float>(bias[channel]);
    output[(plane * length + position) * channels + channel] =
        moonshine_gelu_half(static_cast<half_t>(value));
  }
}

}  // namespace

extern "C" int hipengine_cuda_sm120a_moonshine_tanh_apply_fp16(
    const half_t* input, half_t* output, int64_t elements,
    cudaStream_t stream) {
  if (elements <= 0) return 1;
  const int64_t blocks = (elements + 255) / 256;
  moonshine_tanh_apply_fp16_kernel<<<
      static_cast<unsigned>(blocks), 256, 0, stream>>>(
      input, output, elements);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_bias_gelu_apply_fp16(
    const half_t* input, const half_t* bias, half_t* output,
    int64_t elements, int64_t channels, int64_t length,
    cudaStream_t stream) {
  if (elements <= 0 || channels <= 0 || length <= 0) return 1;
  const int64_t blocks = (elements + 255) / 256;
  moonshine_bias_gelu_apply_fp16_kernel<<<
      static_cast<unsigned>(blocks), 256, 0, stream>>>(
      input, bias, output, elements, channels, length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int hipengine_cuda_sm120a_moonshine_bias_gelu_apply_rowmajor_fp16(
    const half_t* input, const half_t* bias, half_t* output,
    int64_t planes, int64_t channels, int64_t length,
    cudaStream_t stream) {
  if (planes <= 0 || channels <= 0 || length <= 0) return 1;
  const int64_t elements = planes * channels * length;
  const int64_t blocks = (elements + 255) / 256;
  moonshine_bias_gelu_apply_rowmajor_fp16_kernel<<<
      static_cast<unsigned>(blocks), 256, 0, stream>>>(
      input, bias, output, planes, channels, length);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
