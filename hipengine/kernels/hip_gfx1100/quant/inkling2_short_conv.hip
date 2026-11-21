#include <hip/hip_runtime.h>

__device__ __forceinline__ void convolve_channel(float* history, const float* input, const float* taps,
                                                 float* output, int ch, int width, float scale) {
    float* h = history + size_t(ch) * width;
    const float* w = taps + size_t(ch) * width;
    for (int i = width - 1; i > 0; --i) h[i] = h[i-1];
    h[0] = input[ch];
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) sum += h[i] * w[i];
    output[ch] = (input[ch] + sum) * scale;
}

__global__ void inkling2_short_conv(float* history, const float* input, const float* taps,
                                   float* output, int channels, int width, float scale) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch < channels) convolve_channel(history, input, taps, output, ch, width, scale);
}

__global__ void inkling2_short_conv_block(float* history, const float* input, const float* taps,
                                         float* output, int channels, int width, float scale, int rows,
                                         size_t tap_stride) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    for (int row = 0; row < rows; ++row) {
        const size_t offset = size_t(row) * channels;
        convolve_channel(history, input + offset, taps + size_t(row) * tap_stride,
                         output + offset, ch, width, scale);
    }
}

extern "C" int hipengine_inkling2_short_conv_block(float* history, const float* input, const float* taps,
                                                  float* output, int channels, int width, float scale, int rows,
                                                  int dynamic_taps) {
    if (!history || !input || !taps || !output || channels <= 0 || width <= 0 || rows <= 0) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(inkling2_short_conv_block, dim3((channels - 1) / 128 + 1), dim3(128), 0, 0,
                       history, input, taps, output, channels, width, scale, rows,
                       dynamic_taps ? size_t(channels) * width : 0);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_short_conv(float* history, const float* input, const float* taps,
                                            float* output, int channels, int width, float scale) {
    if (!history || !input || !taps || !output || channels <= 0 || width <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_short_conv, dim3((channels - 1) / 128 + 1), dim3(128), 0, 0,
                       history, input, taps, output, channels, width, scale);
    return hipGetLastError();
}
