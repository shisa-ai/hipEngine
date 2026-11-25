#include <hip/hip_runtime.h>

// One row runs the serial kernels: a thread walks its channel, shifts its history and may write
// over the input. A block of several rows runs the row kernels, which give one thread to each
// (row, channel): a row reads its own input and the `width - 1` inputs before it and sums in the
// same order, so the results are bit for bit the serial ones. Such a block needs an output apart
// from its input, because a row's inputs are other rows' outputs otherwise, and the history is
// committed from the untouched input afterwards.

namespace {

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

constexpr int kRowChannels = 64;   // channels per workgroup of the row kernels
constexpr int kRowRows = 8;         // rows per workgroup
constexpr int kMaxWidth = 16;

// `input` and `output` may be the same buffer only at one row.
template <bool PER_ROW>
__global__ void short_conv_rows(const float* __restrict__ history, const float* __restrict__ input,
                                const float* __restrict__ taps, float* __restrict__ output, int channels,
                                int width, float scale, int rows, size_t tap_stride) {
    const int ch = blockIdx.x * kRowChannels + threadIdx.x;
    const int row0 = blockIdx.y * kRowRows;
    if (ch >= channels) return;
    const float* hist = history + size_t(ch) * width;
    const float* base = taps + size_t(ch) * width;
    for (int r = 0; r < kRowRows; ++r) {
        const int row = row0 + r;
        if (row >= rows) break;
        const float* w = base + (PER_ROW ? size_t(row) * tap_stride : 0);
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            const int back = row - i;
            const float x = back >= 0 ? input[size_t(back) * channels + ch] : hist[-back - 1];
            sum += x * w[i];
        }
        const float own = input[size_t(row) * channels + ch];
        output[size_t(row) * channels + ch] = (own + sum) * scale;
    }
}

__global__ void short_conv_dynamic_rows(const float* __restrict__ history, const float* __restrict__ input,
                                        const float* __restrict__ fixed, const float* __restrict__ dynamic,
                                        float* __restrict__ output, int heads, int dim, int width, int rows,
                                        float scale) {
    const int channels = heads * dim;
    const int ch = blockIdx.x * kRowChannels + threadIdx.x;
    const int row0 = blockIdx.y * kRowRows;
    if (ch >= channels) return;
    const int head = ch / dim;
    const float* hist = history + size_t(ch) * width;
    const float* w = fixed + size_t(ch) * width;
    for (int r = 0; r < kRowRows; ++r) {
        const int row = row0 + r;
        if (row >= rows) break;
        const float* d = dynamic + (size_t(row) * heads + head) * width;
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            const int back = row - i;
            const float x = back >= 0 ? input[size_t(back) * channels + ch] : hist[-back - 1];
            const float tap = w[i] + d[i];
            sum += x * tap;
        }
        const float own = input[size_t(row) * channels + ch];
        output[size_t(row) * channels + ch] = (own + sum) * scale;
    }
}

// After a block the history holds the last `width` inputs, the newest first. A block shorter than
// the history keeps the older inputs it did not reach.
__global__ void short_conv_commit(float* history, const float* __restrict__ input, int channels, int width,
                                  int rows) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    float* h = history + size_t(ch) * width;
    float keep[kMaxWidth];
    for (int i = 0; i < width; ++i) {
        const int back = rows - 1 - i;
        keep[i] = back >= 0 ? input[size_t(back) * channels + ch] : h[i - rows];
    }
    for (int i = 0; i < width; ++i) h[i] = keep[i];
}

}  // namespace

__global__ void inkling2_short_conv(float* history, const float* input, const float* taps,
                                   float* output, int channels, int width, float scale) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch < channels) convolve_channel(history, input, taps, output, ch, width, scale);
}

__global__ void inkling2_short_conv_block_serial(float* history, const float* input, const float* taps,
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

__global__ void inkling2_short_conv_dynamic_serial(float* history, float* input, const float* fixed,
                                                   const float* dynamic, int heads, int dim, int width,
                                                   int rows, float scale) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int channels = heads * dim;
    if (ch >= channels) return;
    const int head = ch / dim;
    float* h = history + size_t(ch) * width;
    const float* w = fixed + size_t(ch) * width;
    for (int row = 0; row < rows; ++row) {
        float* x = input + size_t(row) * channels + ch;
        const float* d = dynamic + (size_t(row) * heads + head) * width;
        for (int i = width - 1; i > 0; --i) h[i] = h[i - 1];
        h[0] = *x;
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            const float tap = w[i] + d[i];
            sum += h[i] * tap;
        }
        *x = (*x + sum) * scale;
    }
}

extern "C" int hipengine_inkling2_short_conv_block_serial(float* history, const float* input, const float* taps,
                                                          float* output, int channels, int width, float scale,
                                                          int rows, size_t tap_stride, hipStream_t stream) {
    if (!history || !input || !taps || !output || channels <= 0 || width <= 0 || rows <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_short_conv_block_serial, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream,
                       history, input, taps, output, channels, width, scale, rows, tap_stride);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_short_conv_dynamic_serial(float* history, float* input, const float* fixed,
                                                            const float* dynamic, int heads, int dim, int width,
                                                            int rows, float scale, hipStream_t stream) {
    if (!history || !input || !fixed || !dynamic || heads <= 0 || dim <= 0 || width <= 0 || rows <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_short_conv_dynamic_serial, dim3((heads * dim - 1) / 128 + 1), dim3(128), 0, stream,
                       history, input, fixed, dynamic, heads, dim, width, rows, scale);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_short_conv_block(float* history, const float* input, const float* taps,
                                                  float* output, int channels, int width, float scale, int rows,
                                                  int dynamic_taps, hipStream_t stream) {
    if (!history || !input || !taps || !output || channels <= 0 || width <= 0 || rows <= 0)
        return hipErrorInvalidValue;
    const size_t tap_stride = dynamic_taps ? size_t(channels) * width : 0;
    if (rows == 1) {
        // One row shifts its history itself, so it may convolve in place.
        hipLaunchKernelGGL(inkling2_short_conv_block_serial, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream,
                           history, input, taps, output, channels, width, scale, 1, tap_stride);
        return hipGetLastError();
    }
    if (output == input || width > kMaxWidth) return hipErrorInvalidValue;
    const dim3 grid((channels - 1) / kRowChannels + 1, (rows - 1) / kRowRows + 1);
    if (dynamic_taps) {
        hipLaunchKernelGGL((short_conv_rows<true>), grid, dim3(kRowChannels), 0, stream, history, input, taps, output,
                           channels, width, scale, rows, tap_stride);
    } else {
        hipLaunchKernelGGL((short_conv_rows<false>), grid, dim3(kRowChannels), 0, stream, history, input, taps, output,
                           channels, width, scale, rows, tap_stride);
    }
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(short_conv_commit, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream, history, input,
                       channels, width, rows);
    return hipGetLastError();
}

// A null `output` leaves the result in the input, which one row may do.
extern "C" int hipengine_inkling2_short_conv_dynamic(float* history, const float* input, const float* fixed,
                                                     const float* dynamic, float* output, int heads, int dim,
                                                     int width, int rows, float scale, hipStream_t stream) {
    if (!history || !input || !fixed || !dynamic || heads <= 0 || dim <= 0 || width <= 0 || rows <= 0)
        return hipErrorInvalidValue;
    const int channels = heads * dim;
    if (rows == 1 && (!output || output == input)) {
        hipLaunchKernelGGL(inkling2_short_conv_dynamic_serial, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream,
                           history, const_cast<float*>(input), fixed, dynamic, heads, dim, width, 1, scale);
        return hipGetLastError();
    }
    if (!output || output == input || width > kMaxWidth) return hipErrorInvalidValue;
    const dim3 grid((channels - 1) / kRowChannels + 1, (rows - 1) / kRowRows + 1);
    hipLaunchKernelGGL(short_conv_dynamic_rows, grid, dim3(kRowChannels), 0, stream, history, input, fixed, dynamic,
                       output, heads, dim, width, rows, scale);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(short_conv_commit, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream, history, input,
                       channels, width, rows);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_short_conv(float* history, const float* input, const float* taps,
                                            float* output, int channels, int width, float scale,
                                            hipStream_t stream) {
    if (!history || !input || !taps || !output || channels <= 0 || width <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_short_conv, dim3((channels - 1) / 128 + 1), dim3(128), 0, stream,
                       history, input, taps, output, channels, width, scale);
    return hipGetLastError();
}
