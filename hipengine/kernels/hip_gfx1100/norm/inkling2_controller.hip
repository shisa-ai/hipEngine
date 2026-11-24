#include <hip/hip_runtime.h>
#include <math.h>

// Keep each host operation separately rounded: no fused multiply-add in the blend.
#pragma clang fp contract(off)

namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kKept = 16;

// Reduce two per-thread values over the block. Every thread receives both sums.
__device__ __forceinline__ void block_sums(float& a, float& b, float (*shared)[kWaves]) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        a += __shfl_down(a, offset);
        b += __shfl_down(b, offset);
    }
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[0][wave] = a;
        shared[1][wave] = b;
    }
    __syncthreads();
    const int waves = blockDim.x / warpSize;
    if (threadIdx.x == 0) {
        float total_a = 0.0f, total_b = 0.0f;
        for (int w = 0; w < waves; ++w) {
            total_a += shared[0][w];
            total_b += shared[1][w];
        }
        shared[0][0] = total_a;
        shared[1][0] = total_b;
    }
    __syncthreads();
    a = shared[0][0];
    b = shared[1][0];
}

// One block per (controller, row): [rms(x) | rms(message)] feeds the down projection.
__global__ void inkling2_controller_input(const float* x, const float* message, float* out,
                                          int rows, int width, int count, float eps) {
    __shared__ float shared[2][kWaves];
    const size_t base = size_t(blockIdx.x) * width;
    float sum_x = 0.0f, sum_m = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        sum_x += x[base + j] * x[base + j];
        sum_m += message[base + j] * message[base + j];
    }
    block_sums(sum_x, sum_m, shared);
    const float root_x = sqrtf(sum_x / width + eps);
    const float root_m = sqrtf(sum_m / width + eps);
    float* row = out + size_t(blockIdx.x) * 2 * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        row[j] = x[base + j] / root_x;
        row[width + j] = message[base + j] / root_m;
    }
}

__device__ __forceinline__ float blended(float x, float message, float forget, float gain) {
    const float keep = 1.0f - tanhf(forget);
    const float take = tanhf(gain);
    return keep * x + take * message;
}

// One block per (controller, row): gate, blend and RMS-normalize with the layer weight.
__global__ void inkling2_controller_blend_norm(const float* x, const float* message, const float* forget,
                                               const float* gain, const float* weight, float* out,
                                               int rows, int width, int count, float eps) {
    __shared__ float shared[2][kWaves];
    const int controller = blockIdx.x / rows;
    const size_t base = size_t(blockIdx.x) * width;
    const float* w = weight + size_t(controller) * width;
    float kept[kKept];
    const int stride = blockDim.x;
    const bool register_only = (width + stride - 1) / stride <= kKept;
    float sum = 0.0f;
    int n = 0;
    for (int j = threadIdx.x; j < width; j += stride, ++n) {
        const float merged = blended(x[base + j], message[base + j], forget[base + j], gain[base + j]);
        if (register_only) kept[n] = merged;
        sum += merged * merged;
    }
    float unused = 0.0f;
    block_sums(sum, unused, shared);
    const float root = sqrtf(sum / width + eps);
    n = 0;
    for (int j = threadIdx.x; j < width; j += stride, ++n) {
        float merged;
        if (register_only) {
            merged = kept[n];
        } else {
            merged = blended(x[base + j], message[base + j], forget[base + j], gain[base + j]);
        }
        out[base + j] = merged / root * w[j];
    }
}

}  // namespace

extern "C" int hipengine_inkling2_controller_input(const float* x, const float* message, float* out,
                                                   int rows, int width, int count, float eps, hipStream_t stream) {
    if (!x || !message || !out || rows <= 0 || width <= 0 || count <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_controller_input, dim3(rows * count), dim3(kThreads), 0, stream,
                       x, message, out, rows, width, count, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_controller_blend_norm(const float* x, const float* message,
                                                        const float* forget, const float* gain,
                                                        const float* weight, float* out,
                                                        int rows, int width, int count, float eps, hipStream_t stream) {
    if (!x || !message || !forget || !gain || !weight || !out || rows <= 0 || width <= 0 || count <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_controller_blend_norm, dim3(rows * count), dim3(kThreads), 0, stream,
                       x, message, forget, gain, weight, out, rows, width, count, eps);
    return hipGetLastError();
}
