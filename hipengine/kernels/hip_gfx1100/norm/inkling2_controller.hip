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

// [rms(x) | rms(message)] of one row feeds the down projection.
__device__ __forceinline__ void controller_input_row(const float* x, const float* message, float* row, int width,
                                                     float eps) {
    __shared__ float shared[2][kWaves];
    float sum_x = 0.0f, sum_m = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        sum_x += x[j] * x[j];
        sum_m += message[j] * message[j];
    }
    block_sums(sum_x, sum_m, shared);
    const float root_x = sqrtf(sum_x / width + eps);
    const float root_m = sqrtf(sum_m / width + eps);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        row[j] = x[j] / root_x;
        row[width + j] = message[j] / root_m;
    }
}

// One block per (controller, row) over packed [inputs | messages].
__global__ void inkling2_controller_input(const float* x, const float* message, float* out,
                                          int rows, int width, int count, float eps) {
    const size_t base = size_t(blockIdx.x) * width;
    controller_input_row(x + base, message + base, out + size_t(blockIdx.x) * 2 * width, width, eps);
}

// The same over rows that stay where they are: one input and one message pointer per controller.
__global__ void inkling2_controller_input_direct(const float* x0, const float* x1, const float* x2,
                                                 const float* m0, const float* m1, const float* m2,
                                                 float* out, int rows, int width, int count, float eps) {
    const int controller = blockIdx.x / rows;
    const size_t base = size_t(blockIdx.x - controller * rows) * width;
    const float* x = (controller == 0 ? x0 : controller == 1 ? x1 : x2) + base;
    const float* message = (controller == 0 ? m0 : controller == 1 ? m1 : m2) + base;
    controller_input_row(x, message, out + size_t(blockIdx.x) * 2 * width, width, eps);
}

__device__ __forceinline__ float blended(float x, float message, float forget, float gain) {
    const float keep = 1.0f - tanhf(forget);
    const float take = tanhf(gain);
    return keep * x + take * message;
}

// Gate, blend and RMS-normalize one row with the layer weight.
__device__ __forceinline__ void controller_blend_row(const float* x, const float* message, const float* forget,
                                                     const float* gain, const float* w, float* out, int width,
                                                     float eps) {
    __shared__ float shared[2][kWaves];
    float kept[kKept];
    const int stride = blockDim.x;
    const bool register_only = (width + stride - 1) / stride <= kKept;
    float sum = 0.0f;
    int n = 0;
    for (int j = threadIdx.x; j < width; j += stride, ++n) {
        const float merged = blended(x[j], message[j], forget[j], gain[j]);
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
            merged = blended(x[j], message[j], forget[j], gain[j]);
        }
        out[j] = merged / root * w[j];
    }
}

// One block per (controller, row) over packed [inputs | messages], packed gates and weights.
__global__ void inkling2_controller_blend_norm(const float* x, const float* message, const float* forget,
                                               const float* gain, const float* weight, float* out,
                                               int rows, int width, int count, float eps) {
    const int controller = blockIdx.x / rows;
    const size_t base = size_t(blockIdx.x) * width;
    controller_blend_row(x + base, message + base, forget + base, gain + base,
                         weight + size_t(controller) * width, out + base, width, eps);
}

// The same over input, message and weight rows that stay where they are. `forget` and `gain` hold
// `rows` rows per controller, as the projections write them.
__global__ void inkling2_controller_blend_norm_direct(const float* x0, const float* x1, const float* x2,
                                                      const float* m0, const float* m1, const float* m2,
                                                      const float* forget, const float* gain,
                                                      const float* w0, const float* w1, const float* w2,
                                                      float* out, int rows, int width, int count, float eps) {
    const int controller = blockIdx.x / rows;
    const size_t base = size_t(blockIdx.x - controller * rows) * width;
    const size_t gate = size_t(rows) * width;
    controller_blend_row((controller == 0 ? x0 : controller == 1 ? x1 : x2) + base,
                         (controller == 0 ? m0 : controller == 1 ? m1 : m2) + base,
                         forget + controller * gate + base, gain + controller * gate + base,
                         controller == 0 ? w0 : controller == 1 ? w1 : w2,
                         out + size_t(blockIdx.x) * width, width, eps);
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

extern "C" int hipengine_inkling2_controller_input_direct(const float* x0, const float* x1, const float* x2,
                                                          const float* m0, const float* m1, const float* m2,
                                                          float* out, int rows, int width, int count, float eps,
                                                          hipStream_t stream) {
    if (!x0 || !m0 || !out || rows <= 0 || width <= 0 || count <= 0 || count > 3
            || (count > 1 && (!x1 || !m1)) || (count > 2 && (!x2 || !m2))) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_controller_input_direct, dim3(rows * count), dim3(kThreads), 0, stream, x0, x1, x2,
                       m0, m1, m2, out, rows, width, count, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_controller_blend_norm_direct(
        const float* x0, const float* x1, const float* x2, const float* m0, const float* m1, const float* m2,
        const float* forget, const float* gain, const float* w0, const float* w1, const float* w2, float* out,
        int rows, int width, int count, float eps, hipStream_t stream) {
    if (!x0 || !m0 || !forget || !gain || !w0 || !out || rows <= 0 || width <= 0 || count <= 0 || count > 3
            || (count > 1 && (!x1 || !m1 || !w1)) || (count > 2 && (!x2 || !m2 || !w2)))
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_controller_blend_norm_direct, dim3(rows * count), dim3(kThreads), 0, stream, x0, x1,
                       x2, m0, m1, m2, forget, gain, w0, w1, w2, out, rows, width, count, eps);
    return hipGetLastError();
}
