#include <hip/hip_runtime.h>
#include <stdint.h>

// Row-wise reductions over a hidden width, plus the residual-stack read.
// Every kernel uses the same fixed tree, so repeated runs are bit-identical.

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float block_sum(float value, float* shared) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down(value, offset, 32);
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) shared[wave] = value;
    __syncthreads();
    const int waves = blockDim.x / 32;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < waves; ++w) total += shared[w];
        shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float inverse_root(float sum, int width, float eps) {
    return rsqrtf(sum / float(width) + eps);
}

}  // namespace

// out = x / sqrt(mean(x * x) + eps), one block per row. `out` may alias `x`.
__global__ void inkling2_row_rms(const float* x, float* out, int rows, int width, float eps) {
    __shared__ float shared[8];
    const int row = blockIdx.x;
    const float* source = x + size_t(row) * width;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) sum += source[j] * source[j];
    const float inverse = inverse_root(block_sum(sum, shared), width, eps);
    float* target = out + size_t(row) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) target[j] = __fmul_rn(source[j], inverse);
}

// out = x / sqrt(mean(x * x) + eps) * weight, one block per row.
__global__ void inkling2_row_norm_mul(const float* x, const float* weight, float* out, int rows, int width,
                                      float eps) {
    __shared__ float shared[8];
    const int row = blockIdx.x;
    const float* source = x + size_t(row) * width;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) sum += source[j] * source[j];
    const float inverse = inverse_root(block_sum(sum, shared), width, eps);
    float* target = out + size_t(row) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        target[j] = __fmul_rn(__fmul_rn(source[j], inverse), weight[j]);
    }
}

// out = a + b, one block per row.
__global__ void inkling2_row_add(const float* a, const float* b, float* out, int rows, int width) {
    const size_t base = size_t(blockIdx.x) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) out[base + j] = a[base + j] + b[base + j];
}

// out = softmax((query . values) / sqrt(width)) . values over `blocks` rows, one block per query row.
__global__ void inkling2_residual_read(const float* query, const float* values, float* out, int blocks, int width,
                                       float divisor) {
    extern __shared__ float shared[];          // blocks scores, then the weights
    float* scores = shared;
    float* weights = shared + blocks;
    const int row = blockIdx.x;
    const float* q = query + size_t(row) * width;
    for (int b = 0; b < blocks; ++b) {
        const float* v = values + size_t(b) * width;
        float part = 0.0f;
        for (int j = threadIdx.x; j < width; j += blockDim.x) part += q[j] * v[j];
        // Every lane needs the total before the next block starts.
        part = block_sum(part, weights + blocks);
        if (threadIdx.x == 0) scores[b] = part * divisor;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float maximum = scores[0];
        for (int b = 1; b < blocks; ++b) maximum = fmaxf(maximum, scores[b]);
        float total = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            const float e = expf(scores[b] - maximum);
            weights[b] = e;
            total += e;
        }
        const float inverse = __fdiv_rn(1.0f, total);
        for (int b = 0; b < blocks; ++b) weights[b] = __fmul_rn(weights[b], inverse);
    }
    __syncthreads();
    float* target = out + size_t(row) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < blocks; ++b) acc += __fmul_rn(weights[b], values[size_t(b) * width + j]);
        target[j] = acc;
    }
}

extern "C" int hipengine_inkling2_row_rms(const float* x, float* out, int rows, int width, float eps) {
    if (!x || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_rms, dim3(rows), dim3(kThreads), 0, 0, x, out, rows, width, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_row_norm_mul(const float* x, const float* weight, float* out, int rows, int width,
                                               float eps) {
    if (!x || !weight || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_norm_mul, dim3(rows), dim3(kThreads), 0, 0, x, weight, out, rows, width, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_row_add(const float* a, const float* b, float* out, int rows, int width) {
    if (!a || !b || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_add, dim3(rows), dim3(kThreads), 0, 0, a, b, out, rows, width);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_residual_read(const float* query, const float* values, float* out, int rows,
                                                int blocks, int width) {
    if (!query || !values || !out || rows <= 0 || rows > 65535 || blocks <= 0 || blocks > 64 ||
        width <= 0 || width > 8192) return hipErrorInvalidValue;
    const size_t shared = size_t(2 * blocks + kThreads / 32) * sizeof(float);
    hipLaunchKernelGGL(inkling2_residual_read, dim3(rows), dim3(kThreads), shared, 0, query, values, out, blocks,
                       width, 1.0f / sqrtf(float(width)));
    return hipGetLastError();
}
