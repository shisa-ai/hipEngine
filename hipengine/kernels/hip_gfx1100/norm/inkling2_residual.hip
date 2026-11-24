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

// out[i][r] = softmax((query[i] . values[b][r]) / sqrt(width)) . values[b][r] over `blocks` rows.
// `values` is block-major: block b holds the `rows` normalized rows of `width` elements. The
// output is query-major, so one query's `rows` results are contiguous.
__global__ void inkling2_residual_read(const float* query, const float* values, float* out, int rows, int blocks,
                                       int queries, int width, float divisor) {
    extern __shared__ float shared[];          // blocks scores, then the weights
    float* scores = shared;
    float* weights = shared + blocks;
    const int index = blockIdx.x;
    const int row = index / queries;
    const int which = index % queries;
    const float* q = query + size_t(which) * width;
    const float* base = values + size_t(row) * width;
    const size_t stride = size_t(rows) * width;
    for (int b = 0; b < blocks; ++b) {
        const float* v = base + size_t(b) * stride;
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
    float* target = out + (size_t(which) * rows + row) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < blocks; ++b) acc += __fmul_rn(weights[b], base[size_t(b) * stride + j]);
        target[j] = acc;
    }
}

extern "C" int hipengine_inkling2_row_rms(const float* x, float* out, int rows, int width, float eps, hipStream_t stream) {
    if (!x || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_rms, dim3(rows), dim3(kThreads), 0, stream, x, out, rows, width, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_row_norm_mul(const float* x, const float* weight, float* out, int rows, int width,
                                               float eps, hipStream_t stream) {
    if (!x || !weight || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_norm_mul, dim3(rows), dim3(kThreads), 0, stream, x, weight, out, rows, width, eps);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_row_add(const float* a, const float* b, float* out, int rows, int width,
                                            hipStream_t stream) {
    if (!a || !b || !out || rows <= 0 || rows > 65535 || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_add, dim3(rows), dim3(kThreads), 0, stream, a, b, out, rows, width);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_residual_read(const float* query, const float* values, float* out, int rows,
                                                int blocks, int queries, int width, hipStream_t stream) {
    if (!query || !values || !out || rows <= 0 || blocks <= 0 || blocks > 64 || queries <= 0 || queries > 8 ||
        width <= 0 || width > 8192 || size_t(rows) * size_t(queries) > 65535) return hipErrorInvalidValue;
    const size_t shared = size_t(2 * blocks + kThreads / 32) * sizeof(float);
    hipLaunchKernelGGL(inkling2_residual_read, dim3(rows * queries), dim3(kThreads), shared, stream, query, values, out,
                       rows, blocks, queries, width, 1.0f / sqrtf(float(width)));
    return hipGetLastError();
}

// Graph-replay tick: fill every per-visit KV header from one device counter.
// headers holds nvisits (live, position) int32 pairs; caps[i] is visit i's KV capacity.
// Reads the counter once, publishes (min(c + 1, cap_i), c) per visit, then advances it.
__global__ void inkling2_resident_tick(int* headers, const int* caps, int* counter, int nvisits) {
    const int c = *counter;
    __syncthreads();
    if (threadIdx.x == 0) *counter = c + 1;
    for (int i = threadIdx.x; i < nvisits; i += blockDim.x) {
        headers[2 * i] = min(c + 1, caps[i]);
        headers[2 * i + 1] = c;
    }
}

extern "C" int hipengine_inkling2_resident_tick(int* headers, const int* caps, int* counter, int nvisits,
                                                hipStream_t stream) {
    if (!headers || !caps || !counter || nvisits <= 0 || nvisits > 1024) return hipErrorInvalidValue;
    const int threads = nvisits < 32 ? 32 : (nvisits + 31) / 32 * 32;
    hipLaunchKernelGGL(inkling2_resident_tick, dim3(1), dim3(threads), 0, stream, headers, caps, counter, nvisits);
    return hipGetLastError();
}
