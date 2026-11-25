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

constexpr int kPerThread = 16;   // kThreads * kPerThread >= width
constexpr int kWaves = kThreads / 32;
constexpr int kRows = 8;         // templated row counts of the fused residual step

// The end of a residual step in one launch: h = a + b (when `b` is given), the normalized h is appended
// to the residual stack, and every query reads the stack with that row. Block `which` of the `queries`
// blocks per row performs the read of that query and writes h when there is one (all of them write the
// same values). The results are those of inkling2_row_add, inkling2_row_rms and
// inkling2_residual_read over the stored rows: the new row stays in registers as the FP32 values the
// stack row would hold, and every value keeps the separate kernels' reduction tree. `ROWS` is the number
// of rows read, the new row last.
template <int ROWS>
__device__ __forceinline__ void residual_update_rows(
        const float* a, const float* b, float* h, const float* stack, float* row_out, const float* query,
        float* out, int rows, int blocks, int queries, int width, float eps, float divisor) {
    __shared__ float tree[kWaves][ROWS + 1];   // per-wave sums: ROWS dot products, then the square sum
    __shared__ float total[ROWS + 1];          // block sums, then the softmax weights
    __shared__ float root_shared;
    const int index = blockIdx.x;
    const int row = index / queries;
    const int which = index % queries;
    const size_t base = size_t(row) * width;
    const float* source = a + base;
    // Unrolled so that the row stays in registers: a runtime index would send it to scratch memory.
    float kept[kPerThread];
    float squares = 0.0f;
    #pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
        const int j = threadIdx.x + m * blockDim.x;
        kept[m] = j < width ? (b ? source[j] + b[base + j] : source[j]) : 0.0f;
        if (j < width) squares += kept[m] * kept[m];
    }
    for (int shift = 16; shift; shift /= 2) squares += __shfl_down(squares, shift, 32);
    if (threadIdx.x % 32 == 0) tree[threadIdx.x / 32][ROWS] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int w = 0; w < kWaves; ++w) sum += tree[w][ROWS];
        root_shared = sum;
    }
    __syncthreads();
    const float inverse = inverse_root(root_shared, width, eps);
    if (h || row_out) {
        #pragma unroll
        for (int m = 0; m < kPerThread; ++m) {
            const int j = threadIdx.x + m * blockDim.x;
            if (j < width) {
                if (h) h[base + j] = kept[m];
                if (row_out && which == 0) row_out[base + j] = __fmul_rn(kept[m], inverse);
            }
        }
    }
    const float* q = query + size_t(which) * width;
    const float* values = stack + size_t(row) * width;
    const size_t stride = size_t(rows) * width;
    float parts[ROWS > 1 ? ROWS - 1 : 1];
    #pragma unroll
    for (int r = 0; r < ROWS - 1; ++r) parts[r] = 0.0f;
    float last = 0.0f;
    #pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
        const int j = threadIdx.x + m * blockDim.x;
        if (j < width) {
            const float value = q[j];
            #pragma unroll
            for (int r = 0; r < ROWS - 1; ++r) parts[r] += value * values[size_t(r) * stride + j];
            last += value * __fmul_rn(kept[m], inverse);
        }
    }
    #pragma unroll
    for (int shift = 16; shift; shift /= 2) {
        #pragma unroll
        for (int r = 0; r < ROWS - 1; ++r) parts[r] += __shfl_down(parts[r], shift, 32);
        last += __shfl_down(last, shift, 32);
    }
    if (threadIdx.x % 32 == 0) {
        const int wave = threadIdx.x / 32;
        #pragma unroll
        for (int r = 0; r < ROWS - 1; ++r) tree[wave][r] = parts[r];
        tree[wave][ROWS - 1] = last;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float sum = 0.0f;
            for (int w = 0; w < kWaves; ++w) sum += tree[w][r];
            total[r] = sum * divisor;
        }
        float maximum = total[0];
        #pragma unroll
        for (int r = 1; r < ROWS; ++r) maximum = fmaxf(maximum, total[r]);
        float norm = 0.0f;
        // Rolled on purpose: unrolled, each copy of expf is optimized on its own and some results
        // differ from the separate kernel by one unit.
        #pragma unroll 1
        for (int r = 0; r < ROWS; ++r) {
            const float e = expf(total[r] - maximum);
            total[r] = e;
            norm += e;
        }
        const float norm_inverse = __fdiv_rn(1.0f, norm);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) total[r] = __fmul_rn(total[r], norm_inverse);
    }
    __syncthreads();
    float* target = out + (size_t(which) * rows + row) * width;
    #pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
        const int j = threadIdx.x + m * blockDim.x;
        if (j < width) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < ROWS - 1; ++r) {
                acc += __fmul_rn(total[r], values[size_t(r) * stride + j]);
            }
            acc += __fmul_rn(total[ROWS - 1], __fmul_rn(kept[m], inverse));
            target[j] = acc;
        }
    }
}

// Up to kRows rows read: one instantiation per count, so that the per-row accumulators are registers.
template <int ROWS>
__global__ void inkling2_residual_update_rows(const float* a, const float* b, float* h, const float* stack,
                                              float* row_out, const float* query, float* out, int rows, int blocks,
                                              int queries, int width, float eps, float divisor) {
    residual_update_rows<ROWS>(a, b, h, stack, row_out, query, out, rows, blocks, queries, width, eps, divisor);
}

template <int ROWS>
bool launch_update(const dim3& grid, int rows, const float* a, const float* b, float* h, const float* stack,
                   float* row_out, const float* query, float* out, int blocks, int queries, int width, float eps,
                   hipStream_t stream) {
    if (ROWS == blocks + 1) {
        hipLaunchKernelGGL((inkling2_residual_update_rows<ROWS>), grid, dim3(kThreads), 0, stream, a, b, h, stack,
                           row_out, query, out, rows, blocks, queries, width, eps, 1.0f / sqrtf(float(width)));
        return true;
    }
    return false;
}

// out[i] = source[i] / sqrt(mean(source[i] * source[i]) + eps) * weight[i] for `count` inputs, one block per
// (input, row): the row normalization of inkling2_row_norm_mul for each input, whose `rows` rows are
// consecutive at `out` as they would be from separate launches.
__global__ void inkling2_row_norm_mul_group(const float* s0, const float* s1, const float* s2, const float* w0,
                                            const float* w1, const float* w2, float* out, int rows, int width,
                                            float eps, int count) {
    __shared__ float shared[8];
    const int index = blockIdx.x;
    const int which = index / rows;
    const int row = index - which * rows;
    const float* source = (which == 0 ? s0 : which == 1 ? s1 : s2) + size_t(row) * width;
    const float* weight = which == 0 ? w0 : which == 1 ? w1 : w2;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) sum += source[j] * source[j];
    const float inverse = inverse_root(block_sum(sum, shared), width, eps);
    float* target = out + size_t(index) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        target[j] = __fmul_rn(__fmul_rn(source[j], inverse), weight[j]);
    }
}

extern "C" int hipengine_inkling2_row_norm_mul_group(
        const float* s0, const float* s1, const float* s2, const float* w0, const float* w1, const float* w2,
        float* out, int rows, int width, float eps, int count, hipStream_t stream) {
    if (!s0 || !w0 || !out || rows <= 0 || (count != 1 && count != 3) || (count == 3 && (!s1 || !s2 || !w1 || !w2))
            || width <= 0 || width > 8192) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_row_norm_mul_group, dim3(rows * count), dim3(kThreads), 0, stream, s0, s1, s2, w0,
                       w1, w2, out, rows, width, eps, count);
    return hipGetLastError();
}

// More than kRows rows: one reduction pass per row, each with the separate kernels' tree.
__global__ void inkling2_residual_update_many(const float* a, const float* b, float* h, const float* stack,
                                              float* row_out, const float* query, float* out, int rows, int blocks,
                                              int queries, int width, float eps, float divisor) {
    extern __shared__ float shared[];          // scores, then the weights, then two reduction trees
    float* scores = shared;
    float* weights = shared + blocks + 1;
    float* tree = weights + blocks + 1;
    float* rms_tree = tree + kThreads / 32;
    const int index = blockIdx.x;
    const int row = index / queries;
    const int which = index % queries;
    const size_t base = size_t(row) * width;
    const float* source = a + base;
    float kept[kPerThread];
    float squares = 0.0f;
    #pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
        const int j = threadIdx.x + m * blockDim.x;
        kept[m] = j < width ? (b ? source[j] + b[base + j] : source[j]) : 0.0f;
        if (j < width) squares += kept[m] * kept[m];
    }
    const float inverse = inverse_root(block_sum(squares, rms_tree), width, eps);
    if (h || row_out) {
        #pragma unroll
        for (int m = 0; m < kPerThread; ++m) {
            const int j = threadIdx.x + m * blockDim.x;
            if (j < width) {
                if (h) h[base + j] = kept[m];
                if (row_out && which == 0) row_out[base + j] = __fmul_rn(kept[m], inverse);
            }
        }
    }
    const float* q = query + size_t(which) * width;
    const float* values = stack + size_t(row) * width;
    const size_t stride = size_t(rows) * width;
    for (int block = 0; block <= blocks; ++block) {
        const bool newest = block == blocks;
        const float* v = newest ? nullptr : values + size_t(block) * stride;
        float part = 0.0f;
        #pragma unroll
        for (int m = 0; m < kPerThread; ++m) {
            const int j = threadIdx.x + m * blockDim.x;
            if (j < width) part += q[j] * (newest ? __fmul_rn(kept[m], inverse) : v[j]);
        }
        // Every lane needs the total before the next block starts.
        part = block_sum(part, tree);
        if (threadIdx.x == 0) scores[block] = part * divisor;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float maximum = scores[0];
        for (int block = 1; block <= blocks; ++block) maximum = fmaxf(maximum, scores[block]);
        float norm = 0.0f;
        for (int block = 0; block <= blocks; ++block) {
            const float e = expf(scores[block] - maximum);
            weights[block] = e;
            norm += e;
        }
        const float norm_inverse = __fdiv_rn(1.0f, norm);
        for (int block = 0; block <= blocks; ++block) weights[block] = __fmul_rn(weights[block], norm_inverse);
    }
    __syncthreads();
    float* target = out + (size_t(which) * rows + row) * width;
    #pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
        const int j = threadIdx.x + m * blockDim.x;
        if (j < width) {
            float acc = 0.0f;
            for (int block = 0; block <= blocks; ++block) {
                const bool newest = block == blocks;
                const float value = newest ? __fmul_rn(kept[m], inverse) : values[size_t(block) * stride + j];
                acc += __fmul_rn(weights[block], value);
            }
            target[j] = acc;
        }
    }
}

extern "C" int hipengine_inkling2_residual_update(
        const float* a, const float* b, float* h, const float* stack, float* row_out, const float* query,
        float* out, int rows, int blocks, int queries, int width, float eps, hipStream_t stream) {
    if (!a || !stack || !row_out || !query || !out || rows <= 0 || blocks < 0 || blocks >= 64 || queries <= 0
            || queries > 8 || width <= 0 || width > kThreads * kPerThread || size_t(rows) * size_t(queries) > 65535)
        return hipErrorInvalidValue;
    // Blocks read `a` while every block writes h, so an aliased h needs the single-query form.
    if (h == a && queries != 1) return hipErrorInvalidValue;
    const dim3 grid(rows * queries);
    const float divisor = 1.0f / sqrtf(float(width));
    bool launched = false;
    switch (blocks + 1) {
        case 1: launched = launch_update<1>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 2: launched = launch_update<2>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 3: launched = launch_update<3>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 4: launched = launch_update<4>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 5: launched = launch_update<5>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 6: launched = launch_update<6>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 7: launched = launch_update<7>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        case 8: launched = launch_update<8>(grid, rows, a, b, h, stack, row_out, query, out, blocks, queries, width, eps, stream); break;
        default:
            hipLaunchKernelGGL(inkling2_residual_update_many, grid, dim3(kThreads),
                               size_t(2 * (blocks + 1) + 2 * (kThreads / 32)) * sizeof(float), stream, a, b, h, stack,
                               row_out, query, out, rows, blocks, queries, width, eps, divisor);
            launched = true;
            break;
    }
    return launched ? hipGetLastError() : hipErrorInvalidValue;
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
