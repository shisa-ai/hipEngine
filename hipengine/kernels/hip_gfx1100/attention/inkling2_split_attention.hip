#include <hip/hip_runtime.h>
#include <stdint.h>

// Each wave computes one score with the same 128-lane tree as the tiled kernel.
__global__ void split_scores(const float* q, const float* k, const float* bias, float* scores,
                             const int32_t* offsets, const int32_t* count, const int32_t* positions,
                             const uint8_t* evicted, const int32_t* row, int nq, int nk, int dim,
                             int extent, int stride) {
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + threadIdx.x / 32;
    const int head = blockIdx.y;
    if (i >= *count) return;
    const int lag = *row - positions[i];
    if (evicted[i] || lag < 0) return;
    const int kh = head / (nq / nk);
    const float* key = k + (size_t(offsets[i]) * nk + kh) * dim;
    q += head * dim;
    float part[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int col = lane + 32 * g;
        part[g] = col < dim ? __fmul_rn(q[col], key[col]) : 0.0f;
    }
    float sum = (part[0] + part[2]) + (part[1] + part[3]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
    if (!lane) scores[size_t(head) * stride + i] = sum * rsqrtf(float(dim)) +
        (lag < extent ? bias[head * extent + lag] : 0.0f);
}

__global__ void split_values(const float* value, const float* scores, const float* sink, float* out,
                             const int32_t* offsets, const int32_t* count, const int32_t* positions,
                             const uint8_t* evicted, const int32_t* row, const float* gain,
                             int nq, int nk, int dim, int stride) {
    constexpr int tile = 32;
    const int tid = threadIdx.x, head = blockIdx.x;
    const int kh = head / (nq / nk);
    const int live = *count, position = *row;
    scores += size_t(head) * stride;
    __shared__ float previous[tile], current[tile], final_denom;
    __shared__ int physical[tile];
    float hi = sink[head], denom = 1.0f, acc = 0.0f;
    for (int start = 0; start < live; start += tile) {
        const int n = min(tile, live - start);
        if (tid < tile) {
            const int i = start + tid;
            const bool active = tid < n && !evicted[i] && position >= positions[i];
            const float score = active ? scores[i] : -INFINITY;
            const float tile_hi = __shfl(hi, 0, tile);
            float prefix = fmaxf(tile_hi, score);
            // Only the maximum is scanned. The denominator and value sums
            // below still consume the exact online factors in slot order.
            #pragma unroll
            for (int delta = 1; delta < tile; delta *= 2) {
                const float earlier = __shfl_up(prefix, delta, tile);
                if (tid >= delta) prefix = fmaxf(earlier, prefix);
            }
            const float prior = __shfl_up(prefix, 1, tile);
            const float last = __shfl(prefix, n - 1, tile);
            if (active) {
                previous[tid] = expf((tid == 0 ? tile_hi : prior) - prefix);
                current[tid] = expf(score - prefix);
            }
            physical[tid] = active ? offsets[i] : -1;
            if (!tid) hi = last;
        }
        __syncthreads();
        if (!tid) {
            for (int j = 0; j < n; ++j) {
                if (physical[j] >= 0) denom = denom * previous[j] + current[j];
            }
        }
        if (tid < dim) {
            for (int j = 0; j < n; ++j) {
                if (physical[j] < 0) continue;
                const size_t base = (size_t(physical[j]) * nk + kh) * dim;
                acc = acc * previous[j] + current[j] * value[base + tid];
            }
        }
        __syncthreads();
    }
    if (!tid) final_denom = denom;
    __syncthreads();
    if (tid < dim) {
        const int i = head * dim + tid;
        const float result = acc / final_denom;
        out[i] = gain ? __fmul_rn(result, gain[i]) : result;
    }
}

namespace {
constexpr int value_chunk = 64;

__global__ void split_chunk_values(const float* value, const float* scores, float* partial,
                                   const int32_t* offsets, const int32_t* count, const int32_t* positions,
                                   const uint8_t* evicted, const int32_t* row, int nq, int nk, int dim,
                                   int stride, int splits) {
    const int tid = threadIdx.x, head = blockIdx.y;
    const int kh = head / (nq / nk);
    const int start = blockIdx.x * value_chunk;
    __shared__ float reduced[value_chunk], weights[value_chunk];
    __shared__ int physical[value_chunk];
    float score = -INFINITY;
    if (tid < value_chunk) {
        const int slot = start + tid;
        const bool active = slot < *count && !evicted[slot] && *row >= positions[slot];
        physical[tid] = active ? offsets[slot] : -1;
        score = active ? scores[size_t(head) * stride + slot] : -INFINITY;
        reduced[tid] = score;
    }
    __syncthreads();
    for (int half = value_chunk / 2; half; half /= 2) {
        if (tid < half) reduced[tid] = fmaxf(reduced[tid], reduced[tid + half]);
        __syncthreads();
    }
    const float high = reduced[0];
    __syncthreads();
    if (tid < value_chunk) {
        const float w = physical[tid] >= 0 ? expf(score - high) : 0.0f;
        weights[tid] = reduced[tid] = w;
    }
    __syncthreads();
    float* dst = partial + (size_t(head) * splits + blockIdx.x) * (dim + 2);
    if (tid < dim) {
        float acc = 0.0f;
        for (int j = 0; j < value_chunk; ++j) {
            const int slot = physical[j];
            if (slot >= 0) acc = fmaf(weights[j], value[(size_t(slot) * nk + kh) * dim + tid], acc);
        }
        dst[tid] = acc;
    }
    for (int half = value_chunk / 2; half; half /= 2) {
        if (tid < half) reduced[tid] += reduced[tid + half];
        __syncthreads();
    }
    if (!tid) {
        dst[dim] = high;
        dst[dim + 1] = reduced[0];
    }
}

__global__ void split_merge_values(const float* partial, const float* sink, float* out, const float* gain,
                                   int dim, int splits) {
    const int head = blockIdx.x, tid = threadIdx.x;
    const int stride = dim + 2;
    partial += size_t(head) * splits * stride;
    __shared__ float scalar, scales[32];
    if (!tid) {
        float high = sink[head];
        for (int i = 0; i < splits; ++i) high = fmaxf(high, partial[size_t(i) * stride + dim]);
        scalar = high;
    }
    __syncthreads();
    const float high = scalar;
    float sum = expf(sink[head] - high), acc = 0.0f;
    for (int base = 0; base < splits; base += 32) {
        const int n = min(32, splits - base);
        if (tid < n) scales[tid] = expf(partial[size_t(base + tid) * stride + dim] - high);
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            const float* item = partial + size_t(base + j) * stride;
            if (tid < dim) acc = fmaf(scales[j], item[tid], acc);
            if (!tid) sum = fmaf(scales[j], item[dim + 1], sum);
        }
        __syncthreads();
    }
    if (!tid) scalar = sum;
    __syncthreads();
    if (tid < dim) {
        const int i = head * dim + tid;
        const float result = acc / scalar;
        out[i] = gain ? __fmul_rn(result, gain[i]) : result;
    }
}
}  // namespace

extern "C" int hipengine_inkling2_split_chunked_attention(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, float* scores, const float* gain, float* partial,
    int nq, int nk, int dim, int extent, int live) {
    if (live <= 0 || nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || !partial) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(split_scores, dim3(1 + (live - 1) / 4, nq), dim3(128), 0, 0,
                       q, k, bias, scores, offsets, count, positions, evicted, row, nq, nk, dim, extent, live);
    if (auto status = hipGetLastError()) return status;
    const int splits = 1 + (live - 1) / value_chunk;
    hipLaunchKernelGGL(split_chunk_values, dim3(splits, nq), dim3(128), 0, 0,
                       v, scores, partial, offsets, count, positions, evicted, row, nq, nk, dim, live, splits);
    if (auto status = hipGetLastError()) return status;
    hipLaunchKernelGGL(split_merge_values, dim3(nq), dim3(128), 0, 0, partial, sink, out, gain, dim, splits);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_split_attention(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, float* scores, const float* gain, int nq, int nk, int dim, int extent, int live) {
    if (live <= 0 || nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128) return hipErrorInvalidValue;
    hipLaunchKernelGGL(split_scores, dim3(1 + (live - 1) / 4, nq), dim3(128), 0, 0,
                       q, k, bias, scores, offsets, count, positions, evicted, row, nq, nk, dim, extent, live);
    int status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(split_values, dim3(nq), dim3(128), 0, 0,
                       v, scores, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim, live);
    return hipGetLastError();
}
