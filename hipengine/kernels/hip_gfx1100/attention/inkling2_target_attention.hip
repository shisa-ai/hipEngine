#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void inkling2_target_cache_write(
    const float* key, const float* value, float* keys, float* values,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, int capacity, int width) {
    const int slot = *row % capacity;
    if (slot >= *count || evicted[slot]) return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t dst = size_t(offsets[slot]) * width + idx;
        keys[dst] = key[idx];
        values[dst] = value[idx];
    }
    if (idx == 0) positions[slot] = *row;
}

extern "C" int hipengine_inkling2_target_cache_write(
    const float* key, const float* value, float* keys, float* values,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, int capacity, int width) {
    if (capacity <= 0 || width <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_cache_write, dim3(1 + (width - 1) / 256), dim3(256), 0, 0,
                       key, value, keys, values, offsets, count, positions, evicted, row, capacity, width);
    return hipGetLastError();
}

__global__ void inkling2_target_attention(const float* q, const float* k, const float* v,
                                           const float* bias, const float* sink, float* out,
                                           const int32_t* offsets, const int32_t* count,
                                           const int32_t* positions, const uint8_t* evicted,
                                           const int32_t* row, const float* gain, int nq, int nk, int dim, int extent) {
    constexpr int tile = 128;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int kh = head / (nq / nk);
    const float query = lane < dim ? q[head * dim + lane] : 0.0f;
    __shared__ float reduce[tile];
    __shared__ float factors[2];
    float acc = 0.0f;
    float max_score = sink[head];
    float denom = 1.0f;
    for (int i = 0; i < *count; ++i) {
        const int lag = *row - positions[i];
        if (evicted[i] || lag < 0) continue;
        const size_t base = (size_t(offsets[i]) * nk + kh) * dim;
        reduce[lane] = lane < dim ? query * k[base + lane] : 0.0f;
        __syncthreads();
        for (int stride = tile / 2; stride > 0; stride /= 2) {
            if (lane < stride) reduce[lane] += reduce[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = reduce[0] * rsqrtf(float(dim)) + (lag < extent ? bias[head * extent + lag] : 0.0f);
            const float next = fmaxf(max_score, score);
            factors[0] = expf(max_score - next);
            factors[1] = expf(score - next);
            denom = denom * factors[0] + factors[1];
            max_score = next;
        }
        __syncthreads();
        if (lane < dim) acc = acc * factors[0] + factors[1] * v[base + lane];
        __syncthreads();
    }
    if (lane == 0) factors[0] = denom;
    __syncthreads();
    if (lane < dim) {
        const int i = head * dim + lane;
        const float value = acc / factors[0];
        out[i] = gain ? __fmul_rn(value, gain[i]) : value;
    }
}

__global__ void inkling2_target_attention_tiled(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* gain, int nq, int nk, int dim, int extent) {
    constexpr int tile = 16;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int head = blockIdx.x;
    const int kh = head / (nq / nk);
    const int live = *count, position = *row;
    q += head * dim;
    bias += head * extent;
    __shared__ float score[tile], previous[tile], current[tile];
    __shared__ int physical[tile];
    __shared__ float final_denom;
    float hi = sink[head], denom = 1.0f, acc = 0.0f;
    for (int start = 0; start < live; start += tile) {
        const int n = min(tile, live - start);
        for (int j = wave; j < n; j += 4) {
            const int i = start + j;
            const int lag = position - positions[i];
            const bool active = !evicted[i] && lag >= 0;
            if (lane == 0) physical[j] = active ? offsets[i] : -1;
            if (!active) continue;
            const float* key = k + (size_t(offsets[i]) * nk + kh) * dim;
            float part[4];
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int col = lane + 32 * g;
                part[g] = col < dim ? __fmul_rn(q[col], key[col]) : 0.0f;
            }
            // Match the strict 128-way tree: combine +64, then +32, then the wave.
            float sum = (part[0] + part[2]) + (part[1] + part[3]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down(sum, offset, 32);
            if (lane == 0) score[j] = sum * rsqrtf(float(dim)) + (lag < extent ? bias[lag] : 0.0f);
        }
        __syncthreads();
        if (tid == 0) {
            for (int j = 0; j < n; ++j) {
                if (physical[j] < 0) continue;
                const float next = fmaxf(hi, score[j]);
                previous[j] = expf(hi - next);
                current[j] = expf(score[j] - next);
                denom = denom * previous[j] + current[j];
                hi = next;
            }
        }
        __syncthreads();
        if (tid < dim) {
            for (int j = 0; j < n; ++j) {
                if (physical[j] < 0) continue;
                const size_t base = (size_t(physical[j]) * nk + kh) * dim;
                acc = acc * previous[j] + current[j] * v[base + tid];
            }
        }
        __syncthreads();
    }
    if (tid == 0) final_denom = denom;
    __syncthreads();
    if (tid < dim) {
        const int i = head * dim + tid;
        const float value = acc / final_denom;
        out[i] = gain ? __fmul_rn(value, gain[i]) : value;
    }
}

extern "C" int hipengine_inkling2_target_attention_tiled(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* gain, int nq, int nk, int dim, int extent) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_attention_tiled, dim3(nq), dim3(128), 0, 0,
                       q, k, v, bias, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim, extent);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_target_attention(const float* q, const float* k, const float* v,
                                                   const float* bias, const float* sink, float* out,
                                                   const int32_t* offsets, const int32_t* count,
                                                   const int32_t* positions, const uint8_t* evicted,
                                                   const int32_t* row, const float* gain, int nq, int nk, int dim, int extent) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_attention, dim3(nq), dim3(128), 0, 0,
                       q, k, v, bias, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim, extent);
    return hipGetLastError();
}
