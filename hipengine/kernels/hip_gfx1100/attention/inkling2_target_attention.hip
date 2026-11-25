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
    const int32_t* row, int capacity, int width, hipStream_t stream) {
    if (capacity <= 0 || width <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_cache_write, dim3(1 + (width - 1) / 256), dim3(256), 0, stream,
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

template <bool Block, bool ApplySigmoid = false, bool TableBias = false>
__global__ void inkling2_target_attention_tiled(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* gain, int nq, int nk, int dim, int extent,
    const float* block_k, const float* block_v, int capacity, const float* table, int rel_width) {
    constexpr int tile = 16;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int head = blockIdx.x;
    const int kh = head / (nq / nk);
    const int position = *row + (Block ? blockIdx.y : 0);
    const int live = Block ? min(*count, position + 1) : *count;
    if constexpr (Block) {
        q += size_t(blockIdx.y) * nq * dim;
        if (gain) gain += size_t(blockIdx.y) * nq * dim;
        bias += size_t(blockIdx.y) * nq * (TableBias ? rel_width : extent);
        sink += size_t(blockIdx.y) * nq;
        out += size_t(blockIdx.y) * nq * dim;
    }
    q += head * dim;
    if constexpr (TableBias) {
        // Each head carries one row of `rel_width` scores; the lag table is stored lag by lag.
        bias += head * rel_width;
    } else {
        bias += head * extent;
    }
    const float rel = TableBias && lane < rel_width ? bias[lane] : 0.0f;
    __shared__ float score[tile], previous[tile], current[tile];
    __shared__ int physical[tile];
    __shared__ float final_denom;
    float hi = sink[head], denom = 1.0f, acc = 0.0f;
    for (int start = 0; start < live; start += tile) {
        const int n = min(tile, live - start);
        for (int j = wave; j < n; j += 4) {
            const int i = start + j;
            int staged = -1;
            if constexpr (Block) {
                const int newest = position % capacity;
                const int lag = i <= newest ? newest - i : newest + capacity - i;
                staged = position - lag - *row;
            }
            const int lag = staged >= 0 ? position - *row - staged : position - positions[i];
            const bool active = !evicted[i] && lag >= 0 && (!Block || lag < capacity);
            if (lane == 0) physical[j] = active ? (staged >= 0 ? capacity + staged : offsets[i]) : -1;
            if (!active) continue;
            const float* key = staged >= 0 ? block_k + (size_t(staged) * nk + kh) * dim
                                           : k + (size_t(offsets[i]) * nk + kh) * dim;
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
            // The table dot is a wave reduction too, so every lane must reach it.
            float shift = 0.0f;
            if (lag < extent) {
                if constexpr (TableBias) {
                    float part = lane < rel_width ? __fmul_rn(rel, table[lag * rel_width + lane]) : 0.0f;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) part += __shfl_down(part, offset, 32);
                    shift = part;
                } else {
                    shift = bias[lag];
                }
            }
            if (lane == 0) score[j] = sum * rsqrtf(float(dim)) + shift;
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
                const bool staged = Block && physical[j] >= capacity;
                const int offset = staged ? physical[j] - capacity : physical[j];
                const size_t base = (size_t(offset) * nk + kh) * dim;
                acc = acc * previous[j] + current[j] * (staged ? block_v : v)[base + tid];
            }
        }
        __syncthreads();
    }
    if (tid == 0) final_denom = denom;
    __syncthreads();
    if (tid < dim) {
        const int i = head * dim + tid;
        const float value = acc / final_denom;
        float g = gain ? gain[i] : 1.0f;
        if constexpr (ApplySigmoid) {
            if (gain) {
                const float z = expf(-fabsf(g));
                g = (g >= 0.0f ? 1.0f : z) / (1.0f + z);
            }
        }
        out[i] = gain ? __fmul_rn(value, g) : value;
    }
}

extern "C" int hipengine_inkling2_target_attention_tiled(
    const float* q, const float* k, const float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* gain, int nq, int nk, int dim, int extent, hipStream_t stream) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_attention_tiled<false>, dim3(nq), dim3(128), 0, stream,
                       q, k, v, bias, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim, extent,
                       nullptr, nullptr, 0, nullptr, 0);
    return hipGetLastError();
}

__global__ void inkling2_target_cache_commit_block(
    const float* key, const float* value, float* keys, float* values,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, int capacity, int width, int first) {
    const int i = first + blockIdx.y;
    const int position = *row + i;
    const int slot = position % capacity;
    if (slot >= *count || evicted[slot]) return;
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch < width) {
        const size_t src = size_t(i) * width + ch;
        const size_t dst = size_t(offsets[slot]) * width + ch;
        keys[dst] = key[src];
        values[dst] = value[src];
    }
    if (ch == 0) positions[slot] = position;
}

// Block attention with one 32-lane group per (row, KV head), for 128-wide heads.
//
// `inkling2_target_attention_tiled` gives one 128-thread block to each (query head, row) and
// walks the live context 16 keys at a time with three barriers per step. Here a group serves one
// row and one KV head with its GROUPS query heads. Per pass, four groups of eight lanes each take
// one key: a lane holds sixteen columns of that key and of the query heads, the running softmax
// stays in registers, and the group reads each key and value row once. Scores, the softmax over
// passes of four keys and the value sums follow another order than the tiled kernel's, which
// stays the control.
namespace {
constexpr int kWavesPerBlock = 8;
constexpr int kKeysPerPass = 4;
constexpr int kLanesPerKey = 8;      // kKeysPerPass * kLanesPerKey is the wave's lane count
constexpr int kColumnsPerLane = 16;   // kLanesPerKey * kColumnsPerLane is the head width

__device__ __forceinline__ float accumulate4(float sum, float4 a, float4 b) {
    return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, sum))));
}

__device__ __forceinline__ float4 shuffle4(float4 v, int lane_mask) {
    return make_float4(__shfl_xor(v.x, lane_mask, 32), __shfl_xor(v.y, lane_mask, 32),
                       __shfl_xor(v.z, lane_mask, 32), __shfl_xor(v.w, lane_mask, 32));
}

// The sum over the four keys of a pass: every lane then holds it.
__device__ __forceinline__ float4 sum_over_keys(float4 v) {
    const float4 a = shuffle4(v, 8), b = shuffle4(v, 16), c = shuffle4(v, 24);
    return make_float4((v.x + a.x) + (b.x + c.x), (v.y + a.y) + (b.y + c.y),
                       (v.z + a.z) + (b.z + c.z), (v.w + a.w) + (b.w + c.w));
}
}  // namespace

// Eight wave groups per block: the bound lets each lane hold its query and value columns in registers.
template <int GROUPS>
__global__ __launch_bounds__(kWavesPerBlock * 32) void inkling2_target_attention_waves(
        const float* q, const float* k, const float* v, const float* rel, const float* sink, float* out,
        const int32_t* offsets, const int32_t* count, const int32_t* positions, const uint8_t* evicted,
        const int32_t* row, const float* gate, int nq, int nk, int extent, const float* block_k,
        const float* block_v, int capacity, const float* table, int rel_width, int rows) {
    const int lane = threadIdx.x & 31;
    const int key_lane = lane / kLanesPerKey;   // the key this lane serves in a pass
    const int chunk = lane % kLanesPerKey;      // its sixteen columns
    const int index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int kh = index % nk, r = index / nk;
    if (r >= rows) return;
    const int base = *row;
    const int position = base + r;
    const int live = min(*count, position + 1);
    const int newest = position % capacity;
    const int width = nk * kLanesPerKey * kColumnsPerLane;
    const size_t heads = size_t(r) * nq + size_t(kh) * GROUPS;
    float4 query[GROUPS][4], value_sum[GROUPS][4], relative[GROUPS];
    float highest[GROUPS], denom[GROUPS];
    #pragma unroll
    for (int h = 0; h < GROUPS; ++h) {
        const float* qh = q + (heads + h) * kLanesPerKey * kColumnsPerLane + chunk * kColumnsPerLane;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            query[h][c] = *reinterpret_cast<const float4*>(qh + 4 * c);
            value_sum[h][c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        relative[h] = chunk * 4 < rel_width
            ? *reinterpret_cast<const float4*>(rel + (heads + h) * rel_width + chunk * 4)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        highest[h] = sink[heads + h];
        denom[h] = 1.0f;
    }
    for (int start = 0; start < live; start += kKeysPerPass) {
        const int slot = start + key_lane;
        const float *key = nullptr, *value = nullptr;
        int lag = 0;
        if (slot < live) {
            const int distance = slot <= newest ? newest - slot : newest + capacity - slot;
            const int staged = position - distance - base;
            lag = staged >= 0 ? position - base - staged : position - positions[slot];
            if (!evicted[slot] && lag >= 0 && lag < capacity) {
                const size_t offset = staged >= 0 ? size_t(staged) : size_t(offsets[slot]);
                key = (staged >= 0 ? block_k : k) + offset * width + kh * kLanesPerKey * kColumnsPerLane
                    + chunk * kColumnsPerLane;
                value = (staged >= 0 ? block_v : v) + offset * width + kh * kLanesPerKey * kColumnsPerLane
                    + chunk * kColumnsPerLane;
            }
        }
        float score[GROUPS], weight[GROUPS], rescale[GROUPS];
        #pragma unroll
        for (int h = 0; h < GROUPS; ++h) score[h] = 0.0f;
        if (key) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 kk = *reinterpret_cast<const float4*>(key + 4 * c);
                #pragma unroll
                for (int h = 0; h < GROUPS; ++h) score[h] = accumulate4(score[h], query[h][c], kk);
            }
            // The dot product is scaled; the table bias is not.
            #pragma unroll
            for (int h = 0; h < GROUPS; ++h) score[h] = score[h] * rsqrtf(float(kLanesPerKey * kColumnsPerLane));
            if (lag < extent && chunk * 4 < rel_width) {
                const float4 tt = *reinterpret_cast<const float4*>(table + lag * rel_width + chunk * 4);
                #pragma unroll
                for (int h = 0; h < GROUPS; ++h) score[h] = accumulate4(score[h], relative[h], tt);
            }
        }
        // The eight chunks of this lane's key sum; its eight lanes then hold the same score.
        #pragma unroll
        for (int h = 0; h < GROUPS; ++h) {
            score[h] += __shfl_xor(score[h], 4, 32);
            score[h] += __shfl_xor(score[h], 2, 32);
            score[h] += __shfl_xor(score[h], 1, 32);
            if (!key) score[h] = -INFINITY;
        }
        #pragma unroll
        for (int h = 0; h < GROUPS; ++h) {
            const float a = __shfl_xor(score[h], 8, 32), b = __shfl_xor(score[h], 16, 32),
                        c = __shfl_xor(score[h], 24, 32);
            const float next = fmaxf(fmaxf(highest[h], score[h]), fmaxf(fmaxf(a, b), c));
            rescale[h] = expf(highest[h] - next);
            weight[h] = expf(score[h] - next);
            highest[h] = next;
            #pragma unroll
            for (int c = 0; c < 4; ++c) value_sum[h][c] = value_sum[h][c] * rescale[h];
        }
        if (value) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float4 vv = *reinterpret_cast<const float4*>(value + 4 * c);
                #pragma unroll
                for (int h = 0; h < GROUPS; ++h) {
                    value_sum[h][c].x = fmaf(weight[h], vv.x, value_sum[h][c].x);
                    value_sum[h][c].y = fmaf(weight[h], vv.y, value_sum[h][c].y);
                    value_sum[h][c].z = fmaf(weight[h], vv.z, value_sum[h][c].z);
                    value_sum[h][c].w = fmaf(weight[h], vv.w, value_sum[h][c].w);
                }
            }
        }
        #pragma unroll
        for (int h = 0; h < GROUPS; ++h) {
            // Every lane holds its key's weight; the four of a pass add up along with the sink row.
            float sum = weight[h] + __shfl_xor(weight[h], 8, 32);
            sum += __shfl_xor(sum, 16, 32);
            denom[h] = fmaf(denom[h], rescale[h], sum);
        }
    }
    // Every lane runs the sum: a shuffle reads nothing useful from a lane that skips it.
    float4 total[GROUPS][4];
    #pragma unroll
    for (int h = 0; h < GROUPS; ++h) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) total[h][c] = sum_over_keys(value_sum[h][c]);
    }
    if (key_lane == 0) {
        #pragma unroll
        for (int h = 0; h < GROUPS; ++h) {
            const float* g = gate ? gate + (heads + h) * kLanesPerKey * kColumnsPerLane + chunk * kColumnsPerLane
                                  : nullptr;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float values[4];
                values[0] = total[h][c].x / denom[h];
                values[1] = total[h][c].y / denom[h];
                values[2] = total[h][c].z / denom[h];
                values[3] = total[h][c].w / denom[h];
                float applied[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                if (gate) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float z = expf(-fabsf(g[4 * c + e]));
                        applied[e] = (g[4 * c + e] >= 0.0f ? 1.0f : z) / (1.0f + z);
                    }
                }
                float* o = out + (heads + h) * kLanesPerKey * kColumnsPerLane + chunk * kColumnsPerLane + 4 * c;
                #pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = gate ? __fmul_rn(values[e], applied[e]) : values[e];
            }
        }
    }
}

template <bool ApplySigmoid, bool TableBias>
static int launch_attention_block(
    const float* q, float* k, float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* block_k, const float* block_v, const float* gain,
    int nq, int nk, int dim, int extent, int rows, int capacity, const float* table, int rel_width,
    hipStream_t stream) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0 ||
        rows <= 0 || rows > 65535 || capacity <= 0) return hipErrorInvalidValue;
    if constexpr (TableBias) {
        if (!table || rel_width <= 0 || rel_width > 32) return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL((inkling2_target_attention_tiled<true, ApplySigmoid, TableBias>), dim3(nq, rows), dim3(128),
                       0, stream, q, k, v, bias, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim,
                       extent, block_k, block_v, capacity, table, rel_width);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    const int width = nk * dim;
    const int committed = rows < capacity ? rows : capacity;
    hipLaunchKernelGGL(inkling2_target_cache_commit_block, dim3(1 + (width - 1) / 256, committed), dim3(256), 0, stream,
                       block_k, block_v, k, v, offsets, count, positions, evicted, row, capacity, width,
                       rows - committed);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_target_attention_block(
    const float* q, float* k, float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* block_k, const float* block_v, const float* gain,
    int nq, int nk, int dim, int extent, int rows, int capacity, hipStream_t stream) {
    return launch_attention_block<false, false>(q, k, v, bias, sink, out, offsets, count, positions, evicted,
                                               row, block_k, block_v, gain, nq, nk, dim, extent, rows, capacity,
                                               nullptr, 0, stream);
}

extern "C" int hipengine_inkling2_target_attention_block_sigmoid(
    const float* q, float* k, float* v, const float* bias, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* block_k, const float* block_v, const float* gate,
    int nq, int nk, int dim, int extent, int rows, int capacity, hipStream_t stream) {
    return launch_attention_block<true, false>(q, k, v, bias, sink, out, offsets, count, positions, evicted,
                                            row, block_k, block_v, gate, nq, nk, dim, extent, rows, capacity,
                                            nullptr, 0, stream);
}

// Relative bias comes from one score row per (query, head) and a lag table instead of a dense
// query-by-history tensor: bias[row, head, lag] = sum_j rel[row, head, j] * table[lag, j].
extern "C" int hipengine_inkling2_target_attention_block_table_sigmoid(
    const float* q, float* k, float* v, const float* rel, const float* sink, float* out,
    const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
    const int32_t* row, const float* block_k, const float* block_v, const float* gate, const float* table,
    int nq, int nk, int dim, int extent, int rows, int capacity, int rel_width, hipStream_t stream) {
    return launch_attention_block<true, true>(q, k, v, rel, sink, out, offsets, count, positions, evicted,
                                              row, block_k, block_v, gate, nq, nk, dim, extent, rows, capacity,
                                              table, rel_width, stream);
}

// The same inputs and results up to rounding, through one 32-lane group per (row, KV head).
// Serves up to four query heads per KV head, 128-wide heads and a relative width that is a
// multiple of four up to 32; other shapes are the caller's to send to the tiled kernel.
#define INKLING2_WAVES_ARGS                                                                                       \
    q, k, v, rel, sink, out, offsets, count, positions, evicted, row, gate, nq, nk, extent, block_k, block_v, \
        capacity, table, rel_width, rows
extern "C" int hipengine_inkling2_target_attention_block_waves(
        const float* q, float* k, float* v, const float* rel, const float* sink, float* out,
        const int32_t* offsets, const int32_t* count, int32_t* positions, const uint8_t* evicted,
        const int32_t* row, const float* block_k, const float* block_v, const float* gate, const float* table,
        int nq, int nk, int dim, int extent, int rows, int capacity, int rel_width, hipStream_t stream) {
    if (nq <= 0 || nk <= 0 || nq % nk || nq / nk > 4 || dim != kLanesPerKey * kColumnsPerLane ||
        extent <= 0 || rows <= 0 || rows > 65535 || capacity <= 0 || !table || rel_width <= 0 || rel_width > 32 ||
        rel_width % 4) return hipErrorInvalidValue;
    const dim3 grid(1 + (rows * nk - 1) / kWavesPerBlock), block(kWavesPerBlock * 32);
    switch (nq / nk) {
        case 1: hipLaunchKernelGGL((inkling2_target_attention_waves<1>), grid, block, 0, stream, INKLING2_WAVES_ARGS); break;
        case 2: hipLaunchKernelGGL((inkling2_target_attention_waves<2>), grid, block, 0, stream, INKLING2_WAVES_ARGS); break;
        case 3: hipLaunchKernelGGL((inkling2_target_attention_waves<3>), grid, block, 0, stream, INKLING2_WAVES_ARGS); break;
        default: hipLaunchKernelGGL((inkling2_target_attention_waves<4>), grid, block, 0, stream, INKLING2_WAVES_ARGS); break;
    }
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    const int width = nk * dim;
    const int committed = rows < capacity ? rows : capacity;
    hipLaunchKernelGGL(inkling2_target_cache_commit_block, dim3(1 + (width - 1) / 256, committed), dim3(256), 0, stream,
                       block_k, block_v, k, v, offsets, count, positions, evicted, row, capacity, width,
                       rows - committed);
    return hipGetLastError();
}

extern "C" int hipengine_inkling2_target_attention(const float* q, const float* k, const float* v,
                                                   const float* bias, const float* sink, float* out,
                                                   const int32_t* offsets, const int32_t* count,
                                                   const int32_t* positions, const uint8_t* evicted,
                                                   const int32_t* row, const float* gain, int nq, int nk, int dim, int extent,
                                                   hipStream_t stream) {
    if (nq <= 0 || nk <= 0 || nq % nk || dim <= 0 || dim > 128 || extent <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inkling2_target_attention, dim3(nq), dim3(128), 0, stream,
                       q, k, v, bias, sink, out, offsets, count, positions, evicted, row, gain, nq, nk, dim, extent);
    return hipGetLastError();
}
