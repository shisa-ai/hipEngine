// Grouped WMMA projection for raw GGUF NVFP4 expert banks: Inkling 2 Mini prompt blocks of 192 or more rows.
//
// Routes are grouped by expert ("compact" order): expert_start_compact[e] .. [e + 1] are the compact
// rows of expert e. Each group of 64 compact rows never crosses an expert. A block of eight wave32
// covers 256 output columns by the 64 compact rows of one group, four k-tiles of the rows at a
// time: the activation tile is fetched into shared memory once for the eight waves, the next
// k-tile being fetched under the compute of the current one, and each wave computes its own 32
// columns. NVFP4 sub-blocks (16 values and one E4M3 scale) are dequantized to FP16 in registers,
// with the nine words of a block read one block ahead; activations are FP16 (one value, or a high
// plus a residual part that the same weights multiply); and accumulation is FP32. The row
// fragments past an expert's rows in its last group are padding and take no product. The expert's
// outer scale is applied on store.
//
// Arithmetic differs from the FP32 tile kernels (selected_tile4_f32_f32_out), which stay the
// strict control: this is a production arithmetic candidate behind the profile gate.
//
// The same kernel reads Q5_K and Q8_0 banks, which a mixed file keeps in its last layers. Those
// weights are not exact in FP16, so each comes as a high and a residual part too, and the tile
// takes three products: a high and a low by weight high, and activation high by weight low. The
// product of the two residual parts is below FP32 rounding and is left out.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int TILE_COLS = 32;   // 16 * TM output columns per wave
constexpr int TILE_ROWS = 64;   // 16 * TN compact rows per wave and group
constexpr int TM = TILE_COLS / 16;
constexpr int TN = TILE_ROWS / 16;
enum WeightFormat : int { NVFP4_FORMAT = 0, Q5_K_FORMAT = 1, Q8_0_FORMAT = 2 };
constexpr int kBlockValues[3] = {64, 256, 32};
constexpr int kBlockBytes[3] = {36, 176, 34};

__device__ __forceinline__ float decode_scale(uint8_t code) {
    // GGUF NVFP4 uses unsigned E4M3; its integer FP4 table is doubled.
    const int exponent = (code >> 3) & 15;
    const int significand = (code & 7) + (exponent ? 8 : 0);
    const float power = __uint_as_float(static_cast<uint32_t>(116 + (exponent ? exponent : 1)) << 23);
    return code == 0x7f ? 0.0f : significand * power;
}

__device__ __forceinline__ float decode_value(unsigned code) {
    // Four-bit magnitudes in ascending code order, doubled as decode_scale expects.
    const int value = static_cast<int>((0xc8643210u >> (4 * (code & 7))) & 15);
    return static_cast<float>(code & 8 ? -value : value);
}

// Sixteen consecutive values of one weight row: byte j carries value j in its low nibble and value
// 8 + j in its high nibble, as the FP32 kernels read it. A dequantized value has at most eight
// significant bits, so it is exact in FP16.
__device__ __forceinline__ half16_t dequant16(uint32_t lo, uint32_t hi, float scale) {
    half16_t out;
    const uint32_t words[2] = {lo, hi};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned c = (words[w] >> (8 * j)) & 255u;
            out[w * 4 + j] = static_cast<half_t>(decode_value(c) * scale);
            out[8 + w * 4 + j] = static_cast<half_t>(decode_value(c >> 4) * scale);
        }
    }
    return out;
}

typedef half_t __attribute__((ext_vector_type(2))) half2_t;

// A value as an FP16 high part and the FP16 residual of that rounding.
__device__ __forceinline__ half2_t split16(float value) {
    const half_t hi = static_cast<half_t>(value);
    return half2_t{hi, static_cast<half_t>(value - static_cast<float>(hi))};
}

__device__ __forceinline__ float half_at(const uint8_t* p) {
    return static_cast<float>(*reinterpret_cast<const half_t*>(p));
}

// Sixteen bytes at an even address: four aligned words, or five when the address is two bytes past
// a word. The last two bytes of that case are loaded alone: a word there could pass the weights.
__device__ __forceinline__ uint4 load16_even(const uint8_t* p) {
    const bool shifted = (reinterpret_cast<uintptr_t>(p) & 2) != 0;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(p - 2 * shifted);
    const uint32_t w0 = words[0], w1 = words[1], w2 = words[2], w3 = words[3];
    if (!shifted) return make_uint4(w0, w1, w2, w3);
    const uint32_t w4 = static_cast<uint32_t>(p[14]) | (static_cast<uint32_t>(p[15]) << 8);
    return make_uint4(__byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432),
                      __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432));
}

// Q8_0: 32 signed bytes and one FP16 scale per block. A weight is scale times byte, exact in FP32.
__device__ __forceinline__ void dequant16_q8_0(const uint8_t* row, int64_t kt, half16_t& hi, half16_t& lo) {
    const uint8_t* block = row + (kt >> 1) * kBlockBytes[Q8_0_FORMAT];
    const float scale = half_at(block);
    const uint4 packed = load16_even(block + 2 + 16 * (kt & 1));
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int value = static_cast<int8_t>((words[w] >> (8 * j)) & 255u);
            const half2_t parts = split16(scale * static_cast<float>(value));
            hi[w * 4 + j] = parts[0];
            lo[w * 4 + j] = parts[1];
        }
    }
}

__device__ __forceinline__ uint8_t q5_k_scale(const uint8_t* scales, int sub) {
    if (sub < 4) return scales[sub] & 0x3fu;
    const int index = sub - 4;
    return (scales[8 + index] & 0x0fu) | ((scales[index] >> 2) & 0x30u);
}

__device__ __forceinline__ uint8_t q5_k_min(const uint8_t* scales, int sub) {
    if (sub < 4) return scales[4 + sub] & 0x3fu;
    const int index = sub - 4;
    return (scales[8 + index] >> 4) | ((scales[4 + index] >> 2) & 0x30u);
}

// Q5_K: 256 weights in eight sub-blocks of 32 around one FP16 scale and minimum, each sub-block
// with a 6-bit scale and minimum of its own. The low four bits of a weight are in `qs`, two
// sub-blocks to a byte; the fifth is bit `sub` of `qh[lane]`.
__device__ __forceinline__ void dequant16_q5_k(const uint8_t* row, int64_t kt, half16_t& hi, half16_t& lo) {
    const uint8_t* block = row + (kt >> 4) * kBlockBytes[Q5_K_FORMAT];
    const float d = half_at(block);
    const float dmin = half_at(block + 2);
    const int sub = (kt >> 1) & 7;
    const float scale = d * static_cast<float>(q5_k_scale(block + 4, sub));
    const float minimum = dmin * static_cast<float>(q5_k_min(block + 4, sub));
    const int tile = kt & 1;
    const uint4 packed = *reinterpret_cast<const uint4*>(block + 48 + (sub >> 1) * 32 + 16 * tile);
    const uint4 high = *reinterpret_cast<const uint4*>(block + 16 + 16 * tile);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const uint32_t highs[4] = {high.x, high.y, high.z, high.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned byte = (words[w] >> (8 * j)) & 255u;
            const unsigned low = (sub & 1) ? (byte >> 4) : (byte & 15u);
            const unsigned bit = ((highs[w] >> (8 * j)) >> sub) & 1u;
            const half2_t parts = split16(scale * static_cast<float>(low | (bit << 4)) - minimum);
            hi[w * 4 + j] = parts[0];
            lo[w * 4 + j] = parts[1];
        }
    }
}

constexpr int WAVES = 8;        // waves per block, each on its own 32 columns around one shared activation tile
constexpr int THREADS = 32 * WAVES;

// One k-tile of the block's rows into shared memory: two threads fetch the 32 bytes of one row.
template<int FORMAT>
__device__ __forceinline__ void stage_tile(const half_t* __restrict__ x, const int32_t* __restrict__ rows,
                                           int64_t kt, int64_t in_features, half_t* tile) {
    const int r = threadIdx.x >> 1, half = threadIdx.x & 1;
    if (r >= TILE_ROWS) return;
    reinterpret_cast<uint4*>(tile + r * 16)[half] =
        *reinterpret_cast<const uint4*>(x + rows[r] * in_features + kt * 16 + half * 8);
}

// One k-tile for this wave: the weights of its two column fragments, then a product per row
// fragment that is not padding. The next k-tile of the block's rows is fetched before the compute
// and stored into shared memory after it, so its memory latency passes under the compute. NVFP4
// weights are the nine words of the block that arrived ahead in `cw`; Q5_K and Q8_0 weights are
// read from memory. A wave past the output columns of the last block takes part in the fetch and
// the barrier only.
template<int PARTS, int FORMAT>
__device__ __forceinline__ void k_tile(float8_t (&acc)[TM][TN], uint32_t (&cw)[TM][9], half_t* tile_hi,
                                       half_t* tile_lo, const int32_t* __restrict__ rows,
                                       const uint8_t* __restrict__ expert_base, int64_t row_bytes, int64_t col_start,
                                       int lane16, int first_row, int64_t count, const half_t* __restrict__ x_hi,
                                       const half_t* __restrict__ x_lo, int64_t in_features, int64_t k_tiles,
                                       int64_t kt, bool fetches, bool active) {
    const int buf = static_cast<int>(kt & 1), f = static_cast<int>(kt & 3);
    const int fetch_row = threadIdx.x >> 1, fetch_half = threadIdx.x & 1;
    const bool fetch = kt + 1 < k_tiles && fetches;
    uint4 piece_hi = uint4{0, 0, 0, 0}, piece_lo = uint4{0, 0, 0, 0};
    if (fetch) {
        piece_hi = *reinterpret_cast<const uint4*>(x_hi + rows[fetch_row] * in_features + (kt + 1) * 16 + fetch_half * 8);
        if constexpr (PARTS == 2) {
            piece_lo = *reinterpret_cast<const uint4*>(x_lo + rows[fetch_row] * in_features + (kt + 1) * 16 + fetch_half * 8);
        }
    }
    if (active) {
        half16_t b_hi[TM], b_lo[TM];
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const uint8_t* row = expert_base + (col_start + tm * 16 + lane16) * row_bytes;
            if constexpr (FORMAT == NVFP4_FORMAT) {
                b_hi[tm] = dequant16(cw[tm][1 + 2 * f], cw[tm][2 + 2 * f], decode_scale((cw[tm][0] >> (8 * f)) & 255u));
            } else if constexpr (FORMAT == Q5_K_FORMAT) {
                dequant16_q5_k(row, kt, b_hi[tm], b_lo[tm]);
            } else {
                dequant16_q8_0(row, kt, b_hi[tm], b_lo[tm]);
            }
        }
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            // The row fragments past the expert's rows in its last group are padding: their
            // products are never stored, and the branch is the same for the whole block.
            if (first_row + tn * 16 >= count) continue;
            const half16_t a_hi = *reinterpret_cast<const half16_t*>(tile_hi + (buf * TILE_ROWS + tn * 16 + lane16) * 16);
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi[tm], acc[tm][tn]);
            }
            if constexpr (PARTS == 2) {
                const half16_t a_lo = *reinterpret_cast<const half16_t*>(tile_lo + (buf * TILE_ROWS + tn * 16 + lane16) * 16);
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_hi[tm], acc[tm][tn]);
                }
            }
            if constexpr (FORMAT != NVFP4_FORMAT) {
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_lo[tm], acc[tm][tn]);
                }
            }
        }
    }
    if (fetch) {
        reinterpret_cast<uint4*>(tile_hi + ((buf ^ 1) * TILE_ROWS + fetch_row) * 16)[fetch_half] = piece_hi;
        if constexpr (PARTS == 2) {
            reinterpret_cast<uint4*>(tile_lo + ((buf ^ 1) * TILE_ROWS + fetch_row) * 16)[fetch_half] = piece_lo;
        }
    }
    __syncthreads();
}

template<int PARTS, int FORMAT>
__launch_bounds__(THREADS, 2) __global__ void selected_wmma_kernel(
        const half_t* __restrict__ x_hi, const half_t* __restrict__ x_lo,
        const int64_t* __restrict__ expert_start_compact, const int32_t* __restrict__ group_expert,
        const int32_t* __restrict__ group_first_row, const int32_t* __restrict__ row_source,
        const int32_t* __restrict__ row_route, const float* __restrict__ outer,
        const uint8_t* __restrict__ qweight, float* __restrict__ out, int64_t in_features,
        int64_t out_features, const int32_t* __restrict__ group_count) {
    // Two k-tiles of the block's 64 activation rows, and the token row of each of the 64 rows.
    __shared__ half_t tile_hi[2][TILE_ROWS][16];
    __shared__ half_t tile_lo[2][TILE_ROWS][16];
    __shared__ int32_t rows[TILE_ROWS];
    // A schedule built on the device does not say how many groups it has: extra blocks of the
    // grid's fixed height leave here.
    if (group_count && static_cast<int>(blockIdx.y) >= group_count[0]) return;
    const int expert = group_expert[blockIdx.y];
    const int first_row = group_first_row[blockIdx.y];
    if (expert < 0) return;
    const int tid = threadIdx.x, wave = tid >> 5, lane16 = tid & 15, half_id = (tid >> 4) & 1;
    const bool fetches = (tid >> 1) < TILE_ROWS;
    const int64_t col_start = static_cast<int64_t>(blockIdx.x) * (TILE_COLS * WAVES) + wave * TILE_COLS;
    const bool active = col_start < out_features;
    const int64_t compact_base = expert_start_compact[expert];
    const int64_t count = expert_start_compact[expert + 1] - compact_base;
    const int64_t row_bytes = in_features / kBlockValues[FORMAT] * kBlockBytes[FORMAT];
    const uint8_t* expert_base = qweight + static_cast<int64_t>(expert) * out_features * row_bytes;
    const float scale = outer ? outer[expert] : 1.0f;
    const int64_t k_tiles = in_features / 16;

    // The token row of each tile row. Padding rows of the last group of an expert read compact
    // row 0 and are not stored.
    for (int r = tid; r < TILE_ROWS; r += THREADS) {
        const int64_t local = first_row + r;
        const int64_t compact = local < count ? compact_base + local : 0;
        rows[r] = static_cast<int32_t>(row_source ? row_source[compact] : compact);
    }
    float8_t acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) acc[tm][tn] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    // One NVFP4 block is four k-tiles: four scale bytes, then 32 bytes of codes. The nine words
    // of a block arrive one block ahead, so their memory latency passes under a block's compute.
    uint32_t cw[TM][9];
    if (FORMAT == NVFP4_FORMAT && active) {
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const uint32_t* words =
                reinterpret_cast<const uint32_t*>(expert_base + (col_start + tm * 16 + lane16) * row_bytes);
            #pragma unroll
            for (int w = 0; w < 9; ++w) cw[tm][w] = words[w];
        }
    }
    __syncthreads();
    if (fetches) {
        stage_tile<FORMAT>(x_hi, rows, 0, in_features, &tile_hi[0][0][0]);
        if constexpr (PARTS == 2) stage_tile<FORMAT>(x_lo, rows, 0, in_features, &tile_lo[0][0][0]);
    }
    __syncthreads();
    half_t* hi = &tile_hi[0][0][0];
    half_t* lo = &tile_lo[0][0][0];
    if constexpr (FORMAT == NVFP4_FORMAT) {
        // An NVFP4 input is a whole number of 64-value blocks, so four k-tiles at a time: the block
        // in the registers is consumed while the next is fetched.
        #pragma unroll 1
        for (int64_t kb = 0; kb < k_tiles / 4; ++kb) {
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int64_t kt = 4 * kb + f;
                k_tile<PARTS, FORMAT>(acc, cw, hi, lo, rows, expert_base, row_bytes, col_start, lane16,
                                      first_row, count, x_hi, x_lo, in_features, k_tiles, kt, fetches, active);
                // After the block's last k-tile, its words give way to the next block's.
                if (f == 3 && kt + 4 < k_tiles && active) {
                    #pragma unroll
                    for (int tm = 0; tm < TM; ++tm) {
                        const uint32_t* words = reinterpret_cast<const uint32_t*>(
                            expert_base + (col_start + tm * 16 + lane16) * row_bytes + ((kt + 4) >> 2) * kBlockBytes[FORMAT]);
                        #pragma unroll
                        for (int w = 0; w < 9; ++w) cw[tm][w] = words[w];
                    }
                }
            }
        }
    } else {
        #pragma unroll 1
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            k_tile<PARTS, FORMAT>(acc, cw, hi, lo, rows, expert_base, row_bytes, col_start, lane16,
                                  first_row, count, x_hi, x_lo, in_features, k_tiles, kt, fetches, active);
        }
    }
    if (!active) return;
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int64_t local = first_row + tn * 16 + half_id + j * 2;
            if (local >= count) continue;
            const int64_t compact = compact_base + local;
            const int64_t row = row_route ? row_route[compact] : compact;
            if (row < 0) continue;
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                out[row * out_features + col_start + tm * 16 + lane16] = acc[tm][tn][j] * scale;
            }
        }
    }
}

// x = hi + lo with hi = fp16(x) and lo = fp16(x - hi); lo is written only for two parts.
__global__ void fp16_parts_kernel(const float* __restrict__ x, half_t* __restrict__ hi,
                                  half_t* __restrict__ lo, int64_t count, int parts) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = x[index];
    const half_t high = static_cast<half_t>(value);
    hi[index] = high;
    if (parts == 2) lo[index] = static_cast<half_t>(value - static_cast<float>(high));
}

// The grouped schedule of a block from the per-route expert ids and mix scales of its rows: the
// expert starts over compact rows, the 64-row groups, the token row of each compact row, and the
// slot where each compact row's down output goes so that a row's top-k routes are laid out in
// expert order for the ordered mix (``mix`` is gathered in the same order). One workgroup. Routes
// of one expert land in any order among themselves (LDS atomics): a route's row is computed on its
// own, so the order inside an expert changes no result. ``total_groups`` receives the group count.
__global__ void route_schedule_kernel(const int32_t* __restrict__ ids, const float* __restrict__ scaled,
                                      int routes, int experts, int top, int64_t* __restrict__ start,
                                      int32_t* __restrict__ group_expert, int32_t* __restrict__ group_first,
                                      int32_t* __restrict__ source, int32_t* __restrict__ route,
                                      float* __restrict__ mix, int32_t* __restrict__ total_groups,
                                      int group_capacity) {
    extern __shared__ int32_t shared[];
    int32_t* counts = shared;
    int32_t* cursor = shared + experts;
    int32_t* group_base = shared + 2 * experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < routes; r += blockDim.x) atomicAdd(&counts[ids[r]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t rows = 0;
        int groups = 0;
        for (int e = 0; e < experts; ++e) {
            start[e] = rows;
            cursor[e] = static_cast<int32_t>(rows);
            rows += counts[e];
            group_base[e] = groups;
            groups += (counts[e] + TILE_ROWS - 1) / TILE_ROWS;
        }
        start[experts] = rows;
        total_groups[0] = groups <= group_capacity ? groups : -1;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const int first = group_base[e];
        const int count = (counts[e] + TILE_ROWS - 1) / TILE_ROWS;
        for (int j = 0; j < count; ++j) {
            group_expert[first + j] = e;
            group_first[first + j] = j * TILE_ROWS;
        }
    }
    for (int r = threadIdx.x; r < routes; r += blockDim.x) {
        // The slot of a route in its row: the routes with a smaller expert come first.
        const int first = r - r % top;
        int slot = 0;
        for (int other = first; other < first + top; ++other) slot += ids[other] < ids[r];
        const int32_t position = first + slot;
        mix[position] = scaled[r];
        const int compact = atomicAdd(&cursor[ids[r]], 1);
        source[compact] = r / top;
        route[compact] = position;
    }
}

template<int PARTS, int FORMAT>
int launch_wmma(const void* x_hi, const void* x_lo, const int64_t* expert_start_compact,
                const int32_t* group_expert, const int32_t* group_first_row, const int32_t* row_source,
                const int32_t* row_route, const float* outer, const uint8_t* weights, float* out, int groups,
                int in_features, int out_features, const int32_t* group_count, hipStream_t stream) {
    if (!x_hi || (PARTS == 2 && !x_lo) || !expert_start_compact || !group_expert || !group_first_row || !weights
            || !out || groups <= 0 || in_features <= 0 || in_features % kBlockValues[FORMAT] || out_features <= 0
            || out_features % TILE_COLS) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(static_cast<unsigned>((out_features + TILE_COLS * WAVES - 1) / (TILE_COLS * WAVES)),
                   static_cast<unsigned>(groups));
    hipLaunchKernelGGL((selected_wmma_kernel<PARTS, FORMAT>), grid, dim3(THREADS), 0, stream,
                       static_cast<const half_t*>(x_hi), static_cast<const half_t*>(x_lo), expert_start_compact,
                       group_expert, group_first_row, row_source, row_route, outer, weights, out,
                       static_cast<int64_t>(in_features), static_cast<int64_t>(out_features), group_count);
    return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_gguf_nvfp4_selected_wmma_f16_f32_out(
        const void* x_hi, const void* x_lo, const int64_t* expert_start_compact, const int32_t* group_expert,
        const int32_t* group_first_row, const int32_t* row_source, const int32_t* row_route, const float* outer,
        const uint8_t* weights, float* out, int groups, int in_features, int out_features,
        const int32_t* group_count, hipStream_t stream) {
    return launch_wmma<1, NVFP4_FORMAT>(x_hi, x_lo, expert_start_compact, group_expert, group_first_row, row_source, row_route,
                          outer, weights, out, groups, in_features, out_features, group_count, stream);
}

extern "C" int hipengine_gguf_nvfp4_selected_wmma_f16_parts_f32_out(
        const void* x_hi, const void* x_lo, const int64_t* expert_start_compact, const int32_t* group_expert,
        const int32_t* group_first_row, const int32_t* row_source, const int32_t* row_route, const float* outer,
        const uint8_t* weights, float* out, int groups, int in_features, int out_features,
        const int32_t* group_count, hipStream_t stream) {
    return launch_wmma<2, NVFP4_FORMAT>(x_hi, x_lo, expert_start_compact, group_expert, group_first_row, row_source, row_route,
                          outer, weights, out, groups, in_features, out_features, group_count, stream);
}

#define INKLING2_KQUANT_WMMA_ENTRY(name, parts, format)                                                             \
extern "C" int name(const void* x_hi, const void* x_lo, const int64_t* expert_start_compact,                  \
                    const int32_t* group_expert, const int32_t* group_first_row, const int32_t* row_source,   \
                    const int32_t* row_route, const float* outer, const uint8_t* weights, float* out,         \
                    int groups, int in_features, int out_features, const int32_t* group_count,                 \
                    hipStream_t stream) {                                                                    \
    return launch_wmma<parts, format>(x_hi, x_lo, expert_start_compact, group_expert, group_first_row,        \
                                      row_source, row_route, outer, weights, out, groups, in_features,        \
                                      out_features, group_count, stream);                                     \
}

INKLING2_KQUANT_WMMA_ENTRY(hipengine_gguf_q5_k_selected_wmma_f16_f32_out, 1, Q5_K_FORMAT)
INKLING2_KQUANT_WMMA_ENTRY(hipengine_gguf_q5_k_selected_wmma_f16_parts_f32_out, 2, Q5_K_FORMAT)
INKLING2_KQUANT_WMMA_ENTRY(hipengine_gguf_q8_0_selected_wmma_f16_f32_out, 1, Q8_0_FORMAT)
INKLING2_KQUANT_WMMA_ENTRY(hipengine_gguf_q8_0_selected_wmma_f16_parts_f32_out, 2, Q8_0_FORMAT)

extern "C" int hipengine_nvfp4_route_schedule(
        const int32_t* ids, const float* scaled, int routes, int experts, int top, int64_t* start,
        int32_t* group_expert, int32_t* group_first, int32_t* source, int32_t* route, float* mix,
        int32_t* total_groups, int group_capacity, hipStream_t stream) {
    if (!ids || !scaled || !start || !group_expert || !group_first || !source || !route || !mix
            || !total_groups || routes <= 0 || experts <= 0 || top <= 0 || group_capacity <= 0) {
        return hipErrorInvalidValue;
    }
    const size_t shared = static_cast<size_t>(3 * experts) * sizeof(int32_t);
    hipLaunchKernelGGL(route_schedule_kernel, dim3(1), dim3(256), shared, stream, ids, scaled, routes, experts, top,
                       start, group_expert, group_first, source, route, mix, total_groups, group_capacity);
    return hipGetLastError();
}

extern "C" int hipengine_fp16_parts(const float* x, void* hi, void* lo, int64_t count, int parts, hipStream_t stream) {
    if (!x || !hi || (parts == 2 && !lo) || (parts != 1 && parts != 2) || count < 0) {
        return hipErrorInvalidValue;
    }
    if (count == 0) return hipSuccess;
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    hipLaunchKernelGGL(fp16_parts_kernel, dim3(blocks), dim3(threads), 0, stream, x, static_cast<half_t*>(hi),
                       static_cast<half_t*>(lo), count, parts);
    return hipGetLastError();
}
