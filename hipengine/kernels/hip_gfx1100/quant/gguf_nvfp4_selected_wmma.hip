// Grouped WMMA projection for raw GGUF NVFP4 expert banks: Inkling 2 Mini prompt blocks of 192 or more rows.
//
// Routes are grouped by expert ("compact" order): expert_start_compact[e] .. [e + 1] are the compact
// rows of expert e. Each group of 64 compact rows never crosses an expert. One wave32 computes
// 32 output columns by 64 compact rows of one group: NVFP4 sub-blocks (16 values and one E4M3
// scale) are dequantized to FP16 in registers, activations are FP16 (one value, or a high plus a
// residual part that the same weights multiply), and accumulation is FP32. The expert's outer
// scale is applied on store.
//
// Arithmetic differs from the FP32 tile kernels (selected_tile4_f32_f32_out), which stay the
// strict control: this is a production arithmetic candidate behind the profile gate.
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
constexpr int BLOCK_VALUES = 64;
constexpr int BLOCK_BYTES = 36;

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

template<int PARTS>
__launch_bounds__(32) __global__ void nvfp4_selected_wmma_kernel(
        const half_t* __restrict__ x_hi, const half_t* __restrict__ x_lo,
        const int64_t* __restrict__ expert_start_compact, const int32_t* __restrict__ group_expert,
        const int32_t* __restrict__ group_first_row, const int32_t* __restrict__ row_source,
        const int32_t* __restrict__ row_route, const float* __restrict__ outer,
        const uint8_t* __restrict__ qweight, float* __restrict__ out, int64_t in_features,
        int64_t out_features) {
    const int64_t col_start = static_cast<int64_t>(blockIdx.x) * TILE_COLS;
    const int expert = group_expert[blockIdx.y];
    const int first_row = group_first_row[blockIdx.y];
    const int tid = threadIdx.x, lane16 = tid & 15, half_id = tid >> 4;
    if (expert < 0) return;
    const int64_t compact_base = expert_start_compact[expert];
    const int64_t count = expert_start_compact[expert + 1] - compact_base;
    const int64_t row_bytes = in_features / BLOCK_VALUES * BLOCK_BYTES;
    const uint8_t* expert_base = qweight + static_cast<int64_t>(expert) * out_features * row_bytes;
    const float scale = outer ? outer[expert] : 1.0f;

    int64_t x_row[TN];
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        // Padding rows of the last group of an expert read compact row 0 and are not stored.
        const int64_t local = first_row + tn * 16 + lane16;
        const int64_t compact = local < count ? compact_base + local : 0;
        x_row[tn] = row_source ? row_source[compact] : compact;
    }
    float8_t acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) acc[tm][tn] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    const int64_t k_tiles = in_features / 16;
    #pragma unroll 1
    for (int64_t kt = 0; kt < k_tiles; ++kt) {
        half16_t a_hi[TN], a_lo[TN];
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            a_hi[tn] = *reinterpret_cast<const half16_t*>(x_hi + x_row[tn] * in_features + kt * 16);
            if constexpr (PARTS == 2) {
                a_lo[tn] = *reinterpret_cast<const half16_t*>(x_lo + x_row[tn] * in_features + kt * 16);
            }
        }
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const int64_t col = col_start + tm * 16 + lane16;
            const uint8_t* block = expert_base + col * row_bytes + (kt >> 2) * BLOCK_BYTES;
            const uint32_t* codes = reinterpret_cast<const uint32_t*>(block + 4 + (kt & 3) * 8);
            const half16_t b = dequant16(codes[0], codes[1], decode_scale(block[kt & 3]));
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi[tn], b, acc[tm][tn]);
                if constexpr (PARTS == 2) {
                    acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo[tn], b, acc[tm][tn]);
                }
            }
        }
    }
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

template<int PARTS>
int launch_wmma(const void* x_hi, const void* x_lo, const int64_t* expert_start_compact,
                const int32_t* group_expert, const int32_t* group_first_row, const int32_t* row_source,
                const int32_t* row_route, const float* outer, const uint8_t* weights, float* out, int groups,
                int in_features, int out_features, hipStream_t stream) {
    if (!x_hi || (PARTS == 2 && !x_lo) || !expert_start_compact || !group_expert || !group_first_row || !weights
            || !out || groups <= 0 || in_features <= 0 || in_features % BLOCK_VALUES || out_features <= 0
            || out_features % TILE_COLS) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(static_cast<unsigned>(out_features / TILE_COLS), static_cast<unsigned>(groups));
    hipLaunchKernelGGL((nvfp4_selected_wmma_kernel<PARTS>), grid, dim3(32), 0, stream,
                       static_cast<const half_t*>(x_hi), static_cast<const half_t*>(x_lo), expert_start_compact,
                       group_expert, group_first_row, row_source, row_route, outer, weights, out,
                       static_cast<int64_t>(in_features), static_cast<int64_t>(out_features));
    return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_gguf_nvfp4_selected_wmma_f16_f32_out(
        const void* x_hi, const void* x_lo, const int64_t* expert_start_compact, const int32_t* group_expert,
        const int32_t* group_first_row, const int32_t* row_source, const int32_t* row_route, const float* outer,
        const uint8_t* weights, float* out, int groups, int in_features, int out_features, hipStream_t stream) {
    return launch_wmma<1>(x_hi, x_lo, expert_start_compact, group_expert, group_first_row, row_source, row_route,
                          outer, weights, out, groups, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_nvfp4_selected_wmma_f16_parts_f32_out(
        const void* x_hi, const void* x_lo, const int64_t* expert_start_compact, const int32_t* group_expert,
        const int32_t* group_first_row, const int32_t* row_source, const int32_t* row_route, const float* outer,
        const uint8_t* weights, float* out, int groups, int in_features, int out_features, hipStream_t stream) {
    return launch_wmma<2>(x_hi, x_lo, expert_start_compact, group_expert, group_first_row, row_source, row_route,
                          outer, weights, out, groups, in_features, out_features, stream);
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
