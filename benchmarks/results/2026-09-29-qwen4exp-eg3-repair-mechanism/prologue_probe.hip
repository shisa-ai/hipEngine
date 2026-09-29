#include <hip/hip_runtime.h>
#include <cstdint>

// Variant A: exactly what the repair kernel does today - runtime 64-bit div,
// multiply-subtract modulo, and a per-slot binary search over expert starts.
extern "C" __global__ void prologue_current(
    const int64_t* __restrict__ expert_start, const int* __restrict__ risk_indices,
    int64_t max_risks, int64_t compact_rows, int64_t out_features_total,
    int64_t num_experts, int* __restrict__ out) {
  for (int64_t slot = blockIdx.x; slot < max_risks; slot += gridDim.x) {
    const int64_t idx = risk_indices[slot];
    if (idx < 0 || idx >= compact_rows * out_features_total) continue;
    const int64_t compact_row = idx / out_features_total;
    const int64_t global_col = idx - compact_row * out_features_total;
    int64_t lo = 0, hi = num_experts;
    while (lo < hi) {
      const int64_t mid = (lo + hi) >> 1;
      if (expert_start[mid] <= compact_row) lo = mid + 1; else hi = mid;
    }
    const int64_t expert = lo - 1;
    if (expert < 0 || expert >= num_experts) continue;
    out[slot] = static_cast<int>(global_col + expert);
  }
}

// Variant B: host-computed magic reciprocal for the division (bit-exact for the
// same quotient), and the expert search skipped when the row is unchanged from
// the previous slot the block processed.
extern "C" __global__ void prologue_magic(
    const int64_t* __restrict__ expert_start, const int* __restrict__ risk_indices,
    int64_t max_risks, int64_t compact_rows, int64_t out_features_total,
    int64_t num_experts, uint64_t magic, int shift, int* __restrict__ out) {
  int64_t prev_row = -1, prev_expert = -1;
  for (int64_t slot = blockIdx.x; slot < max_risks; slot += gridDim.x) {
    const int64_t idx = risk_indices[slot];
    if (idx < 0 || idx >= compact_rows * out_features_total) continue;
    // magic/shift reciprocal: quotient of unsigned 64-bit idx by out_features_total
    const uint64_t q = static_cast<uint64_t>(
        (static_cast<__uint128_t>(static_cast<uint64_t>(idx)) * magic) >> (64 + shift));
    const int64_t compact_row = static_cast<int64_t>(q);
    const int64_t global_col = idx - compact_row * out_features_total;
    int64_t expert;
    if (compact_row == prev_row) {
      expert = prev_expert;
    } else {
      int64_t lo = 0, hi = num_experts;
      while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (expert_start[mid] <= compact_row) lo = mid + 1; else hi = mid;
      }
      expert = lo - 1;
      prev_row = compact_row; prev_expert = expert;
    }
    if (expert < 0 || expert >= num_experts) continue;
    out[slot] = static_cast<int>(global_col + expert);
  }
}
