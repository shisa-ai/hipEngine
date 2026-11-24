// Device builder for Inkling 2 expert-route tiles.
//
// Reproduces, on the GPU, exactly what
// hipengine.loading.inkling2_device_experts.build_expert_route_tiles does on
// the host: group route indices by expert (stable within expert), walk
// experts in ascending order, emit fixed-width tiles of route indices padded
// with -1. Each record is
//   [expert, route_0, ..., route_{width-1}]   (width + 1 int32 values)
// so a record's route entries are indices into the flattened ids array.
//
// Single-block kernel; O(routes^2) in the worst case for rank computation,
// which is fine for the decode case (routes <= 32) and acceptable for the
// prefill tails this exists for (a few thousand routes). It also gathers an
// optional per-expert scale table into per-route scales while the ids are hot.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kMaxExperts = 1024;

__global__ void inkling2_route_tiles_kernel(
    const long long* __restrict__ ids,
    int routes,
    int num_experts,
    int width,
    const float* __restrict__ scale_table,
    float* __restrict__ gathered_scales,
    int* __restrict__ records,
    int* __restrict__ tile_count,
    int record_capacity) {
  __shared__ int counts[kMaxExperts];
  __shared__ int offsets[kMaxExperts];
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    counts[e] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < routes; i += blockDim.x) {
    const long long e = ids[i];
    if (e >= 0 && e < num_experts) {
      atomicAdd(&counts[static_cast<int>(e)], 1);
    }
  }
  __syncthreads();
  // Exclusive scan of per-expert tile counts. Single pass over experts;
  // every thread recomputes it so no global round-trip is needed.
  int total_tiles = 0;
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < num_experts; ++e) {
      const int tiles = (counts[e] + width - 1) / width;
      offsets[e] = acc;
      acc += tiles;
    }
    tile_count[0] = acc;
  }
  __syncthreads();
  for (int e = 0; e < num_experts; ++e) {
    total_tiles += (counts[e] + width - 1) / width;
  }
  const int stride = width + 1;
  // Zero-fill the whole caller-provided capacity so tiles past the real tile
  // count read as -1 (skipped by the grouped projection kernels) and the
  // caller never needs the tile count on the host.
  for (int i = threadIdx.x; i < record_capacity * stride; i += blockDim.x) {
    records[i] = -1;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < routes; i += blockDim.x) {
    const long long e64 = ids[i];
    const int e = static_cast<int>(e64);
    if (e64 < 0 || e64 >= num_experts) {
      continue;
    }
    int rank = 0;
    for (int j = 0; j < i; ++j) {
      rank += (ids[j] == e64) ? 1 : 0;
    }
    const int tile = offsets[e] + rank / width;
    const int slot = rank - (rank / width) * width;
    records[tile * stride + 1 + slot] = i;
    if (slot == 0) {
      records[tile * stride] = e;
    }
    if (gathered_scales != nullptr) {
      gathered_scales[i] = scale_table[e];
    }
  }
}

}  // namespace

extern "C" int hipengine_inkling2_route_tiles_from_ids(
    const void* ids,
    int routes,
    int num_experts,
    int width,
    const void* scale_table,
    void* gathered_scales,
    void* records,
    void* tile_count,
    int record_capacity,
    hipStream_t stream) {
  if (routes < 0 || num_experts <= 0 || num_experts > kMaxExperts ||
      (width != 2 && width != 4) || ids == nullptr || records == nullptr ||
      tile_count == nullptr || (scale_table == nullptr) != (gathered_scales == nullptr)) {
    return 1;
  }
  if (record_capacity < routes) {
    return 3;
  }
  if (routes > 0) {
    inkling2_route_tiles_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const long long*>(ids), routes, num_experts, width,
        reinterpret_cast<const float*>(scale_table),
        reinterpret_cast<float*>(gathered_scales),
        reinterpret_cast<int*>(records),
        reinterpret_cast<int*>(tile_count),
        record_capacity);
  } else {
    hipMemsetAsync(tile_count, 0, sizeof(int), stream);
  }
  return hipGetLastError() == hipSuccess ? 0 : 2;
}
