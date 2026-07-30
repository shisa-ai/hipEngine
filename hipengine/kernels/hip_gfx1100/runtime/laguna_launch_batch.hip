// Native host-call batching for exact Laguna decode boundaries.
//
// The shim does not replace or modify either GPU kernel. It invokes the
// existing exported host launch wrappers back-to-back so HIP sees the same
// stream, grids, arithmetic, and dependency order without returning through
// Python between launches.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using q4_shared_down_fn = int (*)(
    const uint16_t*,
    const int32_t*,
    const float*,
    const float*,
    uint16_t*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    hipStream_t);

using q6_shared_down_fn = int (*)(
    const uint16_t*,
    const uint8_t*,
    uint16_t*,
    int64_t,
    int64_t,
    int64_t,
    hipStream_t);

using q4_t16_shared_down_fn = q6_shared_down_fn;

using moe_tail_fn = int (*)(
    const uint16_t*,
    const uint16_t*,
    const uint16_t*,
    const float*,
    uint16_t*,
    uint16_t*,
    int64_t,
    float,
    hipStream_t);

}  // namespace

extern "C" int hipengine_laguna_q4_shared_down_tail_batch(
    void* projection_function,
    void* tail_function,
    const uint16_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    uint16_t* shared_out,
    const uint16_t* routed,
    const uint16_t* post_attention,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* hidden_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    float eps,
    hipStream_t stream) {
  if (projection_function == nullptr || tail_function == nullptr ||
      x == nullptr || qweight == nullptr || scales == nullptr ||
      mins == nullptr || shared_out == nullptr || routed == nullptr ||
      post_attention == nullptr || norm_weight == nullptr ||
      norm_out == nullptr || hidden_out == nullptr || rows != 1 ||
      in_features <= 0 || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto projection =
      reinterpret_cast<q4_shared_down_fn>(projection_function);
  const auto tail = reinterpret_cast<moe_tail_fn>(tail_function);
  const int projection_error = projection(
      x,
      qweight,
      scales,
      mins,
      shared_out,
      rows,
      in_features,
      out_features,
      threads,
      stream);
  if (projection_error != static_cast<int>(hipSuccess)) {
    return projection_error;
  }
  return tail(
      routed,
      shared_out,
      post_attention,
      norm_weight,
      norm_out,
      hidden_out,
      out_features,
      eps,
      stream);
}

extern "C" int hipengine_laguna_q6_shared_down_tail_batch(
    void* projection_function,
    void* tail_function,
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* shared_out,
    const uint16_t* routed,
    const uint16_t* post_attention,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* hidden_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    float eps,
    hipStream_t stream) {
  if (projection_function == nullptr || tail_function == nullptr ||
      x == nullptr || qweight == nullptr || shared_out == nullptr ||
      routed == nullptr || post_attention == nullptr ||
      norm_weight == nullptr || norm_out == nullptr ||
      hidden_out == nullptr || rows != 1 || in_features <= 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto projection =
      reinterpret_cast<q6_shared_down_fn>(projection_function);
  const auto tail = reinterpret_cast<moe_tail_fn>(tail_function);
  const int projection_error = projection(
      x,
      qweight,
      shared_out,
      rows,
      in_features,
      out_features,
      stream);
  if (projection_error != static_cast<int>(hipSuccess)) {
    return projection_error;
  }
  return tail(
      routed,
      shared_out,
      post_attention,
      norm_weight,
      norm_out,
      hidden_out,
      out_features,
      eps,
      stream);
}

extern "C" int hipengine_laguna_q4_t16_shared_down_tail_batch(
    void* projection_function,
    void* tail_function,
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* shared_out,
    const uint16_t* routed,
    const uint16_t* post_attention,
    const float* norm_weight,
    uint16_t* norm_out,
    uint16_t* hidden_out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    float eps,
    hipStream_t stream) {
  if (projection_function == nullptr || tail_function == nullptr ||
      x == nullptr || tiles == nullptr || shared_out == nullptr ||
      routed == nullptr || post_attention == nullptr ||
      norm_weight == nullptr || norm_out == nullptr ||
      hidden_out == nullptr || rows != 1 || in_features <= 0 ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const auto projection =
      reinterpret_cast<q4_t16_shared_down_fn>(projection_function);
  const auto tail = reinterpret_cast<moe_tail_fn>(tail_function);
  const int projection_error = projection(
      x,
      tiles,
      shared_out,
      rows,
      in_features,
      out_features,
      stream);
  if (projection_error != static_cast<int>(hipSuccess)) {
    return projection_error;
  }
  return tail(
      routed,
      shared_out,
      post_attention,
      norm_weight,
      norm_out,
      hidden_out,
      out_features,
      eps,
      stream);
}
