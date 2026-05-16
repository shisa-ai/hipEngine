// Torch-free C ABI bridge for AOTriton compact-varlen forward attention.
//
// AOTriton exposes a C++ ABI (TensorView<N> plus a namespaced function).  The
// Python hot path should call this stable hipENGINE-owned C surface instead of
// dlopening mangled C++ symbols directly.

#include <array>
#include <cstdint>
#include <limits>

#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <hip/hip_runtime.h>

extern "C" {

struct HipengineAotritonTensor1 {
  void* data;
  int64_t sizes[1];
  int64_t strides[1];
  int32_t dtype;
};

struct HipengineAotritonTensor2 {
  void* data;
  int64_t sizes[2];
  int64_t strides[2];
  int32_t dtype;
};

struct HipengineAotritonTensor4 {
  void* data;
  int64_t sizes[4];
  int64_t strides[4];
  int32_t dtype;
};

}  // extern "C"

namespace {

using AOTRITON_NS::DType;
using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;

constexpr hipError_t kInvalidValue = hipErrorInvalidValue;

bool dtype_from_code(int32_t code, DType* out) {
  switch (code) {
    case AOTRITON_NS::kFloat32:
    case AOTRITON_NS::kFloat16:
    case AOTRITON_NS::kBFloat16:
    case AOTRITON_NS::kInt32:
    case AOTRITON_NS::kInt64:
      *out = static_cast<DType>(code);
      return true;
    default:
      return false;
  }
}

bool to_u64(int64_t value, uint64_t* out) {
  if (value < 0) {
    return false;
  }
  *out = static_cast<uint64_t>(value);
  return true;
}

template <int Rank, typename TensorDesc>
bool copy_shape(const TensorDesc& desc,
                std::array<uint64_t, Rank>* sizes,
                std::array<uint64_t, Rank>* strides) {
  for (int i = 0; i < Rank; ++i) {
    if (!to_u64(desc.sizes[i], &(*sizes)[i]) || !to_u64(desc.strides[i], &(*strides)[i])) {
      return false;
    }
  }
  return true;
}

bool make_tensor(const HipengineAotritonTensor1* desc, TensorView<1>* out) {
  if (desc == nullptr || desc->data == nullptr) {
    return false;
  }
  DType dtype;
  std::array<uint64_t, 1> sizes{};
  std::array<uint64_t, 1> strides{};
  if (!dtype_from_code(desc->dtype, &dtype) || !copy_shape<1>(*desc, &sizes, &strides)) {
    return false;
  }
  *out = TensorView<1>(reinterpret_cast<intptr_t>(desc->data), sizes, strides, dtype);
  return true;
}

bool make_tensor(const HipengineAotritonTensor2* desc, TensorView<2>* out) {
  if (desc == nullptr || desc->data == nullptr) {
    return false;
  }
  DType dtype;
  std::array<uint64_t, 2> sizes{};
  std::array<uint64_t, 2> strides{};
  if (!dtype_from_code(desc->dtype, &dtype) || !copy_shape<2>(*desc, &sizes, &strides)) {
    return false;
  }
  *out = TensorView<2>(reinterpret_cast<intptr_t>(desc->data), sizes, strides, dtype);
  return true;
}

bool make_tensor(const HipengineAotritonTensor4* desc, TensorView<4>* out) {
  if (desc == nullptr || desc->data == nullptr) {
    return false;
  }
  DType dtype;
  std::array<uint64_t, 4> sizes{};
  std::array<uint64_t, 4> strides{};
  if (!dtype_from_code(desc->dtype, &dtype) || !copy_shape<4>(*desc, &sizes, &strides)) {
    return false;
  }
  *out = TensorView<4>(reinterpret_cast<intptr_t>(desc->data), sizes, strides, dtype);
  return true;
}

}  // namespace

extern "C" int hipengine_aotriton_check_gpu(void* stream) {
  return static_cast<int>(AOTRITON_NS::v2::flash::check_gpu(Stream(reinterpret_cast<hipStream_t>(stream))));
}

extern "C" int hipengine_aotriton_attn_fwd_compact_varlen(
    const HipengineAotritonTensor4* q,
    const HipengineAotritonTensor4* k,
    const HipengineAotritonTensor4* v,
    const HipengineAotritonTensor1* cu_seqlens_q,
    const HipengineAotritonTensor1* cu_seqlens_k,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    const HipengineAotritonTensor2* softmax_lse,
    const HipengineAotritonTensor4* out,
    float sm_scale,
    int32_t is_causal,
    void* stream) {
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    return static_cast<int>(kInvalidValue);
  }

  TensorView<4> q_view;
  TensorView<4> k_view;
  TensorView<4> v_view;
  TensorView<1> cu_q_view;
  TensorView<1> cu_k_view;
  TensorView<2> lse_view;
  TensorView<4> out_view;
  if (!make_tensor(q, &q_view) || !make_tensor(k, &k_view) || !make_tensor(v, &v_view) ||
      !make_tensor(cu_seqlens_q, &cu_q_view) || !make_tensor(cu_seqlens_k, &cu_k_view) ||
      !make_tensor(softmax_lse, &lse_view) || !make_tensor(out, &out_view)) {
    return static_cast<int>(kInvalidValue);
  }

  const DType scratch_dtype = AOTRITON_NS::kFloat32;
  TensorView<4> null_bias = TensorView<4>::get_null_tensor(scratch_dtype);
  TensorView<4> null_encoded_softmax = TensorView<4>::get_null_tensor(scratch_dtype);
  TensorView<0> null_seed(0, AOTRITON_NS::kInt64);
  TensorView<0> null_offset(0, AOTRITON_NS::kInt64);

  return static_cast<int>(AOTRITON_NS::v2::flash::attn_fwd_compact_varlen(
      q_view,
      k_view,
      v_view,
      cu_q_view,
      cu_k_view,
      max_seqlen_q,
      max_seqlen_k,
      null_bias,
      sm_scale,
      lse_view,
      out_view,
      0.0f,
      null_seed,
      null_offset,
      0,
      null_seed,
      null_offset,
      null_encoded_softmax,
      is_causal != 0,
      Stream(reinterpret_cast<hipStream_t>(stream)),
      nullptr));
}
