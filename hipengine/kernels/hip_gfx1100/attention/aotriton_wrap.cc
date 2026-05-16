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

int64_t dtype_itemsize(DType dtype) {
  switch (dtype) {
    case AOTRITON_NS::kFloat32:
    case AOTRITON_NS::kInt32:
      return 4;
    case AOTRITON_NS::kFloat16:
    case AOTRITON_NS::kBFloat16:
      return 2;
    case AOTRITON_NS::kInt64:
      return 8;
    default:
      return 0;
  }
}

bool slice_tensor4_dim(const HipengineAotritonTensor4& src,
                       int dim,
                       int64_t index,
                       HipengineAotritonTensor4* dst) {
  if (dst == nullptr || src.data == nullptr || dim < 0 || dim >= 4 || index < 0 || index >= src.sizes[dim]) {
    return false;
  }
  DType dtype;
  if (!dtype_from_code(src.dtype, &dtype) || src.strides[dim] < 0) {
    return false;
  }
  const int64_t itemsize = dtype_itemsize(dtype);
  if (itemsize <= 0) {
    return false;
  }
  *dst = src;
  dst->sizes[dim] = 1;
  dst->data = static_cast<void*>(static_cast<char*>(src.data) + index * src.strides[dim] * itemsize);
  return true;
}

bool slice_tensor2_dim(const HipengineAotritonTensor2& src,
                       int dim,
                       int64_t index,
                       HipengineAotritonTensor2* dst) {
  if (dst == nullptr || src.data == nullptr || dim < 0 || dim >= 2 || index < 0 || index >= src.sizes[dim]) {
    return false;
  }
  DType dtype;
  if (!dtype_from_code(src.dtype, &dtype) || src.strides[dim] < 0) {
    return false;
  }
  const int64_t itemsize = dtype_itemsize(dtype);
  if (itemsize <= 0) {
    return false;
  }
  *dst = src;
  dst->sizes[dim] = 1;
  dst->data = static_cast<void*>(static_cast<char*>(src.data) + index * src.strides[dim] * itemsize);
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

extern "C" int hipengine_aotriton_attn_fwd_compact_varlen_gqa_per_q_head(
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
  if (q == nullptr || k == nullptr || v == nullptr || softmax_lse == nullptr || out == nullptr) {
    return static_cast<int>(kInvalidValue);
  }
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    return static_cast<int>(kInvalidValue);
  }
  const int64_t num_q_heads = q->sizes[1];
  const int64_t num_kv_heads = k->sizes[1];
  if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) {
    return static_cast<int>(kInvalidValue);
  }
  if (v->sizes[1] != num_kv_heads || out->sizes[1] != num_q_heads || softmax_lse->sizes[0] != num_q_heads) {
    return static_cast<int>(kInvalidValue);
  }
  if (q->sizes[0] != 1 || k->sizes[0] != 1 || v->sizes[0] != 1 || out->sizes[0] != 1) {
    return static_cast<int>(kInvalidValue);
  }
  if (q->sizes[2] != out->sizes[2] || q->sizes[3] != k->sizes[3] || q->sizes[3] != v->sizes[3] ||
      q->sizes[3] != out->sizes[3]) {
    return static_cast<int>(kInvalidValue);
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;

  for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
    const int64_t kv_head = q_head / kv_group;
    HipengineAotritonTensor4 q_one{};
    HipengineAotritonTensor4 k_one{};
    HipengineAotritonTensor4 v_one{};
    HipengineAotritonTensor4 out_one{};
    HipengineAotritonTensor2 lse_one{};
    if (!slice_tensor4_dim(*q, 1, q_head, &q_one) || !slice_tensor4_dim(*k, 1, kv_head, &k_one) ||
        !slice_tensor4_dim(*v, 1, kv_head, &v_one) || !slice_tensor4_dim(*out, 1, q_head, &out_one) ||
        !slice_tensor2_dim(*softmax_lse, 0, q_head, &lse_one)) {
      return static_cast<int>(kInvalidValue);
    }
    const int err = hipengine_aotriton_attn_fwd_compact_varlen(
        &q_one,
        &k_one,
        &v_one,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        &lse_one,
        &out_one,
        sm_scale,
        is_causal,
        stream);
    if (err != static_cast<int>(hipSuccess)) {
      return err;
    }
  }
  return static_cast<int>(hipSuccess);
}
