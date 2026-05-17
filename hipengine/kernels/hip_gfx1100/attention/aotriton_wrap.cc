// Torch-free C ABI bridge for AOTriton compact-varlen forward attention.
//
// AOTriton exposes a C++ ABI (TensorView<N> plus a namespaced function).  The
// Python hot path should call this stable hipEngine-owned C surface instead of
// dlopening mangled C++ symbols directly.

#include <array>
#include <cstdint>
#include <limits>

#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <hip/hip_runtime.h>

using half_t = _Float16;

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

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ inline float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u32;
    float f32;
  } value;
  value.u32 = static_cast<uint32_t>(bits) << 16;
  return value.f32;
}

__global__ void gate_mul_fp16_inplace_kernel(half_t* attn_out, const half_t* gate, int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gated = static_cast<float>(attn_out[idx]) * sigmoid_f32(static_cast<float>(gate[idx]));
    attn_out[idx] = static_cast<half_t>(gated);
  }
}

__global__ void gate_mul_bf16_to_fp16_kernel(
    const uint16_t* attn_out,
    const half_t* gate,
    half_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gated = bf16_bits_to_float(attn_out[idx]) * sigmoid_f32(static_cast<float>(gate[idx]));
    out[idx] = static_cast<half_t>(gated);
  }
}

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

extern "C" int hipengine_aotriton_gate_mul_fp16_inplace(
    void* attn_out,
    const void* gate,
    int64_t total,
    void* stream) {
  if (attn_out == nullptr || gate == nullptr || total <= 0) {
    return static_cast<int>(kInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  hipLaunchKernelGGL(
      gate_mul_fp16_inplace_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      reinterpret_cast<hipStream_t>(stream),
      reinterpret_cast<half_t*>(attn_out),
      reinterpret_cast<const half_t*>(gate),
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_aotriton_gate_mul_bf16_to_fp16(
    const void* attn_out,
    const void* gate,
    void* out,
    int64_t total,
    void* stream) {
  if (attn_out == nullptr || gate == nullptr || out == nullptr || total <= 0) {
    return static_cast<int>(kInvalidValue);
  }
  const int threads = 256;
  const int64_t blocks64 = (total + threads - 1) / threads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  hipLaunchKernelGGL(
      gate_mul_bf16_to_fp16_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      reinterpret_cast<hipStream_t>(stream),
      reinterpret_cast<const uint16_t*>(attn_out),
      reinterpret_cast<const half_t*>(gate),
      reinterpret_cast<half_t*>(out),
      total);
  return static_cast<int>(hipGetLastError());
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

  // AOTriton 0.11.x v2::flash::attn_fwd_compact_varlen signature:
  //   (T4 q, T4 k, T4 v, T4 b, T1 cu_seqlens_q, T1 cu_seqlens_k,
  //    int32 max_seqlen_q, int32 max_seqlen_k,
  //    float sm_scale, T2 softmax_lse, T4 Out,
  //    float dropout_p,
  //    T0 philox_seed, T0 philox_offset1, int64 philox_offset2,
  //    T0 philox_seed_output, T0 philox_offset_output,
  //    T4 encoded_softmax,
  //    bool is_causal,
  //    T0 atomic_for_causal,
  //    Stream stream, FwdExtraArguments* extargs = nullptr);
  // 0.8.x had b after cu_seqlens_k and no atomic_for_causal; both changed in 0.11.x.
  //
  // When is_causal=true, AOTriton uses an in-kernel atomic to dispatch causal
  // blocks dynamically (added in 0.9b: "Persistent Dynamic for Causal").  The
  // atomic_for_causal tensor must point at a zero-initialized 1-element int32
  // device buffer; passing a null TensorView<0> returns hipErrorInvalidValue.
  // We allocate+zero+free per call; cost is ~5 us, negligible against the
  // attention launch itself.
  const DType scratch_dtype = AOTRITON_NS::kFloat32;
  TensorView<4> null_bias = TensorView<4>::get_null_tensor(scratch_dtype);
  TensorView<4> null_encoded_softmax = TensorView<4>::get_null_tensor(scratch_dtype);
  TensorView<0> null_seed(0, AOTRITON_NS::kInt64);
  TensorView<0> null_offset(0, AOTRITON_NS::kInt64);

  void* atomic_buf = nullptr;
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (is_causal != 0) {
    if (hipError_t err = hipMalloc(&atomic_buf, sizeof(int32_t)); err != hipSuccess) {
      return static_cast<int>(err);
    }
    if (hipError_t err = hipMemsetAsync(atomic_buf, 0, sizeof(int32_t), hip_stream); err != hipSuccess) {
      (void)hipFree(atomic_buf);
      return static_cast<int>(err);
    }
  }
  TensorView<0> atomic_view = is_causal != 0
      ? TensorView<0>(reinterpret_cast<intptr_t>(atomic_buf), AOTRITON_NS::kInt32)
      : TensorView<0>(0, AOTRITON_NS::kInt32);

  hipError_t aot_err = AOTRITON_NS::v2::flash::attn_fwd_compact_varlen(
      q_view,
      k_view,
      v_view,
      null_bias,
      cu_q_view,
      cu_k_view,
      max_seqlen_q,
      max_seqlen_k,
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
      atomic_view,
      Stream(hip_stream),
      nullptr);
  if (atomic_buf != nullptr) {
    (void)hipFree(atomic_buf);
  }
  return static_cast<int>(aot_err);
}

extern "C" int hipengine_aotriton_attn_fwd_v3_compact_varlen(
    const HipengineAotritonTensor4* q,
    const HipengineAotritonTensor4* k,
    const HipengineAotritonTensor4* v,
    const HipengineAotritonTensor1* cu_seqlens_q,
    const HipengineAotritonTensor1* cu_seqlens_k,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    const HipengineAotritonTensor2* softmax_lse,
    const HipengineAotritonTensor4* out,
    void* persistent_atomic_counter,
    float sm_scale,
    int32_t is_causal,
    void* stream) {
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    return static_cast<int>(kInvalidValue);
  }
  if (is_causal != 0 && persistent_atomic_counter == nullptr) {
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

  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  if (is_causal != 0) {
    if (hipError_t err = hipMemsetAsync(persistent_atomic_counter, 0, sizeof(int32_t), hip_stream); err != hipSuccess) {
      return static_cast<int>(err);
    }
  }

  TensorView<4> null_bias = TensorView<4>::get_null_tensor(AOTRITON_NS::kFloat32);
  TensorView<2> null_alibi = TensorView<2>::get_null_tensor(q_view.dtype());
  TensorView<4> null_encoded_softmax = TensorView<4>::get_null_tensor(AOTRITON_NS::kFloat32);
  TensorView<0> null_seed(0, AOTRITON_NS::kInt64);
  TensorView<0> null_offset(0, AOTRITON_NS::kInt64);
  TensorView<0> atomic_view = persistent_atomic_counter != nullptr
      ? TensorView<0>(reinterpret_cast<intptr_t>(persistent_atomic_counter), AOTRITON_NS::kInt32)
      : TensorView<0>(0, AOTRITON_NS::kInt32);

  AOTRITON_NS::v3::flash::attn_fwd_params params;
  params.Q = q_view;
  params.K = k_view;
  params.V = v_view;
  params.B = null_bias;
  params.A = null_alibi;
  params.Sm_scale = sm_scale;
  params.L = lse_view;
  params.Out = out_view;
  params.cu_seqlens_q = cu_q_view;
  params.cu_seqlens_k = cu_k_view;
  params.Max_seqlen_q = max_seqlen_q;
  params.Max_seqlen_k = max_seqlen_k;
  params.dropout_p = 0.0f;
  params.philox_seed_ptr = null_seed;
  params.philox_offset1 = null_offset;
  params.philox_offset2 = 0;
  params.philox_seed_output = null_seed;
  params.philox_offset_output = null_offset;
  params.encoded_softmax = null_encoded_softmax;
  params.persistent_atomic_counter = atomic_view;
  params.causal_type = is_causal != 0
      ? AOTRITON_NS::v3::flash::CausalType::WindowedAttention
      : AOTRITON_NS::v3::flash::CausalType::None;
  params.varlen_type = AOTRITON_NS::v3::flash::VarlenType::CompactVarlen;
  params.window_left = AOTRITON_NS::v3::flash::WindowValue::BottomRightAligned;
  params.window_right = AOTRITON_NS::v3::flash::WindowValue::BottomRightAligned;

  return static_cast<int>(AOTRITON_NS::v3::flash::attn_fwd(
      params,
      AOTRITON_NS::v3::flash::attn_fwd_params::kVersion,
      Stream(hip_stream),
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
