// Host-only fixed-B2 target graph launcher for NativeSpecCycle ABI v1.
//
// The provider owns the graph executable and resolves backend runtime entry
// points once.  This launcher validates the provider-neutral control block,
// submits exactly one target graph, synchronizes its session-owned stream, and
// writes a bounded terminal result.  It contains no device math and owns none
// of the borrowed pointers or handles.

#include "native_cycle_abi.h"

#include <stdint.h>
#include <string.h>

namespace {

using graph_launch_fn = int32_t (*)(void* graph_exec, void* stream);
using stream_synchronize_fn = int32_t (*)(void* stream);

void initialize_result(
    const HipengineNativeSpecCycleControlV1* control,
    HipengineNativeSpecCycleResultV1* result) {
  memset(result, 0, sizeof(*result));
  result->abi_version = HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = HIPENGINE_NATIVE_SPEC_STATUS_FAILED;
  result->error_code = HIPENGINE_NATIVE_SPEC_ERROR_INTERNAL;
  result->failed_stage = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  if (control != nullptr) {
    result->cycle_id = control->cycle_id;
    result->transaction_id = control->transaction_id;
    result->request_count = control->request_count;
  }
}

int32_t fail(
    HipengineNativeSpecCycleResultV1* result,
    uint32_t error_code,
    int64_t backend_error_code = 0) {
  result->status = HIPENGINE_NATIVE_SPEC_STATUS_FAILED;
  result->error_code = error_code;
  result->completed_stage_mask = 0;
  result->failed_stage = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  result->visible_output_count = 0;
  result->backend_error_code = backend_error_code;
  return 0;
}

bool has_required_verify_pointers(const HipengineNativeSpecCycleControlV1* control) {
  return control->metadata_token_ids != 0 &&
         control->metadata_positions != 0 &&
         control->metadata_parent_rows != 0 &&
         control->metadata_draft_depths != 0 &&
         control->metadata_row_to_request != 0 &&
         control->metadata_active_mask != 0 &&
         control->kv_base_offsets != 0 &&
         control->kv_live_counts != 0 &&
         control->state_hidden_seed_rows != 0 &&
         control->output_target_top1 != 0;
}

bool is_fixed_b2_target(const HipengineNativeSpecCycleControlV1* control) {
  return control->stage_mask == HIPENGINE_NATIVE_SPEC_STAGE_VERIFY &&
         control->mode == HIPENGINE_NATIVE_SPEC_MODE_CHAIN &&
         control->request_count == 1 &&
         control->request_capacity >= 1 &&
         control->row_count == 3 &&
         control->active_row_count == 3 &&
         control->row_capacity >= 3 &&
         control->candidate_count == 2 &&
         control->active_candidate_count == 2 &&
         control->candidate_capacity >= 2 &&
         control->candidate_budget == 2 &&
         control->span_count > 0 &&
         control->span_capacity >= control->span_count &&
         control->context_bucket >= control->max_live_count &&
         control->hidden_size > 0 &&
         control->hidden_row_capacity >= 3 &&
         control->output_stride >= 3 &&
         control->metadata_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_INT64 &&
         control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP32 &&
         control->kv_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_BF16;
}

}  // namespace

extern "C" int32_t hipengine_native_spec_target_graph_launch_v1(
    const HipengineNativeSpecCycleControlV1* control,
    HipengineNativeSpecCycleResultV1* result,
    void* graph_exec,
    void* graph_launch_address,
    void* stream_synchronize_address) {
  if (control == nullptr || result == nullptr) {
    return -1;
  }
  initialize_result(control, result);
  if (control->abi_version != HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION ||
      control->struct_size != sizeof(*control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_ABI_MISMATCH);
  }
  if (!is_fixed_b2_target(control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_UNSUPPORTED_SHAPE);
  }
  if (!has_required_verify_pointers(control) || control->stream == 0 ||
      control->deadline_ns != 0 || control->output_cancel_flag != 0 ||
      graph_exec == nullptr || graph_launch_address == nullptr ||
      stream_synchronize_address == nullptr) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_INVALID_CONTROL);
  }

  auto launch = reinterpret_cast<graph_launch_fn>(graph_launch_address);
  int32_t error = launch(graph_exec, reinterpret_cast<void*>(control->stream));
  if (error != 0) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_KERNEL_LAUNCH, error);
  }
  auto synchronize = reinterpret_cast<stream_synchronize_fn>(stream_synchronize_address);
  error = synchronize(reinterpret_cast<void*>(control->stream));
  if (error != 0) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_KERNEL_LAUNCH, error);
  }

  result->status = HIPENGINE_NATIVE_SPEC_STATUS_COMPLETE;
  result->error_code = HIPENGINE_NATIVE_SPEC_ERROR_NONE;
  result->completed_stage_mask = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  result->failed_stage = 0;
  result->visible_output_count = 0;
  result->backend_error_code = 0;
  return 0;
}
