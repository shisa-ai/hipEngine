// Host-only target/proposal graph launcher for NativeSpecCycle ABI v1.
//
// The provider owns the graph executable and resolves backend runtime entry
// points once. This launcher validates either the strict GGUF N1 B1/B2/B3 contract
// or the shared PARO MTP/DFlash B1/B2/B3/B4/B5/B8 target+accept contract,
// submits exactly one graph, synchronizes its selected stream, and
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
    HipengineNativeSpecCycleResultV1* result,
    uint32_t failed_stage) {
  memset(result, 0, sizeof(*result));
  result->abi_version = HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = HIPENGINE_NATIVE_SPEC_STATUS_FAILED;
  result->error_code = HIPENGINE_NATIVE_SPEC_ERROR_INTERNAL;
  result->failed_stage = failed_stage;
  if (control != nullptr) {
    result->cycle_id = control->cycle_id;
    result->transaction_id = control->transaction_id;
    result->request_count = control->request_count;
  }
}

int32_t fail(
    HipengineNativeSpecCycleResultV1* result,
    uint32_t error_code,
    uint32_t failed_stage,
    int64_t backend_error_code = 0) {
  result->status = HIPENGINE_NATIVE_SPEC_STATUS_FAILED;
  result->error_code = error_code;
  result->completed_stage_mask = 0;
  result->failed_stage = failed_stage;
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

bool has_required_accept_pointers(const HipengineNativeSpecCycleControlV1* control) {
  return control->metadata_candidate_counts != 0 &&
         control->output_accepted_counts != 0 &&
         control->output_commit_rows != 0 &&
         control->output_commit_tokens != 0 &&
         control->output_commit_positions != 0 &&
         control->output_next_tokens != 0 &&
         control->output_full_accept != 0 &&
         control->output_committed_output_ids != 0 &&
         control->output_committed_output_lengths != 0;
}

bool has_required_n2_pointers(const HipengineNativeSpecCycleControlV1* control) {
  return control->metadata_candidate_counts != 0 &&
         control->metadata_remaining_decode != 0 &&
         control->state_hidden_seed_dst != 0 &&
         control->output_accepted_counts != 0 &&
         control->output_commit_rows != 0 &&
         control->output_commit_tokens != 0 &&
         control->output_commit_positions != 0 &&
         control->output_next_tokens != 0 &&
         control->output_full_accept != 0 &&
         control->output_committed_output_ids != 0 &&
         control->output_committed_output_lengths != 0 &&
         control->output_output_ids != 0 &&
         control->output_output_lengths != 0 &&
         control->output_last_positions != 0 &&
         control->output_context_lengths != 0 &&
         ((control->state_linear_state_rows != 0 && control->state_linear_state_dst != 0) ||
          (control->state_key_rows != 0 && control->state_value_rows != 0 &&
           control->kv_key_cache != 0 && control->kv_value_cache != 0) ||
          control->state_hidden_seed_dst != 0);
}

bool has_required_provider_commit_pointers(
    const HipengineNativeSpecCycleControlV1* control) {
  return has_required_accept_pointers(control) &&
         control->state_linear_state_rows != 0 &&
         control->state_linear_state_dst != 0 &&
         control->output_output_ids != 0 &&
         control->output_output_lengths != 0 &&
         control->output_last_positions != 0 &&
         control->output_context_lengths != 0;
}

bool is_small_chain_target(const HipengineNativeSpecCycleControlV1* control) {
  constexpr uint32_t kN2Stages = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY |
                                 HIPENGINE_NATIVE_SPEC_STAGE_ACCEPT |
                                 HIPENGINE_NATIVE_SPEC_STAGE_COMMIT |
                                 HIPENGINE_NATIVE_SPEC_STAGE_UPDATE_CURSORS;
  const bool verify_only = control->stage_mask == HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  const bool n2 = control->stage_mask == kN2Stages;
  const uint32_t rows = control->row_count;
  const uint32_t candidates = rows >= 1 ? rows - 1 : 0;
  const bool supported_rows = rows == 2 || rows == 3 || rows == 4;
  return (verify_only || n2) &&
         control->mode == HIPENGINE_NATIVE_SPEC_MODE_CHAIN &&
         control->request_count == 1 &&
         control->request_capacity >= 1 &&
         supported_rows &&
         control->active_row_count == rows &&
         control->row_capacity >= rows &&
         control->candidate_count == candidates &&
         control->active_candidate_count == candidates &&
         control->candidate_capacity >= candidates &&
         control->candidate_budget == candidates &&
         control->span_count > 0 &&
         control->span_capacity >= control->span_count &&
         control->context_bucket >= control->max_live_count &&
         control->hidden_size > 0 &&
         control->hidden_row_capacity >= rows &&
         control->output_stride >= rows &&
         control->metadata_dtype == (verify_only ? HIPENGINE_NATIVE_SPEC_DTYPE_INT64
                                                 : HIPENGINE_NATIVE_SPEC_DTYPE_INT32) &&
         control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP32 &&
         control->kv_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_BF16 &&
         (!n2 || has_required_n2_pointers(control));
}

bool is_provider_chain_target(const HipengineNativeSpecCycleControlV1* control) {
  constexpr uint32_t kVerifyAcceptStages = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY |
                                            HIPENGINE_NATIVE_SPEC_STAGE_ACCEPT;
  constexpr uint32_t kCommitStages = kVerifyAcceptStages |
                                     HIPENGINE_NATIVE_SPEC_STAGE_COMMIT |
                                     HIPENGINE_NATIVE_SPEC_STAGE_UPDATE_CURSORS;
  const bool verify_only = control->stage_mask == HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  const bool verify_accept = control->stage_mask == kVerifyAcceptStages;
  const bool commit = control->stage_mask == kCommitStages;
  const uint32_t rows = control->row_count;
  const uint32_t candidates = rows >= 1 ? rows - 1 : 0;
  const bool supported_budget = candidates == 1 || candidates == 2 ||
                                candidates == 3 || candidates == 4 ||
                                candidates == 5 || candidates == 8;
  return (verify_only || verify_accept || commit) &&
         control->mode == HIPENGINE_NATIVE_SPEC_MODE_CHAIN &&
         control->request_count == 1 &&
         control->request_capacity >= 1 &&
         supported_budget &&
         control->active_row_count == rows &&
         control->row_capacity >= rows &&
         control->candidate_count == candidates &&
         control->active_candidate_count == candidates &&
         control->candidate_capacity >= candidates &&
         control->candidate_budget == candidates &&
         control->span_count > 0 &&
         control->span_capacity >= control->span_count &&
         control->context_bucket >= control->max_live_count &&
         control->hidden_size > 0 &&
         control->hidden_row_capacity >= rows &&
         control->output_stride >= rows &&
         control->metadata_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_INT32 &&
         (control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP16 ||
          control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_BF16) &&
         (!commit || control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP16) &&
         control->kv_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_BF16 &&
         (!(verify_accept || commit) || has_required_accept_pointers(control)) &&
         (!commit || has_required_provider_commit_pointers(control));
}

bool has_required_proposal_pointers(const HipengineNativeSpecCycleControlV1* control) {
  return control->state_hidden_seed_in != 0 &&
         control->state_candidate_token_ids != 0 &&
         control->state_draft_key_cache != 0 &&
         control->state_draft_value_cache != 0;
}

bool is_small_chain_proposal(const HipengineNativeSpecCycleControlV1* control) {
  const uint32_t rows = control->row_count;
  const uint32_t candidates = rows >= 1 ? rows - 1 : 0;
  return control->stage_mask == HIPENGINE_NATIVE_SPEC_STAGE_PROPOSE &&
         control->mode == HIPENGINE_NATIVE_SPEC_MODE_CHAIN &&
         control->request_count == 1 &&
         control->request_capacity >= 1 &&
         (rows == 2 || rows == 3) &&
         control->active_row_count == rows &&
         control->row_capacity >= rows &&
         control->candidate_count == candidates &&
         control->active_candidate_count == candidates &&
         control->candidate_capacity >= candidates &&
         control->candidate_budget == candidates &&
         control->span_count > 0 &&
         control->span_capacity >= control->span_count &&
         control->context_bucket >= control->max_live_count &&
         control->hidden_size > 0 &&
         control->hidden_row_capacity >= rows &&
         control->output_stride >= rows &&
         control->metadata_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_INT64 &&
         control->hidden_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP32 &&
         control->kv_dtype == HIPENGINE_NATIVE_SPEC_DTYPE_FP32 &&
         has_required_proposal_pointers(control);
}

int32_t submit_graph(
    const HipengineNativeSpecCycleControlV1* control,
    HipengineNativeSpecCycleResultV1* result,
    void* graph_exec,
    void* graph_launch_address,
    void* stream_synchronize_address,
    uint32_t failed_stage,
    bool allow_default_stream = false) {
  if ((!allow_default_stream && control->stream == 0) || control->deadline_ns != 0 ||
      control->output_cancel_flag != 0 || graph_exec == nullptr ||
      graph_launch_address == nullptr || stream_synchronize_address == nullptr) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_INVALID_CONTROL, failed_stage);
  }
  auto launch = reinterpret_cast<graph_launch_fn>(graph_launch_address);
  int32_t error = launch(graph_exec, reinterpret_cast<void*>(control->stream));
  if (error != 0) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_KERNEL_LAUNCH, failed_stage, error);
  }
  auto synchronize = reinterpret_cast<stream_synchronize_fn>(stream_synchronize_address);
  error = synchronize(reinterpret_cast<void*>(control->stream));
  if (error != 0) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_KERNEL_LAUNCH, failed_stage, error);
  }
  result->status = HIPENGINE_NATIVE_SPEC_STATUS_COMPLETE;
  result->error_code = HIPENGINE_NATIVE_SPEC_ERROR_NONE;
  result->completed_stage_mask = control->stage_mask;
  result->failed_stage = 0;
  result->visible_output_count = 0;
  result->backend_error_code = 0;
  return 0;
}

}  // namespace

extern "C" int32_t hipengine_native_spec_target_graph_launch_v1(
    const HipengineNativeSpecCycleControlV1* control,
    HipengineNativeSpecCycleResultV1* result,
    void* graph_exec,
    void* graph_launch_address,
    void* stream_synchronize_address) {
  constexpr uint32_t kStage = HIPENGINE_NATIVE_SPEC_STAGE_VERIFY;
  if (control == nullptr || result == nullptr) {
    return -1;
  }
  initialize_result(control, result, kStage);
  if (control->abi_version != HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION ||
      control->struct_size != sizeof(*control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_ABI_MISMATCH, kStage);
  }
  const bool is_small_gguf = is_small_chain_target(control);
  const bool is_provider_target = is_provider_chain_target(control);
  if (!is_small_gguf && !is_provider_target) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_UNSUPPORTED_SHAPE, kStage);
  }
  if (!has_required_verify_pointers(control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_INVALID_CONTROL, kStage);
  }
  return submit_graph(
      control, result, graph_exec, graph_launch_address,
      stream_synchronize_address, kStage, is_provider_target);
}

extern "C" int32_t hipengine_native_spec_proposal_graph_launch_v1(
    const HipengineNativeSpecCycleControlV1* control,
    HipengineNativeSpecCycleResultV1* result,
    void* graph_exec,
    void* graph_launch_address,
    void* stream_synchronize_address) {
  constexpr uint32_t kStage = HIPENGINE_NATIVE_SPEC_STAGE_PROPOSE;
  if (control == nullptr || result == nullptr) {
    return -1;
  }
  initialize_result(control, result, kStage);
  if (control->abi_version != HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION ||
      control->struct_size != sizeof(*control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_ABI_MISMATCH, kStage);
  }
  if (!is_small_chain_proposal(control)) {
    return fail(result, HIPENGINE_NATIVE_SPEC_ERROR_UNSUPPORTED_SHAPE, kStage);
  }
  return submit_graph(
      control, result, graph_exec, graph_launch_address,
      stream_synchronize_address, kStage);
}
