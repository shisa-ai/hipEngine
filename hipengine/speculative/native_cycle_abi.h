#pragma once

// Versioned raw-pointer ABI for provider-neutral speculative-cycle launchers.
//
// Every address is a borrowed 64-bit host/device address. The caller owns the
// pointed-to allocation and must keep it alive until the invocation reaches a
// terminal state. Native launchers may mutate only output/state-destination
// pointees and must not retain or free any address.

#include <stddef.h>
#include <stdint.h>

#define HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION 1u

#define HIPENGINE_NATIVE_SPEC_STAGE_PROPOSE (1u << 0)
#define HIPENGINE_NATIVE_SPEC_STAGE_VERIFY (1u << 1)
#define HIPENGINE_NATIVE_SPEC_STAGE_ACCEPT (1u << 2)
#define HIPENGINE_NATIVE_SPEC_STAGE_COMMIT (1u << 3)
#define HIPENGINE_NATIVE_SPEC_STAGE_UPDATE_CURSORS (1u << 4)

#define HIPENGINE_NATIVE_SPEC_MODE_CHAIN 1u
#define HIPENGINE_NATIVE_SPEC_MODE_TREE 2u

#define HIPENGINE_NATIVE_SPEC_STATUS_CREATED 0u
#define HIPENGINE_NATIVE_SPEC_STATUS_SUBMITTED 1u
#define HIPENGINE_NATIVE_SPEC_STATUS_RUNNING 2u
#define HIPENGINE_NATIVE_SPEC_STATUS_COMPLETE 3u
#define HIPENGINE_NATIVE_SPEC_STATUS_YIELDED 4u
#define HIPENGINE_NATIVE_SPEC_STATUS_CANCELLED 5u
#define HIPENGINE_NATIVE_SPEC_STATUS_DEADLINE_EXCEEDED 6u
#define HIPENGINE_NATIVE_SPEC_STATUS_FAILED 7u

#define HIPENGINE_NATIVE_SPEC_ERROR_NONE 0u
#define HIPENGINE_NATIVE_SPEC_ERROR_ABI_MISMATCH 1u
#define HIPENGINE_NATIVE_SPEC_ERROR_INVALID_CONTROL 2u
#define HIPENGINE_NATIVE_SPEC_ERROR_UNSUPPORTED_SHAPE 3u
#define HIPENGINE_NATIVE_SPEC_ERROR_CANCELLED 4u
#define HIPENGINE_NATIVE_SPEC_ERROR_DEADLINE_EXCEEDED 5u
#define HIPENGINE_NATIVE_SPEC_ERROR_KERNEL_LAUNCH 6u
#define HIPENGINE_NATIVE_SPEC_ERROR_INTERNAL 7u

#define HIPENGINE_NATIVE_SPEC_DTYPE_INT32 1u
#define HIPENGINE_NATIVE_SPEC_DTYPE_INT64 2u
#define HIPENGINE_NATIVE_SPEC_DTYPE_FP16 3u
#define HIPENGINE_NATIVE_SPEC_DTYPE_BF16 4u
#define HIPENGINE_NATIVE_SPEC_DTYPE_FP32 5u
#define HIPENGINE_NATIVE_SPEC_DTYPE_INT8 6u
#define HIPENGINE_NATIVE_SPEC_DTYPE_INT8_PER_TOKEN_HEAD 7u

typedef struct HipengineNativeSpecCycleControlV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t stage_mask;
  uint32_t mode;
  uint64_t cycle_id;
  uint64_t transaction_id;
  uint64_t stream;
  uint64_t deadline_ns;

  uint32_t request_count;
  uint32_t request_capacity;
  uint32_t row_count;
  uint32_t active_row_count;
  uint32_t row_capacity;
  uint32_t candidate_count;
  uint32_t active_candidate_count;
  uint32_t candidate_capacity;
  uint32_t candidate_budget;
  uint32_t span_count;
  uint32_t span_capacity;
  uint32_t max_live_count;
  uint32_t context_bucket;
  uint32_t hidden_size;
  uint32_t hidden_row_capacity;
  uint32_t output_stride;
  uint32_t metadata_dtype;
  uint32_t hidden_dtype;
  uint32_t kv_dtype;

  uint64_t metadata_request_ids;
  uint64_t metadata_token_ids;
  uint64_t metadata_positions;
  uint64_t metadata_parent_rows;
  uint64_t metadata_draft_depths;
  uint64_t metadata_row_to_request;
  uint64_t metadata_active_mask;
  uint64_t metadata_candidate_counts;
  uint64_t metadata_remaining_decode;

  uint64_t kv_base_offsets;
  uint64_t kv_live_counts;
  uint64_t kv_token_positions;
  uint64_t kv_evict_mask;
  uint64_t kv_request_ids;
  uint64_t kv_row_positions;
  uint64_t kv_k_scale;
  uint64_t kv_v_scale;
  uint64_t kv_key_cache;
  uint64_t kv_value_cache;

  uint64_t state_hidden_seed_in;
  uint64_t state_proposal_state;
  uint64_t state_candidate_token_ids;
  uint64_t state_candidate_probabilities;
  uint64_t state_draft_key_cache;
  uint64_t state_draft_value_cache;
  uint64_t state_hidden_seed_rows;
  uint64_t state_linear_state_rows;
  uint64_t state_linear_state_dst;
  uint64_t state_key_rows;
  uint64_t state_value_rows;
  uint64_t state_hidden_seed_dst;

  uint64_t output_target_logits;
  uint64_t output_target_top1;
  uint64_t output_accepted_counts;
  uint64_t output_commit_rows;
  uint64_t output_commit_tokens;
  uint64_t output_commit_positions;
  uint64_t output_next_tokens;
  uint64_t output_full_accept;
  uint64_t output_committed_output_ids;
  uint64_t output_committed_output_lengths;
  uint64_t output_output_ids;
  uint64_t output_output_lengths;
  uint64_t output_last_positions;
  uint64_t output_context_lengths;
  uint64_t output_cancel_flag;
} HipengineNativeSpecCycleControlV1;

typedef struct HipengineNativeSpecCycleResultV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t error_code;
  uint32_t completed_stage_mask;
  uint32_t failed_stage;
  uint32_t request_count;
  uint32_t visible_output_count;
  uint64_t cycle_id;
  uint64_t transaction_id;
  int64_t backend_error_code;
  uint64_t reserved;
} HipengineNativeSpecCycleResultV1;

#if defined(__cplusplus)
static_assert(sizeof(HipengineNativeSpecCycleControlV1) == 496, "native cycle control ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleControlV1, cycle_id) == 16, "cycle_id ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleControlV1, request_count) == 48, "shape ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleControlV1, metadata_request_ids) == 128, "pointer ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleControlV1, state_hidden_seed_in) == 280, "state ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleControlV1, output_target_logits) == 376, "output ABI drift");
static_assert(sizeof(HipengineNativeSpecCycleResultV1) == 64, "native cycle result ABI drift");
static_assert(offsetof(HipengineNativeSpecCycleResultV1, cycle_id) == 32, "result ABI drift");
#endif
