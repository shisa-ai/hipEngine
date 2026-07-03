from __future__ import annotations

from hipengine.kernels.cpu_reference import register_cpu_reference_kernels
from hipengine.speculative import (
    DEFAULT_DRAFT_SELECTION,
    DEFAULT_DRAFT_TOPK,
    DEFAULT_DRAFT_TOPK_KERNEL,
    DEFAULT_NEXTN_ATTN_DECODE_LAYER,
    DEFAULT_NEXTN_ATTN_DECODE_VARIANT,
    DEFAULT_NEXTN_KV_WRITE_LAYER,
    DEFAULT_NEXTN_KV_WRITE_VARIANT,
    GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
    GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
    GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
    GGUF_MTP_METRICS_CONTRACT_READY,
    Qwen35GGUFMTPAcceptStep,
    Qwen35GGUFMTPAcceptStepMetrics,
    Qwen35GGUFMTPTop1AcceptSpec,
    Qwen35GGUFMTPContext,
    Qwen35GGUFMTPKVLiveSpansPlan,
    Qwen35GGUFMTPPerformanceReadiness,
    Qwen35GGUFMTPRuntimeKernelPlan,
    Qwen35GGUFMTPSeedRow,
    Qwen35GGUFMTPVerificationMetrics,
    Qwen35GGUFMTPVerificationResult,
)
from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPAcceptStep as ModuleAcceptStep
from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPAcceptStepMetrics as ModuleAcceptStepMetrics
from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPTop1AcceptSpec as ModuleTop1AcceptSpec
from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPContext as ModuleContext


def test_gguf_mtp_contracts_are_exported_from_speculative_package() -> None:
    register_cpu_reference_kernels(replace=True)

    assert Qwen35GGUFMTPContext is ModuleContext
    assert Qwen35GGUFMTPAcceptStep is ModuleAcceptStep
    assert Qwen35GGUFMTPAcceptStepMetrics is ModuleAcceptStepMetrics
    assert Qwen35GGUFMTPTop1AcceptSpec is ModuleTop1AcceptSpec
    assert DEFAULT_DRAFT_TOPK == 10
    assert DEFAULT_DRAFT_SELECTION == "greedy_top1_from_topk"
    assert DEFAULT_DRAFT_TOPK_KERNEL == ("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h")
    assert DEFAULT_NEXTN_KV_WRITE_LAYER == "paged_kv_write"
    assert DEFAULT_NEXTN_KV_WRITE_VARIANT == "mixed_bf16_spans"
    assert GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE == "computed"
    assert DEFAULT_NEXTN_ATTN_DECODE_LAYER == "paged_attn_decode"
    assert DEFAULT_NEXTN_ATTN_DECODE_VARIANT == "bf16_context_spans"

    seed = Qwen35GGUFMTPSeedRow(token_id=1, position=2, hidden_ptr=0x1000, hidden_size=8)
    plan = Qwen35GGUFMTPKVLiveSpansPlan.from_draft_batch(
        Qwen35GGUFMTPContext(target_session=object()).build_draft_batch(
            request_id=0,
            token_ids=(5,),
            seed_rows=(seed,),
        ),
        block_size=4,
    )
    metrics = Qwen35GGUFMTPVerificationMetrics.from_results(
        (
            Qwen35GGUFMTPVerificationResult(
                proposed_token_ids=(5,),
                target_token_ids=(5,),
                n_accepted=1,
                verify_seed_count=2,
                reseed=Qwen35GGUFMTPSeedRow(token_id=5, position=3, hidden_ptr=0x2000, hidden_size=8),
            ),
        ),
        output_token_count=1,
    )
    runtime_plan = Qwen35GGUFMTPRuntimeKernelPlan.from_registry(backend="hip_gfx1100")
    readiness = Qwen35GGUFMTPPerformanceReadiness.from_gate_inputs(
        parity_precheck=True,
        draft_budget_precheck=True,
        draft_sampling_contract_precheck=True,
        hidden_seed_contract_precheck=True,
        exactness_gate="passed",
        kvlivespans_paged_cache_smoke=True,
        llamacpp_trace_budget_coverage=GGUF_MTP_FULL_TRACE_BUDGET_COVERAGE,
        accepted_per_draft_status=GGUF_MTP_ACCEPTED_DRAFT_COMPARABLE,
        accepted_per_output_status=GGUF_MTP_ACCEPTED_OUTPUT_COMPARABLE,
        native_runtime_kernels_ready=True,
        optimization_kernels_ready=True,
        metrics_contract_status=GGUF_MTP_METRICS_CONTRACT_READY,
    )

    assert plan.as_dict()["token_positions"] == [3]
    assert metrics.as_dict()["kind"] == "hipengine_gguf_mtp_verification_metrics"
    assert metrics.as_dict()["budget_label"] == "B1"
    assert metrics.as_dict()["denominators"] == {
        "accepted_per_draft": "accepted_token_count / draft_token_count",
        "accepted_per_output": "accepted_token_count / output_token_count",
    }
    assert runtime_plan.exactness_oracles_ready is True
    assert readiness.ready is True
