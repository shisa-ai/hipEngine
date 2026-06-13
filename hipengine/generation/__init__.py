"""Generation registries and built-in torch-free generation paths."""

from hipengine.generation.batch_scheduler import (
    BatchGenerateRequest,
    CompactPromptBucket,
    CompactPromptSlab,
    CompletedRequest,
    GeneratedToken,
    GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS,
    GraphBucketCache,
    GraphBucketStats,
    PerRowSamplingParams,
    RequestObservability,
    ResidentBatchScheduler,
    SamplerParamsBlock,
    SpeculativeCommitPlan,
    SpeculativeStateCommitPlan,
    SpeculativeVerifyBufferPlan,
    SpeculativeVerifyPlan,
    SpeculativeVerifyWork,
)
from hipengine.generation.engine_loop import (
    PREFILL_DECODE_POLICIES,
    EngineLoopConfig,
    EngineLoopEvent,
    EngineLoopRunner,
    ResidentEngineLoop,
    SubmitPollTextGenerator,
    add_engine_loop_config_args,
    engine_loop_config_from_args,
    engine_loop_config_from_env,
)
from hipengine.generation.registry import (
    DuplicateGeneratorError,
    GenerationKey,
    GenerationRequest,
    MissingGeneratorError,
    TextGenerator,
    clear_generation_registry_for_tests,
    register_text_generator,
    registered_text_generators,
    resolve_text_generator,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SampleResult,
    SamplerPlan,
    SamplingMode,
    active_processor_names,
    derive_row_seed,
    normalize_logit_bias_pairs,
    plan_sampler,
    row_seed_for_index,
    select_token,
    validate_sampling_params,
)

_BUILTINS_REGISTERED = False


def register_builtin_generators() -> None:
    """Register built-in generation paths lazily.

    Importing ``hipengine`` must remain light and torch-free, so model-specific runtime
    generation modules are imported only when the public API needs generation.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation import qwen35_gguf as _qwen35_gguf  # noqa: F401
    from hipengine.generation import qwen35_paro as _qwen35_paro  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "BatchGenerateRequest",
    "CompactPromptBucket",
    "CompactPromptSlab",
    "CompletedRequest",
    "DuplicateGeneratorError",
    "EngineLoopConfig",
    "EngineLoopEvent",
    "EngineLoopRunner",
    "GeneratedToken",
    "GenerationKey",
    "GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS",
    "GraphBucketCache",
    "GraphBucketStats",
    "GenerationRequest",
    "MissingGeneratorError",
    "PREFILL_DECODE_POLICIES",
    "PerRowSamplingParams",
    "RequestObservability",
    "ResidentBatchScheduler",
    "RowSamplingState",
    "SampleResult",
    "SamplerParamsBlock",
    "SamplerPlan",
    "SamplingMode",
    "ResidentEngineLoop",
    "SubmitPollTextGenerator",
    "SpeculativeCommitPlan",
    "SpeculativeStateCommitPlan",
    "SpeculativeVerifyBufferPlan",
    "SpeculativeVerifyPlan",
    "SpeculativeVerifyWork",
    "TextGenerator",
    "active_processor_names",
    "add_engine_loop_config_args",
    "clear_generation_registry_for_tests",
    "derive_row_seed",
    "engine_loop_config_from_args",
    "engine_loop_config_from_env",
    "normalize_logit_bias_pairs",
    "plan_sampler",
    "register_builtin_generators",
    "register_text_generator",
    "registered_text_generators",
    "resolve_text_generator",
    "row_seed_for_index",
    "select_token",
    "validate_sampling_params",
]
