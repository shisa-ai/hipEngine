"""Generation registries and built-in torch-free generation paths."""

from hipengine.generation.batch_scheduler import (
    BatchGenerateRequest,
    CompletedRequest,
    GeneratedToken,
    GraphBucketCache,
    GraphBucketStats,
    ResidentBatchScheduler,
    SpeculativeCommitPlan,
    SpeculativeVerifyBufferPlan,
    SpeculativeVerifyPlan,
    SpeculativeVerifyWork,
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

_BUILTINS_REGISTERED = False


def register_builtin_generators() -> None:
    """Register built-in generation paths lazily.

    Importing ``hipengine`` must remain light and torch-free, so model-specific runtime
    generation modules are imported only when the public API needs generation.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation import qwen35_paro as _qwen35_paro  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "BatchGenerateRequest",
    "CompletedRequest",
    "DuplicateGeneratorError",
    "GeneratedToken",
    "GenerationKey",
    "GraphBucketCache",
    "GraphBucketStats",
    "GenerationRequest",
    "MissingGeneratorError",
    "ResidentBatchScheduler",
    "SpeculativeCommitPlan",
    "SpeculativeVerifyBufferPlan",
    "SpeculativeVerifyPlan",
    "SpeculativeVerifyWork",
    "TextGenerator",
    "clear_generation_registry_for_tests",
    "register_builtin_generators",
    "register_text_generator",
    "registered_text_generators",
    "resolve_text_generator",
]
