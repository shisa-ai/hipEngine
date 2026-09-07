"""CPU-only artifact qualification for the existing plain GGUF profile plans.

Operation/physical ABI admission is independent and still runs in the loader
and runtime. These named plans do not constitute a generic strict AR plan.
No unknown or UD manifest may borrow their numerical evidence, even if generic
AR admission succeeds. Explicit developer environment overrides are not profile
qualification and are never examined here as evidence of user consent.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hipengine.execution_profiles import RuntimeProfileKey
from hipengine.loading.gguf import GGUFModelInfo


def qualify_plain_gguf_profile(key: RuntimeProfileKey, context: Mapping[str, Any]) -> object:
    from hipengine.loading.qwen35_gguf_admission import (
        qwen35_gguf_artifact_identity_from_info,
    )

    info = context.get('weight_index')
    if not isinstance(info, GGUFModelInfo):
        raise ValueError('GGUF execution-profile qualification requires actual weight_index metadata')
    path = context.get('model_path')
    if path is not None and Path(path).resolve() != info.path.resolve():
        raise ValueError('GGUF execution-profile qualification model_path differs from weight_index')
    fingerprint, preset = qwen35_gguf_artifact_identity_from_info(info)
    if preset is not None:
        raise ValueError(
            f'GGUF execution-profile qualification refused {preset!r}: '
            f'{key.profile.value} plain profile is not certified for this artifact; '
            'no artifact-qualified generic strict profile plan is registered'
        )
    # These are the existing dense-27B / MoE-35B Q4_K_M profile families,
    # not every pinned control. Small-model and Q4_K_S controls retain their
    # independent policies; their inclusion in admission is not certification
    # of these verifier-shaped profile selections.
    supported = {
        'qwen3_5_gguf': ('qwen35', 5120),
        'qwen3_5_moe_gguf': ('qwen35moe', 2048),
    }
    architecture, hidden = supported.get(key.model, ('', 0))
    stamp_quant = 'gguf_' + str(info.file_type_name or '').removeprefix('MOSTLY_').lower()
    if (info.architecture != architecture
            or info.metadata.get(f'{architecture}.embedding_length') != hidden
            or stamp_quant != key.quant):
        raise ValueError('GGUF execution-profile qualification does not cover this model/quant scope')
    return fingerprint, preset
