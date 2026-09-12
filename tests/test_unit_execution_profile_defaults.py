"""The shipped default execution profile is ``production`` where it is certified.

An omitted profile resolves to ``production`` only for a combination with both a
registered strict plan and a certified production plan. Everything else keeps
the migration path, which is not a fourth named profile.
"""

from __future__ import annotations

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    resolve_default_execution_profile,
)

CERTIFIED_COMBINATIONS = (
    ("qwen3_5_gguf", "hip_gfx1100", "gguf_q4_k_m"),
    ("qwen3_5_gguf", "hip_gfx1151", "gguf_q4_k_m"),
    ("qwen3_5_moe_gguf", "hip_gfx1100", "gguf_q4_k_m"),
    ("qwen3_5_moe_gguf", "hip_gfx1151", "gguf_q4_k_m"),
    ("qwen3_5_moe_paro", "hip_gfx1100", "w4_paro"),
    ("qwen4_exp_gguf", "hip_gfx1151", "gguf_q4_k_m"),
    ("qwen4_exp_gguf", "hip_gfx1151", "gguf_ud_q4_k_xl"),
)

UNCERTIFIED_COMBINATIONS = (
    ("laguna_gguf", "hip_gfx1151", "gguf_q4_k_m"),
    ("maple", "hip_gfx1100", "maple_ternary2"),
    ("evie_4p5b", "hip_gfx1151", "fp16"),
    ("timesfm_2p5_200m", "hip_gfx1151", "fp32"),
    ("moonshine_asr", "hip_gfx1151", "fp16"),
)


def _register_shipped_plans() -> None:
    import hipengine.speculative  # noqa: F401  (registers the PARO gfx1100 plans)
    from hipengine.generation.qwen36_gguf_gfx1100_profiles import (
        register_qwen36_dense_gguf_gfx1100_profiles,
        register_qwen36_moe_gguf_gfx1100_profiles,
    )
    from hipengine.generation.qwen36_gguf_profiles import (
        register_qwen36_gguf_gfx1151_profiles,
    )
    from hipengine.generation.qwen38_gguf_profiles import (
        register_qwen38_gguf_gfx1151_profiles,
    )
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )

    register_qwen36_dense_gguf_gfx1100_profiles()
    register_qwen36_moe_gguf_gfx1100_profiles()
    register_qwen36_gguf_gfx1151_profiles()
    register_qwen38_gguf_gfx1151_profiles()
    register_qwen4_exp_gfx1151_profiles()


# Import time, not a fixture: ``tests/conftest.py`` snapshots the profile-plan
# registry after collection and restores that baseline after every test, so a
# module-scoped fixture would be undone by the first test's teardown.
_register_shipped_plans()


@pytest.mark.parametrize(("model", "backend", "quant"), CERTIFIED_COMBINATIONS)
def test_certified_combination_defaults_to_production(
    model: str, backend: str, quant: str
) -> None:
    assert (
        resolve_default_execution_profile(model=model, backend=backend, quant=quant)
        is ExecutionProfile.PRODUCTION
    )


@pytest.mark.parametrize(("model", "backend", "quant"), UNCERTIFIED_COMBINATIONS)
def test_uncertified_combination_keeps_the_migration_path(
    model: str, backend: str, quant: str
) -> None:
    assert (
        resolve_default_execution_profile(model=model, backend=backend, quant=quant)
        is None
    )
