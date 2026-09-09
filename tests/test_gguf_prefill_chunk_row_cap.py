"""Prefill layer chunk sizes must respect the dense scratch row-cap policy.

Regression guard for the pure-INT8 comparison protocol crash
(`copy size 32768 exceeds device buffer size 8192`): the gfx1100 dense
H5120 Q4_K_M scratch row-cap policy pins the bulk prefill scratch at
1,024 rows for capacities above 1K, but the auto chunk policy resolves
full-attention query chunks to 4,096 rows below the 52K low-memory
threshold, so ``for_chunk`` overflows the metadata buffers. The chunk
resolution must clamp to the scratch row cap.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


class _FakeConfig:
    is_moe = False
    ssm_conv_kernel = 4


class _FakeWeights:
    config = _FakeConfig()
    geometry = QWEN35_DENSE_H5120_GEOMETRY
    file_type_name = "MOSTLY_Q4_K_M"


class _FakeRunner:
    weights = _FakeWeights()
    backend = "hip_gfx1100"


def _session_with_fake_runner() -> Qwen35GGUFResidentSession:
    session = object.__new__(Qwen35GGUFResidentSession)
    object.__setattr__(session, "runner", _FakeRunner())
    object.__setattr__(session, "prefill_config", PrefillConfig())
    object.__setattr__(session, "prefill_chunk_size", 0)
    return session


@pytest.mark.parametrize("tokens", [2_048, 8_192, 16_384, 32_768])
def test_full_attention_chunk_clamped_to_scratch_row_cap(tokens: int) -> None:
    session = _session_with_fake_runner()
    chunk = session._full_attention_prefill_layer_chunk_size(tokens)
    assert chunk <= 1_024, (
        f"full-attention chunk {chunk} exceeds the 1,024-row dense scratch cap "
        "for the H5120 MOSTLY_Q4_K_M geometry"
    )


@pytest.mark.parametrize("tokens", [2_048, 8_192, 16_384, 32_768])
def test_linear_chunk_clamped_to_scratch_row_cap(tokens: int) -> None:
    session = _session_with_fake_runner()
    chunk = session._linear_prefill_layer_chunk_size(tokens)
    assert chunk <= 1_024, (
        f"linear-attention chunk {chunk} exceeds the 1,024-row dense scratch cap "
        "for the H5120 MOSTLY_Q4_K_M geometry"
    )


def test_chunk_unclamped_without_policy_identity() -> None:
    """A geometry without a row-cap policy keeps the auto chunk sizes."""

    class _UncappedWeights(_FakeWeights):
        file_type_name = "Q4_0"

    session = object.__new__(Qwen35GGUFResidentSession)
    object.__setattr__(session, "runner", type("R", (), {"weights": _UncappedWeights(), "backend": "hip_gfx1100"})())
    object.__setattr__(session, "prefill_config", PrefillConfig())
    object.__setattr__(session, "prefill_chunk_size", 0)
    chunk = session._full_attention_prefill_layer_chunk_size(16_384)
    assert chunk == 16_384, (
        "uncapped geometry should keep the untuned total-row chunk "
        "(the auto tuning applies at session construction)"
    )
