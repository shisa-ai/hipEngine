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
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence
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


def _gfx1151_q4_k_m_session(capacity: int) -> Qwen35GGUFResidentSession:
    """The shipped gfx1151 dense Q4_K_M owner, which carries no row cap.

    gfx1151's ``GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES`` has only a
    ``MOSTLY_Q4_K_S`` key, so this file type resolves to no clamp. The prefill
    config is the real tuned one for the sequence length (the gfx1151 arch
    profile changes linear/MoE chunks to 256 rows), not the untuned defaults.
    """

    session = object.__new__(Qwen35GGUFResidentSession)
    object.__setattr__(
        session,
        "runner",
        type("R", (), {"weights": _FakeWeights(), "backend": "hip_gfx1151"})(),
    )
    object.__setattr__(session, "prefill_config", _tuned_config(capacity))
    object.__setattr__(session, "prefill_chunk_size", 0)
    return session


def _tuned_config(capacity: int) -> PrefillConfig:
    config, _ = resolve_prefill_config_for_sequence(
        PrefillConfig(),
        max_sequence_length=capacity,
        total_memory_bytes=128 * 1024**3,
        target_arch="gfx1151",
    )
    return config


@pytest.mark.parametrize("capacity", [1_024, 4_096, 8_192, 32_768])
@pytest.mark.parametrize("tail", [0, 1, 3])
def test_uncapped_chunk_never_exceeds_allocated_scratch_rows(
    capacity: int,
    tail: int,
) -> None:
    """Q4_K_M on gfx1151 is safe without a clamp (section I safety question).

    ``for_chunk`` writes ``rows`` entries into buffers sized by
    ``_prefill_scratch_rows``, so the safety invariant is
    ``chunk_rows <= allocated_rows``. Without a cap both sides derive from the
    same chunk resolution, so the invariant holds at every boundary. The
    2026-09-09 overrun required a cap that shrank the allocation below the
    natural chunk; with no cap there is nothing for a chunk site to ignore.
    """

    session = _gfx1151_q4_k_m_session(capacity)
    assert session._dense_prefill_scratch_row_cap(capacity) is None, (
        "this test guards the uncapped gfx1151 route"
    )
    allocated = session._prefill_scratch_rows(capacity)
    for rows in (capacity, max(1, capacity - tail), 1, 2):
        linear = session._linear_prefill_layer_chunk_size(rows)
        full = session._full_attention_prefill_layer_chunk_size(rows)
        assert max(linear, full) <= allocated, (
            f"capacity={capacity} rows={rows}: chunk max({linear}, {full}) "
            f"exceeds the {allocated} allocated scratch rows"
        )
        assert max(linear, full) <= rows


def test_gfx1151_scratch_allocation_tracks_the_auto_query_chunk() -> None:
    """The uncapped allocation is exactly the largest selectable chunk.

    This is the substantive half of the safety argument: gfx1151 sizes the
    bulk scratch by the same 4096-row auto query chunk it later selects, so the
    uncapped route is self-consistent rather than merely unbounded.
    """

    for capacity in (4_096, 8_192, 32_768):
        session = _gfx1151_q4_k_m_session(capacity)
        allocated = session._prefill_scratch_rows(capacity)
        full = session._full_attention_prefill_layer_chunk_size(capacity)
        assert allocated == full == 4_096, (
            f"capacity={capacity}: allocated={allocated} full={full}"
        )


def test_gfx1151_scratch_row_cap_is_absent_for_q4_k_m() -> None:
    """Record the shipped gfx1151 policy shape this decision rests on."""

    from hipengine.kernels.backends import backend_package_capability

    policies = backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES",
        {},
    )
    assert (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M") not in policies
    assert session_cap("hip_gfx1151") is None
    assert session_cap("hip_gfx1100") == 1_024


def test_gfx1100_clamp_is_load_bearing_and_gfx1151_has_nothing_to_ignore() -> None:
    """Contrast the two backends: the clamp matters only where it shrinks.

    On gfx1100 the cap pins the allocation to 1,024 rows while the auto policy
    resolves a 4,096-row query chunk, so every chunk site must honor the cap or
    ``for_chunk`` overruns the 1,024-row metadata buffers (the 2026-09-09
    crash). On gfx1151 there is no cap, so the allocation is the 4,096-row
    chunk itself and no site can ignore it.
    """

    gfx1100 = object.__new__(Qwen35GGUFResidentSession)
    object.__setattr__(
        gfx1100,
        "runner",
        type("R", (), {"weights": _FakeWeights(), "backend": "hip_gfx1100"})(),
    )
    object.__setattr__(gfx1100, "prefill_config", _tuned_config(8_192))
    object.__setattr__(gfx1100, "prefill_chunk_size", 0)
    capped_allocation = gfx1100._prefill_scratch_rows(8_192)
    assert capped_allocation == 1_024
    # The uncapped natural resolution for this sequence is the 4,096-row auto
    # query chunk; that is the value the clamp has to bring down to 1,024.
    assert min(8_192, gfx1100.prefill_config.full_attn_query_chunk_size) == 4_096
    assert gfx1100._full_attention_prefill_layer_chunk_size(8_192) == 1_024
    assert _gfx1151_q4_k_m_session(8_192)._prefill_scratch_rows(8_192) == 4_096


def session_cap(backend: str) -> int | None:
    session = object.__new__(Qwen35GGUFResidentSession)
    object.__setattr__(
        session,
        "runner",
        type("R", (), {"weights": _FakeWeights(), "backend": backend})(),
    )
    object.__setattr__(session, "prefill_config", _tuned_config(32_768))
    object.__setattr__(session, "prefill_chunk_size", 0)
    return session._dense_prefill_scratch_row_cap(32_768)
