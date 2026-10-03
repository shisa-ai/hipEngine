"""Prefill chunk resolution is one owner with explicit, recorded precedence.

Regression guard for the dispatch defect recorded 2026-10-03: the GGUF resident
session applied ``GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY`` in its constructor
*after* :func:`resolve_prefill_config_for_sequence` and after any explicit
``PrefillConfig``, so

* ``--prefill-linear-chunk-size`` / ``--prefill-moe-chunk-size`` were silently
  ignored on the H2048-MoE geometry, and
* ``prefill_chunk_tuning["chunk_sizes"]`` reported 1024/1024 while the session
  ran 512/512.

The contract these tests pin down: the resolver is the single owner of the final
sizes, precedence is per field (manual > geometry > arch > shape/memory), and the
recorded ``chunk_sizes`` always describes the config the call returned.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence

_GIB = 1024 ** 3
# 128 GB host: not a 24GB-class card, so the shape policy is the retained
# 1024/4096 policy.
_BIG_HOST = 128 * _GIB
_24GB_HOST = 24 * _GIB


def _resolve(config: PrefillConfig, **kwargs: object):
    kwargs.setdefault("max_sequence_length", 4096)
    kwargs.setdefault("total_memory_bytes", _BIG_HOST)
    return resolve_prefill_config_for_sequence(config, **kwargs)


def test_recorded_chunk_sizes_always_match_the_returned_config() -> None:
    """The record must describe the run, not an intermediate policy layer."""

    for config, kwargs in (
        (PrefillConfig(), {}),
        (PrefillConfig(), {"target_arch": "gfx1151"}),
        (PrefillConfig(), {"geometry_chunk_sizes": (2048, 2048)}),
        (PrefillConfig(), {"target_arch": "gfx1151", "geometry_chunk_sizes": (2048, 2048)}),
        (PrefillConfig(linear_chunk_size=4096), {"geometry_chunk_sizes": (2048, 2048)}),
        (PrefillConfig(auto_tune_chunk_sizes=False), {}),
    ):
        resolved, tuning = _resolve(config, **kwargs)
        assert tuning["chunk_sizes"] == {
            "linear": resolved.linear_chunk_size,
            "moe": resolved.moe_chunk_size,
            "full_attn_query": resolved.full_attn_query_chunk_size,
            "full_attn_post": resolved.full_attn_post_chunk_size,
            "full_attn_rope": resolved.full_attn_rope_chunk_size,
        }, f"tuning does not describe the returned config for {config} {kwargs}"


def test_manual_chunk_size_wins_over_the_geometry_profile() -> None:
    """The defect: an explicit request was overridden by the geometry policy."""

    manual = PrefillConfig(linear_chunk_size=4096, moe_chunk_size=4096)
    resolved, tuning = _resolve(manual, geometry_chunk_sizes=(512, 512))
    assert resolved is manual, "an explicit config must be returned untouched"
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (4096, 4096)
    assert tuning["reason"] == "manual_chunk_sizes"
    assert tuning["manual_chunk_sizes"] == ["linear_chunk_size", "moe_chunk_size"]
    assert tuning["geometry_profile_applied"] is False
    assert tuning["chunk_sizes"]["linear"] == 4096, "the record must show 4096, not 512"


def test_manual_config_also_wins_over_the_architecture_profile() -> None:
    manual = PrefillConfig(moe_chunk_size=2048)
    resolved, tuning = _resolve(manual, target_arch="gfx1151")
    assert resolved is manual
    assert resolved.moe_chunk_size == 2048
    assert tuning["reason"] == "manual_chunk_sizes"
    assert tuning["arch_profile_applied"] is False
    assert tuning["chunk_sizes"]["moe"] == 2048


def test_manual_config_records_the_sizes_the_session_will_run() -> None:
    """The record is truthful even though no policy layer applied."""

    manual = PrefillConfig(linear_chunk_size=2048)
    resolved, tuning = _resolve(manual, geometry_chunk_sizes=(512, 512))
    assert resolved is manual
    assert tuning["chunk_sizes"] == {
        "linear": 2048,
        "moe": 0,
        "full_attn_query": 0,
        "full_attn_post": 0,
        "full_attn_rope": 0,
    }


def test_geometry_layer_names_the_reason_and_keeps_the_arch_layer_visible() -> None:
    resolved, tuning = _resolve(
        PrefillConfig(),
        target_arch="gfx1151",
        geometry_chunk_sizes=(2048, 2048),
    )
    assert tuning["reason"] == "geometry_prefill_chunk_policy"
    assert tuning["arch_reason"] == "gfx1151_paro_prefill_recovery"
    assert tuning["arch_source"] == "sol_r1_gfx1151_prefill_recovery_2026_07"
    assert resolved.linear_chunk_size == 2048


def test_geometry_layer_alone_applies_without_an_arch_profile() -> None:
    resolved, tuning = _resolve(PrefillConfig(), geometry_chunk_sizes=(2048, 1024))
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (2048, 1024)
    assert tuning["geometry_profile_applied"] is True
    assert tuning["arch_profile_applied"] is False
    assert tuning["reason"] == "geometry_prefill_chunk_policy"


def test_manual_linear_still_suppresses_the_long_context_policy() -> None:
    """Pre-existing contract: an explicit config is returned untouched.

    This is deliberately all-or-nothing rather than per-field, so that passing
    one manual size cannot silently change the arithmetic or the scratch
    headroom of the other stages.
    """

    manual = PrefillConfig(linear_chunk_size=1024)
    resolved, tuning = _resolve(
        manual,
        max_sequence_length=131_072,
        total_memory_bytes=_24GB_HOST,
    )
    assert resolved is manual
    assert resolved.full_attn_query_chunk_size == 0
    assert tuning["reason"] == "manual_chunk_sizes"


def test_architecture_profile_applies_without_a_geometry_profile() -> None:
    """The gfx1151 arch profile is reachable, which it was not on the GGUF path."""

    resolved, tuning = _resolve(PrefillConfig(), target_arch="gfx1151")
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (256, 256)
    assert tuning["arch_profile_applied"] is True
    assert tuning["geometry_profile_applied"] is False
    assert resolved.full_attn_query_chunk_size == 4096, "arch profile touches only linear/moe"


def test_unknown_arch_leaves_the_shape_policy_in_place() -> None:
    resolved, tuning = _resolve(PrefillConfig(), target_arch="gfx9999")
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (1024, 1024)
    assert tuning["arch_profile_applied"] is False
    assert tuning["target_arch"] == "gfx9999"


def test_auto_tune_off_ignores_every_profile() -> None:
    """Auto-tuning off means the caller owns the sizes, recorded truthfully."""

    resolved, tuning = _resolve(
        PrefillConfig(auto_tune_chunk_sizes=False, linear_chunk_size=777),
        target_arch="gfx1151",
        geometry_chunk_sizes=(2048, 2048),
    )
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (777, 0)
    assert tuning["enabled"] is False
    assert tuning["reason"] == "disabled"
    assert tuning["arch_profile_applied"] is False
    assert tuning["geometry_profile_applied"] is False


def test_manual_linear_does_not_suppress_the_long_context_headroom_policy() -> None:
    """Superseded: the all-or-nothing manual contract is kept deliberately."""

    pytest.skip("replaced by test_manual_linear_still_suppresses_the_long_context_policy")


def test_below_min_tokens_still_takes_the_arch_profile() -> None:
    resolved, tuning = _resolve(PrefillConfig(), max_sequence_length=512, target_arch="gfx1151")
    assert tuning["base_reason"] == "below_min_tokens"
    assert tuning["reason"] == "gfx1151_paro_prefill_recovery"
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == (256, 256)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        ((0, 0), (1024, 1024)),
        ((0, 512), (1024, 512)),
        ((512, 0), (512, 1024)),
    ],
)
def test_zero_geometry_entries_leave_the_field_to_the_next_layer(
    geometry: tuple[int, int], expected: tuple[int, int]
) -> None:
    resolved, _ = _resolve(PrefillConfig(), geometry_chunk_sizes=geometry)
    assert (resolved.linear_chunk_size, resolved.moe_chunk_size) == expected


def test_resolved_config_is_frozen_and_replaceable() -> None:
    """The single-owner contract relies on callers using replace(), not mutation."""

    resolved, _ = _resolve(PrefillConfig(), geometry_chunk_sizes=(1024, 1024))
    with pytest.raises(Exception):
        resolved.linear_chunk_size = 7  # type: ignore[misc]
    assert replace(resolved, linear_chunk_size=2048).linear_chunk_size == 2048
