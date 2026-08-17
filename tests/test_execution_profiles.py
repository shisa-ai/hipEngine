from __future__ import annotations

import copy

import pytest

from hipengine.execution_profiles import (
    EXECUTION_PROFILE_SCHEMA_VERSION,
    ExecutionProfile,
    VariantSelection,
    build_variant_manifest,
    manifest_sha256,
    validate_variant_manifest,
)


def _selection(*, selected: str = "online", fallback: str | None = "exact") -> VariantSelection:
    return VariantSelection(
        layer="paged_attn_decode",
        scope="c4/context512",
        selected_variant=selected,
        strict_fallback_variant=fallback,
        evidence_artifact="benchmarks/results/example.json",
    )


def test_variant_manifest_hash_is_canonical_and_order_independent() -> None:
    first = build_variant_manifest(
        profile=ExecutionProfile.PRODUCTION,
        backend="hip_gfx1100",
        model="qwen3_5_moe",
        quant="w4_paro",
        kv_policy="paged_bf16",
        graph_policy="decode-c1-c4",
        selections=(
            _selection(),
            VariantSelection(
                layer="gdn_decode",
                scope="c1-c4",
                selected_variant="peer_wave32",
                strict_fallback_variant="decode_order",
                evidence_artifact="benchmarks/results/gdn.json",
            ),
        ),
    )
    second = build_variant_manifest(
        profile="production",
        backend="hip_gfx1100",
        model="qwen3_5_moe",
        quant="w4_paro",
        kv_policy="paged_bf16",
        graph_policy="decode-c1-c4",
        selections=tuple(reversed(first["selections"])),
    )

    assert first["schema_version"] == EXECUTION_PROFILE_SCHEMA_VERSION
    assert validate_variant_manifest(first) == first
    assert manifest_sha256(first) == manifest_sha256(second)
    assert len(manifest_sha256(first)) == 64


def test_production_manifest_requires_registered_strict_fallback() -> None:
    with pytest.raises(ValueError, match="strict fallback"):
        build_variant_manifest(
            profile=ExecutionProfile.PRODUCTION,
            backend="hip_gfx1100",
            model="qwen3_5_moe",
            quant="w4_paro",
            kv_policy="paged_bf16",
            graph_policy="decode",
            selections=(_selection(fallback=None),),
        )


def test_manifest_validation_rejects_unknown_profile_and_duplicate_scope() -> None:
    manifest = build_variant_manifest(
        profile=ExecutionProfile.STRICT,
        backend="hip_gfx1100",
        model="qwen3_5_moe",
        quant="w4_paro",
        kv_policy="paged_bf16",
        graph_policy="decode",
        selections=(_selection(selected="exact", fallback="exact"),),
    )

    unknown = copy.deepcopy(manifest)
    unknown["execution_profile"] = "relaxed_all"
    with pytest.raises(ValueError, match="execution_profile"):
        validate_variant_manifest(unknown)

    duplicate = copy.deepcopy(manifest)
    duplicate["selections"].append(copy.deepcopy(duplicate["selections"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_variant_manifest(duplicate)
