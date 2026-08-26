from __future__ import annotations

from types import SimpleNamespace

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.kernels.registry import resolve
from hipengine.speculative.paro_mtp_profiles import (
    FAST_VERIFIER_CANDIDATE_VARIANT,
    FULL_ATTN_EXACT_SUFFIX_ENV,
    PARO_MTP_BACKEND,
    GDN_EXACT_ENV,
    LINEAR_EXACT_ENV,
    MOE_EXACT_ENV,
    PARO_MTP_CHAIN_ATTN_MODE_ENV,
    PARO_MTP_CONTRACT_ENV,
    PARO_MTP_MODEL,
    PARO_MTP_MODEL_QUANT,
    PARO_MTP_REGISTRY_QUANT,
    PARO_MTP_ROUTE_ENV,
    PROPOSER_LAYER,
    STRICT_VERIFIER_VARIANT,
    TARGET_CONTRACT_VARIANT,
    VERIFIER_LAYER,
    ParoMtpRoute,
    paro_mtp_profiles_registered,
    register_paro_mtp_gfx1100_profiles,
)


def _resolved(profile: str):
    register_paro_mtp_gfx1100_profiles()
    return resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile=profile,
    )


def test_paro_mtp_strict_and_production_manifests_are_registered() -> None:
    assert paro_mtp_profiles_registered()
    strict = _resolved("strict")
    production = _resolved("production")

    strict_by_layer = {row["layer"]: row for row in strict.manifest["selections"]}
    production_by_layer = {row["layer"]: row for row in production.manifest["selections"]}
    assert strict_by_layer[PROPOSER_LAYER]["selected_variant"] == TARGET_CONTRACT_VARIANT
    assert strict_by_layer[VERIFIER_LAYER]["selected_variant"] == STRICT_VERIFIER_VARIANT
    assert production_by_layer[PROPOSER_LAYER]["strict_fallback_variant"] == TARGET_CONTRACT_VARIANT
    assert production_by_layer[VERIFIER_LAYER]["selected_variant"] == FAST_VERIFIER_CANDIDATE_VARIANT
    assert production_by_layer[VERIFIER_LAYER]["strict_fallback_variant"] == STRICT_VERIFIER_VARIANT
    assert production_by_layer[PROPOSER_LAYER]["evidence_artifact"].endswith(
        "2026-08-24-w7900-paro-fast-d24-3run-default.json"
    )
    assert production_by_layer[VERIFIER_LAYER]["evidence_artifact"].endswith(
        "2026-08-24-w7900-paro-fast-d24-3run-default.json"
    )
    assert production.fell_back_to_strict is False
    assert production.strict_manifest_sha256 == strict.manifest_sha256


def test_fast_verifier_route_is_certified_and_selected_by_production() -> None:
    candidate_factory = resolve(
        backend=PARO_MTP_BACKEND,
        layer=VERIFIER_LAYER,
        quant=PARO_MTP_REGISTRY_QUANT,
        variant=FAST_VERIFIER_CANDIDATE_VARIANT,
    )
    candidate = candidate_factory()
    production = _resolved("production")

    assert candidate == ParoMtpRoute(
        component="verifier",
        variant=FAST_VERIFIER_CANDIDATE_VARIANT,
        target_contract=True,
        verifier_profile="fast",
        certified=True,
    )
    assert any(
        row["selected_variant"] == FAST_VERIFIER_CANDIDATE_VARIANT
        for row in production.manifest["selections"]
    )


def _track_bound_route_environment(monkeypatch) -> None:
    for name in (
        PARO_MTP_CONTRACT_ENV,
        PARO_MTP_ROUTE_ENV,
        PARO_MTP_CHAIN_ATTN_MODE_ENV,
        GDN_EXACT_ENV,
        LINEAR_EXACT_ENV,
        MOE_EXACT_ENV,
        FULL_ATTN_EXACT_SUFFIX_ENV,
    ):
        # The runtime binder intentionally mutates os.environ directly. Prime
        # each key through monkeypatch so pytest restores the pre-test process
        # environment after the binder has overwritten it.
        monkeypatch.setenv(name, "pytest-sentinel")


def test_profile_binder_enables_target_contract_and_records_route(monkeypatch) -> None:
    _track_bound_route_environment(monkeypatch)
    resolution = _resolved("production")
    generator = resolution.construct_generator(lambda: SimpleNamespace())

    assert generator.execution_profile == "production"
    assert __import__("os").environ[PARO_MTP_CONTRACT_ENV] == "1"
    route = __import__("os").environ[PARO_MTP_ROUTE_ENV]
    assert TARGET_CONTRACT_VARIANT in route
    assert FAST_VERIFIER_CANDIDATE_VARIANT in route
    env = __import__("os").environ
    assert env[PARO_MTP_CHAIN_ATTN_MODE_ENV] == "decode_batched"
    assert env[GDN_EXACT_ENV] == "0"
    assert env[LINEAR_EXACT_ENV] == "0"
    assert env[MOE_EXACT_ENV] == "0"
    assert env[FULL_ATTN_EXACT_SUFFIX_ENV] == "0"


def test_strict_profile_binder_restores_exact_route(monkeypatch) -> None:
    _track_bound_route_environment(monkeypatch)
    resolution = _resolved("strict")
    resolution.construct_generator(lambda: SimpleNamespace())
    env = __import__("os").environ

    assert env[PARO_MTP_CHAIN_ATTN_MODE_ENV] == "c1_loop"
    assert env[GDN_EXACT_ENV] == "1"
    assert env[LINEAR_EXACT_ENV] == "1"
    assert env[MOE_EXACT_ENV] == "1"
    assert env[FULL_ATTN_EXACT_SUFFIX_ENV] == "0"
