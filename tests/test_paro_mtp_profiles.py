from __future__ import annotations

from types import SimpleNamespace

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.kernels.registry import resolve
from hipengine.speculative.paro_mtp_profiles import (
    FAST_VERIFIER_CANDIDATE_VARIANT,
    PARO_MTP_BACKEND,
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
    assert production_by_layer[VERIFIER_LAYER]["strict_fallback_variant"] == STRICT_VERIFIER_VARIANT
    assert production_by_layer[PROPOSER_LAYER]["evidence_artifact"].endswith(
        "2026-08-24-w7900-paro-mtp-provider-contract-spike.json"
    )
    assert production.fell_back_to_strict is False
    assert production.strict_manifest_sha256 == strict.manifest_sha256


def test_fast_verifier_route_is_registered_but_not_selected_before_gate() -> None:
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
        certified=False,
    )
    assert all(
        row["selected_variant"] != FAST_VERIFIER_CANDIDATE_VARIANT
        for row in production.manifest["selections"]
    )


def test_profile_binder_enables_target_contract_and_records_route(monkeypatch) -> None:
    monkeypatch.delenv(PARO_MTP_CONTRACT_ENV, raising=False)
    monkeypatch.delenv(PARO_MTP_ROUTE_ENV, raising=False)
    resolution = _resolved("production")
    generator = resolution.construct_generator(lambda: SimpleNamespace())

    assert generator.execution_profile == "production"
    assert __import__("os").environ[PARO_MTP_CONTRACT_ENV] == "1"
    route = __import__("os").environ[PARO_MTP_ROUTE_ENV]
    assert TARGET_CONTRACT_VARIANT in route
    assert STRICT_VERIFIER_VARIANT in route
