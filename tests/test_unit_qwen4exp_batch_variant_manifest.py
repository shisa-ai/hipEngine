"""CPU tests for the Qwen4Exp batch variant manifest.

These tests bind the batch decode path to the live direct callables and registry
keys. They deliberately do not use the c1 profile manifest as evidence: the c1
manifest describes prefill and single-request decode scopes, and the batch
manifest is a separate, structured artifact bound to it by hash.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    manifest_sha256,
    resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen4_exp_linear_attn_conv_decode_indexed_f32,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_batch_manifest as batch_manifest
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from scripts.qwen4exp_ar_batch_candidate import batch_manifest_binding

SOURCE_REVISION = "a" * 40
QUANT = "gguf_ud_q4_k_xl"


@pytest.fixture(autouse=True)
def _registered_kernels():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()


@pytest.fixture
def profile_manifest(_registered_kernels):
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    return resolved.manifest


def _config(*, ple_layers=(0,)):
    return SimpleNamespace(
        hidden_size=2048,
        context_length=4096,
        qsa_compression_ratio=4,
        layer_types=["gdn", "qsa"] * 24,
        ple_layers=list(ple_layers),
    )


def _runner(*, ple_layers=(0,)):
    return SimpleNamespace(
        backend=QWEN4_EXP_BACKEND,
        config=_config(ple_layers=ple_layers),
    )


def _dispatch_records(rows=(1, 2, 4)):
    # The real launcher resolves a row-dependent dense-GEMV key, not a
    # synthetic "linear" key; bind the same shape the GPU smoke records.
    key = {
        "backend": "hip_gfx1100",
        "layer": "dense_gemv",
        "quant": "f32",
        "variant": "f32_hidden_f32_out",
    }
    return [
        {
            "key": dict(key),
            "abi": "generic",
            "rows": int(row),
            "in_features": 2048,
            "out_features": 2048,
            "activation_dtype": "f32",
            "output_dtype": "f32",
        }
        for row in rows
    ]


def _build(profile_manifest, **overrides):
    kwargs = {
        "runner": _runner(),
        "profile_manifest": profile_manifest,
        "source_revision": SOURCE_REVISION,
        "dispatch_records": _dispatch_records(),
        "step_widths": {1: 1, 2: 1, 4: 1},
        "runtime_class": Qwen4ExpGGUFBatchRunner,
        "embedding_quant": "gguf_q6_k",
    }
    kwargs.update(overrides)
    return batch_manifest.build_qwen4_exp_batch_variant_manifest(**kwargs)


def _validate(payload, profile_manifest):
    return batch_manifest.validate_batch_variant_manifest(
        payload,
        expected_profile_manifest_sha256=manifest_sha256(profile_manifest),
        expected_source_revision=SOURCE_REVISION,
        expected_backend=QWEN4_EXP_BACKEND,
        expected_model=QWEN4_EXP_MODEL,
        expected_quant=QUANT,
    )


def _entry(payload, role):
    return next(entry for entry in payload["entries"] if entry["role"] == role)


def _rehash(payload):
    body = copy.deepcopy(payload)
    body.pop("manifest_sha256", None)
    body["manifest_sha256"] = batch_manifest.batch_variant_manifest_sha256(body)
    return body


# -- real binding -----------------------------------------------------------


def test_head_pair_manifest_binds_actual_geometry_and_strict_rows_oracle(profile_manifest):
    from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
        qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32,
    )

    runner = _runner()
    runner.block_size = 256
    runner.config.attention_head_count = 24
    runner.config.attention_kv_head_count = 2
    runner.config.attention_key_length = 256
    built = _build(profile_manifest, runner=runner)
    entry = _entry(built, "qsa_rows_attention")
    assert entry["identity"] == batch_manifest.function_identity(
        qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32
    )
    assert entry["registered_keys"][0]["variant"] == "strict_h256_head_pair_rows_spans"
    assert entry["oracle"]["identity"] == batch_manifest.function_identity(
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32
    )
    assert _validate(built, profile_manifest)["valid"]
    altered = copy.deepcopy(built)
    altered["runtime_config"]["qsa_geometry"]["head_dim"] = 128
    assert "direct_identity_mismatch:qsa_rows_attention" in _validate(
        _rehash(altered), profile_manifest
    )["reasons"]


def test_manifest_binds_real_batch_calls(profile_manifest):
    built = _build(profile_manifest)
    result = _validate(built, profile_manifest)
    assert result["valid"] is True
    assert result["reasons"] == []
    assert len(built["manifest_sha256"]) == 64

    roles = {entry["role"] for entry in built["entries"]}
    assert set(batch_manifest.BATCH_CORE_REQUIRED_ROLES) <= roles
    assert set(batch_manifest.BATCH_PLE_REQUIRED_ROLES) <= roles

    gdn_conv = _entry(built, "gdn_conv_indexed")
    assert gdn_conv["binding"] == "direct"
    assert gdn_conv["identity"] == batch_manifest.function_identity(
        qwen4_exp_linear_attn_conv_decode_indexed_f32
    )
    assert gdn_conv["registered_keys"][0]["variant"] == "qwen4exp_conv_indexed_decode"

    rows_attention = _entry(built, "qsa_rows_attention")
    assert rows_attention["binding"] == "direct"
    assert rows_attention["identity"] == batch_manifest.function_identity(
        qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32
    )

    ple_conv = _entry(built, "ple_indexed_conv")
    assert ple_conv["binding"] == "registry"
    assert ple_conv["registry_key"]["layer"] == "ple_dilated_depthwise_conv"
    assert ple_conv["strict_fallback"]["registry_key"]["variant"] == "strict"

    projection = _entry(built, "projection_dispatch")
    assert projection["binding"] == "registry_dispatch"
    assert projection["registry_key"] is None
    assert projection["rows"] == [1, 2, 4]
    assert {record["rows"] for record in projection["observed"]} == {1, 2, 4}


def test_manifest_reports_unbound_and_unused_roles(profile_manifest):
    built = _build(profile_manifest)
    unbound = {row["role"] for row in built["unbound"]}
    not_used = {row["role"] for row in built["not_used"]}
    assert "prefill" in unbound
    assert "ple_hash_rows" in unbound
    assert "stage_qwen4_exp_ple_rows" in unbound
    assert "dense_paged_decode" in not_used
    assert "qsa_split_norm_mrope" in not_used
    roles = {entry["role"] for entry in built["entries"]}
    assert "dense_paged_decode" not in roles
    assert "qsa_split_norm_mrope" not in roles


# -- changed variant / hash -------------------------------------------------


def test_changed_projection_variant_fails_live_identity(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    record = _entry(forged, "projection_dispatch")["observed"][0]
    record["key"]["variant"] = "bf16_hidden_bf16_out"
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("projection_identity_mismatch" in reason for reason in result["reasons"])


def test_changed_hash_without_recompute_fails(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    forged["source_revision"] = "b" * 40
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert "manifest_sha256_mismatch" in result["reasons"]


def test_duplicate_role_entry_is_rejected(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    duplicate = copy.deepcopy(_entry(forged, "gdn_conv_indexed"))
    # A different identity survives the exact-duplicate normalizer and must
    # still be rejected: two entries for one role make the binding ambiguous.
    duplicate["identity"] = "hipengine.fake.fn"
    forged["entries"].append(duplicate)
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("duplicate_role_entry" in reason for reason in result["reasons"])


def test_changed_direct_function_hash_fails(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    _entry(forged, "gdn_conv_indexed")["function_sha256"] = "0" * 64
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("direct_function_hash_mismatch" in reason for reason in result["reasons"])


# -- missing member ---------------------------------------------------------


def test_missing_required_role_fails(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    forged["entries"] = [
        entry for entry in forged["entries"] if entry["role"] != "qsa_paged_kv_write"
    ]
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("required_role_missing:qsa_paged_kv_write" in reason for reason in result["reasons"])
    assert any(row["role"] == "qsa_paged_kv_write" for row in result["missing"])


def test_missing_ple_role_fails_when_ple_layers_exist(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    forged["entries"] = [
        entry for entry in forged["entries"] if entry["role"] != "ple_indexed_conv"
    ]
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("required_role_missing:ple_indexed_conv" in reason for reason in result["reasons"])


def test_projection_dispatch_missing_step_width_fails_at_build(profile_manifest):
    with pytest.raises(ValueError, match="step width"):
        _build(profile_manifest, dispatch_records=_dispatch_records(rows=(1, 2)))


# -- forged labels ----------------------------------------------------------


def test_direct_role_cannot_be_relabelled_as_registry(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    entry = _entry(forged, "gdn_conv_indexed")
    entry["binding"] = "registry"
    entry["registry_key"] = entry["registered_keys"][0]
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert "registry_entry_for_direct_role:gdn_conv_indexed" in result["reasons"]


def test_registry_role_cannot_be_relabelled_as_direct(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    entry = _entry(forged, "ple_indexed_conv")
    entry["binding"] = "direct"
    entry["registry_key"] = None
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert "direct_role_not_live:ple_indexed_conv" in result["reasons"]


def test_direct_entry_cannot_claim_a_registry_key(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    entry = _entry(forged, "qsa_rows_attention")
    entry["registry_key"] = entry["registered_keys"][0]
    with pytest.raises(ValueError, match="must not claim a registry key"):
        batch_manifest.normalize_batch_variant_manifest(forged)


def test_forged_direct_identity_fails(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    entry = _entry(forged, "qsa_rows_attention")
    entry["identity"] = (
        "hipengine.kernels.hip_gfx1100.attention.paged_attn_decode."
        "qwen35_paged_full_attn_decode_context_bf16_batch_spans"
    )
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert "direct_identity_mismatch:qsa_rows_attention" in result["reasons"]


def test_forged_oracle_identity_fails(profile_manifest):
    built = _build(profile_manifest)
    forged = copy.deepcopy(built)
    entry = _entry(forged, "gdn_conv_indexed")
    entry["oracle"]["identity"] = "not.a.real.oracle"
    forged = _rehash(forged)
    result = _validate(forged, profile_manifest)
    assert result["valid"] is False
    assert any("oracle_identity_not_live" in reason for reason in result["reasons"])


# -- binding and serialization ---------------------------------------------


def test_profile_hash_and_source_revision_are_bound(profile_manifest):
    built = _build(profile_manifest)
    wrong_profile = batch_manifest.validate_batch_variant_manifest(
        built,
        expected_profile_manifest_sha256="0" * 64,
        expected_source_revision=SOURCE_REVISION,
    )
    assert wrong_profile["valid"] is False
    assert "profile_manifest_sha256_mismatch" in wrong_profile["reasons"]

    wrong_source = batch_manifest.validate_batch_variant_manifest(
        built,
        expected_profile_manifest_sha256=manifest_sha256(profile_manifest),
        expected_source_revision="b" * 40,
    )
    assert wrong_source["valid"] is False
    assert "source_revision_mismatch" in wrong_source["reasons"]


def test_manifest_serialization_is_deterministic(profile_manifest):
    first = _build(profile_manifest)
    second = _build(
        profile_manifest,
        dispatch_records=list(reversed(_dispatch_records())),
        step_widths={4: 1, 1: 1, 2: 1},
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["entries"] == second["entries"]
    assert _entry(first, "projection_dispatch") == _entry(
        second, "projection_dispatch"
    )


def test_c1_profile_manifest_is_not_a_batch_manifest(profile_manifest):
    result = batch_manifest.validate_batch_variant_manifest(profile_manifest)
    assert result["valid"] is False
    assert any("missing fields" in reason for reason in result["reasons"])


def test_legacy_c1_binding_still_reports_missing(profile_manifest):
    binding = batch_manifest_binding(profile_manifest)
    assert binding["bound"] is False
    assert binding["missing"]


def test_dispatch_recorder_is_context_local_and_exact():
    from hipengine.runtime.gguf_linear import (
        _record_gguf_linear_dispatch,
        record_gguf_linear_dispatch,
    )

    kwargs = {
        "backend": QWEN4_EXP_BACKEND,
        "layer": "linear",
        "quant": "fp16_weight",
        "variant": "f32_f32_out",
        "abi": "generic",
        "rows": 2,
        "in_features": 2048,
        "out_features": 2048,
        "activation_dtype": "f32",
        "output_dtype": "f32",
    }
    # Off by default: a launch must not pay for a record nobody asked for.
    _record_gguf_linear_dispatch(**kwargs)
    records: list[dict] = []
    with record_gguf_linear_dispatch(records):
        _record_gguf_linear_dispatch(**kwargs)
    assert records == [
        {
            "key": {
                "backend": QWEN4_EXP_BACKEND,
                "layer": "linear",
                "quant": "fp16_weight",
                "variant": "f32_f32_out",
            },
            "abi": "generic",
            "rows": 2,
            "in_features": 2048,
            "out_features": 2048,
            "activation_dtype": "f32",
            "output_dtype": "f32",
        }
    ]
    # The context is gone; another helper call records nothing.
    _record_gguf_linear_dispatch(**kwargs)
    assert len(records) == 1
