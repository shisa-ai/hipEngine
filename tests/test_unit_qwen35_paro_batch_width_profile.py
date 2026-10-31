from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import hipengine.runtime.qwen35_paro_batch_width as profile_module
from hipengine.runtime.qwen35_paro_batch_width import (
    DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE,
    load_qwen35_paro_native_batch_width_profile,
    native_batch_width_domain,
)

GFX1151_WIDTHS, GFX1151_EXACT = native_batch_width_domain(8, row_chunk_widths=(3, 5, 6, 7))
GFX1100_WIDTHS, GFX1100_EXACT = native_batch_width_domain(8, row_chunk_widths=(3, 4, 5, 6, 7, 8))


def _geometry() -> dict[str, object]:
    return {
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "quant": "w4_paro",
        "kv_storage": "bf16",
        "max_batch_size": 8,
        "block_size": 256,
        "max_sequence_length": 4096,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 4096,
        "layer_types": ("linear_attention", "full_attention"),
    }


_DEFAULT_CONFIG_SIGNATURE = _geometry()


def _profile_payload(*, with_geometry: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": 1,
        "kind": "gfx1151_paro_cn_readme_diagnostic_summary",
        "performance_claim": True,
        "native_batch_width_profile": {
            "schema": 2,
            "backend": "hip_gfx1151",
            "target_arch": "gfx1151",
            "model_snapshot": "model-snapshot",
            "quant": "w4_paro",
            "kv_storage": "bf16",
            "native_widths": [2, 3, 4, 5, 6, 7, 8],
            "min_position": 512,
            "max_position": 647,
        },
        "hardware": {
            "backend": "hip_gfx1151",
            "target_arch": "gfx1151",
        },
        "model": {
            "snapshot": "model-snapshot",
            "quant": "w4_paro",
            "kv_storage": "bf16",
        },
        "protocol": {
            "native_partition_widths": [2, 3, 4, 5, 6, 7, 8],
            "evidenced_decode_position_range": {"min": 512, "max": 647},
            "correctness_oracle": "independent_single_request_prefill_decode",
            "packed_prefill_generated_token_equality": True,
            "sparse_slot_generated_token_equality": True,
            "shrinking_batch_generated_token_equality": True,
        },
        "rows": {
            "1": {
                "status": "accepted_reference",
                "decode_step_ms_median_of_run_medians": 15.0,
            },
            **{
                str(rows): {
                    "status": "accepted_exact",
                    "generated_token_equality": True,
                    "primitive_correctness": True,
                    "decode_step_ms_median_of_run_medians": float(rows * 10),
                }
                for rows in range(2, 9)
            },
        },
    }
    if with_geometry:
        payload["geometry"] = _geometry()
    return payload


def _write_profile(tmp_path: Path, payload: dict[str, object]) -> str:
    relative = "benchmarks/results/profile.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return relative


def _load(relative: str, **overrides: object):
    signature = dict(_DEFAULT_CONFIG_SIGNATURE)
    signature["backend"] = overrides.get("backend", "hip_gfx1151")
    signature["target_arch"] = overrides.get("target_arch", "gfx1151")
    signature["kv_storage"] = overrides.get("kv_dtype", "bf16")
    signature["quant"] = overrides.get("quant", "w4_paro")
    kwargs: dict[str, object] = {
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "model_path": Path("/models/model-snapshot"),
        "kv_dtype": "bf16",
        "native_widths": GFX1151_WIDTHS,
        "exact_widths": GFX1151_EXACT,
        "max_native_width": 8,
        "config_signature": signature,
    }
    kwargs.update(overrides)
    return load_qwen35_paro_native_batch_width_profile(relative, **kwargs)


def test_native_batch_width_domain_derives_exact_widths_per_backend() -> None:
    assert GFX1151_WIDTHS == (2, 3, 4, 5, 6, 7, 8)
    assert GFX1151_EXACT == (2, 4, 8)
    assert GFX1100_WIDTHS == (2, 3, 4, 5, 6, 7, 8)
    assert GFX1100_EXACT == (2,)
    assert native_batch_width_domain(1) == ((), ())


def test_qwen35_paro_default_batch_width_profile_is_retained_c248_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    profile = load_qwen35_paro_native_batch_width_profile(
        DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=Path("/models/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"),
        kv_dtype="bf16",
        native_widths=GFX1151_WIDTHS,
        exact_widths=GFX1151_EXACT,
        max_native_width=8,
    )

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.exact_widths == GFX1151_EXACT
    assert tuple(width for width, _cost_ms in profile.native_step_ms) == (2, 4, 8)
    assert profile.min_position == 512
    assert profile.max_position == 647


def test_packaged_qwen35_paro_profile_matches_retained_source() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    packaged_path = profile_module._DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_PACKAGE
    packaged = json.loads(packaged_path.read_text(encoding="utf-8"))

    assert packaged["source_artifact"] == DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE
    assert packaged["source_artifact_sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    for field in (
        "performance_claim",
        "native_batch_width_profile",
        "protocol",
    ):
        assert packaged[field] == source[field]
    assert packaged["hardware"] == {
        "backend": source["hardware"]["backend"],
        "target_arch": source["hardware"]["target_arch"],
    }
    assert packaged["model"] == {
        "snapshot": source["model"]["snapshot"],
        "quant": source["model"]["quant"],
        "kv_storage": source["model"]["kv_storage"],
    }
    for width in ("1", "2", "4", "8"):
        assert packaged["rows"][width] == {
            field: source["rows"][width][field]
            for field in packaged["rows"][width]
        }


def test_qwen35_paro_batch_width_profile_uses_measured_costs_for_matching_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.exact_widths == GFX1151_EXACT
    assert profile.native_step_ms == tuple((rows, float(rows * 10)) for rows in range(2, 9))
    assert profile.serial_row_step_ms == 15.0
    assert profile.min_position == 512
    assert profile.max_position == 647


def test_qwen35_paro_batch_width_profile_model_identity_is_a_cost_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(relative, model_path=Path("/models/another-snapshot"))

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.native_step_ms == ()
    assert any("model snapshot" in note for note in profile.cost_notes)


@pytest.mark.parametrize(
    ("field", "value", "note"),
    [
        ("backend", "hip_gfx1100", "capability fields"),
        ("target_arch", "gfx1100", "capability fields"),
    ],
)
def test_qwen35_paro_batch_width_profile_capability_mismatch_is_a_cost_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    note: str,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(relative, **{field: value})

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.native_step_ms == ()
    assert any(note in reason for reason in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_gfx1100_domain_runs_without_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(
        relative,
        backend="hip_gfx1100",
        target_arch="gfx1100",
        native_widths=GFX1100_WIDTHS,
        exact_widths=GFX1100_EXACT,
    )

    assert profile.blockers == ()
    assert profile.native_widths == GFX1100_WIDTHS
    assert profile.exact_widths == GFX1100_EXACT
    assert profile.native_step_ms == ()


def test_qwen35_paro_batch_width_profile_blocks_real_bf16_kv_capability_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(relative, kv_dtype="int8_per_token_head")

    assert profile.blockers == ("native c>N decode requires BF16 KV",)
    assert profile.native_step_ms == ()


def test_qwen35_paro_batch_width_profile_blocks_real_quant_capability_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = _load(relative, quant="w8_other")

    assert profile.blockers == ("native c>N decode requires the PARO w4 quant",)
    assert profile.native_step_ms == ()


def test_qwen35_paro_batch_width_profile_excludes_present_false_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["rows"]["4"] = {
        "status": "rejected",
        "generated_token_equality": False,
        "primitive_correctness": False,
        "decode_step_ms_median_of_run_medians": 40.0,
    }
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert 4 not in profile.native_widths
    assert profile.native_widths == (2, 3, 5, 6, 7, 8)
    assert profile.exact_widths == (2, 8)
    assert any("excluded recorded failures: c4" in note for note in profile.cost_notes)


def _rejected_row() -> dict[str, object]:
    return {
        "status": "rejected",
        "generated_token_equality": False,
        "primitive_correctness": False,
        "decode_step_ms_median_of_run_medians": 40.0,
    }


def test_qwen35_paro_batch_width_profile_rename_does_not_change_failure_applicability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["rows"]["4"] = _rejected_row()
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile_a = _load(relative, model_path=Path("/models/model-snapshot"))
    profile_b = _load(relative, model_path=Path("/models/another-snapshot"))

    assert profile_a.native_widths == profile_b.native_widths
    assert profile_a.blockers == profile_b.blockers == ()
    assert 4 not in profile_a.native_widths
    assert 4 not in profile_b.native_widths


def test_qwen35_paro_batch_width_profile_rename_does_not_change_end_to_end_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["protocol"]["packed_prefill_generated_token_equality"] = False
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile_a = _load(relative, model_path=Path("/models/model-snapshot"))
    profile_b = _load(relative, model_path=Path("/models/another-snapshot"))

    assert profile_a.blockers == profile_b.blockers
    assert any("packed prefill" in blocker for blocker in profile_a.blockers)


def test_qwen35_paro_batch_width_profile_geometry_mismatch_does_not_apply_false_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["rows"]["4"] = _rejected_row()
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    signature = dict(_DEFAULT_CONFIG_SIGNATURE)
    signature["hidden_size"] = 4096

    profile = _load(relative, config_signature=signature)

    assert profile.blockers == ()
    assert 4 in profile.native_widths
    assert any("geometry does not match" in note for note in profile.cost_notes)
    assert any("c4 records a failure" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_legacy_no_geometry_keeps_false_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload(with_geometry=False)
    payload["rows"]["4"] = _rejected_row()
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert 4 in profile.native_widths
    assert any("declares no geometry" in note for note in profile.cost_notes)
    assert any("c4 records a failure" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_legacy_no_geometry_end_to_end_false_is_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload(with_geometry=False)
    payload["protocol"]["packed_prefill_generated_token_equality"] = False
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert any("provenance only" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_geometry_mismatch_end_to_end_false_is_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["protocol"]["packed_prefill_generated_token_equality"] = False
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    signature = dict(_DEFAULT_CONFIG_SIGNATURE)
    signature["hidden_size"] = 4096

    profile = _load(relative, config_signature=signature)

    assert profile.blockers == ()
    assert any("provenance only" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_present_false_end_to_end_is_a_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["protocol"]["packed_prefill_generated_token_equality"] = False
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert any("failed packed prefill" in reason for reason in profile.blockers)
    assert profile.native_step_ms == ()


def test_qwen35_paro_batch_width_profile_missing_correctness_field_is_not_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["protocol"].pop("packed_prefill_generated_token_equality")
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert any("lacks packed prefill evidence" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_none_correctness_field_is_not_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["protocol"]["sparse_slot_generated_token_equality"] = None
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert any("lacks sparse-slot decode evidence" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_missing_row_correctness_field_keeps_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["rows"]["4"].pop("primitive_correctness")
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert 4 in profile.native_widths
    assert any("c4 lacks accepted exact evidence" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_diagnostic_artifact_keeps_native_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["performance_claim"] = False
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.native_step_ms == ()
    assert any("not an accepted performance claim" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_embedded_identity_disagreement_is_a_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["native_batch_width_profile"]["backend"] = "hip_gfx1100"
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = _load(relative)

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert any("embedded backend" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_missing_default_uses_nominal_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        profile_module,
        "_DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_PACKAGE",
        tmp_path / "missing-packaged-profile.json",
    )

    profile = _load(DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE)

    assert profile.blockers == ()
    assert profile.native_widths == GFX1151_WIDTHS
    assert profile.native_step_ms == ()
    assert any("does not exist" in note for note in profile.cost_notes)


def test_qwen35_paro_batch_width_profile_explicit_missing_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="does not exist"):
        _load("benchmarks/results/missing.json", artifact_is_explicit=True)


def test_qwen35_paro_batch_width_profile_explicit_malformed_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "benchmarks/results/profile.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not readable JSON"):
        _load(relative, artifact_is_explicit=True)


def test_qwen35_paro_batch_width_profile_rejects_non_results_path(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmarks/results"):
        _load(str(path))
