from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine.runtime.qwen35_paro_batch_width import load_qwen35_paro_native_batch_width_profile


def _profile_payload() -> dict[str, object]:
    return {
        "schema": 1,
        "kind": "gfx1151_paro_cn_readme_diagnostic_summary",
        "native_batch_width_profile": {
            "schema": 1,
            "backend": "hip_gfx1151",
            "target_arch": "gfx1151",
            "model_snapshot": "model-snapshot",
            "quant": "w4_paro",
            "kv_storage": "bf16",
            "native_widths": [2, 4, 6, 8],
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
            "native_partition_widths": [2, 4, 6, 8],
            "evidenced_decode_position_range": {"min": 512, "max": 647},
        },
        "rows": {
            "1": {
                "status": "diagnostic_reference",
                "decode_step_ms_median_of_run_medians": 15.0,
            },
            **{
                str(rows): {
                    "status": "diagnostic_exact",
                    "generated_token_equality": True,
                    "primitive_correctness": True,
                    "decode_step_ms_median_of_run_medians": float(rows * 10),
                }
                for rows in (2, 4, 6, 8)
            },
        },
    }


def _write_profile(tmp_path: Path, payload: dict[str, object]) -> str:
    relative = "benchmarks/results/profile.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return relative


def test_qwen35_paro_batch_width_profile_requires_matching_full_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)

    profile = load_qwen35_paro_native_batch_width_profile(
        relative,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=Path("/models/model-snapshot"),
        kv_dtype="bf16",
    )

    assert profile.blockers == ()
    assert profile.native_step_ms == ((2, 20.0), (4, 40.0), (6, 60.0), (8, 80.0))
    assert profile.serial_row_step_ms == 15.0
    assert profile.min_position == 512
    assert profile.max_position == 647


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("backend", "hip_gfx1100", "backend"),
        ("target_arch", "gfx1100", "target architecture"),
        ("model_path", Path("/models/another-snapshot"), "model snapshot"),
        ("kv_dtype", "int8_per_token_head", "KV dtype"),
    ],
)
def test_qwen35_paro_batch_width_profile_blocks_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    blocker: str,
) -> None:
    relative = _write_profile(tmp_path, _profile_payload())
    monkeypatch.chdir(tmp_path)
    kwargs: dict[str, object] = {
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "model_path": Path("/models/model-snapshot"),
        "kv_dtype": "bf16",
    }
    kwargs[field] = value

    profile = load_qwen35_paro_native_batch_width_profile(relative, **kwargs)

    assert any(blocker in reason for reason in profile.blockers)
    assert profile.native_step_ms == ()


def test_qwen35_paro_batch_width_profile_rejects_non_results_path(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmarks/results"):
        load_qwen35_paro_native_batch_width_profile(
            str(path),
            backend="hip_gfx1151",
            target_arch="gfx1151",
            model_path=Path("/models/model-snapshot"),
            kv_dtype="bf16",
        )


def test_qwen35_paro_batch_width_profile_blocks_embedded_identity_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _profile_payload()
    payload["native_batch_width_profile"]["backend"] = "hip_gfx1100"
    relative = _write_profile(tmp_path, payload)
    monkeypatch.chdir(tmp_path)

    profile = load_qwen35_paro_native_batch_width_profile(
        relative,
        backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=Path("/models/model-snapshot"),
        kv_dtype="bf16",
    )

    assert any("embedded backend" in reason for reason in profile.blockers)
    assert profile.native_step_ms == ()
