from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.gguf_mtp_forced_target_probe import _raw_numeric_payload_from_raw
from scripts.gguf_mtp_compare_prefix_state_summaries import build_artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _buffer(
    *,
    digest: str,
    mean: float,
    rms: float,
    first: list[float],
    raw_values: list[float] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "nbytes": len(first) * 4,
        "blake2b_128": digest,
        "numeric_summary": {
            "label": digest,
            "size": len(first),
            "sha256_16": digest[:16],
            "mean": mean,
            "rms": rms,
            "min": min(first),
            "max": max(first),
            "first8": first,
            "last8": first,
        },
    }
    if raw_values is not None:
        raw = np.asarray(raw_values, dtype=np.float32)
        result["raw_data"] = _raw_numeric_payload_from_raw(
            raw.view(np.uint8),
            dtype="fp32",
            label=digest,
        )
        result["blake2b_128"] = result["raw_data"]["blake2b_128"]  # type: ignore[index]
        result["nbytes"] = int(raw.nbytes)
    return result


def _payload(
    *,
    conv_digest: str,
    conv_first: list[float],
    margin: float,
    conv_raw_values: list[float] | None = None,
) -> dict[str, object]:
    recurrent = _buffer(digest="same-recurrent", mean=0.0, rms=1.0, first=[0.0, 0.0])
    return {
        "result": {
            "prefix_state_fingerprint": {
                "position": 72,
                "current_prev": 653,
                "hidden_seed": {
                    "blake2b_128": "hidden",
                    "summary": {
                        "mean": 0.0,
                        "rms": 1.0,
                    },
                },
                "linear_state_layers": [
                    {
                        "layer": 0,
                        "conv": _buffer(
                            digest=conv_digest,
                            mean=sum(conv_first) / len(conv_first),
                            rms=2.0,
                            first=conv_first,
                            raw_values=conv_raw_values,
                        ),
                        "recurrent": recurrent,
                    }
                ],
                "kv_state": {
                    "layers": [
                        {
                            "layer": 3,
                            "key": _buffer(digest="same-key", mean=0.0, rms=1.0, first=[0.0, 0.0]),
                            "value": _buffer(digest="same-value", mean=1.0, rms=2.0, first=[1.0, 1.0]),
                        }
                    ]
                },
            },
            "rows": [
                {
                    "row": 1,
                    "sampled_token": 20 if margin < 0 else 10,
                    "candidate_scores": [
                        {"token_id": 10, "logit": margin, "rank": 1 if margin > 0 else 2},
                        {"token_id": 20, "logit": 0.0, "rank": 2 if margin > 0 else 1},
                    ],
                }
            ],
        }
    }


def test_build_artifact_compares_prefix_state_summaries(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_json(baseline, _payload(conv_digest="base-conv", conv_first=[0.0, 1.0], margin=-0.25))
    _write_json(candidate, _payload(conv_digest="cand-conv", conv_first=[1.0, 3.0], margin=0.5))

    artifact = build_artifact(
        argparse.Namespace(
            baseline_json=baseline,
            candidate_json=candidate,
            baseline_label="default",
            candidate_label="prefill_gdn",
            candidate_tokens="10,20",
            row=1,
            top_n=3,
        )
    )

    assert artifact["performance_claim"] is False
    assert artifact["summary"]["linear_components"] == 2
    assert artifact["summary"]["linear_hash_changed"] == 1
    assert artifact["summary"]["kv_components"] == 2
    assert artifact["summary"]["kv_hash_changed"] == 0
    assert artifact["summary"]["top_linear_summary_deltas"][0]["component"] == "linear_state_conv"
    assert artifact["summary"]["top_linear_summary_deltas"][0]["first8_mae"] == 1.5
    assert artifact["token_margin"]["default"]["10_minus_20"] == -0.25
    assert artifact["token_margin"]["prefill_gdn"]["10_minus_20"] == 0.5


def test_build_artifact_compares_selected_raw_prefix_state(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_json(
        baseline,
        _payload(
            conv_digest="base-conv",
            conv_first=[0.0, 1.0],
            conv_raw_values=[1.0, 2.0, 3.0],
            margin=-0.25,
        ),
    )
    _write_json(
        candidate,
        _payload(
            conv_digest="cand-conv",
            conv_first=[1.0, 3.0],
            conv_raw_values=[1.0, 4.0, 7.0],
            margin=0.5,
        ),
    )

    artifact = build_artifact(
        argparse.Namespace(
            baseline_json=baseline,
            candidate_json=candidate,
            baseline_label="default",
            candidate_label="prefill_gdn",
            candidate_tokens="10,20",
            row=1,
            top_n=3,
        )
    )

    assert artifact["summary"]["raw_linear_components"] == 1
    raw_top = artifact["summary"]["top_linear_pairwise_deltas"][0]
    assert raw_top["component"] == "linear_state_conv"
    assert raw_top["pairwise_delta"]["mean_abs_diff"] == 2.0
    assert raw_top["pairwise_delta"]["max_abs_diff"] == 4.0
