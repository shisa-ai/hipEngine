from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.quant_quality.qwen36_teacher import (
    PROTOCOL_ID,
    compare,
    noninferiority,
    register_raw,
)


def _fixture(path: Path) -> Path:
    payload = {
        "schema": 1,
        "kind": "quant_quality_teacher_fixture",
        "protocol_id": PROTOCOL_ID,
        "teacher_steps": 9,
        "vocab_size": 4,
        "prompt_source_sha256": "prompt-fixture-sha",
        "prompts": [
            {
                "id": "tiny",
                "category": "code",
                "prompt_token_ids": [1, 2],
                "teacher_token_ids": [0, 1, 2, 3, 0, 1, 2, 3, 0],
            }
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _register(
    *,
    fixture: Path,
    raw: Path,
    output: Path,
    model: Path,
    name: str,
) -> Path:
    args = argparse.Namespace(
        fixture=str(fixture),
        raw=str(raw),
        raw_dtype="float32",
        output=str(output),
        name=name,
        runtime="unit-test runtime",
        model=str(model),
        model_sha256="a" * 64,
    )
    assert register_raw(args) == 0
    return output.with_suffix(".manifest.json")


def test_paro_transformers_depthwise_fallback_matches_conv1d() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from scripts.quant_quality.qwen36_teacher import _torch_depthwise_causal_conv1d

    torch.manual_seed(1234)
    hidden_states = torch.randn(2, 6, 11, dtype=torch.float32)
    weight = torch.randn(6, 4, dtype=torch.float32)
    bias = torch.randn(6, dtype=torch.float32)
    expected = F.conv1d(
        hidden_states,
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=hidden_states.shape[1],
    )[:, :, : hidden_states.shape[-1]]

    actual = _torch_depthwise_causal_conv1d(hidden_states, weight, bias)

    assert actual.shape == hidden_states.shape
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_register_raw_and_compare_round_trip(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture.json")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"tiny model")

    reference = np.arange(36, dtype=np.float32).reshape(9, 4) / 10
    candidate = reference.copy()
    candidate[0] = candidate[0, ::-1]
    ref_raw = tmp_path / "ref.raw"
    candidate_raw = tmp_path / "candidate.raw"
    reference.tofile(ref_raw)
    candidate.tofile(candidate_raw)

    ref_manifest = _register(
        fixture=fixture,
        raw=ref_raw,
        output=tmp_path / "ref.npy",
        model=model,
        name="reference",
    )
    candidate_manifest = _register(
        fixture=fixture,
        raw=candidate_raw,
        output=tmp_path / "candidate.npy",
        model=model,
        name="candidate",
    )
    result_path = tmp_path / "comparison.json"
    args = argparse.Namespace(
        fixture=str(fixture),
        reference_manifest=str(ref_manifest),
        candidate_manifest=str(candidate_manifest),
        output=str(result_path),
        top_k=2,
    )

    assert compare(args) == 0
    result = json.loads(result_path.read_text())
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["metrics"]["rows"] == 9
    assert result["metrics"]["mean_kl_nats"] > 0
    assert result["metrics"]["top1_agreement_pct"] < 100
    assert result["candidate"]["model_sha256"] == "a" * 64
    assert result["metrics"]["by_split"]["train"]["rows"] == 9
    assert result["metrics"]["by_prompt"]["tiny"]["rows"] == 9

    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_args = argparse.Namespace(
        fixture=str(fixture),
        reference_manifest=str(ref_manifest),
        baseline_manifest=str(ref_manifest),
        candidate_manifest=str(candidate_manifest),
        output=str(bootstrap_path),
        top_k=2,
        bootstrap_samples=100,
        bootstrap_seed=1234,
        mean_kl_margin=0.005,
        top1_margin_pp=2.0,
        ppl_ratio_margin=0.01,
    )
    assert noninferiority(bootstrap_args) == 0
    bootstrap = json.loads(bootstrap_path.read_text())
    assert bootstrap["verdict"] == "quality-traded"
    assert not bootstrap["gates"]["portable_q4_equivalent"]
