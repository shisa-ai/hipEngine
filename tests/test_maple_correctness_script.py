"""Unit tests for the optional Maple correctness harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maple_correctness.py"
_SPEC = importlib.util.spec_from_file_location("maple_correctness_script", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_args_accepts_explicit_hf_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "hf-cache"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(_SCRIPT),
            "--oracle",
            "hf",
            "--hf-cache-dir",
            str(cache_dir),
            "--hf-match-packed-affine4",
        ],
    )

    args = _MODULE.parse_args()

    assert args.hf_cache_dir == cache_dir
    assert args.hf_match_packed_affine4 is True


def test_hf_expert_offload_map_keeps_only_routed_mlps_on_disk() -> None:
    device_map = _MODULE.maple_hf_expert_device_map(
        layers=2,
        experts=3,
        device="cuda:0",
    )

    assert "" not in device_map
    assert len(device_map) == 18
    assert device_map["model.word_embeddings"] == "cuda:0"
    assert device_map["model.layers.1.self_attn"] == "cuda:0"
    assert device_map["model.layers.0.mlp.gate"] == "cuda:0"
    assert device_map["model.layers.1.mlp.experts.2"] == "disk"


def test_compare_accepts_dense_hf_output_without_hidden_checkpoints(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.npz"
    oracle_path = tmp_path / "oracle.npz"
    logits = np.asarray([[2.0, -1.0, 0.5], [-0.5, 0.25, 3.0]], dtype=np.float32)
    np.savez(
        hip_path,
        logits=logits,
        top_ids=np.asarray([0, 2], dtype=np.int64),
        elapsed_ms=np.asarray([1.0, 1.1]),
        hidden=np.zeros((2, 3, 4), dtype=np.float32),
    )
    np.savez(
        oracle_path,
        logits=logits,
        top_ids=np.asarray([0, 2], dtype=np.int64),
        elapsed_ms=np.asarray([2.0, 2.1]),
        oracle_model=np.asarray("deepgrove/maple-preview"),
        oracle_revision=np.asarray(_MODULE._HF_REVISION),
        transformers_version=np.asarray(_MODULE._HF_TRANSFORMERS_VERSION),
        torch_version=np.asarray("test"),
    )

    result = _MODULE.compare(
        hip_path,
        oracle_path,
        (3, 7),
        oracle_description="dense HF fixture",
    )

    assert result["passed"] is True
    assert result["max_kl"] == pytest.approx(0.0, abs=1e-15)
    assert result["top1_agreement"] == 1.0
    assert result["oracle_revision"] == _MODULE._HF_REVISION
    assert "hidden_labels" not in result


def test_hf_flash_attention_shim_matches_bottom_right_causal_gqa_window() -> None:
    torch = pytest.importorskip("torch")
    query = torch.tensor(
        [[
            [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.7], [0.2, -0.6]],
            [[0.9, 0.2], [-0.3, 0.8], [0.6, 0.1], [-0.4, -0.7]],
        ]],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [[
            [[0.2, 0.3], [-0.4, 0.5]],
            [[0.7, -0.1], [0.6, 0.2]],
            [[-0.2, 0.8], [0.1, -0.9]],
            [[0.5, 0.4], [-0.7, -0.3]],
        ]],
        dtype=torch.float32,
    )
    value = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2)

    actual = _MODULE.torch_flash_attention(
        query,
        key,
        value,
        softmax_scale=0.75,
        causal=True,
        window_size=(1, 0),
    )

    expected = torch.empty_like(query)
    query_offset = key.shape[1] - query.shape[1]
    groups = query.shape[2] // key.shape[2]
    for query_index in range(query.shape[1]):
        absolute_query = query_offset + query_index
        live_keys = range(max(0, absolute_query - 1), absolute_query + 1)
        for query_head in range(query.shape[2]):
            kv_head = query_head // groups
            scores = torch.stack(
                [
                    torch.dot(query[0, query_index, query_head], key[0, key_index, kv_head])
                    * 0.75
                    for key_index in live_keys
                ]
            )
            probabilities = torch.softmax(scores, dim=0)
            expected[0, query_index, query_head] = sum(
                probabilities[offset] * value[0, key_index, kv_head]
                for offset, key_index in enumerate(live_keys)
            )

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
