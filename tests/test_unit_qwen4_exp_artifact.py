from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import qwen4_exp_artifact


def _write_tree(root: Path, revision: str, files: dict[str, tuple[bytes, str]]) -> None:
    manifest = {
        "format_version": 1,
        "files": {
            name: {"size": len(payload), "blob_id": blob_id}
            for name, (payload, blob_id) in files.items()
        },
    }
    path = root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_hf_snapshot_reports_complete_missing_and_wrong_files(tmp_path: Path) -> None:
    revision = "abc123"
    files = {
        "config.json": (b"{}", "config-blob"),
        "model-00001-of-00002.safetensors": (b"one", "shard-1"),
        "model-00002-of-00002.safetensors": (b"two", "shard-2"),
        "model.safetensors.index.json": (b"index", "index-blob"),
    }
    _write_tree(tmp_path, revision, files)
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"bad-size")

    result = qwen4_exp_artifact.verify_hf_snapshot(tmp_path, revision)

    assert result["passed"] is False
    assert result["expected_files"] == 4
    assert result["complete_files"] == 2
    assert result["expected_shards"] == 2
    assert result["complete_shards"] == 1
    assert result["missing"] == ["model.safetensors.index.json"]
    assert result["size_mismatches"] == [
        {
            "path": "model-00002-of-00002.safetensors",
            "actual": 8,
            "expected": 3,
        }
    ]
    assert result["unexpected"] == []
    assert result["tree_manifest_sha256"] == hashlib.sha256(
        (tmp_path / ".cache/huggingface/trees" / f"{revision}.json").read_bytes()
    ).hexdigest()


def test_verify_hf_snapshot_checks_weight_index_contract(tmp_path: Path) -> None:
    revision = "frozen"
    index = json.dumps(
        {
            "metadata": {"total_size": 6},
            "weight_map": {
                "model.a": "model-00001-of-00002.safetensors",
                "model.b": "model-00002-of-00002.safetensors",
            },
        },
        separators=(",", ":"),
    ).encode()
    files = {
        "model-00001-of-00002.safetensors": (b"one", "shard-1"),
        "model-00002-of-00002.safetensors": (b"two", "shard-2"),
        "model.safetensors.index.json": (index, "index-blob"),
    }
    _write_tree(tmp_path, revision, files)
    for name, (payload, _) in files.items():
        (tmp_path / name).write_bytes(payload)

    result = qwen4_exp_artifact.verify_hf_snapshot(tmp_path, revision)

    assert result["passed"] is True
    assert result["weight_index"] == {
        "declared_tensor_bytes": 6,
        "tensor_count": 2,
        "referenced_shards": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "missing_referenced_shards": [],
        "unreferenced_snapshot_shards": [],
    }


def test_qwen4_exp_tensor_roles_are_exhaustive_for_known_families() -> None:
    names = {
        "per_layer_token_embd.weight": "ple_table",
        "token_embd.weight": "root",
        "output.weight": "root",
        "output_hc_norm.weight": "gated_residual",
        "blk.0.hc_attn_down.weight": "gated_residual",
        "blk.0.ple_key.weight": "ple_compute",
        "blk.3.indexer.q_proj.weight": "qsa_indexer",
        "blk.3.attn_q.weight": "full_attention",
        "blk.0.ssm_conv1d.weight": "gdn",
        "blk.0.ffn_gate_exps.weight": "routed_expert",
        "blk.0.ffn_gate_shexp.weight": "shared_expert",
        "blk.0.ffn_gate_inp.weight": "router",
    }
    assert {name: qwen4_exp_artifact.tensor_role(name) for name in names} == names


def test_summarize_qwen4_exp_gguf_groups_bytes_by_role_and_qtype() -> None:
    tensors = (
        SimpleNamespace(name="per_layer_token_embd.weight", ggml_type_name="Q4_0", nbytes=90, shape=(10, 160), byte_shape=(10, 90)),
        SimpleNamespace(name="blk.0.hc_attn_down.weight", ggml_type_name="Q6_K", nbytes=10, shape=(1, 1), byte_shape=(1, 10)),
        SimpleNamespace(name="blk.0.ffn_up_exps.weight", ggml_type_name="Q4_K", nbytes=20, shape=(1, 1, 1), byte_shape=(1, 1, 20)),
    )
    info = SimpleNamespace(
        path=Path("model.gguf"),
        architecture="qwen4exp",
        file_type=15,
        file_type_name="MOSTLY_Q4_K_M",
        tensor_count=3,
        total_tensor_nbytes=120,
        metadata={"general.quantization_version": 2},
        tensors=tensors,
    )

    result = qwen4_exp_artifact.summarize_qwen4_exp_gguf(SimpleNamespace(info=info))

    assert result["passed"] is True
    assert result["tensor_bytes_by_role"] == {
        "gated_residual": 10,
        "ple_table": 90,
        "routed_expert": 20,
    }
    assert result["tensor_bytes_by_type"] == {"Q4_0": 90, "Q4_K": 20, "Q6_K": 10}
    assert result["tensor_bytes_by_role_and_type"] == {
        "gated_residual": {"Q6_K": 10},
        "ple_table": {"Q4_0": 90},
        "routed_expert": {"Q4_K": 20},
    }
    assert result["ple_table"] == {
        "name": "per_layer_token_embd.weight",
        "type": "Q4_0",
        "shape": [10, 160],
        "byte_shape": [10, 90],
        "nbytes": 90,
    }


def test_summarize_qwen4_exp_split_gguf_validates_and_aggregates_parts() -> None:
    tensors = (
        SimpleNamespace(name="per_layer_token_embd.weight", ggml_type_name="Q4_0", nbytes=90, shape=(10, 160), byte_shape=(10, 90)),
        SimpleNamespace(name="blk.0.ffn_up_exps.weight", ggml_type_name="Q4_K", nbytes=20, shape=(1, 1, 1), byte_shape=(1, 1, 20)),
    )
    shared = {
        "general.architecture": "qwen4exp",
        "general.file_type": 15,
        "split.count": 2,
        "split.tensors.count": 2,
    }
    readers = (
        SimpleNamespace(
            info=SimpleNamespace(
                path=Path("model-00001-of-00002.gguf"),
                architecture="qwen4exp",
                file_type=15,
                file_type_name="MOSTLY_Q4_K_M",
                tensor_count=0,
                total_tensor_nbytes=0,
                metadata={**shared, "split.no": 0},
                tensors=(),
            )
        ),
        SimpleNamespace(
            info=SimpleNamespace(
                path=Path("model-00002-of-00002.gguf"),
                architecture=None,
                file_type=None,
                file_type_name=None,
                tensor_count=2,
                total_tensor_nbytes=110,
                metadata={
                    "split.count": 2,
                    "split.no": 1,
                    "split.tensors.count": 2,
                },
                tensors=tensors,
            )
        ),
    )

    result = qwen4_exp_artifact.summarize_qwen4_exp_split_gguf(readers)

    assert result["passed"] is True
    assert result["split"] == {
        "count": 2,
        "part_numbers": [0, 1],
        "declared_tensor_count": 2,
    }
    assert result["tensor_count"] == 2
    assert result["total_tensor_nbytes"] == 110
    assert result["tensor_bytes_by_role"] == {"ple_table": 90, "routed_expert": 20}
    assert result["part_paths"] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]


def test_summarize_qwen4_exp_split_gguf_rejects_missing_parts_and_duplicate_tensors() -> None:
    tensor = SimpleNamespace(name="output.weight", ggml_type_name="Q6_K", nbytes=10, shape=(1, 1), byte_shape=(1, 10))
    metadata = {
        "general.architecture": "qwen4exp",
        "general.file_type": 15,
        "split.count": 3,
        "split.no": 1,
        "split.tensors.count": 2,
    }
    readers = (
        SimpleNamespace(
            info=SimpleNamespace(
                path=Path("part-a.gguf"),
                architecture="qwen4exp",
                file_type=15,
                file_type_name="MOSTLY_Q4_K_M",
                tensor_count=1,
                total_tensor_nbytes=10,
                metadata=metadata,
                tensors=(tensor,),
            )
        ),
        SimpleNamespace(
            info=SimpleNamespace(
                path=Path("part-b.gguf"),
                architecture="qwen4exp",
                file_type=15,
                file_type_name="MOSTLY_Q4_K_M",
                tensor_count=1,
                total_tensor_nbytes=10,
                metadata=metadata,
                tensors=(tensor,),
            )
        ),
    )

    result = qwen4_exp_artifact.summarize_qwen4_exp_split_gguf(readers)

    assert result["passed"] is False
    assert "split part numbers [1, 1] do not cover [0, 1, 2]" in result["errors"]
    assert "duplicate tensor names across split: output.weight" in result["errors"]
