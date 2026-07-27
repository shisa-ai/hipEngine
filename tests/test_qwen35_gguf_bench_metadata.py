from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from scripts import qwen35_gguf_bench as bench


def _tensor(
    name: str,
    *,
    shape: tuple[int, ...] = (2, 4),
    ggml_shape: tuple[int, ...] = (4, 2),
    ggml_type: int = 2,
    ggml_type_name: str = "Q4_K",
    n_elements: int = 8,
    nbytes: int = 144,
    offset: int = 0,
    data_offset: int = 4096,
    byte_shape: tuple[int, ...] = (2, 72),
) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=ggml_shape,
        ggml_type=ggml_type,
        ggml_type_name=ggml_type_name,
        n_elements=n_elements,
        nbytes=nbytes,
        offset=offset,
        data_offset=data_offset,
        byte_shape=byte_shape,
    )


def _model_info(*tensors: GGUFTensorInfo) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("/models/gguf/fake.gguf"),
        version=3,
        alignment=32,
        metadata={"general.architecture": "qwen3moe", "general.file_type": 15},
        tensors=tuple(tensors),
        tensor_data_offset=4096,
    )


def test_exact_command_payload_preserves_argv_and_shell_command() -> None:
    argv = ["python3", "scripts/qwen35_gguf_bench.py", "--model", "path with spaces.gguf"]

    payload = bench._exact_command_payload(argv)

    assert payload["argv"] == argv
    assert shlex.split(payload["command"]) == argv


def test_correctness_fingerprints_are_dtype_and_order_sensitive() -> None:
    logits = np.array([1.0, -0.0, 3.5], dtype=np.float32)
    same = np.array([1.0, -0.0, 3.5], dtype=np.float32)
    changed_bits = np.array([1.0, 0.0, 3.5], dtype=np.float32)

    assert bench._array_sha256(logits) == bench._array_sha256(same)
    assert bench._array_sha256(logits) != bench._array_sha256(changed_bits)
    assert bench._token_ids_sha256([1, 2, 3]) != bench._token_ids_sha256([3, 2, 1])


def test_decode_graph_disabled_reason_tracks_production_graph_capability() -> None:
    class NoGraphSession:
        pass

    class GraphSession:
        def capture_decode_graph(self) -> None:  # pragma: no cover - only capability probe
            raise AssertionError("helper should not call capture_decode_graph")

    class HostEmbeddingGraphSession(GraphSession):
        host_token_embedding_enabled = True

    assert bench._decode_graph_disabled_reason(NoGraphSession(), requested=False) is None
    assert bench._decode_graph_disabled_reason(NoGraphSession(), requested=True) == "capture_decode_graph_unavailable"
    assert bench._decode_graph_disabled_reason(GraphSession(), requested=True) is None
    assert bench._decode_graph_disabled_reason(HostEmbeddingGraphSession(), requested=True) == "host_token_embedding"


def test_gguf_tensor_inventory_hash_is_stable_and_metadata_sensitive() -> None:
    first = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor("blk.0.attn_q.weight", offset=144, data_offset=4240),
    )
    same = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor("blk.0.attn_q.weight", offset=144, data_offset=4240),
    )
    changed_shape = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor(
            "blk.0.attn_q.weight",
            shape=(2, 5),
            ggml_shape=(5, 2),
            n_elements=10,
            nbytes=160,
            offset=144,
            data_offset=4240,
            byte_shape=(2, 80),
        ),
    )
    changed_qtype = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor(
            "blk.0.attn_q.weight",
            ggml_type=8,
            ggml_type_name="Q8_0",
            offset=144,
            data_offset=4240,
        ),
    )

    digest = bench._gguf_tensor_inventory_hash(first)

    assert len(digest) == 64
    assert digest == bench._gguf_tensor_inventory_hash(same)
    assert digest != bench._gguf_tensor_inventory_hash(changed_shape)
    assert digest != bench._gguf_tensor_inventory_hash(changed_qtype)


def test_gguf_tensor_inventory_summary_exposes_baseline_artifact_fields() -> None:
    info = _model_info(
        _tensor("token_embd.weight", nbytes=144, offset=0, data_offset=4096),
        _tensor("output.weight", nbytes=288, offset=144, data_offset=4240),
    )

    summary = bench._gguf_tensor_inventory_summary(info)

    assert summary["path"] == "/models/gguf/fake.gguf"
    assert summary["architecture"] == "qwen3moe"
    assert summary["tensor_count"] == 2
    assert summary["total_tensor_nbytes"] == 432
    assert summary["tensor_data_offset"] == 4096
    assert summary["tensor_inventory_hash_algorithm"] == "sha256"
    assert summary["tensor_inventory_hash"] == bench._gguf_tensor_inventory_hash(info)
