from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import pytest

from hipengine.kvcache.dms_capture import DMSCaptureWriter, load_dms_capture_manifest
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, qwen35_gguf_config_from_metadata
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_gguf_prefill_streams_dms_hidden_qk_chunks(tmp_path: Path) -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    reader = GGUFReader(MODEL)
    config = qwen35_gguf_config_from_metadata(reader.info)
    physical_layers = tuple(
        index
        for index, layer_type in enumerate(config.layer_types)
        if layer_type == FULL_ATTENTION
    )
    writer = DMSCaptureWriter(
        tmp_path / "capture",
        model_path=str(MODEL),
        model_sha256=_sha256(MODEL),
        data_manifest_sha256="d" * 64,
        tokenizer_identity="local-gguf-fixture",
        tokenizer_sha256="e" * 64,
        physical_layer_ids=physical_layers,
        num_q_heads=int(config.head_count),
        num_kv_heads=int(config.head_count_kv),
        head_dim=int(config.key_length),
        hidden_size=int(config.hidden_size),
        teacher_topk=8,
    )
    token_ids = (760, 4087, 760, 4087)
    writer.begin_sequence(
        sequence_id="gguf-fixture",
        token_ids=token_ids,
        category="general_en",
        provenance={"dataset": "local-fixture", "split": "train"},
    )
    with Qwen35GGUFResidentSession(
        MODEL,
        backend="auto",
        max_sequence_length=8,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        result = session.prefill(
            list(token_ids),
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=True,
            dms_capture=writer,
        )
    writer.finish_sequence()
    manifest = load_dms_capture_manifest(writer.finalize(), verify_shards=True)

    assert result.logits.shape == (1, int(config.vocab_size))
    assert manifest["summary"]["token_count"] == len(token_ids)
    assert manifest["summary"]["shard_count"] == len(physical_layers)
    assert {
        shard["physical_layer_id"]
        for shard in manifest["sequences"][0]["shards"]
    } == set(physical_layers)
