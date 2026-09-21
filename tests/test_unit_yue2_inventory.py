"""M0 gate: the checkpoint inventory accounts for file, payload, and header bytes.

``scripts/yue2_inventory.py`` is oracle-side tooling that reads only safetensors
headers. Its byte accounting has to keep three quantities apart - the on-disk
file size the checkpoint's ``weights_manifest.json`` declares, the tensor payload
inside it, and the JSON header between them - because a payload compared against
a file size reports a mismatch equal to the header on every checkpoint.

These tests build a synthetic safetensors shard so the accounting is exercised
without the real 7.8 GB checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_inventory():
    spec = importlib.util.spec_from_file_location(
        "yue2_inventory", REPO / "scripts/yue2_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["yue2_inventory"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory_module():
    return _load_inventory()


def _write_shard(directory: Path, tensors: dict[str, tuple[str, list[int], int]]) -> Path:
    """Write a minimal safetensors shard: ``name -> (dtype, shape, fill)``."""

    sizes = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
    header: dict[str, dict] = {}
    offset = 0
    for name, (dtype, shape, _) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        length = count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    blob = json.dumps(header).encode()
    padding = (-len(blob)) % 8
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob) + padding))
        handle.write(blob + b" " * padding)
        for name, (_, _, fill) in tensors.items():
            length = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            handle.write(bytes([fill]) * length)
    return path


def _config(directory: Path) -> None:
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "yue2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 16,
                "vocab_size": 32,
                "head_dim": 4,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "latent_dim": 4,
                "max_latent_frames": 64,
                "timestep_shift": 1.0,
                "architectures": ["YuE2ForCausalLM"],
            }
        )
    )


@pytest.fixture()
def synthetic_checkpoint(tmp_path):
    directory = tmp_path / "YuE2-3B"
    _write_shard(
        directory,
        {
            "model.embed_tokens.weight": ("BF16", [32, 8], 0),
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8], 1),
            "model.layers.1.self_attn.q_proj.weight": ("BF16", [8, 8], 2),
            "lm_head.weight": ("F32", [32, 8], 3),
        },
    )
    _config(directory)
    return directory


def test_inventory_separates_file_payload_and_header(inventory_module, synthetic_checkpoint):
    entry = inventory_module.inventory(synthetic_checkpoint)
    shard = synthetic_checkpoint / "model.safetensors"
    payload = (32 * 8 * 2) + (8 * 8 * 2) + (8 * 8 * 2) + (32 * 8 * 4)
    assert entry["payload_bytes"] == payload
    assert entry["file_bytes"] == shard.stat().st_size
    assert entry["header_bytes"] == shard.stat().st_size - payload
    assert entry["header_bytes"] > 0
    assert entry["tensor_count"] == 4
    # Components collapse per-layer indices but keep the two layers' bytes.
    assert entry["components"]["model.layers.N.self_attn.q_proj.weight"]["tensors"] == 2
    assert entry["shards"][0]["tensor_bytes"] == payload
    assert entry["shards"][0]["file_bytes"] == shard.stat().st_size


def test_inventory_round_trips_through_compare_without_problems(inventory_module, synthetic_checkpoint):
    first = inventory_module.inventory(synthetic_checkpoint)
    second = inventory_module.inventory(synthetic_checkpoint)
    assert inventory_module.compare(second, None, first) == []


def test_compare_rejects_a_payload_file_confusion(inventory_module, synthetic_checkpoint):
    entry = inventory_module.inventory(synthetic_checkpoint)
    # A pinned file size checked against a payload is exactly the bug this
    # accounting exists to prevent: it must be reported, not silently accepted.
    tampered = dict(entry)
    tampered["payload_bytes"] = entry["file_bytes"]
    problems = inventory_module.compare(tampered, None, entry)
    assert any("payload" in problem for problem in problems)


def test_compare_rejects_a_changed_tensor_shape(inventory_module, synthetic_checkpoint):
    entry = inventory_module.inventory(synthetic_checkpoint)
    changed = json.loads(json.dumps(entry))
    changed["tensors"][0]["shape"] = [1]
    problems = inventory_module.compare(changed, None, entry)
    assert any("changed" in problem for problem in problems)


def test_pinned_identity_declares_file_and_payload_separately(inventory_module):
    for kind, pinned in inventory_module.PINNED.items():
        assert pinned["file_bytes"] is not None, kind
        assert pinned["tensor_bytes"] is not None, kind
        assert pinned["tensor_bytes"] < pinned["file_bytes"], kind
        header = pinned["file_bytes"] - pinned["tensor_bytes"]
        assert 0 < header < 1 << 20, f"{kind} header size {header} is not plausible"


def test_inventory_reports_the_pinned_checkpoints(inventory_module):
    """When the real checkpoints are cached, their bytes must match the pins."""

    cache = Path.home() / ".cache/huggingface/hub"
    model = sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*"))
    vae = sorted(cache.glob("models--m-a-p--YuE2-Vae/snapshots/*"))
    if not model or not vae:
        pytest.skip("YuE2 checkpoints are not cached on this host")
    for kind, directory in (("model", model[-1]), ("vae", vae[-1])):
        if not (directory / "model.safetensors").is_file() and not list(
            directory.glob("*.safetensors")
        ):
            pytest.skip(f"{kind} snapshot has no safetensors file")
        entry = inventory_module.inventory(directory)
        pinned = inventory_module.PINNED[kind]
        assert entry["file_bytes"] == pinned["file_bytes"]
        assert entry["payload_bytes"] == pinned["tensor_bytes"]
        if pinned["tensors"] is not None:
            assert entry["tensor_count"] == pinned["tensors"]
        assert inventory_module.compare(entry, kind, entry) == []
