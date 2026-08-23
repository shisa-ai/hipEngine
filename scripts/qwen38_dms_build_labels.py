#!/usr/bin/env python3
"""Build deterministic future-attention DMS labels from checksummed Qwen3.8 captures."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kvcache.dms_labels import (
    build_dms_label_artifact,
    future_attention_mass_cpu,
    load_dms_label_manifest,
)


def future_attention_mass_torch(
    query: np.ndarray,
    key: np.ndarray,
    *,
    window_size: int,
    device: str,
    query_tile: int,
    query_stride: int = 1,
) -> np.ndarray:
    """Deterministic tiled PyTorch implementation for practical corpus labeling."""

    import torch

    queries = np.asarray(query, dtype=np.float32)
    keys = np.asarray(key, dtype=np.float32)
    if queries.ndim != 3 or keys.ndim != 3:
        raise ValueError("DMS GPU oracle Q/K must be rank-3")
    tokens, q_heads, head_dim = queries.shape
    if keys.shape[0] != tokens or keys.shape[2] != head_dim:
        raise ValueError("DMS GPU oracle Q/K shapes do not align")
    kv_heads = int(keys.shape[1])
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("DMS GPU oracle Q heads must divide into KV groups")
    tile = int(query_tile)
    stride = int(query_stride)
    window = int(window_size)
    if tile <= 0 or stride <= 0 or window < 0:
        raise ValueError(
            "DMS GPU query_tile/query_stride must be positive and window non-negative"
        )
    torch.use_deterministic_algorithms(True)
    q = torch.as_tensor(queries, dtype=torch.float32, device=device)
    k = torch.as_tensor(keys, dtype=torch.float32, device=device)
    mass = torch.zeros((tokens, kv_heads), dtype=torch.float64, device="cpu")
    key_positions = torch.arange(tokens, device=device, dtype=torch.int64)
    group_size = q_heads // kv_heads
    scale = float(head_dim) ** -0.5
    with torch.no_grad():
        for kv_head in range(kv_heads):
            head_keys = k[:, kv_head]
            group = q[:, kv_head * group_size : (kv_head + 1) * group_size]
            for start in range(0, tokens, tile * stride):
                end = min(tokens, start + tile * stride)
                query_positions = torch.arange(
                    start, end, stride, device=device, dtype=torch.int64
                )
                logits = (
                    torch.einsum(
                        "bgd,kd->bgk",
                        group.index_select(0, query_positions),
                        head_keys,
                    )
                    * scale
                )
                causal = key_positions.view(1, 1, tokens) <= query_positions.view(-1, 1, 1)
                logits = logits.masked_fill(~causal, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                old = (
                    query_positions.view(-1, 1) - key_positions.view(1, -1)
                ) > window
                contribution = (
                    probabilities.sum(dim=1) * old.to(dtype=probabilities.dtype)
                ).sum(dim=0)
                mass[:, kv_head] += contribution.to(device="cpu", dtype=torch.float64)
    return mass.numpy()


def _git_provenance(
    *, device: str, query_tile: int, query_stride: int
) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    uname = platform.uname()
    provenance: dict[str, Any] = {
        "source_commit": head,
        "working_tree_clean": not bool(status.strip()),
        "command": [str(value) for value in sys.argv],
        "device": str(device),
        "query_tile": int(query_tile),
        "query_stride": int(query_stride),
        "host": {
            "node": uname.node,
            "system": uname.system,
            "release": uname.release,
            "machine": uname.machine,
        },
    }
    if device != "cpu":
        import torch

        provenance["torch"] = {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(torch.device(device)),
        }
    return provenance


def run(args: argparse.Namespace) -> dict[str, Any]:
    captures = args.captures.expanduser().resolve()
    if captures.is_dir():
        captures = captures / "capture_manifest.json"
    if not captures.is_file():
        raise FileNotFoundError(captures)
    device = str(args.device)
    query_tile = int(args.query_tile)
    query_stride = int(args.query_stride)
    if query_stride <= 0:
        raise ValueError("query_stride must be positive")
    if device == "cpu":
        if query_stride != 1:
            raise ValueError("query_stride subsampling requires a GPU torch device")
        mass_builder = future_attention_mass_cpu
        backend = "cpu_numpy_float64"
        score_dtype = "float64"
    else:
        mass_builder = lambda query, key, *, window_size: future_attention_mass_torch(
            query,
            key,
            window_size=window_size,
            device=device,
            query_tile=query_tile,
            query_stride=query_stride,
        )
        backend = f"torch_{device}_float32_tiled_query_stride{query_stride}"
        score_dtype = "float32_device_float64_host"
    label_manifest = build_dms_label_artifact(
        captures,
        args.output_dir,
        target_compression_ratio=int(args.target_cr),
        window_size=int(args.window_size),
        mass_builder=mass_builder,
        compute_backend=backend,
        compute_score_dtype=score_dtype,
        compute_provenance=_git_provenance(
            device=device,
            query_tile=query_tile,
            query_stride=query_stride,
        ),
    )
    manifest = load_dms_label_manifest(label_manifest, verify_shards=True)
    return {
        "status": "labeled",
        "label_manifest": str(label_manifest),
        "label_manifest_sha256": label_manifest.with_suffix(
            label_manifest.suffix + ".sha256"
        ).read_text(encoding="ascii").strip(),
        "objective": manifest["objective"],
        "compute": manifest["compute"],
        "summary": manifest["summary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--target-cr", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-tile", type=int, default=128)
    parser.add_argument("--query-stride", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
