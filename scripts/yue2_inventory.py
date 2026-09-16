#!/usr/bin/env python3
"""YuE2 checkpoint inventory: per-component tensor and byte accounting.

Reads only the safetensors headers (O(header) I/O, no payload), so it is cheap
enough to run before every residency decision. ``--verify-hash`` additionally
checks the payload against the checkpoint's ``weights_manifest.json`` sha256,
which is the identity recorded in ``docs/MODEL-YUE2.md``.

Usage:
    python3 scripts/yue2_inventory.py --model <dir> --vae <dir> --out artifacts.json
    python3 scripts/yue2_inventory.py --model <dir> --diff artifacts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes  # noqa: E402

# Expected checkpoint identity from docs/MODEL-YUE2.md.
PINNED = {
    "model": {
        "repo": "m-a-p/YuE2-3B",
        "revision": "29b3558dd46954a0cd9021dc76d5c91864a0f1c7",
        "sha256": "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
        "tensors": 628,
        "bytes": 7261441640,
    },
    "vae": {
        "repo": "m-a-p/YuE2-Vae",
        "revision": "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a",
        "sha256": None,
        "tensors": None,
        "bytes": 530512720,
    },
}


def _component(name: str) -> str:
    """Collapse per-layer indices so the inventory groups by component."""
    if name.startswith("model.layers."):
        return "model.layers.N." + name.split(".", 3)[3]
    if name.startswith("time_embedder.mlp."):
        return "time_embedder.mlp." + name.split(".")[-2]
    for prefix in ("decoder.layers.", "encoder.layers."):
        if name.startswith(prefix):
            return prefix + "N." + name.split(".", 3)[3]
    return name


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(path: Path) -> dict[str, str]:
    manifest_path = path / "weights_manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text())
    return {
        str(name): str(entry.get("sha256"))
        for name, entry in data.get("files", {}).items()
        if isinstance(entry, dict) and entry.get("sha256")
    }


def inventory(path: Path, *, verify_hash: bool = False) -> dict:
    index = load_weight_index(path)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    tensors = []
    dtypes: dict[str, int] = defaultdict(int)
    for name, info in index.tensors.items():
        nbytes = info.nbytes
        if nbytes is None:
            raise ValueError(f"tensor {name!r} has an unsupported dtype {info.dtype!r}")
        group = _component(name)
        groups[group]["tensors"] += 1
        groups[group]["bytes"] += nbytes
        dtypes[info.dtype] += nbytes
        tensors.append(
            {"name": name, "dtype": info.dtype, "shape": list(info.shape), "bytes": nbytes}
        )

    payload_bytes = sum(info.nbytes or 0 for info in index.tensors.values())
    manifest = _manifest_sha256(index.model_path)
    shards = []
    for shard in index.shards:
        entry = {
            "name": shard.name,
            "file_bytes": shard.stat().st_size,
            "tensor_bytes": sum(
                info.nbytes or 0 for info in index.tensors.values() if info.shard_path == shard
            ),
            "declared_sha256": manifest.get(shard.name),
        }
        if verify_hash:
            entry["sha256"] = sha256_file(shard)
            entry["sha256_matches_manifest"] = entry["declared_sha256"] in (None, entry["sha256"])
        shards.append(entry)

    return {
        "path": str(index.model_path),
        "config": index.config,
        "shards": shards,
        "tensor_count": len(index.tensors),
        "payload_bytes": payload_bytes,
        "dtypes": dict(sorted(dtypes.items())),
        "components": {
            key: dict(value) for key, value in sorted(groups.items())
        },
        "tensors": sorted(tensors, key=lambda item: item["name"]),
    }


def _expected_kind(path: Path, forced: str | None) -> str | None:
    if forced:
        return forced
    text = str(path)
    if "YuE2-Vae" in text or "YuE2-Vae" in text.replace("-", "-"):
        return "vae"
    if "YuE2-3B" in text:
        return "model"
    return None


def compare(entry: dict, kind: str | None, previous: dict) -> list[str]:
    problems: list[str] = []
    if entry["tensor_count"] != previous["tensor_count"]:
        problems.append(
            f"tensor count {entry['tensor_count']} != {previous['tensor_count']}"
        )
    if entry["payload_bytes"] != previous["payload_bytes"]:
        problems.append(
            f"payload bytes {entry['payload_bytes']} != {previous['payload_bytes']}"
        )
    old = {item["name"]: item for item in previous["tensors"]}
    new = {item["name"]: item for item in entry["tensors"]}
    for name in sorted(set(old) - set(new)):
        problems.append(f"missing tensor {name}")
    for name in sorted(set(new) - set(old)):
        problems.append(f"unexpected tensor {name}")
    for name in sorted(set(old) & set(new)):
        if old[name]["dtype"] != new[name]["dtype"] or old[name]["shape"] != new[name]["shape"]:
            problems.append(
                f"tensor {name} changed: {old[name]['dtype']}{old[name]['shape']} -> "
                f"{new[name]['dtype']}{new[name]['shape']}"
            )
    if kind in PINNED:
        pinned = PINNED[kind]
        if pinned["tensors"] is not None and entry["tensor_count"] != pinned["tensors"]:
            problems.append(
                f"tensor count {entry['tensor_count']} != pinned {pinned['tensors']}"
            )
        if pinned["bytes"] is not None and entry["payload_bytes"] != pinned["bytes"]:
            problems.append(
                f"payload bytes {entry['payload_bytes']} != pinned {pinned['bytes']}"
            )
        for shard in entry["shards"]:
            if pinned["sha256"] and shard.get("declared_sha256") not in (None, pinned["sha256"]):
                problems.append(
                    f"manifest sha256 {shard['declared_sha256']} != pinned {pinned['sha256']}"
                )
            if shard.get("sha256_matches_manifest") is False:
                problems.append(f"shard {shard['name']} payload sha256 does not match manifest")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="YuE2-3B directory (or safetensors file)")
    parser.add_argument("--vae", help="YuE2-Vae directory")
    parser.add_argument("--out", help="write the inventory JSON here")
    parser.add_argument("--diff", help="compare against a previously written inventory JSON")
    parser.add_argument("--verify-hash", action="store_true", help="sha256 the payloads")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result: dict[str, dict] = {}
    for kind, path in (("model", args.model), ("vae", args.vae)):
        if path is None:
            continue
        entry = inventory(Path(path), verify_hash=args.verify_hash)
        result[kind] = entry
        if not args.quiet:
            print(f"== {kind}: {entry['path']}")
            print(
                f"   tensors={entry['tensor_count']} payload={entry['payload_bytes'] / 2**20:.1f} MiB "
                f"dtypes={entry['dtypes']}"
            )
            for shard in entry["shards"]:
                extra = ""
                if "sha256" in shard:
                    extra = f" sha256={'ok' if shard['sha256_matches_manifest'] else 'MISMATCH'}"
                print(
                    f"   shard {shard['name']}: {shard['tensor_bytes'] / 2**20:.1f} MiB of "
                    f"{shard['file_bytes'] / 2**20:.1f} MiB{extra}"
                )
            for name, value in entry["components"].items():
                print(f"   {name:60s} {value['tensors']:4d} {value['bytes'] / 2**20:10.2f} MiB")

    problems: list[str] = []
    if args.diff:
        previous = json.loads(Path(args.diff).read_text())
        for kind, entry in result.items():
            problems.extend(
                f"[{kind}] {problem}" for problem in compare(entry, kind, previous[kind])
            )
        if problems:
            print("DIFF PROBLEMS:")
            for problem in problems:
                print("  " + problem)
        elif not args.quiet:
            print("diff: inventory matches")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not args.quiet:
            print(f"wrote {args.out}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
