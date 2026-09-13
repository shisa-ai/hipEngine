#!/usr/bin/env python3
"""TP shard planning + byte-preservation evidence for a GGUF model.

Emits one JSON artifact per run:

  * per-degree shard manifest hashes and per-rank byte totals
  * per-tensor byte-exactness of shard -> reconstruct round trips
  * per-quant-type and per-split-kind byte accounting

This is the Packet 2 gate: no weight may be dequantized, requantized, or
reordered, so a shard set is only usable when every tensor round-trips
bit-exactly against the source GGUF payload.

Usage:
    python3 scripts/tp_shard_plan_report.py --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --world-size 1 --world-size 2 --world-size 4 \
        --output benchmarks/results/tp2_shard_plan_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf_shards import (  # noqa: E402
    ShardPlanError,
    build_shard_manifest,
    streaming_memory_report,
    verify_shard_bytes,
)

GIB = 1024**3


def file_sha256(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    """Stream a file hash so binding the manifest to the artifact is cheap."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a .gguf model file")
    parser.add_argument(
        "--world-size",
        type=int,
        action="append",
        default=None,
        help="TP degree to plan and verify; repeatable (default: 1 2 4)",
    )
    parser.add_argument("--output", default=None, help="write the JSON artifact here")
    parser.add_argument(
        "--skip-bytes",
        action="store_true",
        help="plan only; skip the byte-preservation round trip",
    )
    parser.add_argument("--owner-rank", type=int, default=0, help="rank owning embedding/lm_head")
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip the model SHA-256 (streamed once, ~10 s for a 16 GiB file)",
    )
    parser.add_argument(
        "--stream-rank",
        type=int,
        default=None,
        help="stream this rank's shards and report peak memory (proves the loader needs one tensor, not a full copy)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2
    degrees = args.world_size or [1, 2, 4]

    started = time.time()
    info = scan_gguf(model_path)
    reader = GGUFReader(model_path)
    artifact: dict = {
        "kind": "tp_shard_plan_report",
        "model": str(model_path),
        "model_bytes": int(model_path.stat().st_size),
        "model_sha256": None if args.skip_hash else file_sha256(model_path),
        "tensor_count": len(info.tensors),
        "owner_rank": int(args.owner_rank),
        "degrees": {},
        "rejected_degrees": {},
    }

    for degree in degrees:
        entry: dict = {"world_size": int(degree)}
        try:
            manifest = build_shard_manifest(
                info,
                world_size=degree,
                owner_rank=int(args.owner_rank),
                model_hash=str(artifact["model_sha256"] or ""),
            )
        except ShardPlanError as error:
            artifact["rejected_degrees"][str(degree)] = str(error)
            if not args.quiet:
                print(f"[N={degree}] REJECTED: {error}")
            continue
        entry["manifest_hash"] = manifest.manifest_hash()
        entry["tensor_count"] = len(manifest.tensors)
        entry["rank_bytes"] = [manifest.rank_bytes(rank) for rank in range(degree)]
        entry["rank_gib"] = [round(manifest.rank_bytes(rank) / GIB, 3) for rank in range(degree)]
        entry["rank_bytes_by_kind"] = [manifest.rank_summary(rank) for rank in range(degree)]
        entry["notes"] = list(manifest.notes)
        entry["total_sharded_bytes"] = sum(plan.source_nbytes for plan in manifest.tensors)
        if not args.quiet:
            print(
                f"[N={degree}] tensors={entry['tensor_count']} "
                f"rank_gib={entry['rank_gib']} hash={entry['manifest_hash'][:12]}"
            )
        if not args.skip_bytes:
            check_started = time.time()
            report = verify_shard_bytes(reader, manifest)
            entry["byte_check"] = report
            entry["byte_check_seconds"] = round(time.time() - check_started, 2)
            if not args.quiet:
                status = "bit-exact" if report["bit_exact"] else f"MISMATCH {report['mismatches'][:3]}"
                print(
                    f"[N={degree}] bytes: {report['tensors_checked']} tensors "
                    f"{report['source_bytes'] / GIB:.2f} GiB {status} "
                    f"({entry['byte_check_seconds']}s)"
                )
        if args.stream_rank is not None and 0 <= int(args.stream_rank) < int(degree):
            stream_started = time.time()
            stream = streaming_memory_report(reader, manifest, rank=int(args.stream_rank))
            stream["seconds"] = round(time.time() - stream_started, 2)
            entry["streaming_rank"] = stream
            if not args.quiet:
                print(
                    f"[N={degree}] stream rank {args.stream_rank}: {stream['tensors']} tensors "
                    f"{stream['local_bytes'] / GIB:.2f} GiB, largest {stream['largest_tensor_bytes'] / 1024**2:.1f} MiB, "
                    f"anonymous growth {stream['anonymous_growth_bytes'] / 1024**2:.1f} MiB "
                    f"vs {stream['full_model_copy_bytes'] / GIB:.2f} GiB full copy ({stream['seconds']}s)"
                )
        artifact["degrees"][str(degree)] = entry

    artifact["elapsed_seconds"] = round(time.time() - started, 2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        if not args.quiet:
            print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
