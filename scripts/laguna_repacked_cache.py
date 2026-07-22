#!/usr/bin/env python3
"""Build or validate a versioned Laguna replacement-layout host cache."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf_materialize import (
    build_laguna_repacked_cache,
    open_laguna_repacked_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--selected-slot", action="append", dest="selected_slots")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _proc_read_bytes() -> int | None:
    try:
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split(":", 1)[1].strip())
    except (FileNotFoundError, OSError, ValueError):
        pass
    return None


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def main() -> int:
    args = _parser().parse_args()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    if not args.model.is_file():
        raise SystemExit(f"model is not a file: {args.model}")

    reader = GGUFReader(args.model)
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    read_before = _proc_read_bytes()

    if args.validate_only:
        cache = open_laguna_repacked_cache(
            args.cache,
            reader,
            source_sha256=args.source_sha256,
        )
        manifest = dict(cache.manifest)
    else:

        def progress(index, total, spec) -> None:
            if index == 1 or index % args.progress_every == 0 or index == total:
                print(
                    f"CACHE {index}/{total} elapsed_s={time.perf_counter() - started:.3f} "
                    f"slot={spec.slot_path} layout={spec.layout} "
                    f"resident={spec.resident_nbytes}",
                    flush=True,
                )

        manifest = build_laguna_repacked_cache(
            reader,
            args.cache,
            selected_slots=args.selected_slots,
            source_sha256=args.source_sha256,
            progress=progress,
        )

    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema": 1,
        "command": [sys.executable, *sys.argv],
        "mode": "validate" if args.validate_only else "build",
        "model": str(args.model.resolve()),
        "cache": str(args.cache.resolve()),
        "seconds": time.perf_counter() - started,
        "physical_read_bytes": _delta(read_before, _proc_read_bytes()),
        "minor_faults": int(usage_after.ru_minflt - usage_before.ru_minflt),
        "major_faults": int(usage_after.ru_majflt - usage_before.ru_majflt),
        "max_rss_bytes": int(usage_after.ru_maxrss) * 1_024,
        "manifest": {
            "schema": manifest["schema"],
            "layout_version": manifest["layout_version"],
            "source": manifest["source"],
            "plan_fingerprint": manifest["plan_fingerprint"],
            "entry_count": manifest["entry_count"],
            "resident_nbytes": manifest["resident_nbytes"],
            "cacheable_layouts": manifest["cacheable_layouts"],
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
