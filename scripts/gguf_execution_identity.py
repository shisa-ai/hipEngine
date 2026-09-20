#!/usr/bin/env python3
"""Print the GGUF execution identity of one or more artifacts.

    python3 scripts/gguf_execution_identity.py ~/models/model.gguf

The printed fingerprint is what a serving-evidence row's artifact axis binds
(`artifact_execution_fingerprint`).  It covers the tensor table and the
routing-relevant metadata, so a revision that differs only in tokenizer content
or provenance strings keeps the same identity, while a different quantization or
layer layout does not.

Only the header and tensor table are read, so this is fast on multi-GB files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import (  # noqa: E402
    gguf_execution_fingerprint,
    scan_gguf,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="GGUF files to identify",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object per artifact instead of aligned text",
    )
    args = parser.parse_args(argv)

    status = 0
    for artifact in args.artifacts:
        path = Path(artifact).expanduser()
        try:
            info = scan_gguf(path)
        except OSError as error:
            print(f"{path}\tERROR\t{error}", file=sys.stderr)
            status = 1
            continue
        fingerprint = gguf_execution_fingerprint(info)
        size_bytes = path.stat().st_size
        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "path": str(path),
                        "size_bytes": size_bytes,
                        "tensor_count": info.tensor_count,
                        "execution_fingerprint": fingerprint,
                    },
                    sort_keys=True,
                )
            )
            continue
        print(f"{fingerprint}  {size_bytes:>13}  {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
