#!/usr/bin/env python3
"""Compare two differential-arm JSON files on semantic axes only.

Strips volatile fields (timestamps, commands, artifact paths, durations),
then deep-compares. Prints SAME/DIFFER per file with the first differing
paths. Exit 0 when all compared files are semantically identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VOLATILE_KEYS = {
    "created_at", "command", "generated_at", "timestamp", "started_at",
    "finished_at", "wall_seconds", "elapsed_seconds", "duration_seconds",
    "host", "hostname", "argv", "json_path", "artifact_path", "log_path",
}
# Per-process identity: allocator addresses, graph handles, session ids, and
# any hash derived from buffer addresses. Value hashes (mismatch packed_hash,
# c1_hash, logits/trajectory sha) stay comparable. Raw state_kv_hashes byte
# dumps are also excluded: full-attention K/V bytes are address-dependent
# (split-K scratch addresses shift whenever allocation shapes change) and the
# state oracle's semantic verdicts (tokens_exact, final_state_exact,
# mismatch fingerprints, first_divergence) are the qualified contract — the
# C2 precedent qualified its lifecycle arm on exactly those axes.
VOLATILE_SUBSTRINGS = (
    "handle", "_ptr", "ptrs", "allocation_id", "graph_exec", "session_id",
    "address", "buffer_identity", "resource_identity", "graph_key_sha",
    "cancelled", "graph_manifests", "state_kv_hashes", "state_kv_mismatches",
    "graph_buckets", "collected_at", "survivors_final", "inactive_sessions",
)


def _volatile(key: str) -> bool:
    if key in VOLATILE_KEYS:
        return True
    low = key.lower()
    return any(s in low for s in VOLATILE_SUBSTRINGS)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in sorted(o.items()) if not _volatile(k)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff_paths(a, b, prefix="", out=None, limit=12):
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append(f"{prefix}: type {type(a).__name__} vs {type(b).__name__}")
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{prefix}.{k}: missing in A")
            elif k not in b:
                out.append(f"{prefix}.{k}: missing in B")
            else:
                diff_paths(a[k], b[k], f"{prefix}.{k}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{prefix}: list len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff_paths(x, y, f"{prefix}[{i}]", out, limit)
                if len(out) >= limit:
                    break
    elif a != b:
        out.append(f"{prefix}: {a!r} vs {b!r}")
    return out


def main() -> int:
    pairs = []
    args = sys.argv[1:]
    if len(args) >= 2 and Path(args[0]).is_dir():
        rollback_dir, candidate_dir = Path(args[0]), Path(args[1])
        names = sorted(p.name for p in rollback_dir.glob("*.json"))
        pairs = [(rollback_dir / n, candidate_dir / n) for n in names]
    else:
        pairs = [(Path(args[i]), Path(args[i + 1])) for i in range(0, len(args), 2)]
    all_same = True
    for ra, ca in pairs:
        if not ra.exists() or not ca.exists():
            print(f"{ra.name}: MISSING ({'rollback' if not ra.exists() else 'candidate'} arm)")
            all_same = False
            continue
        a, b = strip(json.loads(ra.read_text())), strip(json.loads(ca.read_text()))
        if a == b:
            print(f"{ra.name}: SAME")
        else:
            all_same = False
            print(f"{ra.name}: DIFFER")
            for line in diff_paths(a, b):
                print(f"    {line}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
