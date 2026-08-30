"""Classify packed-vs-c1 state mismatches from the packed-AR state oracle.

The oracle stores blake2b hashes per (row, component, part, layer), so a mismatch
list can answer a question the pass/fail flag cannot: is the packed state *wrong*,
or is it written to the wrong *slot*? Per (phase, component, part, layer) this
reports rows that match at the same index, rows whose packed value equals some other
row's reference value (a slot-index/permutation defect), and rows whose packed value
matches no reference (unwritten or corrupted).

Usage: gguf_state_slot_hash_match.py ORACLE_JSON [--verbose]
"""

from __future__ import annotations

import argparse
import collections
import json


def classify(phase: str, mismatches: list[dict]) -> collections.Counter:
    """Bucket mismatches by how the packed value relates to the reference set."""

    by_group: dict[tuple, list[dict]] = {}
    for row in mismatches:
        key = (phase, row.get("component"), row.get("part"), row.get("layer"))
        by_group.setdefault(key, []).append(row)

    tally: collections.Counter = collections.Counter()
    for key, rows in sorted(by_group.items(), key=lambda kv: tuple(map(str, kv[0]))):
        packed_by_row = {int(r["row"]): r.get("packed") for r in rows}
        reference_by_row = {int(r["row"]): r.get("c1") for r in rows}
        if None in packed_by_row.values() or None in reference_by_row.values():
            tally[("missing-field", key[0], key[1])] += 1
            continue
        reference_values = {v: r for r, v in reference_by_row.items()}
        identity = permuted = unmatched = 0
        for row, packed in sorted(packed_by_row.items()):
            if packed == reference_by_row.get(row):
                identity += 1
            elif packed in reference_values:
                permuted += 1
                print(f"  {key} row {row}: packed equals c1 row {reference_values[packed]}")
            else:
                unmatched += 1
        if permuted:
            klass = "permuted"
        elif unmatched and not identity:
            klass = "unmatched"
        else:
            klass = "same-index"
        tally[(klass, str(key[0]), str(key[1]))] += 1
        print(
            f"{str(key):54s} rows={len(rows):3d} same-index={identity:3d} "
            f"permuted={permuted:3d} unmatched={unmatched:3d}"
        )
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("oracle_json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    doc = json.load(open(args.oracle_json))

    print(
        f"status={doc.get('status')} mode={doc.get('prefill_mode')}/{doc.get('decode_mode')} "
        f"tokens_exact={doc.get('tokens_exact')} initial_state_exact={doc.get('initial_state_exact')} "
        f"final_state_exact={doc.get('final_state_exact')}"
    )
    first = doc.get("first_divergence")
    if first:
        print("first_divergence:", json.dumps(first)[:220])

    tally: collections.Counter = collections.Counter()
    for phase, key in (("prefill", "initial_mismatches"), ("post-decode", "final_mismatches")):
        rows = doc.get(key) or []
        print(f"\n{phase}: {len(rows)} mismatches")
        tally.update(classify(phase, rows))

    print("\ntotals:")
    for (klass, phase, component), count in sorted(tally.items()):
        print(f"  {phase:12s} {str(component):10s} {klass:14s} {count}")
    if not doc.get("tokens_exact", True):
        print("\nWARNING: token equality did not hold; state analysis is secondary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
