"""Compare two gguf_mtp_c1c8_server_bench packets at one width.

MTP draft-cycle counts drift between runs on this suite (greedy output is
identical, acceptance/wave interleaving is not), so an aggregate MTP wall
comparison is noisy. This script reports the aggregates and then the paired
comparison over prompts whose draft-cycle count is unchanged, which isolates
per-cycle cost. Usage: gguf_server_packet_diff.py CONTROL CANDIDATE [WIDTH]
"""

from __future__ import annotations

import json
import statistics
import sys


def _cells(packet: dict, width: int, arm: str) -> dict[str, tuple[float, int | None]]:
    out: dict[str, tuple[float, int | None]] = {}
    for cell in packet["cells"]:
        if cell["width"] != width:
            continue
        run = cell[arm]["rows"][0]
        cycles = None
        if arm == "mtp":
            spec = run.get("mtp") or {}
            cycles = spec["draft_cycles"] if spec.get("used") else None
        out[cell["prompt_id"]] = (cell[arm]["wall_seconds"], cycles)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    control_path, candidate_path = argv[1:3]
    width = int(argv[3]) if len(argv) > 3 else 8
    control = json.load(open(control_path))
    candidate = json.load(open(candidate_path))
    print(
        f"width C{width}  control={control_path.split("/")[-1]}  "
        f"candidate={candidate_path.split("/")[-1]}"
    )
    for arm in ("ar", "mtp"):
        base, cand = _cells(control, width, arm), _cells(candidate, width, arm)
        shared = sorted(set(base) & set(cand))
        wb = statistics.mean(base[p][0] for p in shared)
        wc = statistics.mean(cand[p][0] for p in shared)
        n_tok = next(
            c[arm]["generated_tokens"] for c in candidate["cells"] if c["width"] == width
        )
        print(
            f"  {arm.upper()}: wall {wb:.4f} -> {wc:.4f} ({(wb - wc) / wb * 100:+.1f}%), "
            f"tok/s {n_tok / wb:.2f} -> {n_tok / wc:.2f}"
        )
        pairs = [
            (p, base[p][1], base[p][0], cand[p][0])
            for p in shared
            if base[p][1] is not None and base[p][1] == cand[p][1]
        ]
        if not pairs:
            continue
        deltas = [(wc - wb) / wb * 100 for _, _, wb, wc in pairs]
        print(
            f"    same-cycle paired n={len(pairs)}: mean {(statistics.mean(deltas)):+.1f}%, "
            f"better on {sum(1 for d in deltas if d < 0)}/{len(pairs)} prompts"
        )
        for p, cycles, wb, wc in pairs:
            print(
                f"      {p[:28]:28s} {cycles:>3d} cyc  {wb:.3f} -> {wc:.3f}  "
                f"{(wb - wc) / wb * 100:+5.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
