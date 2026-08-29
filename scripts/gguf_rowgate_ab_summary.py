"""Summarize the unequal-dual prefill row-gate A/B from the probe artifacts."""

from __future__ import annotations

import json
import sys

ROWS = (16, 32, 45, 96, 192)


def summary(path: str) -> dict:
    doc = json.load(open(path))
    s = doc["summary"]
    prefill = s["prefill_seconds"]
    return {
        "prefill": float(prefill["median"]) if isinstance(prefill, dict) else float(prefill),
        "id": {int(t) for t in s.get("final_token_ids", ())},
        "logit_finite": bool(s.get("finite_final_logits_all", False)),
    }


def main() -> None:
    rows = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or ROWS
    print(f"{'rows':>5} {'ctrl s':>9} {'cand s':>9} {'delta':>8}  ids/logits")
    for r in rows:
        try:
            c = summary(f"/tmp/he-unequal-{r}-ctrl.json")
            k = summary(f"/tmp/he-unequal-{r}-cand.json")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"{r:>5} pending ({type(exc).__name__})")
            continue
        same = c["id"] == k["id"] and c["logit_finite"] and k["logit_finite"]
        print(
            f"{r:>5} {c['prefill']:>9.4f} {k['prefill']:>9.4f} "
            f"{(c['prefill'] - k['prefill']) / c['prefill'] * 100:>+7.1f}%  "
            f"{'ids match' if same else 'MISMATCH'} ctrl={sorted(c['id'])} cand={sorted(k['id'])}"
        )


if __name__ == "__main__":
    main()
