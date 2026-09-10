"""Empirical first-pair error envelope; diagnostic, not a confidence bound."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HOST = "55ea6c509d0b49eea8de7094a1023668"


def observations(packet):
    if packet["host"]["machine_id"] != HOST or packet["protocol"]["prefill_chunk_size"] != 1024:
        raise ValueError("requires Framework chunk1024 lane")
    samples = packet["samples"]
    if len(samples) != 72:
        raise ValueError("requires canonical72 samples")
    groups = {}
    for row in samples:
        if row["mode"] not in ("before", "after"):
            raise ValueError("invalid arm")
        for key in ("prefill_ms", "decode_ms"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("invalid timing")
        groups.setdefault(row["case_id"], []).append(row)
    if len(groups) != 12:
        raise ValueError("requires twelve cases")
    result = []
    for name, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["sequence_slot"])
        if [r["sequence_slot"] for r in rows] != list(range(6)):
            raise ValueError("invalid counterbalanced slots")
        if {r["mode"] for r in rows[:2]} != {"before", "after"}:
            raise ValueError("first pair must contain both arms")
        first=rows[0]["mode"]
        other="after" if first=="before" else "before"
        if [r["mode"] for r in rows] != [first,other,other,first,first,other]:
            raise ValueError("requires canonical counterbalanced order")
        for mode in ("before", "after"):
            if sorted(r["repetition"] for r in rows if r["mode"] == mode) != [0,1,2]:
                raise ValueError("invalid repetitions")
        if len({r["output_token_ids_sha256"] for r in rows}) != 1:
            raise ValueError("trajectory mismatch")
        for metric in ("prefill", "request"):
            def total(selected, mode):
                return sum(r["prefill_ms"] + (r["decode_ms"] if metric == "request" else 0)
                           for r in selected if r["mode"] == mode)
            first = total(rows[:2], "before") / total(rows[:2], "after")
            aggregate = total(rows, "before") / total(rows, "after")
            result.append(dict(case_id=name,metric=metric,first_ratio=first,
                aggregate_ratio=aggregate,log_error=math.log(first/aggregate)))
    return result


def calibrate(training, heldout):
    if not training or not heldout:
        raise ValueError("requires nonempty training and heldout packets")
    bound = {m:max(abs(r["log_error"]) for rows in training for r in rows if r["metric"] == m)
             for m in ("prefill", "request")}
    evaluations=[]
    for rows in heldout:
        decisions=[]
        for row in rows:
            b=bound[row["metric"]]
            log_ratio=math.log(row["first_ratio"])
            label="win" if log_ratio>b else "loss" if log_ratio < -b else "uncertain"
            decisions.append(dict(**row,decision=label,outside_envelope=abs(row["log_error"])>b,
                wrong_sign=(label=="win" and row["aggregate_ratio"]<=1) or
                           (label=="loss" and row["aggregate_ratio"]>=1)))
        evaluations.append(dict(rows=decisions,
            all_case_early_win=all(r["decision"]=="win" for r in decisions),
            violations=sum(r["outside_envelope"] for r in decisions),
            wrong_signs=sum(r["wrong_sign"] for r in decisions)))
    return dict(log_error_envelope=bound,
                multiplicative_envelope={k:math.exp(v) for k,v in bound.items()},
                heldout=evaluations)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,action="append",required=True)
    p.add_argument("--heldout",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    sources={}
    identities=[]
    def load(paths):
        rows=[]
        for path in paths:
            raw=path.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if digest in sources:
                raise ValueError("duplicate or overlapping train/heldout packet")
            packet=json.loads(raw)
            sources[digest]=str(path)
            identities.append(packet["model_identity"])
            rows.append(observations(packet))
        return rows
    training=load(args.train)
    heldout=load(args.heldout)
    if any(identity != identities[0] for identity in identities):
        raise ValueError("model identities differ")
    result=calibrate(training,heldout)
    result.update(schema=1,sources=sources,train=[str(p) for p in args.train],
        heldout_paths=[str(p) for p in args.heldout],promotion_policy_changed=False,
        limits="Empirical envelope from a small selected historical sample. Three-pair means are noisy, not ground truth. This does not estimate confidence, certify early promotion, or cover unseen thermal/clock/interference conditions.")
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
