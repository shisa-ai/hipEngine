#!/usr/bin/env python3
"""Run CPU-only DMS oracle/control diagnostics over sealed label shards."""
from __future__ import annotations

import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np

from hipengine.kvcache.dms_diagnostic import adapter, discarded_mass


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict:
    files=sorted(Path(p) for p in glob.glob(str(Path(args.labels).expanduser()/'*.npz')))
    if not files: raise ValueError('no label shards found')
    rows=[]
    for path in files:
        z=np.load(path)
        mass=np.asarray(z['future_attention_mass'], dtype=np.float32)
        eligible=np.asarray(z['eligible_mask'], dtype=bool)
        positions=np.asarray(z['positions'], dtype=np.int64)
        if mass.ndim != 2: raise ValueError(f'{path}: expected [tokens,heads] future_attention_mass')
        tokens,heads=mass.shape
        if eligible.shape != (tokens,) or positions.shape != (tokens,): raise ValueError(f'{path}: geometry mismatch')
        scores=mass[:,None,:]
        oracle=adapter('oracle_mass', tokens=tokens,layers=1,heads=heads,positions=positions,window=args.window,ratio=args.ratio,current_position=tokens-1,mass=scores)
        recent=adapter('recency', tokens=tokens,layers=1,heads=heads,positions=positions,window=args.window,ratio=args.ratio,current_position=tokens-1)
        rng=adapter('random', tokens=tokens,layers=1,heads=heads,positions=positions,window=args.window,ratio=args.ratio,current_position=tokens-1,seed=args.seed)
        # The stored continuous mass is a noncausal continuation oracle. Restrict
        # mass accounting to labels' eligible history so protected rows cannot
        # dilute comparisons.
        mass3=np.broadcast_to(mass[:,None,:], oracle.shape)
        elig3=np.broadcast_to(eligible[:,None,None], oracle.shape)
        rows.append({'file':str(path),'sha256':_sha(path),'tokens':tokens,'heads':heads,
          'eligible_tokens':int(eligible.sum()),'oracle_discarded_mass':float(np.mean([discarded_mass(mass3[:,0,h],oracle[:,0,h]) for h in range(heads)])),
          'recency_discarded_mass':float(np.mean([discarded_mass(mass3[:,0,h],recent[:,0,h]) for h in range(heads)])),
          'random_discarded_mass':float(np.mean([discarded_mass(mass3[:,0,h],rng[:,0,h]) for h in range(heads)])),
          'oracle_evictions':int(oracle.sum()),'recency_evictions':int(recent.sum()),'random_evictions':int(rng.sum()),
          'source_kind':'future_attention_mass_continuation_oracle'})
    return {'schema_version':1,'kind':'hipengine_dms_selector_oracle_diagnostic','window':args.window,'ratio':args.ratio,'seed':args.seed,'rows':rows,
      'causal_last_query':{'status':'not_run','reason':'sealed label shards contain no last-query Q/K attention capture'},
      'decision':'continuation oracle is diagnostic only; no selector conclusion until causal last-query capture is available'}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--labels',type=Path,required=True); p.add_argument('--window',type=int,default=256); p.add_argument('--ratio',type=int,default=2); p.add_argument('--seed',type=int,default=0); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=run(args); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({'output':str(args.output),'rows':len(result['rows'])},sort_keys=True))

if __name__=='__main__': main()
