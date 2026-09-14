# Journey Execution Checkpoint

Status: incomplete campaign. Actual experiments have run; this packet does
not label the remaining inventory as tested.

## Adopted

- `47e9f0b9e`: mapping-only random PLE advice, UD-Q4_K_XL production.
  Matched cold PP improves3.095/2.198/1.287x at512/1K/4K, while the full
  warm matrix and targeted five-pair replication are neutral. Both72-sample
  matrices preserve IDs/final logits/state and zero teardown.
- `04653266c`: exact DPP for the existing tiled-GDN suffix. Full72-sample
  model A/B improves weighted PP0.125%/0.265%/0.262%; no decode-speed claim.
  Kernel CPU-reference, carried-state, determinism and cached trace pass.

These gains have different denominators. Do not multiply the ratios or
quote the cold-cache PLE result as a warm throughput improvement.

## Measured But Not Promoted

| Experiment | Outcome |
| --- | --- |
| Incumbent production, full594 rows | Fails maxKL0.054642 and prefill-last/category scopes; deterministic, clean teardown |
| Q8-down/GDN strict-fallback repair | MaxKL improves0.024408 but prefill-last mean/p95 still fail |
| Multi-column GDN | Owner1.34-2.69x faster, deterministic/numerical smoke; full594 Japanese/prefill scopes fail; unpromoted |
| Q8 residual-weight selected down | Raw-weight MSE improves~100x, costs up to19% vs original WMMA; full594 top1 587/594 and prefill-last mean fail; runtime path removed |
| Sorted PLE dedup | Strong repeated-row gains, all-unique regressions; not universal default |
| Copy-elision model screen | Full12 one-pair exact outputs/state; timing mixed near zero; no promotion |
| Serial/persistent-worker pread | Warm unique rows regress; workers improve cold unique gathers6.3-6.7x versus random mmap, but no safe cache-residency policy is qualified |
| Chunk2048/context4352 | Allocation and teardown succeed; exceeds4GiB modeled scratch by478,952,276 bytes |
| Chunk4096/context4352 | Allocation and teardown succeed; exceeds modeled scratch by3,285,240,660 bytes |

Q8 numerical attribution was corrected during execution: the selected-WMMA
down branch is explicitly nonexact in source. Its omission from an early
family-off control did not demonstrate a broken exact kernel. All raw
localization rows remain available; none is substituted for the full gate.

The repaired Q8 GPU oracle fixed scale-byte packing, raw shape, missing
tile IDs, BF16 output reads and reference rounding. It passes its original
tolerance. Model-level failures are independent of that earlier broken test.

## Current Owner Census

Cached-only role/graph traces, all four canonical categories at4K. No claim
that profiler timings equal unprofiled request throughput:

- Prefill: MoE6.10-6.27s, non-GR linear3.46-3.47s, GR1.46-1.47s,
  QSA1.33-1.34s, GDN0.79-0.85s. GPU PLE kernels~16ms exclude CPU gathers.
- Decode: MoE~17.3ms, other linear~17.1ms, GR~6.6ms, GDN~2.35ms,
  QSA~2.6ms. Forty-eight graphs per step.
- Interval-safe non-kernel prefill window662-827ms; decode~5.6-5.9ms.
  Intra-graph gap~1.3ms is not guaranteed removable time or a PM4 speedup.

All eight prefill/decode captures completed and closed tracked ownership.
The broad external lineage command still encounters the absent hb-pr11
checkout; the newly used pinned DPP lineage was checked independently.

## Remaining Work

The numerical baseline remains the main arithmetic-promotion blocker.
Remaining tasks include real-routing/repair census, allocation-model repair
before larger-chunk defaults, additional quant-native matrix/HC fusions,
cache-aware PLE and asynchronous staging, ordered BF16 QSA depth work,
exact-pin isolated PM4 qualification, batched target MTP verification,
external compatibility gaps and final paired campaign closure.

`artifact.json` compacts completed raw observations with hashes, commands,
host/model/source identity and original verdicts. It preserves failures
and scope limitations. Regenerate:

```bash
python3 benchmarks/results/2026-09-14-journey-progress/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

No raw logits, model weights, compiled libraries or profiler dumps are committed.
