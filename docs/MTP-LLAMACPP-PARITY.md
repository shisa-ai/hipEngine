# GGUF MTP / llama.cpp Parity Dashboard

Last reviewed: 2026-07-11.

This file is the current decision surface for GGUF MTP parity. The verbatim
experiment notebook is preserved in
[`MTP-LLAMACPP-PARITY-HISTORY.md`](MTP-LLAMACPP-PARITY-HISTORY.md). Labels such
as “current,” completion checklists, and concurrency rates inside that dated
notebook describe the revision at which they were written; they do not override
this dashboard or [`benchmarks/README.md`](../benchmarks/README.md).

## Current Status

The two hipEngine GGUF columns retained by the canonical scoreboard exercise
different semantic contracts. The llama.cpp HIP column is included as an
external diagnostic comparator, not promoted as a repository topline.

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Headline route | B5, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural24 diagnostic |
| Headline MTP decode | 61.98 tok/s (1.1312x own AR) | 71.52 tok/s (1.3055x own AR) | 71.91 tok/s (1.3835x own AR) |
| Matched natural24 B2 MTP decode | 52.04 tok/s (diagnostic) | 71.52 tok/s | 71.91 tok/s |
| Matched natural24 own AR | 54.80 tok/s | 54.79 tok/s | 51.98 tok/s |
| Matched natural24 cycle wall/output | 19.248 ms | 14.005 ms | 14.269 ms |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The exact/default route remains the semantic control. Its retained 61.98 tok/s
row uses a fixed-cycle horizon and cannot be ranked directly against the
natural24 columns. `llama-compat` is the closer 1:1 performance comparison with
llama.cpp because both use the B2 natural24 shape, but it is not
serial-prefix-equivalent and remains an opt-in replication lane rather than the
production default. The locally instrumented llama.cpp stage rerun has dirty
source provenance and `performance_claim=false`; it is a comparison target,
not an eligible standalone topline.

Artifacts: [retained exact B5](../benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json),
[exact natural24 B1-B5](../benchmarks/results/2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json),
[retained `llama-compat` B2](../benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json),
and [llama.cpp HIP B2 stage rerun](../benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json).

The underlying gfx1151 eager target control is now correctness-certified by
[`SOL-G1`](SOL-OPTIMIZATION.md): for the exact Q4_K_M model and
`[9707] * 512`, llama.cpp and hipEngine emit the same five-token repeated
trajectory, and four teacher-forced transitions match fresh serial prefixes
byte-for-byte across hidden rows, Conv/GDN state, and live K/V. This validates
the repeated stream; it does not refresh any MTP or AR speed row.

## Server And Concurrency Status

There is no eligible OpenAI MTP server timing row.

`SOL-E1`, `SOL-E2`, `SOL-E3`, and `SOL-S2` now provide exact generated IDs
across every choice, one owner for batch-scoped timing, canonical provenance,
and separate route-cap, queue-group, backend-width, and verifier-width shapes.
`SOL-E5` proves direct/HTTP exact-token parity for the shared 512/128 route.
Those contracts postdate the July 6 server measurements, so the old c1/c2/c4/c8
rates remain historical diagnostics. A c8 client request under the current cap
is two four-request queue/backend groups; it is not evidence for one width-8
verifier.

The next server headline must come from a fresh exact-ID rerun. It must report
full-request throughput separately from owned backend/verifier timing and must
not reconstruct completion counts from decoded text.

## Open Work

| Priority | Item | Current state | Exit gate |
| ---: | --- | --- | --- |
| 0 | Exact natural-horizon economics | Open | The full multi-prompt category suite beats the true same-protocol AR control at the requested output horizon with exact/default state semantics. |
| 0 | Exact-ID OpenAI c1/c2/c4/c8 refresh | Awaiting rerun | One clean artifact joins exact IDs, provenance, queue/backend/verifier shapes, owned timing, request wall, and same-server AR/MTP controls. |
| 1 | Current verifier-stage attribution | Awaiting the corrected rerun | Profile the final child process after cache warmup; rank target verify, LM-head/sample, proposal/update, commit/scatter, and host synchronization by owned wall. |
| 1 | Compatibility semantic decision | Open | Either preserve `llama-compat` as an explicitly accuracy-traded mode or produce an exact state lifecycle with the same end-to-end advantage. |
| 2 | gfx1100 portability | Blocked on hardware | Rerun the same exact/default and compatibility contracts on W7900; do not transfer gfx1151 magnitudes. |

No new kernel or route is retained from a single prompt. Acceptance, speed, and
quality changes use the complete category suite plus held-outs, as required by
[`BENCHMARK.md`](BENCHMARK.md).

## Canonical Reruns

Exact/default fixed-cycle suite:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-exact-full.json
```

`llama-compat` natural24 suite:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 --cycles 24 --max-output-tokens 24 \
  --record-cycle-stage-timings \
  --output benchmarks/results/<date>-ar-mtp-llama-compat-natural24.json
```

Use [`scripts/mtp_verifier_rocprof.py`](../scripts/mtp_verifier_rocprof.py) for
verifier profiling. Do not wrap the parent prompt-suite/economics harness in
`rocprofv3`.

## References

- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [Dated parity notebook](MTP-LLAMACPP-PARITY-HISTORY.md)
- [MTP design](MTP.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
