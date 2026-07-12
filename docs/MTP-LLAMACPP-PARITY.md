# GGUF MTP / llama.cpp Parity Dashboard

Last reviewed: 2026-07-12.

This file is the current decision surface for GGUF MTP parity. The verbatim
experiment notebook is preserved in
[`MTP-LLAMACPP-PARITY-HISTORY.md`](MTP-LLAMACPP-PARITY-HISTORY.md). Labels such
as “current,” completion checklists, and concurrency rates inside that dated
notebook describe the revision at which they were written; they do not override
this dashboard or [`benchmarks/README.md`](../benchmarks/README.md).

## Current Status

The two hipEngine GGUF routes exercise different semantic contracts on both
supported RDNA backends. Exact/default is serial-prefix preserving;
`llama-compat` uses direct partial commit/dp4a and remains an explicit,
accuracy-traded replication lane. llama.cpp HIP is an external diagnostic
comparator, not a promoted hipEngine topline.

### gfx1100 current transfer, Radeon Pro W7900

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: | ---: |
| Route | No MTP | B3, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural24 diagnostic |
| Decode | 34.49 fixed / 34.28 natural24 tok/s | 45.96 tok/s | 52.58 tok/s | 119.05 tok/s |
| MTP / own AR | 1.0000x | 1.3328x | 1.5337x | 1.4612x |
| Draft acceptance | n/a | 73.53% | 82.95% | 81.18% |
| Accepted draft/output | n/a | 50.00% | 60.83% | 57.50% |
| Cycle/backend wall per output | n/a | 21.847 ms | 19.045 ms | 8.400 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The full ten-prompt gate passes for both hipEngine routes. `llama-compat` beats
its true AR control in every category and on heldouts: full/train/heldout are
**1.5337x / 1.5761x / 1.4744x**, while train/heldout draft acceptance is
**88.12% / 76.00%**. The W7900 oracle independently passes four byte-exact
serial-prefix transitions across hidden state, all 30 Conv/GDN families, and
all 10 live K/V families. The exact fixed-cycle row and natural24 columns use
different output horizons and must not be ranked as one protocol. llama.cpp
uses prebuilt HIP binary `263cc04a5`/build 9600 and remains
`performance_claim=false`. Its `119.05/81.47 tok/s` values are native
llama.cpp reporting, not transition-normalized cross-engine rates: that run
requested 24 outputs, but llama.cpp starts `predicted_ms` after the first
output is sampled. Do not rank that column directly until W7900 is rerun with
the `N+1` transition contract below.

Artifact: [W7900 GGUF MTP transfer](../benchmarks/results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json).

### gfx1151 current refresh, Radeon 8060S

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2, natural24/cyclecap24 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 51.81 tok/s (0.9571x own AR) | **69.50 tok/s (1.2776x own AR)** | 69.44 tok/s native (1.3752x; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **69.38 tok/s** | 66.66 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **54.40 tok/s** | 48.47 tok/s |
| Cross-engine MTP / own AR | n/a | 1.2755x | 1.3752x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Complete wall/output or timed transition | 19.360 ms/output | 14.413 ms/output | 15.001 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp compatibility target |

The current correctness/state-lifecycle pass changes the exact/default result
materially: B5 is **51.81 versus 54.14 AR tok/s (0.9571x)**, so exact MTP no
longer wins on gfx1151. The old 61.98 tok/s row remains historical evidence,
not a current headline. Replay/commit increased from 0.019 to 2.766 ms/output,
accounting for most of the 16.162 to 19.360 ms/output cycle-wall increase.

Explicit `llama-compat` remains a full-suite win: full/train/heldout are
**1.2776x / 1.3034x / 1.2408x**, train/heldout acceptance is
**82.08% / 71.79%**, and all four categories beat their own true AR controls.
It remains opt-in because direct partial commit is not serial-prefix-equivalent.

At matched decode boundaries, hipEngine `llama-compat` is **69.38 tok/s** and
llama.cpp is **66.66 tok/s** over exactly 24 timed transitions per prompt.
hipEngine uses BF16 KV and llama.cpp uses F16 KV, so this is timer-matched but
not identical model execution. The llama.cpp source is dirty but its complete
instrumentation patchset is retained; the binary hash is authoritative and the
external row remains `performance_claim=false`.

The clean HIP 7.15 gfx1151 oracle repeats llama.cpp token `9707` and passes four
byte-exact serial-prefix transitions across hidden state, all 40 layer outputs,
all 30 Conv/GDN state families, and all 10 live K/V families.

Artifacts: [current gfx1151 refresh](../benchmarks/results/2026-07-12-gfx1151-gguf-mtp-refresh.json),
[llama.cpp patchset](../benchmarks/llama.cpp/README.md),
[historical exact B5](../benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json),
and [historical `llama-compat` B2](../benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json).

## Cross-Engine Decode Timing Boundary

The canonical boundary is maintained in
[`benchmarks/README.md`](../benchmarks/README.md#cross-engine-gguf-decode-timing-contract).
In compact form:

1. hipEngine AR counts returned post-prefill `session.step()` transitions;
2. hipEngine MTP cross-engine rates use complete `cycle_wall_ms` rather than
   the narrower legacy draft+target stage sum;
3. llama.cpp samples its first output before `t_start_generation`, so native
   `predicted_n / predicted_ms` counts one untimed token per request;
4. request `N+1` llama.cpp outputs and report
   `sum(predicted_n - 1) * 1000 / sum(predicted_ms)` for `N` comparable timed
   transitions;
5. client/HTTP wall and direct decode wall remain separate scopes.

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
| 2 | gfx1100 portability | **Closed 2026-07-12** | W7900 oracle plus full exact/default, `llama-compat`, and llama.cpp HIP comparator are published; rerun only when the named route/build/protocol changes. |

No new kernel or route is retained from a single prompt. Acceptance, speed, and
quality changes use the complete category suite plus held-outs, as required by
[`BENCHMARK.md`](BENCHMARK.md).

## Canonical Reruns

Exact/default fixed-cycle suite:

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=<device> HIPENGINE_HIP_ARCH=<gfx1100-or-gfx1151> \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-exact-full.json
```

`llama-compat` natural24 suite:

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=<device> HIPENGINE_HIP_ARCH=<gfx1100-or-gfx1151> \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 --cycles 24 --max-output-tokens 24 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-llama-compat-natural24.json
```

llama.cpp transition-matched natural prompts (`N=24`):

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /path/to/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx-size 8192 --concurrency 1 --gpu-layers 99 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --draft-max 2 --mode both --protocol natural \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-tokens 25 --seed 12345 --temperature 0 \
  --top-k 1 --top-p 1 --min-p 0 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --output benchmarks/results/<date>-llamacpp-natural25.json
```

Use the emitted `aggregate_decode_transition_per_second` and
`transition_normalized_*` fields for cross-engine tables; retain native
`predicted_per_second` only as llama.cpp self-reporting.

Use [`scripts/mtp_verifier_rocprof.py`](../scripts/mtp_verifier_rocprof.py) for
verifier profiling. Do not wrap the parent prompt-suite/economics harness in
`rocprofv3`.

## References

- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [Dated parity notebook](MTP-LLAMACPP-PARITY-HISTORY.md)
- [MTP design](MTP.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
