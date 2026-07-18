# GGUF MTP / llama.cpp Parity Dashboard

Last reviewed: 2026-07-19.

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

### gfx1100 current refresh, Radeon Pro W7900

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR | llama.cpp HIP bundled MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2 natural24, reusable B1/B2 target graphs | Natural25 request / 24 timed transitions | B2, natural25 request / 24 timed transitions |
| Decode | **98.75 fixed / 96.75 natural24 tok/s** | 68.50 tok/s | **122.67 tok/s** | 78.05 tok/s transition-normalized | 115.44 tok/s transition-normalized |
| MTP / own AR | 1.0000x | 0.6936x | **1.2679x** | n/a | **1.4791x** |
| Draft acceptance | n/a | 73.53% | 80.45% | n/a | 81.56% |
| Accepted draft/output | n/a | 50.00% | 60.00% | n/a | 58.40% |
| Complete wall per output/transition | 10.336 ms natural24 | 14.696 ms | **8.186 ms** | 12.812 ms | 8.662 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native autoregressive | native compatibility target |

The reusable native target boundary closes the W7900 parity gap. One
fixed-address graph per B1/B2 bucket consumes live device token, position,
context, and cursor metadata; five two-row tails use B1 and four true one-row
cycles stay on AR. Two clean full-suite processes at `0d7b86e7` measure
**123.33 and 122.67 tok/s** (0.54% spread). The conservative run is **1.2679x**
its **96.75 tok/s** true graph AR and **6.26% above** llama.cpp's **115.44
tok/s / 8.662 ms-transition** floor, with complete wall **5.50% lower** at
**8.186 ms/output**.

Acceptance is unchanged at **80.45% draft / 60.00% accepted-output**. Both clean
runs preserve all 240 output IDs and all 96 cycle semantics from the prior
eager-target `llama-compat` route. Full/train/heldout are **1.2679x / 1.2973x /
1.2257x** their true AR controls; every category is at least **1.1990x AR**.
The real 35B target graph oracle is byte-exact across two B2 positions plus B1
for target top-1, 16,384 hidden values, all captured/resident Conv/GDN state,
all full-attention K/V, and cursors.

A cached six-step replay trace records zero measured captures and **18.67 ms
host / 13.67 ms kernels / 5.00 ms residual**, versus the prior eager verifier's
**52.42 / 14.01 / 38.41 ms**. It sees the expected dynamic-metadata,
cursor-advance, and top-1 widening leaves. N2 accept/commit is now the next
ownership step, not the parity blocker that target submission was.

Exact remains the semantic control; `llama-compat` remains explicit and
accuracy-traded. This retained gfx1100 result does not promote automatic exact
MTP or imply gfx1151 support. hipEngine uses BF16 KV and llama.cpp F16 KV; the
external row remains `performance_claim=false`.

Artifacts: [retained reusable route](../benchmarks/results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json),
[prior hipEngine baseline](../benchmarks/results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json),
and [current llama.cpp MTP floor](../benchmarks/results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json).

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
| 0 | W7900 `llama-compat` vs llama.cpp MTP | **Closed 2026-07-19:** conservative **122.67 vs 115.44 tok/s**, 8.186 vs 8.662 ms/output/transition | Preserve the retained B1/B2 graph oracle and rerun after measured-path changes; do not transfer to gfx1151 or automatic exact MTP without independent gates. |
| 0 | N2 device accept/commit | Open; target parity no longer blocks it | Consume target top-1 and produce accepted count, commit/reseed rows, state/KV transaction, and bounded outputs on device without intermediate host policy/readback. |
| 0 | Exact natural-horizon economics | Open | The full multi-prompt category suite beats the true same-protocol AR control at the requested output horizon with exact/default state semantics. |
| 0 | Exact-ID OpenAI c1/c2/c4/c8 refresh | Awaiting rerun | One clean artifact joins exact IDs, provenance, queue/backend/verifier shapes, owned timing, request wall, and same-server AR/MTP controls. |
| 1 | Complete-cycle/public adapter | Open | N3 owns proposal, target, accept/commit, cursor, and bounded output under one native call; public c1 and later coalesced routes preserve exact requested IDs and timing ownership. |
| 1 | Compatibility semantic decision | Open | Preserve `llama-compat` as explicitly accuracy-traded or produce an exact state lifecycle with the same end-to-end advantage. |
| 2 | gfx1100 portability | **Closed 2026-07-12** | W7900 24-step graph state gate, full graph-AR/exact/`llama-compat` suites, and transition-matched rebuilt llama.cpp base comparator are published; rerun only when the named route/build/protocol changes. |

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

Retained gfx1100 reusable-native `llama-compat` natural24 suite (gfx1151 keeps
its existing route until an independent native-target gate):

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=<device> HIPENGINE_HIP_ARCH=gfx1100 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-native-cycle \
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

Use [`scripts/gguf_mtp_verifier_rocprof.py`](../scripts/gguf_mtp_verifier_rocprof.py)
for this GGUF target route. Do not wrap the parent prompt-suite/economics
harness in `rocprofv3`.

## References

- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [Dated parity notebook](MTP-LLAMACPP-PARITY-HISTORY.md)
- [MTP design](MTP.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
