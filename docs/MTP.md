# hipEngine MTP Native Implementation Plan

> Status: implementation plan. This is the sister document to
> [`DFLASH.md`](DFLASH.md). MTP should be implemented **after** the shared
> native verifier/commit infrastructure from DFlash exists, not as a separate
> c=1 native-loop tuning lane.

## Thesis

Qwen3.5/Qwen3.6 MTP has the same core bottleneck that blocked the Python
DFlash harness: it can improve accepted-token density, but it cannot beat AR if
target verification remains a sequence of near-c=1 decode steps plus rollback
or accepted-prefix re-forward.

The hipEngine MTP path should therefore be a thin speculative-draft plugin on
top of the same native infrastructure planned in `DFLASH.md`:

- `DraftBatch` candidate metadata;
- `TargetVerifyBatch` / verifier-internal root row materialization;
- `KVLiveSpans` with `span_role="verify_chain"`;
- transactional KV/state scratch and commit;
- GPU target top1/accept summaries;
- graph buckets keyed by verify shape;
- exact greedy equality against MTP-disabled AR.

MTP differs from DFlash only in the **proposal provider**. DFlash uses a
separate block-draft model plus draft context KV. MTP uses target-attached MTP
weights to propose a short chain. The verifier, accept, commit, graph, and
benchmark contracts should be shared.

## Alignment with existing hipEngine design

hipEngine already has the right abstract contracts:

| Existing design piece | MTP use |
| --- | --- |
| `hipengine.speculative.DraftBatch` | Carries candidate draft rows, depths, request ids, and optional tree parents. MTP emits chain candidates; it does **not** include the already-committed root row. |
| `hipengine.speculative.DraftModel` | MTP head/plugin implements `propose(...)`. It is model-attached, not an external draft model. |
| `hipengine.speculative.Verifier` | Shared target verifier verifies a `DraftBatch` by internally materializing `[root, candidates...]`. |
| `hipengine.speculative.AcceptResult` | Stores accepted counts/tokens plus optional transaction, selected-row, and target next-token provenance per request; should be extended later with compact device-summary provenance. |
| `KVLiveSpans(span_role="verify_chain")` | Full-attention verify rows write into scratch/journal spans, not canonical KV. |
| `KVPolicy.begin_transaction/commit/rollback` | Accepted prefix commit and rejected suffix discard. |
| Graph bucket key `(mode, draft/tree shape, C, context bucket, ...)` | MTP fixed-depth buckets such as B=1/2/3/5. |

Design clarification shared with `DFLASH.md`: `DraftBatch.candidate_tokens` are
candidate rows only. The target verifier builds a runtime `VerifyBatch` with a
root row at slot 0 and candidate rows at slots `1..N-1`. Docs and tests should
not encode the root token as a `DraftBatch` candidate.

## Current evidence from `~/amd-gpu-tuning`

Source plan: `~/amd-gpu-tuning/PLAN-MTP.md`.

Retained native MTP rows show useful correctness and acceptance, but not speed:

| Evidence | Result |
| --- | --- |
| Best B=5 native-loop / target-graph-replay row | exact same-session equality, MTP `83.88 tok/s` vs AR `120.04 tok/s` = `0.699x` |
| Acceptance | accepted `20/29` drafts, average committed output `2.91` tokens/iteration |
| Scalar syncs | still ~60 scalar D2H reads in retained row |
| Longer B=5/B=6 windows | around `82 tok/s` vs `~120 tok/s`; accepted depth rose but speed did not |
| True bulk torch verifier | argmax-correct but far too slow; torch grouped MoE dominated profile |
| Profile orientation | W8A16 linear/MoE/lm-head family ~51.5% kernel time; full attention ~1.9% |

Budget reminder from the parent plan:

```text
T_iter <= A_out * T_AR / target_speedup
```

At AR `~120 tok/s`, `T_AR ~= 8.33 ms`. With `A_out ~= 2.91`, a `1.10x`
speedup needs the full MTP iteration under `~22 ms`. Current correct rows are
closer to `~35 ms/iteration`. The missing piece is not another MTP policy
sweep; it is a faster exact target verifier and cheap state commit.

## MTP model boundary

The Qwen3.5/Qwen3.6 MTP module is target-attached:

```text
input:  RMSNorm(token_embedding) + RMSNorm(target_hidden)
        -> concat
        -> mtp.fc
        -> one decoder layer (full attention + MoE, same family as target)
        -> mtp.norm
        -> shared target lm_head
output: next-token logits / top1
```

Local bring-up artifact from the parent workspace:

```text
/models/qwen36-quant/Qwen3.6-35B-A3B-Quark-W8A8-INT8-MTP-BF16
```

That artifact hardlinks the Quark W8A8 target shards and adds BF16 `mtp.*`
tensors. It is a bring-up artifact, not necessarily the final quantized MTP
layout.

Important external facts:

- vLLM's `Qwen3_5MultiTokenPredictor` defines the same boundary:
  normalized embedding + normalized target hidden -> `mtp.fc` -> one decoder
  layer -> final norm -> shared lm-head.
- vLLM warns that using one MTP layer for `num_speculative_tokens > 1` repeats
  the same predictor and can reduce acceptance. hipEngine should measure small
  depths first rather than assuming deeper is better.
- llama.cpp-style speculative decode verifies `[last, draft0, ...]` in one
  target batch and crops/restores to the committed prefix. hipEngine should do
  the same logically, but with transactional scratch/commit instead of a
  measured accepted-prefix re-forward.

## Shared infrastructure with DFlash

MTP should not create a parallel verifier stack. The following dependencies are
shared and should land through `DFLASH.md` first:

1. **Native `TargetVerifyBatch`.**
   A fixed-shape batch of root + candidate rows with token ids, positions,
   parent/depth metadata, output hidden/final rows, logits/top1 buffers, and
   per-layer state scratch.

2. **Exact selectable target state.**
   For every verify row the target forward must expose enough state to commit
   that row without target re-forward:
   - full-attention K/V rows;
   - linear-attention Conv/GDN state;
   - hidden taps / final hidden needed by the next proposal step;
   - target top1/logit summary.

3. **Transactional KV/state commit.**
   Rejected rows never touch canonical KV/state. Accepted rows are copied or
   journal-committed through the scheduler-owned transaction.

4. **GPU target top1 and accept summary.**
   MTP must not copy full logits to host per row. It needs a compact summary:
   accepted depth, committed token ids, first correction/bonus token, and graph
   validation/status flags.

5. **Graph bucket discipline.**
   Fixed MTP depths are graph shapes. Initial buckets should be small:
   `B={1,2,3}` and only then `B=5` if the measured split supports it.

6. **Measurement schema.**
   Same fields as DFlash: exact equality, finite logits, AR tok/s, spec tok/s,
   target verify rows/time, draft/proposal time, commit/replay time, rows/output,
   accepted-depth histogram, scalar D2H count, graph status, and peak memory.

## MTP runtime flow

For one request, with chain draft depth `B`:

```text
1. AR or previous commit has produced the current root token and target hidden.
2. MTP propose step uses (root embedding, target hidden, MTP KV/state if any)
   to emit draft token d1.
3. For depth > 1, either repeat the MTP predictor using the newly proposed token
   and MTP state, or use an explicitly supported multi-depth MTP path. Measure;
   do not assume repetition is profitable.
4. MTP plugin returns DraftBatch candidates [d1, d2, ... dB] with depths 1..B.
5. Shared verifier internally builds [root, d1, ... dB] and runs target verify.
6. Device accept compares target top1 at row i to candidate d{i+1}.
7. Commit root + accepted draft prefix + correction/bonus, selecting target
   state from the verified accepted row.
8. Expose the committed target hidden/final row for the next MTP proposal.
```

For c>1, the same flow is row-mapped by `request_id` and physical batch slot;
MTP-specific code must not assume one global scalar root or one global scalar
position.

## Phased implementation plan

### Phase M0 — Wait for the shared verifier baseline

Do not start a speed lane until the DFlash/native verifier work has landed at
least these pieces:

- native topk=1 chain `TargetVerifyBatch`;
- exact selectable per-row target state;
- no accepted-prefix target re-forward in the measured fast path;
- GPU target top1 + chain accept summary;
- benchmark rows proving the verifier is faster than serial c=1 verification.

MTP can use CPU/reference tests before M0, but retained speed work should wait.

### Phase M1 — Model metadata and native MTP proposal oracle

Goal: prove hipEngine can load and execute the MTP proposal module outside the
main generation loop.

- Add model-plugin metadata for target-attached MTP tensors.
- Extend loading/materialization for `mtp.*` tensors from the bring-up artifact.
- Implement or port MTP proposal kernels in parent-compatible order:
  - embedding/root input preparation;
  - hidden/embedding RMSNorm;
  - `mtp.fc`;
  - one MTP decoder layer;
  - `mtp.norm`;
  - shared lm-head + GPU top1/topk.
- Compare native MTP top1/topk against the parent Python/native-loop harness on
  fixed hidden/token fixtures.
- No speed claim yet; this is a proposal correctness gate.

### Phase M2 — DraftModel plugin and DraftBatch chain output

Goal: wire MTP as a `DraftModel` provider without touching target commit logic.

- Implement `MtpDraftProvider.propose(...) -> DraftBatch`.
- Emit candidate tokens only; do not include the root token in `DraftBatch`.
- Fill `request_ids`, `candidate_tokens`, `parent_positions`, `draft_depths`,
  `row_to_request`, `active_mask`, and `mode="verify_chain"`.
- Add CPU fixtures for forced accept patterns at B=1/2/3/5.
- Add telemetry: proposal depth, draft time, proposal top1/top2 margin where
  available, and MTP KV/state bytes.

### Phase M3 — Shared verifier integration

Goal: make MTP consume the same exact verifier as DFlash.

- Feed the MTP `DraftBatch` into the shared chain verifier.
- Use verifier-internal root row + candidates for target logits/state.
- Use the shared chain accept kernel or a small MTP-specialized variant if the
  summary layout differs.
- Commit selected target state without target re-forward.
- Ensure the accepted row exposes target hidden/final hidden for the next MTP
  proposal.
- Correctness gate: exact greedy equality vs MTP-disabled AR on fixed prompts,
  including reject-at-depth-0, partial accept, and full accept cases.

### Phase M4 — Fixed-depth graph buckets and small-depth sweep

Goal: find the useful MTP depth after verifier economics improve.

Initial buckets:

| Bucket | Purpose |
| --- | --- |
| B=1 | baseline one MTP proposal; should be cheap and exact |
| B=2 | likely first real speed candidate if verifier row cost is low |
| B=3 | check acceptance-depth tradeoff |
| B=5 | compare to parent retained B5 evidence only after B1-B3 are understood |

Report for every bucket:

- same-session AR tok/s;
- MTP tok/s and vs AR;
- average committed output tokens/iteration;
- accepted-depth histogram;
- target verify rows/output and verify eta;
- MTP proposal time;
- accept/commit/host overhead;
- scalar/vector D2H count;
- peak memory and MTP KV/state bytes;
- graph direct vs replay validation.

### Phase M5 — MTP cache/state and long-context gate

Goal: avoid short-context wins that collapse with prompt length.

- Keep MTP KV/state append-only where possible; do not rebuild prompt-side MTP
  cache in the measured loop.
- Measure prompt lengths `512`, `4096`, `16384`, `32768`, and `65536` where
  memory permits after a short-context row wins.
- Classify regressions as verifier cost, MTP prompt-cache rebuild, MTP KV memory
  growth, acceptance/position shift, or target attention/KV cost.

### Phase M6 — Policy, top-k rescue, and tree variants

Only after M3/M4 show the verifier is cheap and proposal cost is not dominant:

- root top-k / margin guard;
- adaptive depth cap;
- target-guided top-k oracle for upper bounds;
- MTP tree/branching experiments using `mode="verify_tree"` and the DDTree
  compiler/accept infrastructure from `DFLASH.md`.

Branching must remain parent-linked and prefix-closed: no accepted path may use a
state row whose ancestors were not accepted.

## Correctness gates

MTP has stricter continuation requirements than a pure draft-only provider:

- exact greedy output equality vs MTP-disabled AR;
- target verifier rows finite;
- MTP proposal logits finite;
- accepted prefix state equals serial target state at the first token after the
  commit;
- target hidden/final hidden selected for the accepted row matches serial c=1;
- rejected suffix does not leak into target KV, linear-attention state, MTP KV,
  or output ring;
- disabling MTP produces byte-identical deterministic output to the normal
  target path on fixtures.

Layer-ladder debug is required before promotion: on a mismatch, compare serial
c=1 vs bulk verify row at each layer boundary and report the first failing
layer/family.

## Promotion gates

A retained hipEngine MTP row must satisfy:

| Gate | Requirement |
| --- | --- |
| Correctness | exact same-session AR equality, finite target/MTP logits, no state leak |
| Speed | >1.10x same-session AR on a short retained prompt before policy work |
| Verifier | target verify eta low enough to explain the speedup; no accepted-prefix re-forward |
| Accounting | rows/output, accepted histogram, proposal/verify/commit split, D2H count |
| Memory | MTP weights + target + target KV + MTP state/KV under the active gate |
| Artifacts | compact JSON under `benchmarks/results/`, rollup/changelog updated per `docs/BENCHMARK.md` |

If MTP does not beat AR after the shared verifier is fast, run a fresh split
before optimizing: proposal cost may dominate, or acceptance may be insufficient
for the measured prompt class.

## Do-not-chase list

Until the shared verifier is exact and faster than serial c=1 verification, do
not spend iterations on:

- deeper fixed-B sweeps;
- margin guards or root-topk rescue;
- adaptive depth policy;
- draft-token/sec headlines;
- allocator-only buffer cleanup;
- graph dry-runs that still replay c=1 target steps;
- final quantized-MTP packaging;
- attention-only tuning when profile says linear/MoE/lm-head dominates;
- any speed claim that omits same-session AR and exact equality.

## First concrete hipEngine tasks after DFlash

1. Confirm `docs/DFLASH.md` D1-D2 verifier/accept pieces are landed and faster
   than serial c=1 on chain B=1/2/4/8.
2. Add MTP tensor metadata/materialization for the Quark W8A8 + BF16 MTP
   bring-up artifact.
3. Add native MTP proposal oracle tests against parent fixed fixtures.
4. Implement `MtpDraftProvider` producing `DraftBatch` chain rows.
5. Feed MTP drafts into the shared chain verifier and accept kernel.
6. Add exact commit-state tests for reject, partial, and full accept.
7. Benchmark B=1/2/3, then B=5 only if the split supports it.
8. Revisit top-k/tree policy only after a flat-chain MTP row beats AR.
