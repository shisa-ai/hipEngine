# hipEngine MTP Native Implementation Plan

> Status (2026-05-21): shared ABI + metadata/loading scaffold landed; a local
> PARO+MTP-BF16 artifact is assembled for bring-up. A single-step native
> proposal smoke matches the torch-reference top-1. E2E correctness is exact at
> B=2/3/5 on the stable quicksort prompt. **Speed is blocked at 0.87× AR (B=3,
> chain_tloop, gfx1151)**; closing the gap to llama.cpp’s 1.5–2× wins requires
> new HIP kernels (Tasks #45–#52, see WORKLOG.md 2026-05-19/2026-05-20).
> This is the sister document to [`DFLASH.md`](DFLASH.md). MTP must reuse the
> shared native verifier/commit infrastructure from DFlash, not fork a separate
> c=1 native-loop tuning lane.

> **Top priority for the next push:** batched selected-expert MoE GEMV with an
> ids-tensor ABI (see ["Closing the gap with llama.cpp MTP"](#closing-the-gap-with-llamacpp-mtp--kernel-roadmap-2026-05-21)).
> Expected savings ~10–20 ms of the ~45 ms verifier kernel time, which is
> enough to clear the >14% break-even bar at B=3 perfect-accept.

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

For the current `dflash` branch, the DFlash-built verifier is now exact enough
for MTP integration work: chain B+1 verification, GPU accept summaries,
transactional state/KV commit, batched full-attention verification, and DDTree
verification are all landed as shared infrastructure.  It is **not** yet a speed
win over serial c=1: the true-batched chain verifier is 6-8% faster than
`c1_loop` at B=2/4 but remains `2.0-5.0x` slower than serial c=1, and the first
real branching DDTree proposer beats chain/tree baselines at B=2/4/8 but still
loses to serial.  MTP speed work therefore remains blocked on the same verifier
row-cost wall.

What is now landed for MTP (2026-05-19):

- provider-neutral chain metadata in `hipengine.speculative.chain`:
  `ChainDraftRequest`, `ChainDraftCompiler`, and `compile_chain_draft()`;
- `hipengine.speculative.mtp` with `MtpProposalContext`, `MtpDraftProvider`,
  `Qwen35MtpDraftProvider`, `MtpChainCompiler`, and `compile_mtp_chain()`;
- target-attached Qwen3.5/Qwen3.6 `mtp.*` metadata/loading in
  `hipengine.loading.mtp`, including validation and `load_qwen35_mtp_bf16_weights()`;
- `scripts/mtp_chain_e2e_bench.py`, a readiness diagnostic that records the MTP
  chain ABI and refuses to fake a speed row when tensors are missing.

The original shisa packed PARO target snapshot
`501ef8635e5cfb5a7497d232358ca8d1afc0c66e` contains `0/19` expected `mtp.*`
tensors; the retained artifact
`benchmarks/results/2026-05-19-hipengine-mtp-chain-readiness-missing-tensors-diagnostic.json`
records that initial `blocked_missing_mtp_tensors` state.

A local bring-up artifact now exists at
`/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16`.  It reuses
(symlinks) the packed PARO trunk and adds `mtp-bf16.safetensors` generated from
`Qwen/Qwen3.6-35B-A3B-FP8` `mtp.safetensors`: BF16 tensors are copied as-is,
FP8 block-128 projection/expert tensors are dequantized to BF16, and per-expert
`gate_proj`/`up_proj` are fused into the hipEngine runtime layout
`mtp.layers.0.mlp.experts.gate_up_proj`.  The retained assembly diagnostic is
`benchmarks/results/2026-05-19-hipengine-qwen36-paro-mtp-bf16-assembly-diagnostic.json`.
Validation now sees all `19/19` required tensors.  Native MTP proposal bring-up
has started: `hipengine_mtp_fuse_inputs_f16_bf16` covers token-embedding +
target-hidden pre-fc RMSNorm/concat, and `scripts/mtp_input_fc_smoke.py` applies
`mtp.fc` with the existing BF16 dense GEMV.  A full one-layer torch-reference
proposal smoke also exists in `scripts/mtp_torch_proposal_smoke.py`; it runs
MTP attention, MoE/shared expert, final `mtp.norm`, shared lm-head/top1, and
emits candidate-only `DraftBatch` rows for `verify_chain_bulk_and_commit`.

The native proposal-chain smoke is now `scripts/mtp_native_decode_step_smoke.py`.
It reuses native BF16 dense/QKV/GQA/lm-head kernels plus MTP-specific helper
kernels for zero-centered RMSNorm, q/gate split, BF16 gate multiply, router
softmax, MoE accumulation, and FP32-to-BF16 finalization.  On gfx1151 with the
assembled artifact, `--draft-budget 2 --torch-compare` produced candidate chain
`[12, 4773]` in both native and torch-reference paths and emitted verifier rows
`[root, d1, d2]` with parent rows `[-1, 0, 1]`.  The same smoke can now consume a
BF16 target hidden row captured by `Qwen35ParoResidentSession.step_with_hidden_taps`
using `--target-hidden-source target_session`; that path produced native/torch
candidate chain `[27399, 220]`.  This remains a diagnostic smoke, not a speed row:
selected expert ids are host-orchestrated.  The first shared-verifier E2E smoke
is `scripts/mtp_chain_e2e_smoke.py`; it feeds native MTP candidate rows into
`Qwen35ParoResidentSession.verify_chain_bulk_and_commit` and matched exact AR on
a 3-token gfx1151 smoke.  A persistent native provider now lives in
`hipengine/speculative/mtp_native.py`; it keeps MTP weights/cache resident and
uses device-resident target hidden rows.  On the stable quicksort prompt with
B=5 it matched exact AR and accepted all proposed draft tokens for the 8-token
sample (`accepted_lengths=[5,1]`), but still lost the decode speed gate
(`31.34 tok/s` vs AR `52.98 tok/s`) because the current bulk verifier is slower
than serial AR and expert dispatch still copies selected expert ids to host.  A
follow-up single-chain linear-attention verifier t-loop now avoids parent-row
global state reloads while still materializing row states for exact partial
accepts; the same B=5 sample stayed exact with `linear_attn_mode=chain_tloop` and
measured `33.13 tok/s` vs AR `52.63 tok/s`, so the speed blocker moved from
linear-state materialization specifically to the broader target-verifier
launch/row-cost wall.

A subsequent verifier graph-capture experiment fixed two blockers:
1. `_verify_capture_staging_tensor` rejected `width <= 0`, breaking graph paths
   where hidden-tap capture is empty (`capture_layer_ids=()`).
2. `_should_use_chain_tloop_linear_verify` disabled `chain_tloop` whenever
   `graph_mode != "off"`, forcing the slower `tree_tloop` fallback during capture.

After those fixes, B=2 and B=3 graph capture with `chain_tloop` validates
exactly against the direct path.  Measured on the stable quicksort prompt with
32 decode tokens and `graph_mode=auto`:

| mode | AR decode tok/s | MTP decode tok/s | verify sec/14 cycles | avg ms/cycle |
|---|---|---|---|---|
| graph=off | 61.5 | 28.6 | 0.806 | 57.6 |
| graph=auto | 61.2 | 26.1 | 0.912 | 65.1 |

Graph replay is **not faster**; it is slightly slower.  The verifier per-cycle
execution time (~57-65 ms for B=2) is dominated by GPU kernel work, not CPU
launch overhead.  Capture/validation overhead for new buckets (`rows=3`,
`rows=2`) negates any replay savings when buckets change frequently.  Even with
13 replays and 1 capture over 14 cycles, the total verify time is higher than
the direct path.  This means graph capture alone will not close the MTP/AR gap.
The remaining speed work must attack kernel execution cost (fused verifier layers,
batched row processing) rather than launch overhead.

### GPU-fast token accept

`verify_chain_bulk_and_commit` and `verify_tree_bulk_and_commit` historically
read target top1 tokens back to CPU (`_read_verify_top1`), ran the CPU oracle
`batch.accept_from_top1()`, and then compared with a GPU-side accept-summary
kernel (`dflash_accept_chain_i32`) for validation.  The GPU kernel already
computes the full acceptance result (commit row, commit token, next token,
full-accept flag); the CPU path is redundant overhead.

A GPU-fast accept path was added:
- `TargetAcceptSummary.from_gpu_payload(batch, gpu_payload)` reconstructs
  accepted tokens by walking `batch.parent_rows` from the commit row back to
  the root, producing a semantically identical result to the CPU oracle.
- `_verify_gpu_accept_enabled()` checks `HIPENGINE_VERIFY_GPU_ACCEPT`.
  - `0` / `false` / `no` / `off` → disabled (default).
  - `1` / `true` / `yes` / `on` → fast path: skip CPU top1 read and CPU
    accept computation; trust the GPU accept-summary kernel directly.
  - `validate` → run both paths and compare; fall back to CPU on mismatch.
- When enabled, `verify_chain_bulk_and_commit` reads only the small GPU accept
  payload (`~28 bytes/request`) instead of the larger top1 buffer
  (`rows * 8 bytes`), and avoids the CPU tree-walk entirely.

Validation on the stable quicksort prompt with B=2, 32 decode tokens:
- `HIPENGINE_VERIFY_GPU_ACCEPT=validate` → exact AR match, all cycles
  `gpu_accept_match_cpu=True`.
- Speed impact is **neutral to noise**: the CPU top1 read + accept walk is
  `<0.1 ms/cycle`, negligible next to the `~60 ms/cycle` verifier forward pass.
  The fast path is architecturally cleaner but does not measurably improve
  throughput.  The dominant verifier cost remains the forward-pass kernel
  execution, not host-side acceptance logic.

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

## Prior W7900/Quark evidence from `~/amd-gpu-tuning`

Source plan: `~/amd-gpu-tuning/PLAN-MTP.md`.

The rows below are prior W7900/gfx1100 + Quark/W8A8/BF16-MTP evidence. They are
useful for verifier break-even math, but they are not a baseline for the current
packed `gfx1151` DFlash lane. gfx1151 has a higher compute-per-byte balance than
W7900 (roughly 48% of W7900 compute but ~30% of its memory bandwidth), so bytes
are more expensive and native row reuse may matter even more. Do not promote an
MTP speed claim until it is re-measured on the packed target with the shared
native verifier.

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
tensors. It is a bring-up artifact, not the current DFlash target layout. The
`dflash` branch should not hardcode it into benchmark paths; the target baseline
is the shisa packed PARO model, and the DFlash drafter is
`z-lab/Qwen3.6-35B-A3B-DFlash`. MTP metadata/loading should be revisited after
the shared verifier exists and should validate whatever packed-target-attached
MTP artifact we actually retain.

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

### Phase M0 — Shared verifier baseline

The shared verifier/accept/commit baseline has landed for correctness and ABI
reuse, but not for speed promotion:

- native topk=1 chain `TargetVerifyBatch` exists;
- exact selectable per-row target state exists;
- accepted-prefix target re-forward is avoided in native commit paths;
- GPU target top1 + chain accept summary exists;
- batched chain and tree verifier variants exist.

The unresolved M0 speed gate remains: the verifier is still slower than serial
c=1 on chain-shaped work.  MTP integration may proceed as metadata/proposal
bring-up, but retained MTP speed claims must wait until either verifier row cost
falls or MTP acceptance density clearly overcomes it.

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

## Closing the gap with llama.cpp MTP — kernel roadmap (2026-05-21)

With Tasks #45–#52 done, the picture is no longer “does MTP work?” but “why is
llama.cpp MTP 1.5–2× over AR while ours is 0.87×?” This section is the
focused plan to close that gap. It refines (not replaces) Phase M3/M4 above.

### Architectural lessons from llama.cpp (`/home/lhl/llama.cpp/llama.cpp-hip`)

llama.cpp MTP for Qwen3.5/3.6 has the same model boundary we already model
([qwen35moe.cpp `graph_mtp`](../../../llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp)
lines ~555–727) and the same draft/verify dance
([common/speculative.cpp `common_speculative_impl_draft_mtp`](../../../llama.cpp/llama.cpp-hip/common/speculative.cpp)
lines ~409–775). What it has that we don’t:

1. **One `mul_mat_id` kernel per MoE op.** Their `ggml_cuda_mul_mat_vec_f`
   (and `mul_mat_vec_q` for quantized weights) launches a single 3D grid
   `(rows_in_dst, n_expert_used, n_tokens)` and reads `ids[token, slot]` inside
   the kernel to pick which expert tile to GEMV against. There is no
   host-side dispatch loop; all 60 expert GEMVs at B=3 collapse to **1 launch
   per (layer, MoE op)** instead of our 60 tiny launches per pass.
   References: `ggml/src/ggml-cuda/mmvf.cu` (kernel), `ggml/src/ggml-cuda/mmid.cu`
   (the slow-path token-sort helper used only when `n_tokens > 8` per expert).
   The fast path is mmvf/mmvq with the ids tensor itself — no sort, no scatter.
2. **Fused gate+up matmul.** `gate_up_proj` is a single expert weight stack;
   `build_moe_ffn` issues one `mul_mat_id` for gate_up, then views it as two
   halves. We already pack `mtp.layers.0.mlp.experts.gate_up_proj` the same
   way in `loading/mtp.py`; the runtime currently splits it back into two
   GEMVs because dispatch doesn’t have a fused selected-expert path.
3. **One captured graph per verifier row count.** The entire trunk forward
   (RMSNorm → GQA → MoE → residual … ×30 layers + lm_head) compiles once
   per `(B+1, kv_pos_bucket)` and replays as a single HIP graph each cycle.
   We have graph capture (Task #47) but it didn’t move the needle because the
   *kernel* time — not launch time — dominates.
4. **MTP head is a standalone “draft model”.** llama.cpp’s
   `convert_hf_to_gguf.py --mtp` produces a separate `mtp-*.gguf` containing
   only the MTP block + shared head. `ctx_dft` loads it as a normal model
   with `LLAMA_CONTEXT_TYPE_MTP` and its own KV cache; the draft loop is just
   `llama_decode(ctx_dft, batch)` per step. This is the clean separation our
   shared-verifier story already promises — we just need our proposer to ride
   the same kernel registry as the target.
5. **Backend sampling.** `llama_sampler_chain_add(… top_k(10))` runs as a ggml
   op on-device; the host gets only the sampled id back. Our GPU-fast accept
   (Task #48) is the equivalent on the verify side; we still do a host loop
   to greedy-pick draft tokens. Move that to a tiny GPU top-1 + write.

### Verifier break-even math (gfx1151, packed PARO+MTP-BF16)

From Task #52 profiling, B=3, perfect-accept assumption:

```
T_AR_step      ~= 22 ms        # AR decode tok/s ~= 45.5
T_verify_B3    ~= 52 ms wall   # ~45 ms kernel + ~7 ms host
MTP/AR (B=3)   ~= 22×3 / 52 = 1.27×   # ceiling, perfect accept
MTP/AR measured = 0.87×              # i.e. avg_accepted < 3 per cycle
```

To hit a **≥1.5×** target at B=3 the verifier must drop below
`22×3 / 1.5 = 44 ms` wall — ~15% off today. To hit **≥2.0×** it must drop
below `33 ms` — ~37% off today.

#### Master scoreboard — verifier wall-time budget (B=3, 4 rows, gfx1151)

Updated as each phase lands. “Baseline (ms)” was Task #52’s informal estimate;
M7.0’s rocprofv3 trace (`benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`) replaces it with measured per-pass kernel times.
“Target (ms)” is the post-phase projection. “Actual (ms)” is filled in from the
rocprofv3 + wall-time artifact when the phase commits. Negative deltas mean
we beat the target.

| Cost component                      | Phase | Task #52 est. (ms) | M7.0 measured (ms) | Target (ms) | Actual (ms) | Δ vs baseline | Δ vs target | Artifact / source |
|-------------------------------------|:-----:|-------------------:|-------------------:|------------:|------------:|---------------:|-------------:|-------------------|
| MoE chain (gate_up + down + rotate + silu + router + combine + w4_dual) | M7 | ~20 | **17.0** | 11–13 | _TBD_ | _TBD_ | _TBD_ | M7.0 artifact |
| GDN chain t-loop (4 rows)           | M7.B*  |          ~10  |          **13.1** |    10–11  |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact (new phase, see below) |
| Small-batch prefill kernels (QKV / shared-expert / dense MLP @ tokens=4) | M7.C* |        n/a  |          **11.0** |     2–3   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact (new phase, see below). Originally mis-labeled “runtime_memset” due to classifier bug. |
| LM head W8A16 (4 rows)              | M9    |          ~7.5 |           **9.9** |     6–7   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| Pre-attention chain (QKV + RoPE + norms) | M8 |          ~5  |           **~3.3** (w4_dual_gemv + rmsnorm + paged_kv + decode_attn + attn_gate) |     2.5   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact. Likely no-op once M7.C lands (the QKV prefill→GEMV switch *is* the pre-attention win). |
| Host-side Python overhead           | M10   |           ~7  |          **~9** (host_window − kernel = 65−56) |     2–3   |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| Other (runtime_copy + paro_rotate + misc) |  —  |          —   |          **~4.6** |     ~4    |       _TBD_ |             —  |          —   | M7.0 artifact |
| **Total verifier wall (host)**      |       |             ~52  |             **~65** | **~35–40** |       _TBD_ |          _TBD_ |        _TBD_ | M7.0 artifact |
| **MTP/AR ceiling @ B=3 (22×3/wall)** |       |          **1.27×**|          **1.02×** | **1.65–1.88×** |   _TBD_ |          _TBD_ |        _TBD_ |                   |
| **MTP/AR measured @ B=3**           |       |          **0.87×**|          **~0.6×** (16-tok decode, 50% accept) | **1.0–1.4×** | _TBD_ |        _TBD_ |        _TBD_ | M7.0 + smoke      |

\* M7.B and M7.C were added after M7.0 revealed the actual top costs differ
from Task #52’s estimate. Both are simpler than the M7 MoE kernel work and may
land first.

Real-world MTP/AR depends on accepted-token density (50–80% measured on the
stable quicksort prompt at B=3, mean ~64% in M7.0 run; the persistent-b3
run on 2026-05-19 with 8-token decode hit 100%). The M7.0 measured ceiling
(1.02×) is below the Task #52 estimate (1.27×) because per-pass kernel time
is 56 ms vs Task #52’s 45 ms, plus ~9 ms host — i.e. “45 ms kernel + 7 ms
host” understates today’s actual cost by ~10 ms. The post-M7…M10 ceiling
range (1.65–1.88×) and measured target (1.0–1.4×) are correspondingly more
conservative than the original 2.06–2.36× / 1.3–1.7× projection. A 1.5×
measured row is now contingent on landing M7.C (memset) and M9 (LM head) in
addition to M7 itself.

### Phase M7 — batched selected-expert GEMV (HIGHEST PRIORITY)

Goal: replace the 60 tiny per-pass MoE GEMV launches with O(layer) launches.

ABI (registry key `("hip_gfx1151", "moe_selected_expert", quant, variant)`):

```text
moe_selected_expert_gemv(
    A:    [n_tokens, n_embd]                 fp16/bf16,           # token rows
    W:    [n_experts, n_out, n_embd]         bf16 | awq_q4 stack, # expert weight stack
    ids:  [n_tokens, n_expert_used]          int32,               # selected experts
    Y:    [n_tokens, n_expert_used, n_out]   fp16/bf16,           # per-slot output
    bias_or_scale:  optional,
) -> Y
```

Weighted-combine and shared-expert add stay outside the kernel; they are
cheap pointwise ops we already have.

#### M7 tracker

Projected savings target: **~4–6 ms** of the 17 ms MoE chain at B=3 / 30
layers (revised down from 8–12 after M7.0). Each row is a separately
landable unit with its own correctness + rocprofv3 gate. Fill `Actual (ms)`
and `Status` as each row commits.

**Pre-condition (M7.0):** code inspection (2026-05-21) confirmed the existing
`gemv_awq_selected_dual_pack8_transposed_bf16` is already a llama.cpp-style
mul_mat_id kernel — grid `(out_packed_a+out_packed_b, rows)` with
`rows = tokens * num_experts_per_tok`, reading `selected[row]` inside the
kernel. So “60 tiny launches” in Task #52’s analysis is “30 layers × 2 fused
MoE ops”, not “1 launch per expert”. M7 work is small-batch tile retuning
of the existing kernel, not a brand-new layer-array kernel.

##### M7.0 measured per-pass breakdown (B=3, 4 rows, gfx1151)

Artifact: `benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`.
Command: `python3 scripts/mtp_verifier_rocprof.py --prompt-tokens "$(cat
quicksort-tokens)" --decode-tokens 24 --candidate-budget 3 --steady-state-skip 2`.

Run: 9 verifier passes total, **7 kept** after dropping the 2 cold cycles.
Acceptance 64% mean (vs 100% on the 8-token persistent_b3 diagnostic from
2026-05-19). Per-pass numbers:

| Family                                  | calls/pass | ms/pass | share | avg μs | max μs | Notes |
|-----------------------------------------|-----------:|--------:|------:|-------:|-------:|-------|
| linear_attention_gdn_decode             |        30  |   13.07 | 23.4% |  435.7 |  513.8 | #1 cost; already chain_tloop. → M7.B |
| w8a16_linear (lm_head)                  |         1  |    9.93 | 17.7% | 9930.4 | 10326.8| Single big launch, bandwidth-bound. → M9 |
| **w4_dual_prefill_smallbatch** (QKV / dense MLP / shared-expert gate+up @ tokens=4) | 60 | **7.40** | 13.2% | 123.4 |  239 | **Wrong kernel for small batch**. Sites: `project_full_attention_qkv_fp16` / `project_linear_attention_qkv_z_fp16` / `shared_expert_paro_w4_fp16` / `dense_mlp_paro_w4_fp16` gate `if tokens == 1: GEMV; else: PREFILL` — lower the threshold to e.g. `tokens > 7`. → M7.C |
| moe_down_gemv                           |        70  |    5.99 | 10.7% |   85.6 |  307.3 | Already mul_mat_id; tile retune. → M7.4 |
| moe_gate_up_dual_gemv                   |        70  |    5.93 | 10.6% |   84.7 |  283.5 | Already fused gate+up. → M7.4 |
| **w4_single_prefill_smallbatch** (shared-expert down / dense-MLP down @ tokens=4) | 60 | **3.61** | 6.5% | 60.2 | 162 | Same kernel-choice bug as above, single-tensor path. → M7.C |
| moe_paro_rotate_in                      |       310  |    1.78 |  3.2% |    5.8 |   27.3 | Many small launches; low margin. |
| w4_dual_gemv (small-token QKV / shared) |        80  |    1.69 |  3.0% |   21.1 |  101.2 | The kernel we WANT the M7.C sites to use. |
| decode_attention                        |        40  |    1.11 |  2.0% |   27.7 |   77.8 | Lean already. |
| router                                  |       140  |    1.05 |  1.9% |    7.5 |   78.9 | Lean already. |
| (runtime_copy + rmsnorm + silu + combine + other small ops) | ~720 | ~5.6 | ~10% | | | |
| **TOTAL per pass**                      |     1838   | **56.0**| 100%  |        |        | Host window: **~65 ms** (kernel 56 + host ~9). |

**Important classifier correction (2026-05-21):** the original `_family`
classifier in `scripts/mtp_verifier_rocprof.py` matched bare `"fill"` which
false-matched `"prefill"`. The two `awq_fusedw4_prefill_*` kernels (11 ms
combined) were therefore mis-labeled `runtime_memset` in the first M7.0
report. The artifact at
`benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json`
has been re-processed with the corrected classifier; the kernel CSV / wall
times are unchanged. **There is no memset bottleneck.** M7.C is
rechartered around the small-batch kernel-choice bug below.

##### M7.0 findings vs. Task #52 plan assumptions

1. **MoE is NOT the dominant bottleneck.** Combined MoE chain (gate_up +
   down + rotate + silu + router + combine + w4_dual_gemv attn) = **17.0 ms**
   (~30% of pass), not 20 ms / 44% as Task #52 estimated. The existing
   kernel is already llama.cpp-style. M7 reach drops from 8–12 ms to
   **4–6 ms** (small-batch tile retune of the existing kernel).
2. **GDN chain_tloop is the actual #1 cost: 13.1 ms.** Plan had this as
   “unchanged / already chain_tloop”. 30 launches × 436 μs avg = real
   bottleneck. Added new phase **M7.B** with ~2–3 ms reach (chain length /
   tile / wave-group sweep).
3. **The “memset 11 ms” finding was a classifier bug.** It is actually two
   `awq_fusedw4_prefill_*_fp16` kernels (60 + 60 calls/pass = 11.0 ms)
   firing because `project_full_attention_qkv_fp16`,
   `project_linear_attention_qkv_z_fp16`,
   `shared_expert_paro_w4_fp16` and `dense_mlp_paro_w4_fp16` are gated
   `if tokens == 1: GEMV; else: PREFILL`. At tokens=4 the prefill kernel
   fires even though `gemv_awq_dual_pack8_kernel`’s grid is
   `(out_packed_a + out_packed_b, row)` and already supports `rows > 1`.
   The MoE selected GEMV runs at 85 μs avg; the prefill path runs at 60–123
   μs avg for the same row count. **M7.C is now a small-batch threshold
   fix** with the same ~8–9 ms reach — but it’s a few lines of dispatch
   changes (`if tokens <= 7` instead of `if tokens == 1`), not a runtime
   rewrite. **Still the highest ROI per LoC.**
4. **LM head is 9.9 ms** (vs Task #52’s 7.5 ms estimate). M9 reach adjusted
   to ~3 ms.
5. **Pre-attention chain (M8) is only ~3.3 ms total** (w4_dual_gemv +
   rmsnorm + decode_attention + paged_kv + attn_gate). Task #52’s “~5 ms”
   was high; M8 reach now ~1–1.5 ms. **Note**: M8 was “fused pre-attention
   composite” in the original plan, but M7.C absorbs the bulk of that
   reach (the prefill→GEMV switch *is* the pre-attention QKV win), so M8
   may reduce to a no-op once M7.C lands.
6. **Host-side Python overhead is ~9 ms** vs Task #52’s 7 ms estimate. M10
   ~5 ms reach still plausible.

Revised total reachable savings: **~23–27 ms** (was 18.5–22.5 ms in the
original plan).
Verifier wall: 65 ms → **~38–42 ms**.
Ceiling MTP/AR @ B=3 perfect-accept: **~1.65–1.88×** (was 2.06–2.36×).
Measured MTP/AR @ B=3 60% accept: **~1.0–1.4×** (was 1.3–1.7×).
**A 1.5× measured row is now contingent on landing M7 + M7.C + M9** — not
M7 alone. M7.C is the highest single-phase ROI; landing it first puts us
at ~57 ms verifier wall before any kernel work.

##### M7.0 tooling notes (for next rocprofv3 run)

- **rocprofv3 1.1.0 silently drops `--selected-regions true`** output on this
  gfx1151 host, even when roctxProfilerResume/Pause symbols are correctly
  resolved via the therock SDK overlay. Workaround: full
  `--kernel-trace --marker-trace`, post-process by filtering kernel-CSV
  Start_Timestamp against the roctxRangePush window ns boundaries (markers
  named `mtp_verify_pass_N`).
- **Marker CSV uses `Function` column**, not `Marker_Name` / `Marker_Text`
  (rocprofv3 1.1.0 schema).
- **JIT compile under rocprofv3 spawns subprocesses** that each attach as
  separate rocprofv3 instances (hipcc → clang-23 → lld), producing
  hundreds of “tool initialization” / “tool finalization” log lines and
  breaking output. Fix: set `HIPENGINE_COMPILER_VERSION_FILE` env var so
  the JIT cache key stays stable; pre-warm the build cache by running the
  smoke once before rocprofv3.
- **SDK ROCTX library needs sysdeps on LD_LIBRARY_PATH**: the therock
  `librocprofiler-sdk-roctx.so.1` depends on `librocm_sysdeps_dw.so.1`,
  which lives under `<sdk_core>/lib/rocm_sysdeps/lib`. Without it,
  `ctypes.CDLL('libroctx64.so')` succeeds against the legacy library but
  has no `roctxProfilerResume` symbol and the marker calls silently no-op.
- `scripts/mtp_verifier_rocprof.py` handles all of the above automatically.
  Re-run with `--steady-state-skip N` to drop more cold cycles.

| #   | Sub-task                                                                                   | Variant            | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|--------------------------------------------------------------------------------------------|--------------------|-----------------------:|----------|--------------------:|------------------|
| M7.0| **rocprofv3 re-baseline** — LANDED. 7 steady-state B=3 verifier passes traced via roctxRangePush markers; per-pass: 56 ms kernel + ~9 ms host = ~65 ms wall. Top families: GDN 13.1, memset 11.0, lm_head 9.9, MoE down 6.0, MoE gate_up 5.9. | n/a | 0 (diagnostic) | ✅ **Landed** | 0 | `benchmarks/results/2026-05-21-hipengine-mtp-verifier-rocprof-baseline.json` |
| M7.1| CPU-reference fixture: 4-tok / 30-layer + 8-tok routed MoE, KL≤0.05 / top-1≥0.90 oracle    | n/a                |                     0  | _Pending_|                _TBD_| _TBD_            |
| M7.2| Tune existing `gemv_awq_selected_dual_pack8_transposed_bf16` for 32-row batch (per M7.0: already llama.cpp-style, just needs small-batch micro-tuning). Variant `dense_bf16` for the MTP proposer side. | `dense_bf16` | 1–2 | _Pending_ | _TBD_ | _TBD_ |
| M7.3| Land `dense_bf16` for MTP proposer (down_proj + gate_up_proj fused), route via registry    |`dense_bf16`        |                  ~1–2  | _Pending_|                _TBD_| _TBD_            |
| M7.4| AWQ pack8 small-batch tile retune: 70 calls/pass at 86 μs avg; target 60 μs avg via LDS / wave-tile sizing for the 32-row case. | `awq_q4_pack8` | ~3–4 | _Pending_ | _TBD_ | _TBD_ |
| M7.5| Skip: existing `dual_pack8_transposed` already fuses gate+up (one launch / layer / 2×n_ff_exp output). Reclassify as no-op. | n/a | 0 | ✅ **Landed (n/a)** | 0 | M7.0 confirmed |
| M7.6| Verify: rocprofv3 shows MoE chain runs ≤11 ms total at B=3/30 layers, KL≤0.05 vs CPU-ref    | both               |                     0  | _Pending_|                _TBD_| _TBD_ (gate only) |
| **M7 total** |                                                                                |                    |              **4–6**   |          |             **_TBD_**|                  |

#### M7.B tracker — GDN chain t-loop tuning (NEW, surfaced by M7.0)

Projected savings target: **~2–3 ms** of the 13.1 ms GDN chain t-loop budget.
GDN decode runs 30 calls/pass at 436 μs avg — the per-launch cost is high
and the kernel is already chain_tloop. Possible wins: chain length tile,
shared-LDS K-tile, or wave-group sizing.

| #   | Sub-task                                                                                | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M7.B.1| Confirm GDN chain_tloop launch parameters match the 4-row B=3 case (not stuck on 8-row defaults) |                  ~1   | _Pending_|                _TBD_| _TBD_            |
| M7.B.2| Tile-size sweep on the 32-context decode shape; promote per CPU-ref correctness gate     |                  ~1–2  | _Pending_|                _TBD_| _TBD_            |
| **M7.B total** |                                                                               |              **~2–3** |          |             **_TBD_**|                  |

#### M7.C tracker — small-batch prefill→GEMV switch (RECHARTERED 2026-05-21)

Projected savings target: **~8–9 ms** of the 11.0 ms combined budget for the
two `awq_fusedw4_prefill_*_fp16` kernels. This phase was originally framed as
“runtime_memset elimination” — see M7.C.1 below for what the M7.0 trace actually
showed once the classifier bug was fixed.

Dispatch sites to change (all in `hipengine/runtime/qwen35_paro.py`,
`if tokens == 1: ... else: awq_fusedw4_prefill_*` blocks):

| Site                                          | Line(s) | Kernels removed @ tokens=4 | Replacement |
|-----------------------------------------------|--------:|---------------------------|-------------|
| `project_full_attention_qkv_fp16`             |   ~2087 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `project_linear_attention_qkv_z_fp16`         |   ~3391 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `shared_expert_paro_w4_fp16` (gate+up)        |   ~5025 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `shared_expert_paro_w4_fp16` (down)           |   ~5089 | `awq_fusedw4_prefill_fp16`      | `gemv_awq_pack8_transposed_fp16`      |
| `dense_mlp_paro_w4_fp16` (gate+up)            |   ~5218 | `awq_fusedw4_prefill_dual_fp16` | `gemv_awq_dual_pack8_transposed_fp16` |
| `dense_mlp_paro_w4_fp16` (down)               |   ~5282 | `awq_fusedw4_prefill_fp16`      | `gemv_awq_pack8_transposed_fp16`      |

The replacement kernels already accept `rows > 1` (grid is
`(out_packed_a + out_packed_b, row)` in `gemv_awq_dual_pack8_kernel`). The
BF16 paths (`shared_expert_paro_w4_bf16`, etc.) already do exactly this for
all token counts — per the BF16 docstring: “BF16 has no fused prefill kernel,
so the same dual GEMV (which accepts rows > 1) is used for every tokens
value”. M7.C extends the same logic to the FP16 path for small batches.

| #   | Sub-task                                                                                | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M7.C.1| **Identify culprit** — LANDED. Six dispatch sites listed above use prefill kernels for `tokens > 1`. The 11 ms “memset” was a `_family` classifier substring match against “fill” in “prefill”. | 0 (diagnostic) | ✅ **Landed** | 0 | this section + corrected M7.0 artifact |
| M7.C.2| Add a `_small_batch_decode_threshold` constant + env override; change six dispatch sites from `if tokens == 1` to `if tokens <= _small_batch_decode_threshold()` |             ~6–8 | _Pending_ |               _TBD_ | _TBD_            |
| M7.C.3| Cross-check: prefill batches (16+ tokens) still take the prefill kernel; verify with a `--rocprof-warmup-cycles 0 --prefill-only` smoke run | 0 | _Pending_ | _TBD_ | _TBD_ (gate only) |
| M7.C.4| Correctness: full B=3 chain still exact-AR-match on the quicksort fixture; KL/top-1 unchanged | 0 | _Pending_ | _TBD_ | _TBD_ (gate only) |
| M7.C.5| Re-run M7.0 rocprof; new per-pass kernel ms drops by ~7–8 ms (the prefill kernels fall out, replaced by ~85 μs / 4 μs/row GEMVs at < 3 ms total). | 0 | _Pending_ | _TBD_ | _TBD_            |
| **M7.C total** |                                                                               |              **~6–8** |          |             **_TBD_**|                  |

Design notes:
- Grid `(n_out_tiles, n_expert_used, n_tokens)`. Each block reads `expert = ids[token, slot]` and indexes `W[expert, tile, :]`.
- For BF16 dense weights (MTP proposer): straight fp16/bf16 vec-mat reduction with a 32-thread warp per row tile. Variant `"dense_bf16"`.
- For AWQ pack8 weights (target verifier): same kernel structure, swap inner dequant. Variant `"awq_q4_pack8"`. Reuse the dequant microcode from `gemv_awq_selected_*`.
- For n_tokens ≤ 8: keep `ids` in registers; no shared-memory scratch. Matches llama.cpp’s `MMVF_MAX_BATCH_SIZE = 8` fast path.
- Fused gate+up: weights `[n_experts, 2*n_ff_exp, n_embd]`; output `[n_tokens, n_expert_used, 2*n_ff_exp]`; SwiGLU stays as the existing pointwise op.

Plugin discipline:
- Registry key uses four-axis form (`backend, layer, quant, variant`). Routing lives in `hipengine/kernels/registry.py` and `hipengine/dispatch/fusion.py`, **not** in `if backend ==` / `if quant ==` branches.
- Both `dense_bf16` and `awq_q4_pack8` variants register under the same layer key `"moe_selected_expert"`; proposer and verifier pick the same kernel family with different variant tags.
- The unfused per-expert GEMV chain stays registered as fallback.

### Phase M8 — fused pre-attention sub-path (SECOND PRIORITY)

Goal: collapse the ~90 launches per verifier pass for RMSNorm → QKV → RoPE → q_norm/k_norm into one composite launch per layer.

ABI (registry key `("hip_gfx1151", "pre_attention_fused", quant, variant)`):

```text
pre_attention_fused(
    H_in:     [n_tokens, n_embd]    bf16,        # post-prev-residual
    W_qkv:    AWQ-packed Q/K/V
    W_q_norm, W_k_norm:  bf16,
    rope_cos, rope_sin: fp16,
    pos_ids:  int32,
) -> Q, K, V    # already RoPE’d, q/k normalized
```

Unfused fallback: RMSNorm → QKV GEMV → split → rotate → q_norm/k_norm. Stays registered; the fused composite registers as a separate layer key.

#### M8 tracker

Projected savings target: **~3 ms** out of the ~5 ms pre-attn chain at B=3 / 30 layers. The composite is correct-by-construction iff each step matches the unfused fallback on the same fixture; promote per-step.

| #   | Sub-task                                                                              | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|---------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M8.1| CPU-reference fixture: 4-tok / 30-layer pre-attn input → (Q, K, V) post-RoPE oracle    |                     0  | _Pending_|                _TBD_| _TBD_            |
| M8.2| Fused RMSNorm + QKV GEMV (skip RoPE) — collapse ~60 launches → ~30                    |                  ~1–1.5| _Pending_|                _TBD_| _TBD_            |
| M8.3| Add RoPE inside the kernel (cos/sin from constant buffer)                              |                  ~1   | _Pending_|                _TBD_| _TBD_            |
| M8.4| Add q_norm / k_norm inside the kernel (collapse final ~30 launches)                    |                  ~0.5–1| _Pending_|                _TBD_| _TBD_            |
| M8.5| Verify: rocprofv3 shows 1 launch per layer pre-attn at B=3, KL≤0.05 vs unfused        |                     0  | _Pending_|                _TBD_| _TBD_ (gate only)|
| **M8 total** |                                                                          |               **~3**   |          |             **_TBD_**|                  |

### Phase M9 — parallelized LM head over verifier rows (THIRD PRIORITY)

Goal: cut the 7.5 ms 4-row W8A16 lm_head projection by ~30%.

Switch from current `gemv_w8a16` chained over rows to a row-parallel split-k variant that streams the 248320-row weight matrix once per pass and computes all 4 rows in parallel.

Same ABI as today’s lm_head; new variant under `("hip_gfx1151", "lm_head", "w8a16", "row_parallel")`. Promote only if it beats the current path on B ∈ {2, 3, 4, 8}.

#### M9 tracker

Projected savings target: **~2.5 ms** of the ~7.5 ms LM head at B=3, 4 verifier rows. Bandwidth-bound — the win comes from streaming the weight matrix once, not from arithmetic.

| #   | Sub-task                                                                          | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|-----|-----------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M9.1| Row-parallel split-k kernel: grid `(n_out_tiles, n_tokens)`, single weight stream  |                  ~2–2.5| _Pending_|                _TBD_| _TBD_            |
| M9.2| Promote per `B ∈ {2, 3, 4, 8}` sweep; gate via existing lm_head correctness test  |                  ~0   | _Pending_|                _TBD_| _TBD_ (gate only)|
| **M9 total** |                                                                      |              **~2.5** |          |             **_TBD_**|                  |

### Phase M10 — align proposer with target dispatch

Once Phase M7 lands the `dense_bf16` variant of the MoE GEMV, the native MTP proposer (`hipengine/speculative/mtp_native.py`) plugs into the same registry path. This phase also eliminates host-side overhead (~7 ms/pass per Task #52).

No new model quant required — BF16 MTP weights stay. If we later quantize MTP, register an `awq_q4_pack8` variant under the same layer key and the proposer picks it up via the variant axis (no code branching).

#### M10 tracker

Projected savings target: **~5 ms** of host-side overhead (the ~7 ms baseline minus an irreducible ~2 ms for batch prep + sampling read).

| #    | Sub-task                                                                                                   | Projected savings (ms) | Status   | Actual savings (ms) | Notes / artifact |
|------|------------------------------------------------------------------------------------------------------------|-----------------------:|----------|--------------------:|------------------|
| M10.1| Route `mtp.layers.0.mlp.experts.gate_up_proj` through `moe_selected_expert` / `dense_bf16` (removes Task #50 blocker) |              ~2     | _Pending_|                _TBD_| _TBD_            |
| M10.2| Keep selected-expert ids on-device throughout the proposer chain (only D2H sync per draft step = sampled tok) |              ~1.5–2 | _Pending_|                _TBD_| _TBD_            |
| M10.3| GPU top-1 + write kernel for the next draft seed (proposer never reads top1 to host)                       |                  ~1–1.5| _Pending_|                _TBD_| _TBD_            |
| M10.4| Re-capture HIP graph for the post-M7/M10 proposer chain (one captured graph per draft depth)               |                  ~0.5–1| _Pending_|                _TBD_| _TBD_            |
| **M10 total** |                                                                                              |              **~5**   |          |             **_TBD_**|                  |

### Phase M11 — fixed-depth chain bucket sweep on the fast verifier

After M7 lands (the only mandatory phase for a 1.5× row), re-run the B=1/2/3 sweep with same-session AR. Pick the operating point that maximizes measured MTP/AR.

Keep the existing exact-equality gate. The benchmark rollup (`benchmarks/README.md` + `benchmarks/CHANGELOG.md` + JSON artifact) is the only path to a retained speed claim, per `AGENTS.md`.

#### M11 tracker — operating-point sweep (post-M7…M10)

This phase doesn’t add per-kernel savings; it picks the chain depth and acceptance policy that maximize end-to-end MTP/AR on the fast verifier. Projected MTP/AR is the perfect-accept ceiling at the listed B given the post-M10 verifier wall; “measured target” assumes 60–80% acceptance.

| #    | Operating point                                                  | Projected ceiling MTP/AR | Projected measured MTP/AR | Status   | Actual measured MTP/AR | Artifact / source |
|------|------------------------------------------------------------------|-------------------------:|--------------------------:|----------|-----------------------:|-------------------|
| M11.1| Chain B=2 (drafts/verify=2, target rows=3)                       |                  ~1.5×  |              ~1.2–1.4×    | _Pending_|                  _TBD_ | _TBD_             |
| M11.2| Chain B=3 (drafts/verify=3, target rows=4) — default candidate   |                  ~2.0×  |              ~1.3–1.7×    | _Pending_|                  _TBD_ | _TBD_             |
| M11.3| Chain B=4 (drafts/verify=4, target rows=5)                       |                  ~2.0×  |              ~1.2–1.6×    | _Pending_|                  _TBD_ | _TBD_ (only run if M11.2 < 1.3×) |
| M11.4| DDTree B=4/8 (tree drafts)                                       |                _≥2.0×_  |                    _TBD_  | _Pending_|                  _TBD_ | _TBD_ (only run if any chain row ≥ 1.3×) |
| **M11 retained row** |                                                          |               **≥1.5×** |                  **≥1.3×** |          |             **_TBD_**  | benchmarks/README.md row + benchmarks/CHANGELOG.md entry |

### Out-of-scope (don’t pre-build)

- Cross-arch (CUDA / gfx1100) variants of the new kernels. Land on gfx1151
  first, get a retained row, then port. Backend tree is peer-structured so
  porting is a per-arch task.
- Quantizing MTP weights. Not required for a 1.5× row; revisit if MoE
  bandwidth becomes the new bottleneck after M7.
- Tree-shaped MTP drafts. Chain at B ∈ {2,3} is enough for the first row;
  DDTree is a separate axis covered by `docs/DFLASH.md`.
- Long-context tuning. Get the short-prompt row first; long-context is a
  separate validation matrix.

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
   than serial c=1 on chain B=1/2/4/8 for the shisa packed target on native
   `gfx1151`.
2. Add MTP tensor metadata/materialization for the retained packed-target MTP
   artifact; use the Quark W8A8 + BF16 MTP artifact only as a bring-up/reference
   source unless it becomes the measured target.
3. Add native MTP proposal oracle tests against parent fixed fixtures.
4. Implement `MtpDraftProvider` producing `DraftBatch` chain rows.
5. Feed MTP drafts into the shared chain verifier and accept kernel.
6. Add exact commit-state tests for reject, partial, and full accept.
7. Benchmark B=1/2/3, then B=5 only if the split supports it, always with
   same-session packed-target AR.
8. Revisit top-k/tree policy only after a flat-chain MTP row beats AR.
