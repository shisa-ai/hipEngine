---
status: current
owns: The October 20 Flash-Next Gufo gap analysis and implementation campaign for prefill, decode, speculative execution, and concurrent serving.
---
# Qwen3.8-Flash-Next: close the Gufo performance gap

State: **in progress; execution started 2026-10-21**.
Review date: 2026-10-20. hipEngine review source: `a4c7ca473`.
Execution baseline starts at `926c1728c`, including the subsequent prefix-cache
work. `MTP-PREFILL` implementation is complete. Canonical measurements, native
served phase accounting and the ordered-depth runs are available. At the
lead's direction, expanded baseline matrices are deferred while implementation
progresses; their final validation and quality bindings remain on the checklist.
No packet is closed until its implementation and validation evidence are recorded.

The next work is not another wholesale port of Gufo's prefill kernels. Those
ports already removed most of the earlier prefill deficit. Prioritize bulk
multi-token prediction (MTP) prompt catch-up, a cheaper small-row target
verifier, draft-head reduction and prompt lookup, then concurrent MTP and the
remaining decode/prefill kernel differences. Refresh a matched baseline first:
"half the speed" is not established across all three phases by the available
measurements.

This document defines the execution order for the Gufo comparison. The
[optimization ledger](../QWEN4EXP-OPTIMIZATION-LEDGER.md) remains the experiment
and rejection registry; older entries describe their original compositions.
The [serving follow-up](QWEN38FN-SERVING-FOLLOWUP.md) owns its existing work and
results. Reuse its implementation rather than starting another scheduler or
reopening completed metadata work. This plan neither resumes nor closes the
detached serving loop and changes no runtime default or numerical threshold.

## 1. What the evidence establishes

### Gufo's published target

Sources were accessed on 2026-10-20 at Gufo
[`f020bd3a037d58aa42138ca11fa895524e43cdd6`][gufo-tree]. Its
[benchmark page][gufo-bench] and raw artifacts identify the measured binary as
`fec4759d064585d72f87c6e11af5cfa0e1fa95c7`, October 18, Nix release, ELF
SHA-256 prefix `1d822fb97b4639c61`. Later main-source changes are implementation
leads, not explanations of that binary's rates. A direct tree diff also finds
a Q8_1 block-sum correctness change in `mmq/quantize.hip.cpp`: current main
stores `d * sum(q)` rather than the unquantized `sum(x)` for the affected
quantizers. Recheck their actual engagement and current-pin quality; do not
attribute old rates to today's arithmetic or assume that all published rows
are invalidated by this difference.

Published configuration: Ryzen AI Max+ 395 / Radeon 8060S, `gfx1151`, 128 GB
unified memory, Linux 7.1.8, ROCm 7.2.3; performance/accelerator-performance
profiles and Gufo's patched ROCr runtime. The artifact's platform fingerprint
is `fe5929d418cfe1f7e3aa4a09199012c1fa064859f1fc762a8cbcda970358f158`.
This is not the identity of our Framework machine. Weights are Unsloth
`UD-Q4_K_XL` and the **shared Q8_0** MTP sidecar from revision
`38bb39ee97821de2c9009abb7e93950eec396e66`; full shard and sidecar hashes are in
[model identities][gufo-identities]. Gufo uses F16 attention storage; our
starting configuration uses BF16 KV. Do not change KV format merely to make
the comparison look identical.

All rates below are **author-reported**, not rerun for this review. AR means
autoregressive decoding without speculative proposals. HTTP, greedy, thinking
off, penalties disabled; approximately 2,048 new prompt
tokens and 128 output tokens. Depth means an already-cached prefix, not a
fresh prompt of that size.

| Published cell | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| AR, depth 0 | 1801.01 | 28.29 |
| AR, depth 131,072 | 1557.18 | 27.85 |
| MTP, depth 0 | 1734.96 ± 0.85 | mixed 35.79; repetitive 116.93 ± 0.10 |
| MTP, depth 131,072 | 1481.19 | mixed 39.87; repetitive 108.37 |
| Eight prepared AR sessions | not timed in this phase | 125.48 aggregate |
| Eight prepared MTP sessions | not timed in this phase | mixed 127.02; repetitive 245.81 aggregate |

Interpretation constraints from [QUALITY.md][gufo-quality] and
[bench.json][gufo-config]:

- Single-user rows normally have one warmed sample. Repetitive MTP through
  64K includes a focused repeat and reports sample standard deviation. MTP
  prefill is the **maximum across mixed and repetitive workloads**, not a
  common-prompt mean; report both arms when reproducing it.
- Concurrency sums individual request decode rates after every session has
  prefilled. That is not total generated tokens divided by whole cohort wall.
  The latter includes admission, prefill, scheduling and drain.
- The full ordered prefix history determines actual prompts and cached counts.
  A shortened depth sweep is not an exact replay of the published history.
- Repetitive MTP includes request-local **prompt lookup**, sometimes bypassing
  the neural predictor. It is not a measurement of neural MTP alone.
- Gufo records exact native replay, independent operator checks and many
  functional checks, but not equivalence to hipEngine's strict teacher or
  original unquantized-model accuracy. Some functional timing flags remain
  failed or inconclusive. Its historical llama.cpp MTP reference failed its
  own AR-equivalence check; do not use that comparator to establish quality.

### Our latest relevant evidence

The [October 19 topline packet][hip-topline] runs on Framework Desktop,
Radeon 8060S, machine `55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0,
source `e29d34459`, production, BF16 KV, warm per-layer embedding (PLE),
4,096-token chunks. Exact commands, manifests, model fingerprints and checks
are in that packet. It is not a new measurement of `a4c7ca473`.

| Local measurement | Result | Timing basis |
| --- | --- | --- |
| Canonical AR prefill at 512 / 1,024 / 4,096 tokens | 1036.217 / 1323.865 / 1650.651 tok/s | Four categories; one warmup and three repetitions; token/time weighted |
| Canonical AR decode at those lengths | 23.677 / 22.716 / 23.244 tok/s | 128 transitions after prefill |
| MTP, full ten-prompt suite, budget 3 | 28.930 tok/s; matched AR 22.670 | 64 output tokens; whole-request wall; a second suite run agrees |
| MTP, eight category heldouts | 28.164 tok/s; matched AR 22.569 | Same in-process request protocol |

The [October 12 comparison][hip-cross] measured Gufo `v0.12.0` on this
physical host at 1095.0 / 1482.2 / 1736.4 prefill tok/s for the canonical
lengths, with a declared tokenizer-mismatch exclusion, and about 28.1 tok/s
AR decode at 4K. These older comparator rows and our later rows show why a
uniform twofold deficit is an unreliable starting assumption. They are **not
a fresh paired comparison with current Gufo**. In particular, comparing our
512-token result with Gufo's pp2048 headline exaggerates the prefill gap.

The [latest serving comparison][hip-serving] uses the same Framework host,
production/BF16, 18 suite/heldout waves, 64 tokens per request, widths 1 and 8,
one measured repeat, and client-observed whole-wave wall. Its final arm gives
20.95 tok/s AR and 30.25 tok/s MTP at width 1; width-8 AR is 52.65 tok/s and
one MTP request beside seven AR peers is 47.36 tok/s. **Do not divide Gufo's
125.48 or 245.81 by these values:** prefill inclusion, aggregation, prompts,
output lengths and speculative request counts all differ. The file is dated
October 21 but its embedded captures and committing worklog are October 20;
use the embedded source/diff and timestamp fields.

A direct [host-overhead diagnostic][hip-host] prices batch8 at 75.537 ms per
forced-token step, versus 40.427 ms for c1. This is another timing boundary,
not a substitute for either HTTP result. The new baseline must reconcile
kernel, direct-step, prepared HTTP and complete-request costs.

**Assessment:** missing MTP mechanisms and inefficient shared verification are
source-verified. Their recoverable milliseconds are not yet measured on the
current composition. The magnitude of the current cross-engine gap is open.

## 2. Reuse prior work; do not repeat the old campaign

| Already implemented | Consequence for this campaign |
| --- | --- |
| Routed F16 matrix kernels, all-layer production dense F16 routes, matrix prefill attention, fused hyper-connection operations, row-split GDN prefill and decode | Do not propose these as missing. The [October 14 prefill][hip-prefill] and [decode][hip-decode] packets identify the ports and gates. Diff the remaining Gufo changes instead. |
| State-row target verification and accepted-prefix commit without full target replay | `Qwen4ExpGGUFResidentModelRunner.verify_target_block_state_rows` and `commit_target_block_rows` are the foundation, not work to reimplement. |
| Compact neural proposals, device argmax, device-resident carry between draft tokens, and head-free catch-up | Keep them. The remaining draft prompt loop is still scalar, and proposal scoring still runs the full vocabulary. |
| Shared AR batch, packed metadata, graph-replayed stateless segments and output head | Metadata copies already fell from 40 to 5 per step with no persuasive end-to-end gain. Do not budget that reduction again. |
| Completed-layer prefill time sharing | Keep the shipped decode-latency improvement. Measure idle prefill separately from prefill beside active decodes. |
| Mixed request/depth layout, GDN/QSA/PLE transactions, shared target execution and persistent stateless scratch | Reuse them. Provisional state is still allocated per pass, state movement is row-local, and mixed execution is eager. |

The latest shared-target implementation loses to separate execution on its
measured workloads, so its existing opt-in has a concrete cause. Improving it
requires more than enabling the switch. Conversely, historical claims that
MTP is capped at 1K, that all target verification is serial, or that the old
300 tok/s prefill composition is current must not become new work orders.

## 3. Implementation packets, in priority order

### Source map

Gufo paths below are relative to `src/models/qwen38_flash_next/` at the
review pin. The named bulk catch-up, subset head, lookup and rollback mechanisms
also exist in the measured tree; eligibility and controller details differ.
These are implementation facts, not isolated speedup measurements.

| Mechanism | Gufo source | hipEngine starting point |
| --- | --- | --- |
| Bulk predictor K/V and final carry | [`Executor::MtpForward`, `MtpBody`, `kv_only`][gufo-executor]; prompt caller near line 3105, cache-only branch near 4320 | `runtime/qwen4_exp_mtp.py::prime_prompt_chunk`; `generation/qwen4_exp_mtp.py::iter_generate` |
| Subset head | [`draft_vocab.hpp`][gufo-vocab]; `DraftLogits`, `DraftTable`, `SyncDraftVocab`; `batch.cpp::MtpHeads` | `runtime/qwen4_exp_mtp.py::forward`, full `config.vocab_size` |
| Copy proposals and length policy | [`prompt_lookup.hpp`][gufo-lookup], [`mtp_policy.hpp`][gufo-policy], `engine.cpp::Session` | `generation/qwen4_exp_mtp_stages.py::prepare_proposal`; `speculative/mtp_budget.py` |
| Compact recurrent rollback | [`GdnRollbackRowFloats`][gufo-kernel-api], [`RestoreGdnStateKernel`][gufo-kernels], `Executor::EnsureRollback` | `runtime/qwen4_exp_mixed_gdn.py`; c1 `_verify_rows_workspace` and `commit_target_block_rows` |
| Greedy result and graph controls | `Executor::GreedyMtpPredictions`, `Run`, `Session::Control` in [executor source][gufo-executor] | c1 device argmax already exists; mixed target still downloads all logits |
| Physical batched target/draft | [`ForwardBatch`, `MtpForwardBatch`, `MtpHeads`, `HcMixBatch`, `MoeBatch`][gufo-batch] | `runtime/qwen4_exp_batch_runner.py`, `runtime/qwen4_exp_mixed_target.py`, `generation/qwen4_exp_serving_mixed.py` |
| Routed, dense, sparse attention and producer staging | [`RoutedF16GEMMKernel`, `CompactSelectionKernel`, `GdnEpilogueKernel`][gufo-kernels]; `Executor::ReadyQ8` | `kernels/hip_gfx1100/quant/qwen4_exp_routed_f16.hip`, attention/GDN/GR families and their gfx1151 registrations |
| PLE, weight placement and host pinning | [`ngram.cpp`][gufo-ngram], `kernels/rocm/device_model.cpp`, `src/core/hip/` and the Nix ROCr patch | `loading/qwen4_exp_materialize.py`, PLE staging and `core/memory.py` |

All hipEngine paths in the last column are under `hipengine/`. Preserve the
peer backend registry: gfx1151 can register a shared kernel implementation
without making engine dispatch depend on a backend-name branch.

The [Gufo experiment record][gufo-experiments] also supplies useful
counterevidence: whole-batch shared-stage graph capture, extra transfer queues,
blanket SVM changes, persistent extra packed weights, several larger tiles and
confidence-based draft stopping lost their evaluations. These are not universal
impossibility results, but reopening one needs a different measured mechanism.
No external code was executed for this review.

### Execution order

Packet names below are checklist identifiers, not new runtime components.
`BASE` establishes attribution; `QUALITY` runs alongside performance work.
Independent packets may proceed after source/host coordination. Exact changes
can be evaluated while inherited numerical defects are being isolated; those
defects cannot be silently waived in the final quality claim.

### BASE — one comparable measurement and attribution harness

Extend existing canonical and HTTP harnesses rather than building an unrelated
benchmark. Reproduce both the canonical protocol and Gufo's published HTTP
protocol on **one physical host**, one engine resident at a time. First try the
published binary with its matching Nix/ROCr closure in an isolated process,
without replacing the host runtime. If that closure is unavailable or cannot
run on the host, build the published source tree with the local toolchain and
record the deviation as a same-source comparator, not an exact reproduction
of the published binary. Current main is a separate arm. Record compiler,
ROCr patch, library hashes, memory policy, power state and temperature for each
arm; never load both ROCm stacks into one process.

Required harness repairs/extensions:

- Verify full target-shard and sidecar identities. Our sampled directory/file
  fingerprints and Gufo's full SHA-256 hashes are not directly comparable.
  A filename ending in `Q8_0` does not prove sidecar equivalence.
- Align tokenization, template, thinking, penalties, EOS behavior and first-token
  accounting. Preserve mismatched cases with reasons instead of dropping them
  silently. Record actual new/cached/generated token counts and prompt hashes.
- Implement prepared-session decode and ordered cached-prefix replay for
  hipEngine if the existing harness cannot express them. Report Gufo's
  sum-of-request-rates **and** cohort tokens/wall, separately.
- Audit the AR baseline's capture settings. The existing
  `qwen4exp_mtp_inprocess_bench.py` calls `runner.prefill/step` with their
  defaults, while MTP uses compact proposals. Retain historical rows, but add
  a public-path-equivalent compact AR control; do not price diagnostics as
  unavoidable AR work.
- Capture target prefill, draft priming, proposal, verifier, state commit,
  selection/readback, cache handling and scheduler wall separately. Retain GPU
  interval union, copies, launches, host gaps and unattributed time; overlapping
  host API durations are not additive GPU cost.

**Exit:** reproducible baseline artifacts for the matrix in section 4, with
source/route witnesses and the cross-engine protocol's G0–G4 columns for every
reported speed row: identity/tokenization conformance, greedy output comparison,
distribution comparison where available, repeated-run determinism and the task
battery. Gufo G2 is explicitly not applicable when no logits/logprob surface
exists; do not substitute output hashes for it. Rerun applicable characterization
for each new pin, including the published-HTTP depth and concurrency rows;
identify the exact workload scope of the linked quality packet. Rank subsequent
kernel work by the current measured owner cost, not September's family totals.
Implement harness contract tests for denominators, cache hits, tokenizer
exclusions, quality-column bindings and route fallbacks before using new rows
for decisions.

### QUALITY — repair the failures exposed by the serving follow-up

The [October 20 numerical packet][hip-numerics] has repeatable failures in all
four scheduler arms, including true AR. The separate short c1 composition
reported in the [integration worklog][hip-integration] reaches only 0.9865
top-1 against the 0.99 floor. The
[prefill-isolation packet][hip-attribution] still fails the 12,287-token
mean/max limits after disabling dense-wide on top of other recovery overrides.
Neither successful batching nor agreement between mixed and batch logits
resolves those failures.

**October 22 checkpoint:** the Q4 decode binder now selects the strict owner
already declared in its manifest, while other production routes stay on.
Public generation confirms that selection; three short-suite repeats and state
fingerprints agree. The complete short composition still fails its top-1 floor,
so this packet remains open. A two-plane correction was tested and set aside
without a production promotion. See the
[scoped repair evidence](../../benchmarks/results/20261022-qwen38fn-q4-decode-repair.json).

1. Reproduce current c1 and serving controls with identical input tokens,
   chunking and forced continuation. Keep same-chunk strict controls separate
   from chunk-boundary changes. Retain effective, finite BF16-rounding controls.
2. Isolate the Q4 DP4A decode contribution and the residual prefill contribution
   by layer/role and actual selected variant. The strict-Q4/Q8-down tile-map bug
   was fixed in `f433514ed`; verify that fix before attributing stale mapping
   errors to floating-point drift.
3. Repair the responsible implementation or select a passing registered
   arithmetic variant for the affected operation. Disabling roughly 30 routes
   in a diagnostic is not the desired production implementation.
4. Recheck short, long, transition, isolation and task gates on the complete
   candidate composition. The [54-answer review][hip-tasks] explicitly reports
   failed non-inferiority and two candidate-specific regressions, as well as
   defects shared with controls. Separate those findings; do not summarize all
   task defects as inherited or replace semantic review with matching hashes.

**Exit:** the binding gates in section 5 pass, with causal reproducers for code
repairs. No threshold relaxation or artifact-identity admission list is part
of this packet. Do not turn a historical task answer into a prompt-specific fix.

### MTP-PREFILL — bulk predictor K/V construction and final-row catch-up

**Starting gap at the execution baseline:**
`hipengine/runtime/qwen4_exp_mtp.py::prime_prompt_chunk` loops over prompt tokens
and calls the whole predictor layer with `run_head=False`. This skips the
vocabulary head, but not attention, feed-forward or repeated host dispatch.
Target hidden rows also cross the host boundary. Gufo implements K/V-only prompt
projection and final-tile catch-up instead of computing outputs it will discard.

**First increment complete:** scalar prompt priming now omits unused attention
and feed-forward outputs except on each chunk's final carry row. The existing
QSA restore cursor also no longer marks uncomputed pooled blocks valid. Five
real-sidecar cache/carry/full-logit cases and twenty matched public response
pairs pass; see the [scoped diagnostic artifact](../../benchmarks/results/20261021-qwen38fn-mtp-cache-only.json).
**Bulk implementation complete:** bounded prompt projections and synchronous
device hidden handoff run on the default path. Sixteen real-sidecar cases
cover scalar/workspace/device execution, unfused and matrix-kernel overrides,
cache state, final carry and complete proposal logits. Cancellation commits
only checked rows. The short public resource probe and all twenty matched
HTTP response pairs pass, including a 2K suffix on a 32K cached prefix.
All five scoped prefill medians improve against the preceding scalar increment;
see the [bulk implementation artifact](../../benchmarks/results/20261021-qwen38fn-mtp-bulk.json)
and [worklog](../../worklog/entries/20261021T224428.114242Z-qwen38fn-gufo-qwen38fn-mtp-bulk-5bdac7.md).
This closes this packet, not the campaign's broader quality or concurrency gates.

Implement a batched prompt path that consumes the shifted target-hidden
sequence, writes every required predictor KV/index row, and computes only the
final carry needed by proposal. Keep that carry and target hidden rows on the
device where the ownership contract permits. Preserve the first-row zero seed,
previous-chunk seed, sparse-boundary behavior and cancellation progress. Retain
the scalar path as the oracle/fallback, not as the default for supported bulk
inputs.

**Exit:** full predictor versus bulk catch-up tests on complete/tail chunks,
one-token tails, dense/sparse transitions and restored continuations; identical
declared cache/carry/candidate results or a passing declared numerical contract;
real HTTP MTP prefill including catch-up improves. Test fresh long prompts as
well as a 2K suffix on a deep cached prefix. This is the highest-confidence
structural prefill opportunity; no speedup estimate is assigned before `BASE`.

### VERIFY — specialize few-row execution and remove avoidable transaction cost

**Verified gaps:** the c1 verifier uses prefill-shaped kernels for 2–8 rows.
`hipengine/runtime/qwen4_exp_mixed_target.py` allocates per-layer provisional
GDN/QSA state, dispatches layers eagerly, and always copies full logits plus
hidden rows to the host. `qwen4_exp_mixed_gdn.py` gathers/scatters operands and
copies recurrent snapshots per row and depth wave. Stateless scratch reuse is
already implemented and is not the same as persistent provisional state.

Deliver these as separately measured increments:

1. Give provisional state and metadata stable owner-managed storage, with
   logical transaction reset, bounded growth and explicit failed-owner cleanup.
   Batch state movement with indexed kernels or pointer tables. Evaluate
   Gufo's compact rollback representation—one base state plus exact update
   operands—against full snapshots; charge reconstruction time and bytes.
2. Add a compact greedy mixed result: device argmax/accepted-row selection and
   device hidden-seed handoff. Keep full logits where sampling, grammar,
   logprobs or diagnostics require them; preserve every control's semantics.
3. Select few-row dense, routed-expert, hyper-connection and recurrent kernels
   by geometry and row layout. Gufo's small-row integer-matrix projections and
   batched expert weight sharing are leads, not automatically equivalent math.
4. Capture stable verifier/depth-wave work where profiling shows submission
   cost. Preserve graph pointer lifetimes, slot epochs, active masks, rejection
   and reuse. Measure graph versus eager; do not assume a whole-step graph wins.

**Exit:** full-cycle committed tokens/time improves at widths 1/2/4/6/8 and
across rejection depths, with exact request/rollback ownership. A faster target
sub-window alone does not establish profitable MTP. The initial c1 target and
shared-target variants must be priced independently.

### DRAFT-HEAD — score a request-local vocabulary subset

**Verified gap:** each `Qwen4ExpGGUFMTPDraftRunner.forward` proposal calls
`launch_gguf_linear` with `config.vocab_size`. Gufo uses a 98,304-row prefix
plus a 1,024-row request-local table, with fallback to the full head; it does
not replace the target vocabulary or require another quantized head artifact.

Implement a registered indexed draft-head operation and request-owned table
built only from permitted committed context. Preserve the full target head and
existing deterministic-proposal verification. Do not import Gufo's sampled
neural p/q policy as an incidental part of a head optimization. Cover Japanese,
Unicode, tools, rare tokens and constrained output; acceptance loss can erase
a faster head. Restore/rebuild table state with the request.

**Exit:** lower complete proposal/cycle cost on the full suite and heldouts,
correct target sampling, bounded memory and safe fallback. No benchmark-token
allowlist, language-based admission rule or fixed measured-prompt table.

### LOOKUP — add prompt-derived proposals, then cost-aware depth

Gufo's largest repetitive-text gain comes from session-local copying with full
target verification. Our Flash-Next proposal stage always asks the neural
runner; no prompt-lookup provider is wired into that stage.

Add request-local matching against committed history and route copied suffixes
through the same proposal/verify/commit interfaces. Start with greedy copying
that can skip the neural head; then validate sampled copies under our existing
point-mass proposal semantics. Respect request boundaries, EOS, maximum output,
stops, grammar, cancellation and rollback. Never treat a common chat wrapper as
evidence that a new answer is predictable. Cross-request hint pools are not
part of the first implementation.

Next adapt draft length to **accepted tokens per complete cycle cost**, not
acceptance percentage alone. Reuse the request-owned policy interfaces in
`hipengine/speculative/mtp_budget.py` where they fit; Flash-Next currently
supplies its configured candidate budget. Separate neural and copied-proposal
costs, occupancy and context bins, and preserve sampled replay policy state.
Expanding beyond the current 1–4 draft budget requires implementing its actual
state/buffer limits, not just raising the constant. Freeze controller training
and heldout inputs before tuning. Gufo's confidence-stopping experiment was
rejected; do not copy it as a shipped success.

**Exit:** both mixed/new-text and repetitive/edit workloads pass quality and
sampling checks; copied/neural proposed and accepted counts are distinct.
Repetitive throughput is reported separately, never as general MTP speed.

### BATCH — make concurrent speculation profitable

Build on `VERIFY` and the existing staged provider/mixed layout. Batch neural
proposal projections and heads across requests; the current serving mixed path
calls `prepare_proposal` separately for each speculative request. Combine
independent requests with candidate-depth dependencies in one target layout,
not a loop of independent c1 verifications presented as physical batching.

Price all-AR, all-MTP, one-MTP-plus-AR and copying-plus-new-text cohorts. Keep
request width distinct from total verifier rows. Include width 6 and ragged
widths, not only power-of-two buckets. Bound copied-chain widths beside ordinary
requests so one long copy does not cause unacceptable peer token gaps. Preserve
prefill time sharing and explicit-request behavior. Automatic route policy may
choose among working implementations based on measured costs; it must not
silently disable an explicit request.

**Exit:** shared execution beats true AR in its claimed workload without hiding
peer latency or prefill regressions. Removing the existing selection flag
`HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET` also requires its full
[REFACTOR clearing condition](../REFACTOR.md#flash-next-staged-mtp-physical-width-and-controls):
run `scripts/qwen38fn_mixed_economics.py --widths 1,2,4,8 --arms ar,mixed`
against `hipengine serve` with the flag set and unset, and require the shared
arm to meet or beat separate execution **at every width** before flipping the
default and deleting the variable. Add width 6 for this campaign's comparison.
A win at one width does not clear that condition. A cost-based dispatcher may
keep separate execution where it demonstrably wins; changing that design needs
its own explicit recorded decision, not a silent narrowing of the clearing gate.

### DECODE — close the remaining per-step kernel gap

Compare the live owner census with Gufo's current decode source. Candidate
mechanisms are producer-written Q8_1 activations reused by consumers, grouped
small projections, Q8 hyper-connection prefetch, Q4 weight reuse across rows,
Q5 high-bit expansion, fused short convolution/history, and sparse attention
head grouping plus one-pass selection compaction. October 20 F16/BF16 vector
loads for small projections are a **post-benchmark** lead, not part of the
October 18 result.

Start with the largest measured residual; keep dense GEMV and routed MoE
separate. At long depth, measure selector/compaction growth separately from
attention over selected keys. Preserve selected sets and `KVLiveSpans`.
Introducing quantized activation staging changes our arithmetic even when
Gufo's change was exact relative to its own parent.

**Exit:** same-host direct and served AR gains at shallow and deep contexts,
with C1 and wider rows non-regressive under the declared contract. PLE hashing
was below 1 ms across eight width-1 steps in the latest host profile; a rewrite
is not the default answer to a roughly 40 ms step.

### PREFILL — port only remaining profitable kernel differences

After bulk MTP priming, profile idle AR prefill at matched chunk size and depth.
Evaluate the delta from our already-ported kernels to Gufo's wave-owned paired
weights, coalesced Q5_1 down loads, projection/convolution fusion, register-only
deferred residuals and producer-written next-consumer representations. Inspect
our current source before each port; some equivalent fusions already exist.
Use real expert routing distributions, not synthetic concentrated populations.

For attention, compare dense and sparse paths separately: software-pipelined
QK/softmax/PV, vector staging, bounded sparse fragment scheduling and selection
scratch. Final-layer AR row pruning must preserve every cache/state row and is
not applicable when MTP needs the full target-hidden sequence. Compare 1K/2K/4K
idle chunks with active-decode time sharing; larger chunks must not undo its
latency benefit.

**Exit:** complete prefill gains, actual kernel engagement and the applicable
numerical/task gates. Do not bring back a second full expert-weight layout
without a memory/traffic justification: the old compensated-T16 residency
approach is not the missing foundation of the current F16 path.

### MEMORY — investigate stalls only when reproduced locally

Gufo's package protects runtime queue-save/launch/code buffers from host
compaction, pins owned host buffers, prepares cold functions, and grows cache
backing outside sensitive work. These explain observed Gufo stalls; they do
not prove hipEngine has the same defect or account for steady kernel speed.

Instrument page faults, KFD eviction/queue suspension, memory pressure, first-use
module loads and host allocation during cold/warm and cached continuation runs.
If a stall correlates, reproduce it with an ordered control before a bounded
runtime/allocation change. Preserve coherent managed memory, pin-failure
fallbacks and shared-buffer lifetime. No blanket SVM disable, new transfer
queue pool, system runtime replacement or PLE reader-pool rewrite without
local evidence. A no-stall result closes this packet as unnecessary for the
measured workload; it does not assert universal absence of stalls.

## 4. Measurement matrix and commands

Follow [CROSS-ENGINE-PROTOCOL](../CROSS-ENGINE-PROTOCOL.md) for the canonical
comparison. The published-HTTP reproduction is an additional protocol, not a
replacement. Freeze prompts, metrics and controller policy before timing.

| Matrix | Required rows | What it resolves |
| --- | --- | --- |
| Canonical AR | Four categories × p512/p1024/p4096; 128 decode transitions; one warmup, at least three repeats | Continuity with prior same-host work; preserve the common-case set and report exclusions |
| Published HTTP depth | Exact ordered depths 0/4096/8192/12288/16384/32768/65536/131072; capacity 133760; approximately pp2048/tg128; AR, mixed MTP, repetitive MTP | Current Gufo target, actual cached/new counts, MTP catch-up inclusion; G0–G4 bindings |
| Prepared concurrency | C1/C2/C4/C6/C8; context capacity 4096 per session; pp2048 prepared before tg128; all-AR, all-MTP mixed, all-MTP repetition | Decode-only scaling, sum of request rates versus cohort tokens/wall; G0–G4 bindings |
| Real serving | Full ten-prompt suite plus eight heldouts; fresh and cached; all-AR/all-MTP/one-MTP-plus-peers; blocking and streaming | Queue, time to first token, inter-token median/p95/p99/max, complete request wall and route selection |
| State/shape boundaries | 997/1023/1025/2051/2052/4093/12287 tokens; tail chunks; widths 1–8 and dynamic admit/cancel/refill; each rejection depth | Behavior on both sides of dispatch/chunk/sparse thresholds and unrelated lengths |
| Long context and lifecycle | Numerical controls plus retrieval; 32K/64K/128K serving; cold/warm PLE, cancellation, reuse and memory pressure | Quality, capacity and state stability rather than throughput alone |

Retain an exact publication replay separately from the repeated comparison.
Gufo's prepared-concurrency path prepares the cohort before every round; this
replaces its configured decode warmup. The publication uses one measured
repetition. The campaign's decision run uses at least three measured repetitions
per arm; do not relabel that extension as the original single-sample protocol.
The ordered depth replay retains its own warmup and prefix-preparation history.

Do not run full-prompt ladders to find allocation bounds. Use the Tier-1 probe
in [HARNESSES](../../benchmarks/HARNESSES.md) first, then complete the named
long workloads. Missing measurements do not prohibit running supported inputs.

Existing entry points, inspected for this review. Set `MODEL` to the target
GGUF directory and `SIDECAR` to the MTP GGUF file; verify their identities before
measurement. The function below applies the
[THEROCK clean-process setup](../reference/THEROCK.md#clean-process-wrapper)
using the validated gfx1151 interpreter, not the CPU-only project environment.
It also binds imports to the current checkout. The first canonical command
prebuilds every shape before the cached-only measurement.

```bash
ENV_PREFIX=~/miniforge3/envs/therock
PY="$ENV_PREFIX/bin/python"
ROCM_ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
OUT=/tmp/qwen38fn-gufo-baseline
mkdir -p "$OUT"
"$ROCM_ROOT/bin/hipcc" --version > "$OUT/hipcc-version.txt"

run_hip() {
  env -i HOME="$HOME" USER="$USER" \
    PATH="$ENV_PREFIX/bin:$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib" \
    HIP_PATH="$ROCM_ROOT" ROCM_PATH="$ROCM_ROOT" HIP_LIB_PATH="$ROCM_ROOT/lib" \
    HIP_INCLUDE_PATH="$ROCM_ROOT/include" \
    HIP_DEVICE_LIB_PATH="$ROCM_ROOT/lib/llvm/amdgcn/bitcode" \
    HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
    HIPENGINE_COMPILER_VERSION_FILE="$OUT/hipcc-version.txt" \
    PYTHONPATH="$PWD" "$PY" "$@"
}
run_hip -c 'import ctypes, hipengine; ctypes.CDLL("libamdhip64.so"); print(hipengine.__file__)'
run_hip scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --execution-profile production \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --warmups 0 --repetitions 1 \
  --compiler-version-file "$OUT/hipcc-version.txt" --output "$OUT/prebuild.json"
run_hip scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --execution-profile production \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --warmups 1 --repetitions 3 \
  --compiler-version-file "$OUT/hipcc-version.txt" \
  --require-cached-build --output "$OUT/canonical-ar.json"

run_hip scripts/qwen4exp_mtp_inprocess_bench.py \
  --model-root "$MODEL" --sidecar "$SIDECAR" \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-tokens 64 --budget 3 --execution-profile production \
  --output "$OUT/mtp-suite.json"

run_hip scripts/qwen4exp_canonical_ar_bench.py openai --help
run_hip scripts/qwen4exp_llamacpp_prefill_comparator.py --help
run_hip scripts/qwen38fn_mixed_economics.py --help
run_hip scripts/qwen4exp_layer2_profile_gate.py --help
run_hip scripts/qwen4exp_long_context_gate.py --help
```

Check [THEROCK](../reference/THEROCK.md) if the installed SDK differs. These
commands isolate hipEngine's ROCm libraries; launch the Gufo comparator with
its own recorded environment, not through `run_hip` with inherited TheRock
libraries. Canonical and in-process compact AR controls are implemented.
The HTTP economics harness supports explicit AR/all-MTP/mixed arms, repeated
counterbalanced measurements and hipEngine preparation through `--prepared`.
It renders each chat prompt once, prefills its exact token sequence minus the
final token before the timed cohort, then requires that checkpoint to be reused.
This is a declared one-token-suffix adaptation, not Gufo's identical-prompt
preparation. A missing checkpoint invalidates the row rather than silently
pricing full prefill as prepared decode. Native phase metrics distinguish
output-count decode rates, transition-count rates and complete cohort wall.

For a running hipEngine server, set `PROMPTS` to frozen JSONL rows with `id`,
`category`, and either `messages` or `prompt_token_ids`. Use the pp2048 prompt
set for the prepared matrix, not the shorter task battery. Set `SERVER_MANIFEST`
to the captured server command, source, model and host identities; the harness
records its own source separately.

```bash
run_hip scripts/qwen4exp_http_mtp_economics.py \
  --url http://127.0.0.1:18400 --model qwen38fn-capon \
  --prompts "$PROMPTS" --server-manifest "$SERVER_MANIFEST" \
  --prepared --width 8 --max-tokens 128 --arms ar,mtp,mixed \
  --repetitions 3 --output "$OUT/prepared-http.json"
```

`scripts/qwen4exp_gufo_http_replay.py` calls the pinned Gufo checkout's prompt,
calibration, prefix-reply and adaptive retry implementation without editing that
checkout. It journals every request and prefix reply. It first checks allocation
with a short request, then starts a fresh server to preserve the ordered replay's
history. Set `GUFO_SOURCE`, `GUFO_BIN` and `GUFO_SIDECAR` to the reviewed source,
its built executable and the verified shared sidecar. Run in that binary's
isolated library environment, not through `run_hip`:

```bash
python3 scripts/qwen4exp_gufo_http_replay.py \
  --source-root "$GUFO_SOURCE" --binary "$GUFO_BIN" \
  --model "$MODEL_FIRST_SHARD" --sidecar "$GUFO_SIDECAR" \
  --output-dir "$OUT/gufo-published-depth" --repetitions 1
```

`MODEL_FIRST_SHARD` names the target's first GGUF shard. The default replay
retains all eight ordered depths, capacity 133760, all three workloads and
native count/timing checks. `--probe-only` stops after the allocation checks;
`--depths` subsets and repeat extensions are labelled separately. The output is
diagnostic until its per-workload G0–G4 characterization is complete; unavailable
native output IDs are not replaced by text hashes.

`scripts/qwen4exp_hip_http_replay.py` uses the same pinned prompt, calibration,
prefix-reply and retry recipe through hipEngine's public endpoints. It adds an
untimed checkpoint of the common token prefix between the rendered history and
continuation. The measured response must report exactly that cached count.
This is a prompt-checkpoint adaptation, not Gufo's live reply-state reuse.
Natural EOS is retained; prefix replies may be short, but measured responses
must contain all 128 native tokens.

Start a fresh hipEngine server for each `--arm` (`ar`, `mtp-prose`, or
`mtp-repetition`), with context capacity 133760 and one active request slot.
The MTP arms need the shared sidecar/provider configuration. Set `MODEL_ALIAS`
to its served name and capture its command and identities in `SERVER_MANIFEST`.
Run the Tier-1 allocation probe before the full depth workload.

```bash
run_hip scripts/qwen4exp_hip_http_replay.py \
  --source-root "$GUFO_SOURCE" --url http://127.0.0.1:18400 \
  --model "$MODEL_ALIAS" --server-manifest "$SERVER_MANIFEST" \
  --arm ar --output-dir "$OUT/hipengine-ar-depth"
```

The hipEngine ordered-depth runs are complete. `BASE` still needs prepared
cross-engine replay and quality-column bindings. Existing
`qwen38fn_mixed_economics.py` measures all-AR and one-MTP-plus-peers, not the
all-MTP prepared table.

Use counterbalanced same-session arms and unchanged-code controls to distinguish
small gains from thermal/run-order variation. Profile only a warmed serving
child with cached kernels; never a parent that launches inference processes.
Store exact commands, hashes, raw-sample summaries and failures with each unit.

## 5. Correctness, promotion and completion

Follow [OPTIMIZATION](../OPTIMIZATION.md),
[EXECUTION-PROFILES](../EXECUTION-PROFILES.md) and [TESTING](../TESTING.md).
Strict parity applies only to variants that declare it. Production candidates
may change arithmetic but need the complete applicable numerical and task gate;
Gufo's exactness against Gufo does not establish parity against our parent.

For each implementation unit:

- Add a targeted RED test where practical; test CPU formulas and invalid
  ownership before GPU work. GPU tests require explicit HIP-availability guards.
- Preserve exact positions, masks, `KVLiveSpans`, PLE history, recurrent state,
  accepted-prefix commit, sampler/grammar state and failure containment.
- Measure full-vocabulary strict-relative rows with aligned teacher tokens,
  same-schedule repeats and neighbor isolation. Apply the short global envelope
  and scope top-1 floors, plus the existing valid long-context round-off floor
  and retrieval gate. Do not silently extend a short gate to certify deep rows.
- Cover the full category suite and heldouts, task review and dynamic request
  transitions. Fix test defects without weakening the requested contract.
- Keep registered strict fallbacks, torch-free runtime and four-axis dispatch.
  Before a kernel port run
  `python3 scripts/check_lineage.py --kind kernel --diff stat` and inspect the
  relevant parent differences. Register new source files and exact baselines in
  `docs/source_lineage.json`, preserving existing port provenance, and update
  the `docs/KERNELS.md` path map as needed. Its existing Gufo entry names
  `88090afb`, not either reviewed October pin. Record the port's file, license
  and exact source commit in the implementation unit.
- Confirm actual selection through `LLM.generate()` or `hipengine serve`, then
  publish compact results and the required benchmark rollup. Update the ledger
  with the decision and a specific reopening condition for rejected work.

**Parity target:** hipEngine/Gufo throughput at least 1.00 for each matched
claimed phase/shape/workload, supported by repeated same-host measurements and
controls; no substituting repetitive MTP for mixed text or one aggregate for
failed cells. Report uncertainty near equality rather than inventing a win.
Use identical timing boundaries, report both engines' quality characterization,
and require hipEngine's own quality gates. A speed match that fails those gates
or hides worse peer latency is not campaign completion. Predeclare any
latency tradeoff before candidate timing; existing wins are not rejected merely
because they fall short of final parity.

### Completion checklist

- [ ] `BASE`: pinned same-host canonical and published-HTTP comparison, complete
  timing accounting, model/sidecar identities and route witnesses.
- [ ] `QUALITY`: inherited numerical and candidate-specific task failures
  resolved under the unchanged applicable gates.
- [x] `MTP-PREFILL`: bulk prompt/catch-up implementation, boundary tests and
  complete HTTP prefill measurement including predictor work.
- [ ] `VERIFY`: persistent transactional state, compact result/handoff,
  few-row kernel and graph decisions supported by full-cycle evidence.
- [ ] `DECODE`: current owner census and a measured disposition for each
  prioritized decode/sparse-selection mechanism.
- [ ] `DRAFT-HEAD`: subset-head implementation or a measured rejection with
  acceptance, sampling and heldout evidence.
- [ ] `LOOKUP`: request-local proposals and adaptive-depth decisions, with
  separate neural/copy accounting and mixed/repetitive results.
- [ ] `BATCH`: physical proposal/target execution and dynamic lifecycle at
  C1/C2/C4/C6/C8, with true AR and separate-route controls.
- [ ] `PREFILL`: remaining source deltas evaluated by current owner cost;
  idle and decode-active prefill both preserved.
- [ ] `MEMORY`: local stall evidence and fix, or a documented unnecessary result.
- [ ] `CLOSE`: final-source matrix, public-surface route checks, numerical/task
  gates, benchmark artifacts/rollups and immutable worklogs. Run the applicable
  full milestone suite once; use focused repair after isolated failures as
  required by repository policy. Report parity cell by cell and any exact
  remaining gap. Completed experiments alone do not mean the parity goal met.

A direct source review used the committed hipEngine snapshot named above.
Prefix reuse landed concurrently in `22fa2661b`; it also included the initial
staged version of this campaign. Its runtime changes were not authored or
validated by this review. Refresh source and ownership before execution, and
reuse that prefix-cache implementation when building the prepared/depth harness.

The October 20 planning review ran no GPU experiment and added no benchmark row.
Execution began October 21; its measurements and implementation outcomes will be
linked from the checklist. The review itself makes no new measured speedup claim.

[gufo-tree]: https://github.com/gufo-org/gufo/tree/f020bd3a037d58aa42138ca11fa895524e43cdd6
[gufo-executor]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/kernels/rocm/executor.cpp
[gufo-vocab]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/draft_vocab.hpp
[gufo-lookup]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/prompt_lookup.hpp
[gufo-policy]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/mtp_policy.hpp
[gufo-kernel-api]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/kernels/rocm/kernels.hpp
[gufo-kernels]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
[gufo-batch]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/kernels/rocm/batch.cpp
[gufo-ngram]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/src/models/qwen38_flash_next/ngram.cpp
[gufo-experiments]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/docs/models/qwen3.8-flash-next/EXPERIMENTS.md
[gufo-bench]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/docs/models/qwen3.8-flash-next/BENCHMARKS.md
[gufo-quality]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/docs/models/qwen3.8-flash-next/QUALITY.md
[gufo-config]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/docs/models/qwen3.8-flash-next/artifacts/bench.json
[gufo-identities]: https://github.com/gufo-org/gufo/blob/f020bd3a037d58aa42138ca11fa895524e43cdd6/docs/models/qwen3.8-flash-next/artifacts/model-identities.json
[hip-topline]: ../../benchmarks/results/2026-10-19-gfx1151-flashnext-topline/README.md
[hip-cross]: ../../benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1/README.md
[hip-prefill]: ../../benchmarks/results/2026-10-14-qwen4exp-prefill-hc-gdn-batch/README.md
[hip-decode]: ../../benchmarks/results/2026-10-14-qwen4exp-decode-batch/README.md
[hip-serving]: ../../benchmarks/results/2026-10-21-gfx1151-flashnext-serving-economics-ab.json
[hip-host]: ../../benchmarks/results/2026-10-20-gfx1151-flashnext-host-overhead-diagnostic.json
[hip-integration]: ../../worklog/entries/20261020T192706.014051Z-serving-followup-qwen38fn-serving-integration-b53ed9.md
[hip-numerics]: ../../benchmarks/results/2026-10-20-gfx1151-flashnext-serving-numerical-failure.json
[hip-attribution]: ../../benchmarks/results/2026-10-20-gfx1151-flashnext-prefill-route-attribution.json
[hip-tasks]: ../../benchmarks/results/2026-10-20-gfx1151-flashnext-serving-task-review.json
