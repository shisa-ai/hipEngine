# SPECDEC2 / MTP2 Research and Architecture Proposal

Status: **research complete; architecture approved; normative implementation plan is [`SPECDEC2.md`](SPECDEC2.md)**

Date: **2026-08-24**

Scope: continuous speculative serving across GGUF MTP, PARO MTP, DFlash/DFlash2,
chain/tree providers, and the hipEngine `gfx1100` / `gfx1151` targets.

## Executive decision

hipEngine should build a unified **SPECDEC2** serving path. **MTP2 should be the
first provider on SPECDEC2, not another generation API or model-owned loop.**

This is an orchestration and execution-owner redesign, not a request to discard
all existing speculative work. The current tree already has strong migration
assets:

- provider-neutral `DraftBatch`, `TargetVerifyBatch`, and `AcceptResult` records;
- exact CPU/fake transaction simulation;
- reusable GGUF NativeSpecCycle N1R/N2/N3 components;
- exact GGUF selected-state/KV commit and rollback paths;
- PARO chain/tree verifier and GPU accept-summary infrastructure;
- Laguna DFlash draft/target ownership and streaming semantics; and
- C2 request, slot, output, resource, KV, prefix, graph, and cancellation
  ownership.

The missing layer is the one that matters for `SpecDec + c=N`: speculative work
is not actually driven incrementally by the C2 scheduler. The current dense GGUF
server route finishes a complete speculative request while holding the model-loop
submission lock, then publishes a `VERIFY_CHAIN` submission whose work is already
done. The generic records describe the right future but do not own the current GPU
hot path.

The recommended architecture is a **target frontier**:

- every due request contributes one rooted target-work graph;
- ordinary AR is a root-only graph (`K=0`);
- MTP/DFlash chains are root plus candidate nodes;
- tree methods use the same parent-indexed node representation;
- provider proposal batches are formed first;
- compatible graphs are packed into target-frontier batches;
- target verify, accept, selected state/KV commit, cursor update, and output
  publication remain independently owned per request; and
- the policy may choose `K=0` before mutation whenever concurrency, context,
  memory, or SLO economics make speculation a loss.

For `gfx1151`, start with the retained queue2 policy and one serial execution
stream. The just-completed AR campaign found no marked multi-stream overlap and
ended mechanically qualified but product-blocked at c32: **10.590 aggregate
tok/s, 18.617 s TTFT p95, 2.125 s ITL p99, 0/3 SLO, zero goodput**. SPECDEC2 must
not assume overlap or a speculative multiplier will fix that c32 latency. Its
first purpose is to make the scheduler capable of choosing and efficiently
executing the right target work—often MTP at low C and AR (`K=0`) at high C.

## What this research answers

### Do other engines really support speculative decoding with c=N?

Some do, but “support” spans materially different designs.

| Engine | Multiple live requests plus SpecDec | What actually happens | Qualification |
| --- | --- | --- | --- |
| **vLLM** | **Yes** | One token-progress scheduler covers prefill, AR, and speculative tokens; it reserves lookahead KV, batches draft work, verifies packed per-request rows, and contracts accepted results independently. Dynamic K may reach zero at high batch. | Full continuous architecture, with backend/method-specific restrictions. |
| **SGLang** | **Yes** | A scheduler-owned speculative worker runs draft → target verify → draft extend against request/token pools, per-request accept counts, and shape-specific graph bundles. Adaptive tiers vary with batch size; zero-step uses AR-equivalent target work. | Full continuous architecture; algorithms have different overlap/prefill/parallelism limits. |
| **llama.cpp server** | **Yes, on current main** | One `common_speculative` owner tracks per-sequence state. Active slots draft together, root+draft rows from all slots enter one target `llama_batch`, and acceptance/rollback occurs independently per slot. Non-spec slots can share the target batch. | Real continuous server support; simpler ownership, unified-cache and checkpoint/re-evaluation costs remain. |
| **MTPLX** | **True physical c=N MTP in sealed fixed-width cohorts; no late admission/refill** | A Qwen3.6-35B-A3B K1 lane seals 2–3 requests onto B3/T2 (`M6`) and 4–8 onto B8/T2 (`M16`). Draft and target work are physically batched; ragged KV, GDN state selection, sampling, cancellation, and output are row-owned. The cohort width and members remain fixed until drain. | Strong model/hardware-specific proof of batched MTP and selected recurrent-state commit; not full continuous SpecDec or a generic provider architecture. |
| **mlx-vlm** | **Fixed-cohort batch, not true late-admission continuous SpecDec** | Pending requests are coalesced, prefetched, drafted/verified as a batch, and filtered on finish. The speculative loop runs that cohort to completion before collecting the next arrivals. | Useful batched verifier/reference, but no refill into an active speculative cohort. |
| **oMLX** | **Partial** | Normal MTP is singleton. An opt-in multi-row path extracts each row, runs one singleton verify cycle per row, and merges caches; measured c2/c4 aggregate is below normal batched AR, so it is default-off. DFlash is serialized. | Correct lifecycle experiment, not an efficient shared verifier. |
| **vMLX** | **No for model-based MTP/draft/DFlash2** | `should_use_speculative(is_batched=True)` returns false; native MTP also rejects batch size >1. Prompt-lookup verification removes/reinserts one row and is not shared batched verification. DFlash2 is a direct simple-engine route. | AR continuous batching and separate singleton speculation. |
| **hipEngine today** | **No production hot path** | C2 AR is real. Spec submission invokes complete model-owned generation synchronously; dense GGUF requests are serial, PARO/Laguna cycles are single-request, and generic `VERIFY_CHAIN` metadata is not executed by `ResidentEngineLoop`. | Strong records/oracles and single-request kernels, but SPEC-C5 is correctly product-blocked. |

The useful comparison is therefore not “who has a flag?” It is whether the
scheduler owns proposal, verifier packing, provisional state/KV, per-request
acceptance, commit, refill, cancellation, and output as bounded work.

## Research method and source pins

The external conclusions above come from current shallow source audits, not
feature-list inference. Exact revisions:

| Project | Audited commit | Primary files |
| --- | --- | --- |
| [oMLX](https://github.com/jundot/omlx) | `404c059f442fbb09a8a7690789dcf2d80c82b7a3` | `omlx/patches/mlx_lm_mtp/batch_generator.py`, `omlx/engine/dflash.py`, `omlx/scheduler.py` |
| [vMLX](https://github.com/jjang-ai/vmlx) | `0abdd34cd3caef76c629c557ca36cb52eef2b9ae` | `vmlx_engine/speculative.py`, `vmlx_engine/patches/mlx_lm_mtp/batch_generator.py`, `vmlx_engine/scheduler.py`, `vmlx_engine/dflash2_runtime.py` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `f280b26983ad0fdb705a0d9ebf0503e76f2899b0` | `tools/server/server-context.cpp`, `common/speculative.{h,cpp}`, `docs/speculative.md` |
| [MTPLX](https://github.com/youssofal/MTPLX) | `bd4421567f9e16ce957c6ef97708b072dcd73937` | `mtplx/server/mtp_batch.py`, `mtplx/a3b_mtp_batch.py`, `mtplx/batching/`, `docs/concurrency/qwen35b-mtp-batch.md` |
| [vLLM](https://github.com/vllm-project/vllm) | `7797b6022c129b862e45ae6aed08822e65d1bccb` | `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/spec_decode/`, `vllm/v1/worker/mamba_utils.py` |
| [SGLang](https://github.com/sgl-project/sglang) | `586211bc461c4dbd8df9932bf709aa3d018945d1` | `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/speculative/`, speculative docs |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | `16feaebc904671d477f5c1cf01456608e02a4420` | `mlx_vlm/speculative/{dflash,mtp,eagle3}.py`, `mlx_vlm/server/generation.py` |

External numerical rows are used only to explain those projects' own policy
choices. They are not hipEngine performance baselines. hipEngine evidence remains
subject to the repository's same-host/model/quant/protocol policy.

Relevant systems literature reviewed for design constraints:

- [The Synergy of Speculative Decoding and Batching](https://arxiv.org/abs/2310.18813)
- [SmartSpec](https://arxiv.org/abs/2406.14066)
- [BASS](https://aclanthology.org/2024.findings-acl.489/)
- [MagicDec](https://arxiv.org/abs/2408.11049)
- [Batch Speculative Decoding Done Right](https://arxiv.org/abs/2510.22876)

The consistent lesson is that draft depth is an execution-policy variable, not a
model constant. Increasing C consumes the same spare arithmetic that speculation
tries to exploit. Correctness also requires per-request positions, attention
visibility, RNG, state, and KV to remain aligned after different requests accept
different depths.

## External architecture lessons

### vLLM: schedule token progress, not a separate speculative request type

The decisive vLLM scheduler comment is that there is no separate decode or
prefill phase: each request has `num_computed_tokens` and
`num_tokens_with_spec`, and scheduling makes the former catch up to the latter.
That representation naturally covers prompt chunks, AR roots, and speculative
candidates.

Useful mechanisms:

1. **Lookahead allocation before execution.** Target KV capacity for candidate
   rows is allocated before the target forward. Accepted rows are retained;
   rejected rows reduce computed-token ownership rather than requiring a second
   target forward.
2. **One packed target forward.** Per-request speculative token lists become
   query ranges in one model runner batch.
3. **Draft output feeds the next scheduler iteration.** The target step samples
   committed tokens; the draft worker then proposes candidates and publishes
   bounded `DraftTokenIds` for scheduling.
4. **Dynamic depth keyed by batch size.** A configured table selects K from the
   actual scheduled batch and may select K=0.
5. **Hybrid-state specialization.** Mamba/linear-attention state uses per-token
   state columns and accepted-token-biased GPU postprocessing rather than
   pretending KV rollback alone is sufficient.
6. **Graph shape is part of execution.** Full/piecewise graph selection depends
   on target tokens, request count, and speculative width.

Costs and cautions:

- lookahead rows consume KV/page and global token budget;
- dynamic K is primarily batch-wide rather than arbitrary per-request depth;
- method/backend combinations still carry restrictions;
- the target and draft phases remain a complex coupled runner; and
- a generally correct architecture does not guarantee a win at high QPS.

### SGLang: stage-specific workers and runtime bundles

SGLang's speculative worker owns a decode cycle as:

```text
activate batch-size tier
    -> draft
    -> prepare target verify rows / cache locations
    -> target verify + per-request acceptance
    -> draft extend / provider-state catch-up
```

Useful mechanisms:

1. **The scheduler remains the request owner.** The draft worker is subordinate
   to the target scheduler and shares request/token pool identities.
2. **Separate provider and target pools.** Draft KV/state can have its own model
   runner while target KV stays scheduler-owned.
3. **Distinct draft, verify, and draft-extend graph bundles.** Runtime tier
   switching selects prebuilt graph/backend state instead of recapturing online.
4. **Per-request acceptance with one physical verify batch.** A target batch may
   advance each request by a different accepted count.
5. **Adaptive batch-size tiers.** Acceptance EMA is tracked independently by
   batch-size range. High batch-size tiers narrow depth; an effective zero-step
   path keeps target execution AR-equivalent.
6. **Overlap is an optimization, not semantics.** `FutureMap` and V2 workers
   pipeline CPU/device stages, but the same worker can be driven synchronously.

Costs and cautions:

- one algorithm is generally selected for the server, rather than arbitrary
  per-request providers;
- DFlash and n-gram paths disable some overlap/mixed-prefill features;
- top-k/tree shapes multiply graph and pool complexity; and
- adaptive support is narrower than the full algorithm matrix.

### llama.cpp: one global per-sequence spec owner and one target batch

Current llama.cpp is the simplest strong reference for the exact question.
`server_context` collects all generating slots, asks one `common_speculative`
owner to draft for the enabled sequence IDs, appends every root and draft token
to one target `llama_batch`, and then accepts/rolls back each slot independently.
The provider implementations themselves batch active sequences:

- simple draft models run one draft-model batch per draft depth;
- MTP does the same with per-sequence hidden state;
- DFlash/DSpark put all noise blocks into one draft-model decode; and
- n-gram providers share the same per-sequence interface.

Useful mechanisms:

1. AR and speculative slots can coexist in one target batch.
2. Proposal state is per sequence but proposal execution is cross-sequence.
3. Target verification uses existing batch token/position/sequence metadata.
4. Acceptance and context restoration are per slot.
5. Provider chaining is possible through one common interface.

Costs and cautions:

- enum/switch registration is not hipEngine's plugin architecture;
- checkpoint/context restoration can require re-evaluating draft work;
- speculative rows must not be split across an unsupported target sub-batch;
- unified KV may compute masked cross-sequence attention; and
- output exactness is not hipEngine's execution-profile evidence contract.

### MTPLX: the closest same-family proof of fixed-width MTP batching

MTPLX is especially relevant because its shipped lane targets Qwen3.6-35B-A3B,
a hybrid full-attention/GDN MoE family close to hipEngine's PARO work. It does
not merely run one singleton MTP loop per request:

- one active row drafts K1 with a physical `B x T1` MTP shape;
- the target verifies `[primary, draft]` using `B x T2`, flattened to `M=2B`;
- installed B3 and B8 graph/lane variants serve real widths 2–3 and 4–8;
- target and MTP KV use row-specific ragged offsets;
- each GDN layer captures both T2 recurrent rows and commits row 0 or row 1
  independently using the per-request accept decision;
- greedy token IDs remain device-side on the optimized path;
- stochastic rejection/residual sampling keeps independent request RNG; and
- completed/cancelled rows become inactive without contaminating survivors.

The row commit is a concrete analogue of SPECDEC2's proposed selected-state
operation: full-attention offsets advance by each row's keep count, while Conv
and GDN states select the accepted T2 row and mask inactive rows back to their
base state.

Its construction discipline is also worth copying conceptually:

1. model, quant, MTP, width, context, cache, and kernel contracts bind once;
2. each physical width owns a construction-time numerical self-check;
3. the server reports the actual executed width, not only the configured mode;
4. the narrowest installed width seals a cohort, avoiding B8 padding for c2/c3;
5. throughput, balanced, and B1-exact profiles state their arithmetic contract;
   and
6. a route fails startup rather than falling back silently inside a cycle.

MTPLX's own M5 Max evidence reports roughly 1.6–2.25x per-lane improvement over
its prior AR-batch path and a 349.064 aggregate tok/s B8 greedy row. These are
external Apple/MLX/model-artifact results, not hipEngine baselines. Their value
is demonstrating that K1, physical M16 target verification, ragged KV, and
selected GDN state can be made economically useful together.

The limitations are equally important:

- a cohort is immutable after its 20 ms gather/seal window;
- arrivals during decode wait for the next cohort;
- no finished row is replaced, and no mid-flight width change occurs;
- prompt prefill is request-local before caches are merged;
- all active rows use MTP K1—there is no mixed AR/K=0 target frontier;
- the production service uses its dedicated `MTPBatchGenerationService`; the
  generic `MTPContinuousScheduler` is exercised only by foundation tests at this
  revision, so it does not add late admission to the shipped lane;
- the implementation is a large Qwen35B/MLX-specific lane rather than a generic
  provider/target SPI; and
- B3/B8 arithmetic can drift from B1, requiring explicit profile-level quality
  evidence.

MTPLX therefore strengthens—not replaces—the SPECDEC2 recommendation. A sealed
fixed-width MTP2 cohort is a valid intermediate GPU milestone, but C2 product
support still needs late admission, refill, K=0/mixed policy, resource-ledger
integration, and provider-neutral ownership.

### mlx-vlm: a useful batched verifier, but a fixed cohort

mlx-vlm has real batch MTP, EAGLE, and DFlash round loops. Target verification is
batched, acceptance is per row, hybrid caches are rolled back per row, and
finished rows are filtered. This validates the basic tensor/state mechanics.

However, its speculative server path collects pending requests only when idle,
prefills that cohort, and runs it to completion. Requests arriving during the
cycle wait for the next cohort. Filtering without refill is static batching, not
C2 continuous serving.

### oMLX: correctness-preserving row extraction is the wrong performance end-state

oMLX's opt-in row-wise MTP path is a valuable negative reference. It extracts
each active row's cache, runs a proven singleton cycle, and merges the rows back.
Ownership survives ragged positions and batch filtering. But each request still
runs a separate target backbone forward, so normal batched AR wins at c2/c4 even
with high acceptance. The implementation therefore stays default-off.

Its more useful lifecycle contribution is exact **handoff and recovery**:

- an active singleton MTP request yields to standard AR before a late join;
- a multi-row batch runs the ordinary shared path; and
- when the batch shrinks to one compact row, that survivor may reactivate MTP.

SPECDEC2 needs equivalent transition semantics, but its steady c>N path must
physically batch provider and target work.

### vMLX: explicit non-support is better than a misleading flag

vMLX's classic draft-model route explicitly rejects `is_batched=True`; its
native MTP patch says batch size >1 is off by design. Prompt lookup may remove a
single request from the active batch, verify it, and reinsert it, but the code
records high-concurrency impact as unmeasured. DFlash2 runs through a direct
simple-engine generator.

This is not the desired architecture, but its fail-closed behavior is better
than reporting route coalescing as physical speculative batching.

## Current hipEngine audit

### What is already architecturally strong

The following should be retained and generalized:

- C2 stable request ID, resident slot, execution row separation;
- one `EngineService` request/output lifecycle;
- bounded stream mailboxes and independent terminal reclaim;
- token-budget, fair, and fit-aware scheduling foundations;
- `KVLiveSpans` as the attention ABI;
- atomic resource claims and explicit transaction lifetimes;
- `DraftBatch` candidate-only semantics;
- `TargetVerifyBatch` root materialization and parent mapping;
- `TargetAcceptSummary` and compact GPU accept payloads;
- strict reject/partial/full simulator coverage;
- GGUF N1R/N2 target graphs and selected-state ownership;
- PARO parent-indexed chain/tree verifier work;
- exact unfused/serial fallbacks; and
- the full category/heldout anti-gaming benchmark policy.

### The execution gap

#### 1. `ResidentEngineLoop` never schedules speculative work

`hipengine/generation/engine_loop.py::ResidentEngineLoop._tick_once()` selects
only prefill and decode work. `WorkKind.VERIFY_CHAIN` and `VERIFY_TREE` exist,
and `ResidentBatchScheduler.next_speculative_verify_work()` can construct
metadata, but no real engine tick invokes a proposal, verifier, or transaction
commit.

#### 2. speculative submission performs the work before admission returns

`SubmitPollTextGenerator.submit_speculative_many_detailed()` enters the
submission-priority/model-loop lock and calls:

```python
self._inner.generate_speculative_mtp_detailed(combined)
```

Only after all outputs exist does it allocate synthetic backend request IDs and
build a shared `WorkItem`. `EngineService.submit_speculative_children()` therefore
uses the normal child/result table but not continuous model scheduling. New AR or
MTP work cannot interleave while that call owns the model.

This directly explains the retained SPEC-C5 result: the direct B3 route is
**53.521 tok/s / 1.8012x true AR**, yet the public C=10 path is **10.150 s MTP
versus 5.874 s AR (0.579x)**. The public path serializes already-complete provider
requests behind one service command.

#### 3. the dense GGUF route is request-serial

`Qwen35GGUFGenerator._generate_dense_speculative_mtp_detailed()` opens one
target/provider and loops over prompts. Each `Qwen35GGUFMTPDecodeSession`
generates one request to completion before the next request starts.

The llama-compatible GGUF path has more advanced private multi-slot machinery:
per-slot draft streams, packed target verify chunks, and selected commit. But it
still owns a whole batch inside a model call and publishes outputs only after
that owner finishes. It is also an accuracy-traded policy variant, not the
strict universal control.

#### 4. PARO and Laguna are single-request owners

- `Qwen35ParoResidentSession.verify_chain_bulk_and_commit()` rejects
  `len(batch.request_ids) != 1`.
- `LagunaDFlashResidentCycle` and `LagunaDFlashTextProvider` own one target,
  drafter, request, and full generation loop under the target lock.

Their transaction math is useful. Their host ownership is not c=N.

#### 5. the provider registry owns complete text generation

`SpeculativeTextProvider` exposes `generate_detailed()` and `stream_detailed()`.
That makes a provider a second generation engine. The target design in
`docs/CONCURRENCY2.md` requires the opposite: providers may propose and update
their own state, but cannot own target scheduling, visible tokens, or request
lifecycle.

#### 6. generic contracts are disconnected from real device ownership

`SpeculativeCycleSimulator` correctly models claims, rollback, cancellation, and
cursor conservation. Real GGUF/PARO/Laguna paths use separate provider-specific
state owners. There is no device-native `CandidateGraph` / target-frontier object
connecting the simulator's semantics to those owners.

## Why c=N plus SpecDec is not “turn on MTP inside C2”

Let:

- `C` = due target requests in one fairness pass;
- `k_i` = candidate count for request i (`0` means AR);
- `R = Σ_i (1 + k_i)` = total target frontier rows; and
- `μ_i = 1 + E[accepted_drafts_i]` for speculative requests, otherwise 1.

Then:

```text
G_spec = Σ_i μ_i /
         (T_propose + T_target_frontier + T_accept_commit + T_scheduler)

G_AR   = C / T_AR(C)

speedup = G_spec / G_AR
```

The numerator alone is not sufficient. A high accepted-token count can still
lose when the target verifies too many rows or the provider is request-serial.
The longest cycle wall also bounds worst-request ITL: a rejecting request emits
one token after paying the complete frontier wall.

For the current gfx1151 dense lane, c8 with K=3 implies `R=32`. The finalized AR
profile says c32 is already near four linear c8 owners; the structural c32 lane
was skipped because composition is already nearly linear. If SPECDEC2 verifies
that frontier as four independent c8 target sweeps, it probably gives back the
weight-read amortization that makes single-request MTP win. A verifier-specific
row path is therefore a prerequisite, not a follow-up.

## Proposed architecture

### Names and boundaries

- **SPECDEC2**: shared scheduling, resource, transaction, target-frontier,
  acceptance, commit, output, and policy architecture.
- **MTP2**: model-attached NextN/MTP provider family implemented on SPECDEC2.
- **DFlash providers**: independent draft models on the same SPECDEC2 path.
- **NativeSpecCycle**: retained native component/ABI lineage; adapted as device
  execution inside SPECDEC2 rather than a parallel scheduler.

“From scratch” applies to the hot-path orchestration. Existing exact kernels,
loaders, graphs, fixtures, and oracles remain source components and strict
fallbacks.

### One target frontier

A target frontier is a bounded rooted DAG for one or more requests:

```text
TargetFrontier
  operation_id / cycle_id
  target execution/profile/transaction keys
  request_ids[C]
  root_rows[C]
  node_offsets[C+1]
  token_ids[R]                  device resident
  row_to_request[R]
  resident_slots[R]
  parent_rows[R]                -1 or canonical-root sentinel for roots
  positions[R]
  draft_depths[R]
  active_mask[R]
  candidate/probability metadata
  KVBatchView / KVLiveSpans
  provisional target state/KV owner
```

Examples:

```text
AR request:           root
MTP/DFlash chain:     root -> d1 -> d2 -> d3
Tree provider:        root -> {a,b}; a -> {c,d}; ...
```

The scheduler may pack root-only AR and speculative requests into one target
frontier when a target capability declares that mixed shape exact and efficient.
Otherwise the backend lowers them into separate target groups in the same
fairness round. The abstraction is unified; physical fusion is evidence-driven.

### Device-native provider result

Keep host `DraftBatch` for tests and slow fallbacks, but add a device-native
candidate contract:

```text
CandidateGraph
  provider/method/policy keys and artifact hash
  request_ids[C]
  candidate row count V
  device token/probability buffers
  row offsets / parent/depth / active metadata
  provider transaction owner and checkpoints
  target hidden-tap dependencies
  complete resource claim identity
```

No normal provider copies full logits or candidate arrays to Python. A host or
n-gram provider may upload a bounded result into the same slab.

### Provider SPI

Replace whole-request provider ownership with staged methods equivalent to:

```text
capabilities(target, request semantics) -> SpeculativeCapability
resource_claims(request states, proposed shape) -> ResourceClaimSet
prepare_requests(...)                  # prefill/reseed, bounded
propose_batch(ProposalPlan, stream) -> CandidateGraph
commit_batch(SpecCycleResult, stream)
rollback_batch(transaction, stream)
close_requests(request_ids)
```

The provider may own weights, state, KV, graphs, and streams. It may not:

- publish visible tokens;
- mutate canonical target state outside a target transaction;
- await a frontend;
- choose unbounded repeated cycles; or
- run an independent request scheduler.

### Capability composition

A resolved provider/target capability must declare:

- provider attachment: target-attached, independent, host, or remote;
- chain/tree and variable-depth support;
- maximum C, V, depth, and physical proposal/target buckets;
- provider and target transaction modes;
- required target hidden taps or probabilities;
- compatible target model/backend/quant/KV/profile keys;
- graph/static slab/workspace/result requirements;
- sampling and grammar support;
- strict fallback and pre-launch rejection rules; and
- measured cost-table identities.

Engine/model code must not branch on backend, quant, or provider.

### Transaction modes

Do not force one rollback mechanism on every KV backend.

1. **Reserved append / live-count commit**
   - reserve future canonical token slots;
   - write provisional K/V there;
   - commit by advancing the request's live count to the selected prefix;
   - reject by leaving tail slots dead/reclaimable.

2. **Packed scratch / selected copy**
   - verifier writes target K/V/state into a packed operation slab;
   - selected rows/prefixes copy to canonical ownership at commit;
   - useful when page rounding or a compact codec makes reserved append costly.

3. **Reversible journal**
   - canonical writes are permitted only with a complete exact restoration
     journal and injected-failure gates.

Linear-attention state has a parallel selectable-row contract:

- each candidate node writes Conv/GDN/SSM state derived from its parent;
- a GPU commit selects the accepted node per request; and
- rejected peer rows never affect another request's state.

Provider state may use reseed-from-target-hidden, append-and-trim, selected copy,
or a journal. Both target and provider modes are part of one atomic transaction.

### Engine round

A first implementation can remain serial on one stream:

1. drain submit/cancel/deadline commands;
2. finish pending commit/rollback and terminal reclaim;
3. admit fitting requests;
4. classify each due request as AR or one provider/depth policy;
5. reserve the complete target/provider/transient claim set;
6. execute provider-batched proposal groups;
7. pack compatible candidate graphs plus eligible AR roots into target frontiers;
8. execute target verify;
9. run device acceptance, stop/output-limit handling, and selected state/KV
   commit independently per request;
10. update provider state/cursors;
11. read one bounded committed result slab, publish request-keyed token events,
    and release the transaction; and
12. yield to the next C2 fairness round.

Every due request receives at most one AR transition or one speculative cycle per
fairness pass before any peer receives a second cycle.

### Mixed AR and speculative requests

`K=0` is a normal policy decision, not an exception. A mixed round may contain:

- requests with no provider;
- provider requests whose current cell selects K=0;
- requests whose provider is temporarily behind or rebuilding;
- different provider proposal groups feeding the same compatible target
  frontier; and
- requests using a different target profile/transaction mode, which lower to a
  separate target group.

An AR neighbor must not pay an unbounded speculative tail. The planner prices the
complete frontier cycle against the earliest ITL deadline. If mixing harms the
AR neighbor, the physical planner splits the groups or chooses K=0.

### Streaming and completion

Acceptance returns a bounded committed payload:

```text
SpecCycleResult
  request_ids[C]
  accepted draft counts
  visible token IDs and lengths
  selected target/provider rows
  correction/bonus IDs
  target/provider cursor deltas
  RNG deltas
  stop/EOS/length outcomes
  transaction and route status
```

The C2 output router emits only committed tokens in canonical order. A cycle may
produce multiple chunks or one chunk with several IDs; one slow consumer remains
isolated. Stop strings keep the existing token/state semantics and text holdback.
Cancellation becomes pending during device work and resolves at the next safe
commit/rollback boundary.

### Graph and submission ownership

Use separate reusable bundles first:

- proposal;
- target frontier;
- accept/selected commit; and
- provider repair/update.

A graph key includes target/profile, method/policy, C, physical R bucket,
depth/tree class, context/page bucket, transaction modes, hidden taps, sampler,
and variant-manifest hash. Dynamic request pointers, positions, `KVLiveSpans`,
row maps, and active masks are replay inputs bound through stable slabs.

Do not start with a multi-cycle device loop. Do not combine proposal and target
submissions until separate bounded cycles are correct and complete-wall evidence
shows a reason. A larger native-ownership label is not a performance result.

## Physical verifier design for Qwen3.8 on gfx1151

### Required rows

For uniform chain depth K:

| C | K | target rows R = C × (K+1) |
| ---: | ---: | ---: |
| 1 | 3 | 4 |
| 2 | 3 | 8 |
| 4 | 3 | 16 |
| 8 | 3 | 32 |
| 17 | 1 | 34 |
| 32 | 1 | 64 |

This is why AR width maps cannot price verifier work. The first useful physical
matrix is not “c1/c8”; it is `(C, R, depth, context, transaction)`.

### Dense projection

Dense target projection owns 88–93% of the final profiled c8/c17/c32 GPU duration.
A target frontier needs verifier-specific rows that reuse one encoded weight
stream across as many compatible nodes as the hardware can profitably process.
The strict fallback may decompose to existing c1–c8 rows, but a decomposed route
must report every weight sweep and cannot support a speed claim merely because it
is correct.

Initial target buckets should be measured at `R in {1,2,4,8,16,32}`. An R32
implementation may be tiled internally; the gate is whether its complete target
wall beats four independent R8 sweeps, not whether it carries a “row32” name.

### Conv/GDN recurrence

Verifier nodes are causally related within a request. The linear-attention owner
must consume `parent_rows`:

- root starts from canonical resident state;
- each child starts from its parent node's provisional state;
- different requests execute independently; and
- selected commit copies one node per request.

A correct first implementation may keep the depth loop serial while batching C
requests at each depth. A faster implementation may stage independent dense
projections across all R rows and run a parent-indexed recurrent t-loop. Both
must expose exact row ownership.

### Full attention

Each node attends to:

- its request's canonical committed prefix; and
- only its candidate ancestors.

`KVLiveSpans` remains mandatory. A frontier-specific attention view supplies the
canonical pages plus provisional rows and an ancestor/causal description. No
cross-request visibility is allowed even if rows share one physical kernel.

### MoE / PARO target

PARO MTP is not just a quant-port of the dense path. Efficient C×K verification
requires:

- device-resident router/expert IDs;
- sorting/grouping `(expert, node)` work across the frontier;
- one selected-expert weight stream serving all routed nodes where possible;
- provider and target MoE ledgers separately; and
- exact parent-indexed linear state.

The current BF16 MTP sidecar proposal and W4 PARO target therefore come after the
dense GGUF target-frontier path proves the orchestration.

### Queue and overlap policy

Retain `GPU_MAX_HW_QUEUES=2`. Current gfx1151 traces observed one queue ID and no
marked overlap. Run proposal → target → commit serially at first. Explore overlap
only after profiler evidence shows separate proposal/target phases leave
recoverable device slack. SGLang's overlap architecture is a useful later model,
not a starting assumption for this hardware.

## Provider migration order

### MTP2-1: exact dense GGUF NextN

This is the first provider because:

- target-attached proposal state is smaller and easier to reserve;
- the current single-request direct route already demonstrates strong true-AR
  economics;
- N1R/N2 provide reusable target and selected-state components;
- candidate depths B1/B2/B3 are bounded; and
- no independent 3–4 GiB drafter must be admitted.

MTP2 exact/default and `llama-compat` remain separate execution profiles. The
accuracy-traded direct-partial-commit route cannot silently become the strict
provider.

### MTP2-2: PARO MTP

Adapt the persistent BF16 MTP provider to the staged SPI. Remove host selected-
expert orchestration and add C-batched proposal. Reuse the target frontier only
after independent PARO verifier buckets and exact profile gates pass.

MTPLX provides a particularly relevant external execution template for this
stage: same-family Qwen3.6-35B-A3B K1 uses physical B3/B8 draft rows, M6/M16
target rows, ragged per-request KV, and selected T2 GDN state. Do not port MLX
kernels or its numerical constants into the PARO lane; use the mechanism to
predeclare hipEngine's own `(C,K=1,R=2C)` exact/profile gates and narrowest-fit
physical buckets.

### SPECDEC2-DFlash: Laguna and Qwen DFlash2

Laguna supplies the first independent-provider lifecycle adapter. Its public
whole-request provider is decomposed into prepare/propose/commit stages.

Qwen3.8 DFlash2 follows after the dense target frontier is efficient. The closed
campaign found acceptance near MTP parity but a much higher cost: the drafter,
selector, hidden-tap capture, and >4-row verifier cliff, not acceptance, caused
the loss. Do not compare it again until:

- the same target file/protocol is used for AR, MTP2, and DFlash2;
- provider proposal is batched and profiler-attributed;
- target rows through at least R8 are exact and amortized; and
- the five target hidden taps are part of the declared frontier cost.

### Trees, EAGLE, n-gram, and remote providers

The parent-indexed frontier supports trees from the beginning, but chain
performance should close first. A cheap host n-gram provider can validate staged
orchestration. EAGLE-style hidden providers then use the same target frontier.
Remote/shared provider scheduling remains unsupported until it has a bounded
credit, cancellation, and failure protocol.

## Adaptive policy and cost model

### Cost-table key

A retained cost cell includes at least:

```text
backend / target model / quant / execution profile
provider / method / policy fingerprint
context and KV/page bucket
C_ar / C_spec
depth histogram or tree-shape class
physical R decomposition
provider and target transaction modes
graph/eager route
```

Record:

- proposal wall and kernel union;
- target-frontier wall and kernel families;
- acceptance/commit/provider-update wall;
- scheduler and result-readback wall;
- encoded bytes/weight sweeps where measurable;
- transaction and persistent memory;
- accepted output distribution; and
- complete service TTFT/ITL/goodput.

### Initial policy

Start deterministic and conservative:

1. offline profile powers-of-two C and bounded K;
2. build a hardware/model/profile-specific LUT;
3. use online acceptance EMA only inside measured safe cells;
4. choose before mutation; and
5. report an exact reason for K or K=0.

Uniform K per target group is the first graph-friendly implementation. Variable
per-request depth may follow using active masks or global verification-budget
selection, but it must show a complete-wall win. Never use prompt text, token
IDs, or benchmark categories as a routing feature.

### Attached versus independent provider switching

K=0 is cheap for target-attached MTP when the provider can reseed from the next
target hidden row. An independent provider may fall behind while AR runs. Its
capability must declare one of:

- cheap target-output catch-up every AR step;
- bounded reseed/re-prefill cost;
- request-lifetime one-way downgrade to AR; or
- unsupported dynamic disable.

The policy prices that transition. It may not “turn DFlash back on” using stale
provider KV.

### Likely gfx1151 policy shape (hypothesis only)

A reasonable measurement order is:

- c1: K in {1,2,3};
- c2/c4: K in {1,2,3};
- c8: K in {0,1,2,3};
- c17/c32: K in {0,1};
- repeat by context bucket.

This is not a default recommendation. The expected result may be MTP at low C
and K=0 at c8+, especially for short contexts. Long-context KV pressure can
shift the crossover, so context is a policy axis.

## Migration plan

### S0 — freeze the AR and current speculative controls

Start only after the completed gfx1151 AR campaign commit is selected as the new
campaign base. Record same-host, same-model, same-quant, same-protocol:

- C2 AR c1/c2/c4/c8/c17/c32;
- direct exact MTP B1/B2/B3;
- current public speculative path; and
- proposal/target/commit attribution.

No new architectural claim comes from historical mixed protocols.

### S1 — device-native contracts and RED simulator

Add the target-frontier, candidate-graph, staged-provider, policy, and result
records. Extend fake transactions across:

- mixed AR (`K=0`) and speculative requests;
- variable C/K/tree shapes;
- provider+target atomic claims;
- independent reject/partial/full accepts;
- one peer canceling at every stage;
- EOS/stop/length tails;
- slot permutation and refill; and
- final conservation.

No GPU or performance claim.

### S2 — real C2 one-cycle scheduling

Teach `ResidentEngineLoop` to execute proposal/target/commit work. A fake or CPU
provider must prove:

- admission returns before generation finishes;
- late AR/spec arrivals join future rounds;
- one speculative request can stream several committed tokens;
- cancellation waits for a safe transaction boundary;
- mixed AR/spec fairness holds; and
- there is no provider-owned request loop.

Deprecate—not yet delete—the current synchronous submission path.

### S3 — exact dense GGUF MTP2 c1 adapter

Expose the existing exact NextN/N1R/N2 work as one-cycle staged methods. Match:

- direct generated IDs and cycle semantics;
- hidden/Conv/GDN/full-KV/cursors after every cycle;
- following-AR continuity;
- graph/eager fallback; and
- direct complete wall within a predeclared tolerance (suggested ≤5% overhead)
  before c>N work.

Once this gate passes, the old whole-request method becomes a strict
compatibility oracle, not a server route.

### S4 — physical c2/c4 MTP2

Batch proposal and target frontiers for C={2,4}, K={1,2,3}. Require actual
provider and target call counts proving no request-serial loop. Cover ragged
acceptance, staggered retirement, refill, and neighbor isolation.

### S5 — gfx1151 verifier buckets and dynamic K

Implement/qualify target R buckets through at least R32, parent-indexed GDN,
frontier full attention, device accept, and selected commit. Populate the
C/context/K cost table and select K=0 wherever required.

### S6 — product serving closure

Run fixed, ragged, Poisson, overload, recovery, and soak with mixed AR/spec,
streaming, cancellation, prefix reuse, pressure, and graph churn. Only this phase
can promote an `auto` scope.

### S7 — PARO MTP2

Add independent provider/target capabilities and MoE/frontier kernels. No GGUF
performance constants transfer.

### S8 — Laguna DFlash and Qwen DFlash2

Decompose the whole-request provider, batch proposal, and qualify target hidden
taps/provider KV. Retain default-off if economics lose.

### S9 — trees and optional overlap

Enable tree frontiers and then test proposal/target overlap or a bounded native
multi-cycle owner. Neither is required for basic continuous SpecDec support.

## Likely code changes

| Area | Current seam | Proposed direction |
| --- | --- | --- |
| `hipengine/speculative/interfaces.py` | Host tuple records | Retain CPU records; add device `CandidateGraph`, `TargetFrontier`, and bounded cycle result. |
| `hipengine/speculative/registry.py` | Provider owns `generate_detailed` / `stream_detailed` | Register staged draft-provider factories and composed capabilities; deprecate whole-request provider protocol. |
| `hipengine/speculative/simulator.py` | Correct but fake-only transaction | Rebase on the same records/resource modes used by real execution. |
| `hipengine/dispatch/batch.py` | `VERIFY_*` metadata but no executed stages | Add proposal/target-frontier work metadata; eventually lower AR decode as root-only target work. |
| `hipengine/generation/batch_scheduler.py` | Can materialize verify metadata | Add policy decision, complete-claim planning, frontier packing, and multi-token committed event accounting. |
| `hipengine/generation/engine_loop.py` | Runs only prefill/decode | Execute bounded proposal → target → commit stages; remove synchronous full-generation ownership. |
| `hipengine/generation/engine_service.py` | Separate speculative submit command | One child submission path with a cold speculative policy/plan; no generation inside command handling. |
| `hipengine/generation/qwen35_gguf.py` | Dense serial loop and private llama-compat slot loop | Adapt cycle components to MTP2 provider/target SPI; keep old loops only as controls during migration. |
| `hipengine/runtime/qwen35_gguf_{nextn,mtp}.py` | Strong single-request native components | Add C-batched proposal buffers and target-frontier adapters. |
| `hipengine/runtime/qwen35_paro_runner.py` | Single-request target verifier | Add multi-request parent/state/KV metadata and physical verifier buckets. |
| `hipengine/generation/laguna_dflash.py` | Whole-request provider under target lock | Split prepare/propose/provider-commit; EngineService owns cycles and output. |
| KV backends | Provider-specific journals | Register target/provider transaction modes and packed provisional slabs without hidden dense shadows. |

Do not remove the old routes until the replacement proves direct and public
correctness plus wall economics. Track each compatibility flag/path in
`docs/REFACTOR.md` when implementation starts.

## Correctness and production gates

### Host and mechanical

- C={1,2,4,8}, K={0,1,2,3}, variable K, chain and bounded tree metadata;
- unique request/slot ownership under permutation and compaction;
- atomic provider+target+transient claims;
- reject/partial/full/correction/bonus;
- cancel at reserved, provider-open, drafted, target-open, verified, accepted,
  and commit boundaries;
- no request-lifetime model lock;
- one request's rollback never affects a peer; and
- final resources/transactions/collectors are zero.

### Device and model

For each backend/model/quant/profile/provider cell:

- target top1/accept matches CPU oracle;
- strict exact generated IDs where the profile requires them;
- target hidden, Conv/GDN/SSM, full-attention K/V, positions, live counts, and
  cursors are exact after every cycle and the following AR step;
- provider hidden/KV/state/cursors are exact after reject/partial/full;
- batch-composition isolation and deterministic repeats pass;
- eager/graph and graph-miss fallbacks agree;
- injected failures restore both owners;
- no hidden allocation or dense KV shadow exists; and
- profiler traces show the expected provider, target, accept, and commit owners.

Production-arithmetic variants use the binding calibrated profile gate in
`docs/EXECUTION-PROFILES.md`, not merely the broad KL/top-1 floor.

### API and lifecycle

- blocking/SSE committed IDs and finish details agree;
- several accepted tokens stream once, in order;
- stop/EOS/length can cut through a cycle;
- sampling RNG is request-owned and batch-permutation invariant under the
  declared profile;
- disconnect/backpressure affects only that request;
- staggered arrival, refill, prefix COW/eviction, pressure, and teardown pass;
- circuit breaker and pre-launch fallback reasons are observable; and
- mixed AR/spec requests retain their exact route/accounting.

### Quality and anti-gaming

Use the full committed multi-prompt category and heldout suite, plus applicable
long/task fixtures. Compare against a true no-MTP AR baseline from the same
protocol. No prompt-conditioned depth, token-ID branch, or suite-specific route
is admissible.

## Performance and promotion gates

Every cell reports:

- true AR and SpecDec complete request wall;
- aggregate and per-request tok/s;
- SLO goodput;
- TTFT, ITL median/p95/p99, E2E and queue delay;
- accepted tokens/cycle and per-depth acceptance;
- proposal, target, accept/commit, provider update, scheduler, and readback wall;
- physical C/R decomposition and weight-sweep count;
- graph capture/replay/fallback counts;
- persistent/transaction memory high-water; and
- cancellation/drain/health outcomes.

Promotion rules:

1. Explicit functionality may be retained when correctness passes but speed
   loses; it stays default-off and carries no speed claim.
2. `auto` requires **>1.10x true same-protocol AR** plus non-regressive quality,
   TTFT/ITL/SLO, memory, pressure, and load gates in every admitted cell.
3. The project target remains >1.3x; do not lower it after results.
4. A provider may promote low-C cells while high-C cells select K=0.
5. A speculative cohort may not improve aggregate throughput by regressing AR
   neighbors beyond their SLO.
6. A single-request direct win never authorizes c>N.
7. A decomposed target route cannot hide multiple weight sweeps behind the
   logical `R` label.
8. gfx1100/gfx1151 and different physical hosts remain separate evidence lanes.

## What not to do

- Do not extend `submit_speculative_many_detailed()` with more synchronous batch
  code.
- Do not treat a post-hoc `WorkItem(C,V)` as physical C×V execution.
- Do not run one singleton MTP loop per C2 request and call that batching.
- Do not price verifier rows from AR D2/width cost maps.
- Do not make DFlash, MTP, and PARO separate request schedulers.
- Do not make overlap/multiple hardware queues a prerequisite on gfx1151.
- Do not promote fixed K globally.
- Do not reopen rejected Q4/queue/c32 AR candidates without a new measured
  premise.
- Do not require every provider before shipping the first exact MTP2 scope.

## Recommended first campaign

Architecture approval is now recorded in [`SPECDEC2.md`](SPECDEC2.md). Its
normative S0-S7 punchlist supersedes this research-stage recommendation. The
original bounded first unit was:

After explicit architecture approval, open a bounded **SPECDEC2-S0/S1** campaign:

1. freeze the finalized gfx1151 AR base (`fd9dd35df` or a later explicitly
   selected clean commit);
2. add no device kernel;
3. land the device-native records, staged-provider SPI, target-frontier simulator,
   mixed K=0 semantics, and resource/rollback RED tests;
4. add a fake-runner EngineLoop test proving one speculative cycle—not a complete
   request—is C2 work; and
5. stop for review before adapting the real dense GGUF route.

That first unit establishes the architecture without racing the just-finished
AR tuning campaign or spending GPU time before the ownership model is reviewable.
