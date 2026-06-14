# Tensor Parallel Serving Plan

Last updated: 2026-06-15

This document is the concrete design gate for `docs/AGENTIC.md` P6.3. It turns
the high-level multi-GPU strategy in `docs/PLAN.md` and the forward-compatibility
guardrails in `docs/CONCURRENCY.md` into the server/runtime contract that must
hold before tensor parallelism (TP) can land.

Current status: not implemented. The OpenAI-compatible server reports
`parallelism.tensor_parallel.enabled=false`, `topology.mode="single_process"`,
`world_size=1`, rank/local-rank `0`, no collective backend, and the unsupported
feature ids listed below. This is intentional until a multi-GPU host validates
the runtime path.

## Default Rule

No TP code may affect the default single-GPU path without hardware validation on
a multi-GPU ROCm host. A TP experiment must be opt-in, have an unfused/single-GPU
comparison, and keep dispatch behind the existing `(backend, layer, quant,
variant)` registry model. Do not add backend/quant branches to engine, model, or
server routing code.

## Ownership Boundaries

Rank 0 owns the control plane:

- HTTP request admission and routing;
- generation scheduler policy;
- prompt/session/continuation metadata;
- RNG seed derivation and sampler state;
- final response assembly and error taxonomy;
- readiness/capability reporting for the TP group.

Worker ranks execute per-rank model shards in lockstep. They do not make
admission, fallback, sampler, session-commit, or response-shaping decisions.

## Weight Shards

The first TP implementation should match the plan in `docs/PLAN.md`:

- column-parallel QKV projections;
- column-parallel gate/up projections;
- row-parallel `o_proj`;
- row-parallel `down_proj`;
- replicated router and normalization weights;
- MoE expert sharding only after the dense TP-2 path passes correctness and
  latency gates.

The sharded loader must record per-rank tensor metadata: original shape,
sharded shape, axis, dtype/quant scheme, source file/span, and checksum or
length guard. Quantized sidecar caches must either be rank-specific with a
topology key or generated from the same canonical source tensor on each rank.

## KV Cache

Phase 1 replicates KV cache on every rank. This keeps the existing
`KVLiveSpans` ABI unchanged and avoids sharded attention semantics while the TP
math path is being validated.

Required rules:

- each rank owns a local dynamic KV pool;
- scheduler admission uses the minimum per-rank capacity;
- KV mutation remains transactional at rank-0 commit points;
- `KVLiveSpans` remains the attention/KV-write ABI for every rank;
- no `(block_table, context_len)` shortcut is introduced for TP.

Distributed KV is future work and remains unsupported as `kv_cache_sharding`.

## Collectives

The collective backend should be RCCL through a torch-free binding first, with
MPI or another backend only as an explicitly documented fallback. Collectives
must remain visible to the engine loop rather than hidden inside math kernels.

Initial collective points:

- all-reduce after `o_proj`;
- all-reduce after `down_proj`;
- all-reduce or all-gather after shared expert output when MoE TP lands.

Future vocab-parallel or expert-parallel designs may need reduce-scatter,
all-gather, or all-to-all, but these are not part of the first default path.
Collective failures fail the active generation group; partial-rank output is not
served.

## Sampler Output

Rank 0 owns sampling. A TP model must present the same logical logits contract
to the existing sampler stack that the single-GPU model presents today.

Initial acceptable designs:

- replicate the LM head and sample on rank 0 after hidden-state reductions; or
- gather complete logits to rank 0 before applying processors and sampling.

Future vocab-sharded sampling is allowed only if it produces deterministic
rank-0-equivalent top-k/logprob/logit-bias behavior. Until then, `logit_bias`,
penalties, token suppression, forced-token queues, stop-token logic, logprobs,
and seeded sampling stay rank-0 sampler responsibilities after the full-logit
contract has been restored.

Speculative MTP remains disabled for TP until target verification can consume
the same processed sampler contract. Current server capabilities already mark
`logit_bias` and the other processor fields as incompatible with MTP serving.

## Graph Capture

TP graph capture is future work. The first implementation should run uncaptured
or with per-rank capture only after the communicator, streams, event ordering,
and bucket keys are stable.

Rules for any future capture:

- bucket keys are rank-agnostic;
- all ranks use the same logical bucket decision;
- collective launch order is fixed within a bucket;
- capture misses and replay failures are reported at the TP group level;
- a failed rank disables graph replay for the group until explicitly reset.

`multi_gpu_graph_capture` remains unsupported in the current manifest.

## Sessions And Snapshots

Current sessions are app-local transcript state, not resident KV snapshots. TP
does not change that model.

Before cross-rank resident snapshots exist:

- session commits happen only after rank 0 has a successful result from all
  ranks;
- hidden reasoning stays stripped from visible-only commits;
- exported snapshots include transcript and tokenizer metadata only;
- topology changes invalidate any future resident TP snapshot.

Cross-rank resident-state snapshots remain unsupported as
`cross_rank_session_snapshots`.

## Routing And Multi-Model Serving

A TP group is one logical served model. Routing chooses the model/group before
any rank-local work starts. No implicit fallback is allowed from a failed TP
group to a single-GPU model unless the request explicitly opts into a future
fallback policy.

Future route metadata should expose requested model, served model, TP enabled
state, world size, rank-0 device, collective backend, fallback status, and the
same failure reason taxonomy used by the single-model server today.

Multiple resident models and TP can compose only after per-model VRAM admission
accounts for every rank in the group.

## Failure Handling

Startup failures:

- failed communicator creation keeps the process live but unready;
- `/ready` reports the failed stage and redacted guidance;
- `/v1/hipengine/capabilities` keeps TP disabled when the TP group is absent.

Runtime failures:

- collective, rank, stream, or device failure aborts active requests with a
  stable server error;
- no partial-rank generations are returned;
- session and continuation commits are skipped for failed generations;
- the TP group is marked unhealthy until reinitialized.

## Smallest Measurable Smoke

The first retained TP smoke requires a real multi-GPU ROCm host:

1. Start a TP-2 toy dense model with deterministic weights and a replicated KV
   path.
2. Compare hidden states, logits, token ids, and finish details against the
   single-GPU path for fixed prompts and seeds.
3. Exercise `/v1/models`, `/v1/hipengine/capabilities`, `/ready`,
   `/v1/completions`, and `/v1/chat/completions`.
4. Capture profiler evidence that the expected collective calls run at the
   documented boundaries.
5. Record model, quant, workload shape, GPUs, command, correctness result, and
   timing in `WORKLOG.md` and a retained artifact if it becomes a benchmark
   claim.

No benchmark rollup row is accepted until correctness is green and the
single-GPU default path is non-regressed.

## Unsupported Feature Ids

The server manifest uses stable ids so clients can reason about current TP
limits:

| Feature id | Meaning |
| --- | --- |
| `world_size_gt_1` | Serving is single-process, single-rank only. |
| `weight_sharding` | No sharded weight loader or per-rank tensor metadata exists. |
| `kv_cache_sharding` | KV is not distributed; future TP starts with replicated KV. |
| `collective_reduce_scatter_all_gather` | No collective runtime path is wired. |
| `multi_gpu_graph_capture` | Graph capture is single-GPU only. |
| `cross_rank_session_snapshots` | Session snapshots do not include resident cross-rank state. |

