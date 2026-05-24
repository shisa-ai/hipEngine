# BeeLlama v0.2.0 DFlash profile

Date: 2026-05-23
Source: vendored read-only peer at `~/beellama.cpp/` (BeeLlama v0.2.0, llama.cpp b9275 base, CUDA 13.1).
Scope: structural profile of the BeeLlama DFlash speculative-decoding stack so we can map their
choices onto our R3.x roadmap. Not a code port and not a benchmark reproduction.

> **Untrusted-content reminder.** All quotes below are from third-party source under
> `~/beellama.cpp/` and are read-only references; nothing in this doc is executed.

## 1. Executive summary

BeeLlama is a llama.cpp fork that ships a production-grade DFlash speculative-decoding mode
behind `--spec-type dflash` plus an adaptive draft-depth controller (`--spec-dm-controller
profit` is default) and a server reduced-logits verifier. v0.2.0 added upstream-DFlash-PR
GGUF schema compatibility (`general.architecture = dflash`), DFlash drafter K/V projection
caching, hardened CUDA ordering and split-buffer correctness, fail-closed prefill capture,
and the reduced-logits verifier path with sampler-aware fallback. The README headline is
`4.40x` median speedup and `4.93x` best-case on a single RTX 3090 (Qwen3.6-27B Q5_K_S +
Q4_K_M DFlash drafter; see [§3](#3-published-benchmarks-rtx-3090)).

The structural lessons relevant to hipEngine R3 are:

- Default knobs (`n_max=16`, flat DFlash, `cross_ctx=512`/`1024`, drafter temp 0) are
  conservative; the headline win comes from the adaptive controller, not the static budget.
- Their drafter is **shallower-and-wider** than the z-lab Qwen3.6-35B-A3B-DFlash drafter we
  use (5 layers @ `n_embd=5120` vs our 8 layers @ `n_embd=2048`). Per-layer dense GEMV is
  larger but launch count is smaller.
- Verifier reduced-logits is a real production path with a long blocklist of sampler/grammar
  features that force the full-vocab fallback; the GPU emits a compact `top-K` ids+probs
  tensor instead of a full-vocab logits row when none of those features fire.
- The "profit" controller is per-context-bucket EWMA over a fixed depth ladder
  `{0,1,2,3,4,5,6,7,8,10,12,14,16,base}` with hysteresis and a periodic baseline reprobe.

## 2. Default knobs (`docs/beellama-args.md`, `common/common.h`, `src/llama-cparams.h`)

| Knob | Default | What it does |
| --- | ---: | --- |
| `--spec-draft-n-max` | `16` | Main-path draft token ceiling per cycle (their `B`) |
| `--spec-branch-budget` | `0` | DDTree extra branch nodes beyond the main path; `0` = flat DFlash |
| `--spec-draft-top-k` | `1` | Per-position candidate count; forced to 1 when flat (`branch_budget=0`) |
| `--spec-draft-p-split` | `0.10` | Probability threshold for adding tree branches |
| `--spec-draft-p-min` | `0.0` | Minimum draft probability gate |
| `--spec-draft-temp` | `0.0` | Greedy drafter; `auto` mirrors target temp; `>0` enables Gumbel/rejection-sampling path |
| `--spec-dflash-cross-ctx` | `512` (recommended `1024`) | Recent target hidden-state tokens visible to drafter cross-attention |
| `--spec-dflash-max-slots` | `1` | DFlash is single-slot by default; extra slots fall back to non-speculative |
| `cparams.dflash_topk` | `1` | Drafter graph emits `argmax_ext` (`topk==1`) or `topk_ext` (`topk>1`) directly |

The recommended Qwen3.6-27B launch (`docs/quickstart-qwen36-dflash.md`) uses:

```
--spec-type dflash --spec-draft-n-max 16 --spec-branch-budget 0
--spec-dflash-cross-ctx 1024  --temp 0.6 --top-k 20 --top-p 1.0
```

i.e. `B=16` flat DFlash with greedy drafter + sampled target. This is the exact regime the
4.4x headline result is measured in.

## 3. Published benchmarks (RTX 3090)

From `~/beellama.cpp/README.md` (Qwen3.6-27B Q5_K_S target + Q4_K_M DFlash drafter, RTX 3090):

| Prompt class | Baseline tok/s | DFlash median tok/s | DFlash best | Speedup | Accept (proposed→accepted / accepted→generated) |
| --- | ---: | ---: | ---: | ---: | --- |
| Task store module    | `37.2`  | `163.9` | `181.9` | **4.40x** | `67.7% / 89.2%` |
| KV report module     | `34.6`  | `157.7` | `162.5` | **4.56x** | `58.8% / 88.9%` |
| Doubly-linked list   | `36.8`  | `130.8` | `154.1` | **3.56x** | `50.4% / 86.8%` |
| Multi-turn coding    | `33.3`  | `64.6`  | `65.4`  | `1.94x`   | `24.9% / 72.9%` |
| Prompt processing    | `1229.5`| `1214.4`| `1221.7`| `0.99x`   | n/a (prefill) |

Notes:

- Prompt-processing (prefill) is essentially free of DFlash overhead (`-1%`), confirming the
  controller never engages during pure prefill.
- The `4.4x` headline is the *median* speedup on small generative coding tasks where the
  drafter accepts ~67% of proposed tokens.
- The `1.94x` multi-turn-coding number with `~25%` accept rate is the case where the
  adaptive controller is most necessary; this is also the regime where chain DFlash
  regresses on hipEngine without the R3.1 controller.

## 4. Drafter architecture (`src/llama-arch.cpp`, `src/llama-model.cpp`, `SD-080-benchmark-notes.txt`)

BeeLlama supports two GGUF schemas: `dflash-draft` (Bee/buun) and `dflash` (upstream
llama.cpp DFlash PR). v0.2.0 added the latter and uses
`general.architecture` to disambiguate. Both decode through the same draft graph.

Per-layer tensors loaded by `LLM_ARCH_DFLASH` / `LLM_ARCH_DFLASH_DRAFT` (`llama-model.cpp:7930`):

```
attn_norm, attn_post_norm | wq, wk, wv, wo | attn_q_norm, attn_k_norm
ffn_gate, ffn_down, ffn_up
```

Plus one-shot `dflash_fc` (target-feature fusion) and `dflash_hidden_norm`. Token embedding
and LM head are **shared from the target at runtime** (`tok_embd` / `output` are
`TENSOR_NOT_REQUIRED`). Cross-attention to recent target hidden states is wired through
`dflash_kv_update` and the per-slot cross-context bucketed graph (`src/llama-context.cpp`
`cross_bucket()` ladder: `<=16 -> 16, <=128 -> next pow2, >128 -> 128-aligned`). This is
the same bucketing we ported in our R2.3 `--drafter-bucket cross_bucket` opt-in.

`SD-080-benchmark-notes.txt` records the production Qwen3.5-27B drafter shape:

```
Drafter model: 5 layers, 32 heads, 8 kv heads, n_embd=5120, head_dim=128
n_target_features = 25600 (5 target layers x 5120)
```

vs the z-lab Qwen3.6-35B-A3B-DFlash drafter we use:

| Drafter | Layers | `n_embd` | q heads × head_dim | kv heads × head_dim | intermediate | target_layers tapped |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| BeeLlama Qwen3.5-27B   | 5 | 5120 | 32×128 | 8×128 | n/a (Q4_K_M GGUF) | 5 |
| z-lab Qwen3.6-35B-A3B  | 8 | 2048 | 32×128 | 4×128 | 6144 | (config-driven) |

Per-layer weight footprint scales as `n_embd^2 + n_embd * n_ff`. At `n_embd=5120` the
BeeLlama drafter has roughly **4-5x the per-layer dense work** of ours but **~5/8** the
layer count, so total drafter work per cycle is comparable in FLOPs while the launch count
is **40% lower**. Combined with `cudaGraphLaunch` overhead being a real lever on CUDA, that
plausibly explains why their flat-DFlash drafter wall is small relative to verifier.

## 5. Verifier batching and fusion (`tools/server/server-context.cpp`, `src/llama-context.cpp`)

### Flat verify is `B+1` rows in one ubatch

BeeLlama runs the verifier as one llama.cpp `ubatch` containing the root + `B` candidate
rows. Multi-slot servers can additionally `can_batch_multiseq` together when every slot is
in the token-generation phase (`tools/server/server-context.cpp:4748`):

```c++
const bool can_batch_multiseq = (n_tg_tokens == batch.n_tokens && n_tg_tokens > 0
    && params_base.speculative.type == COMMON_SPECULATIVE_TYPE_DFLASH);
if (can_batch_multiseq) llama_set_force_split_seq(ctx, false);
```

This is a **multi-request-as-multi-seq** batching, not within-cycle fusion. hipEngine
single-request 9-prompt benchmarks don't benefit from this path; it is an extra
multi-tenant lever for server deployments.

### Reduced-logits verifier (`set_dflash_verify_logits`)

When `--spec-type dflash` is active and the per-slot sampler is "boring" (no
grammar/penalty/dry/xtc/top-n-sigma/logit-bias/n_probs/rejection-sampling), the server
calls `llama_set_dflash_verify_logits(ctx, true, top_k)` so the GPU graph emits **only** a
compact `[B+1 × top_k]` tensor of `(token_id, log_prob)` pairs instead of a full
`[B+1 × vocab_size]` FP logits row (`llama-context.cpp:6402`):

```c++
const bool dflash_reduced_logits_only =
    cparams.dflash_reduced_consumer_active && cparams.dflash_verify_logits;
bool has_logits = !dflash_reduced_logits_only;
...
logits.size = has_logits ? n_vocab*n_outputs_max : 0;
```

For greedy sampling `top_k=1`. The eligibility predicate is enumerated in
`dflash_select_reduced_verify_plan()` (`tools/server/server-context.cpp`) — every disabling
case has a named reason (`grammar`, `rejection`, `prob-reporting`, `logit-bias`,
`penalties`, `dry`, `top-n-sigma`, `xtc`, `tree`).

The reduced kernel writes into a per-cycle `t_logits_argmax` tensor, then the server reads
`llama_get_logits_argmax_ith(ctx, idx)` and `llama_get_logits_argmax_probs_ith(ctx, idx)`
to score draft tokens (`dflash_sample_reduced_verify`).

### CUDA hidden capture and replay

The full-attention drafter cross-context lives in a **GPU-resident ring buffer**
(`hidden_gpu`/`gpu_tape`) per slot, updated D2D when the target produces a new accepted
token. Recurrent (Qwen3.5 / Qwen3.5-MoE / Gemma4-ISWA) hidden capture is also handled per-arch
through an explicit `tape_replay` path that rolls back conv + recurrent state on rejection.
The fail-closed prefill capture and per-slot/per-view plans were added in v0.2.0 to make
this safe under partial / mismatched captures.

This is a structurally different cross-attention model from our R3.4 drafter, which uses
projected-context + per-layer rotated-K/V caches written by the verifier `commit_context_rows`
(R2.5 already-present). Both achieve "drafter never re-projects context KV per cycle"; the
mechanism differs.

## 6. Adaptive controller (`tools/server/server-adaptive-dm.h`, 725 lines)

Two controllers are implemented; `profit` is the default (`--spec-dm-controller profit`).
`fringe` is a simpler tail-acceptance controller; `profit` is what the README's headline
acceptance is measured under.

### Profit-controller mechanics

Per-cycle observations are EWMA-aggregated per `(context_bucket, base_n_max, branch_budget,
draft_topk, dflash_cross_ctx, draft_temp, p_min)` config key (`profit_config_key`) so the
controller doesn't blend stats across regimes (e.g. cross-bucket switches reset). Bucketing
of `n_past`:

```
[0..8K) -> 0,  [8K..16K) -> 1,  [16K..32K) -> 2,
[32K..64K) -> 3,  [64K..96K) -> 4,  >=96K -> 5
```

For each bucket the controller maintains:

1. `profit_pos_accept_ewma[pos]` — survival probability of each draft position (i.e.
   probability of accepting at least `pos+1` tokens in a row), EWMA over draft cycles.
2. `profit_depth[d]` — per-depth `(draft_ms, verify_ms, accept_ms, cycle_ms)` EWMA, plus
   a separate `profit_baseline` for `d==0` (no-spec) probes.
3. A fixed depth ladder `{0,1,2,3,4,5,6,7,8,10,12,14,16,base_n_max}` from
   `server_adaptive_dm_build_candidates`.

Per cycle, score each candidate as `(1 + expected_accept) / cycle_ms`, where:

```c++
expected_accept(d) = sum_{p=0}^{d-1} pos_accept_ewma[p]
score(d)           = (1 + expected_accept(d)) * 1000 / cycle_ms(d)
```

Pick the best ready candidate; apply hysteresis margins
(`dm_profit_raise_margin=0.05`, `dm_profit_lower_margin=0.05`) so a single noisy sample
can't flip-flop. If the no-spec baseline beats the best speculative depth by at least
`dm_profit_min=0.05`, increment `profit_consecutive_below_profit`; after `dm_off_dwell=8`
consecutive below-baseline cycles, **disable speculation** (`adaptive_n_max=0`). A periodic
baseline reprobe every `dm_profit_baseline_interval=1024` active cycles refreshes the
no-spec timing.

### Why this matters for hipEngine

This is the **per-prompt routing** that R3.5 (DDTree on gfx1100, currently functional but
non-promotable) needs to satisfy its acceptance gate. BeeLlama's controller switches
DFlash off entirely on prompts where the baseline wins (their multi-turn-coding `1.94x`
result vs `~25%` accept is exactly the regime we currently regress on at `0.327x` AR for
`code:quicksort_prefix`). Our R3.1 implementation has a `profit_window` decision rule and
EWMA but the GPU fallback path is currently blocked on the native-bulk → c=1 AR slot-state
handoff (`docs/DFLASH.md` R3.1 row); BeeLlama proves the controller design works in
practice.

### Fringe controller (alternative)

`fringe` uses a 32-cycle ring of `(n_accepted, n_draft)` pairs and clamps `n_max` based on
the rolling fringe acceptance rate (`dm_fringe_min=0.30`, `dm_fringe_max=0.50`). Simpler
state, weaker decisions; not the recommended default.

## 7. Mapping to hipEngine R3.x

| BeeLlama design point | hipEngine status (post R3.4) |
| --- | --- |
| Default `n_max=16` flat DFlash with greedy drafter | We default to `B=4` flat chain in benches; sweep B is on each row of `dflash_chain_e2e_bench.py`. Bumping default benchmark `B` to 16 only helps if accept-survival stays high enough; per R3.5 we measured tree K=2 B=8 strictly worse on W7900. |
| Adaptive `profit` controller with depth ladder + hysteresis + per-bucket EWMA | **R3.1** has the same controller shape and the native-bulk→c=1 slot-state handoff is fixed: bulk verifies on a branch slot, then accepted rows replay through the canonical c=1 slot before drafter context commit. The strict probe guard now starts in `AR_PROBE`, requires an extra 64-token amortization margin before startup/retry probes, and keeps D32/D160 diagnostics exact inside the safety band (`0.991x` and `0.995x` AR, `draft_calls=0`). BeeLlama's ladder/hysteresis remains the model for future profitable long-horizon probing. |
| Reduced-logits verifier (`set_dflash_verify_logits`) with sampler/grammar fallback | **R3.7** landed default-off as `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD=on`: the fused W8A16 LM-head + argmax rows path removes the full-vocab logits writeback and passes the rocprof gate, but is slower on W7900 (`0.637x → 0.621x` AR; verifier `27.04 → 28.00 ms/cycle`) because the fused stage-1 grid under-occupies RDNA3. Future reduced-logits work needs either true reduced-vocab sampling, a different stage-1 work schedule, or enough verifier rows to restore occupancy. |
| Drafter graph capture with `cross_bucket()` graph cache key | **R2.3** landed bit-equivalent on W7900 but did not produce a perf win (drafter wall on hipEngine is GPU-kernel-bound, not host-launch-bound, after R3.4 WMMA brought drafter to `9.09 ms/cycle`). BeeLlama's win is consistent with their drafter being shallower (5 layers) and on RTX 3090 where graph launch overhead is a higher fraction of cycle. |
| Drafter K/V projection caching for cross-attention window | **R2.5** already-present in our `NativeDFlashChainDrafter` (`commit_context_rows`); `R2.2 sync` confirmed `context_projection_rebuild_rows=0`, drafter context append-only. No remaining lever here. |
| Shallow-and-wide drafter (5 layers @ 5120 hidden) | We use the z-lab 8-layer @ 2048 hidden drafter and have measured 16x2048x{2048,6144} dense kernels at `0.056-0.091 ms/op` on W7900 with R3.4 WMMA. A shallower drafter is **out of scope for this round** (would require retraining), but is worth noting as a model-side lever for future Round-4 work. |
| Single GGUF schema for upstream + Bee | We load via the `z-lab/Qwen3.6-35B-A3B-DFlash` config and `dflash.target_layer_ids` already; v0.2.0's upstream schema is the same shape. |

## 8. Benchmark comparability

| Axis | BeeLlama RTX 3090 (README) | hipEngine W7900 (R3.4 9-prompt) |
| --- | --- | --- |
| Hardware peak BF16 / FP16 TFLOP/s (decode) | RTX 3090 ~`71` (FP16) | W7900 ~`123` (BF16/FP16 WMMA) |
| Hardware peak BW | RTX 3090 `~936 GB/s` | W7900 `864 GB/s` |
| Target | Qwen3.6-27B Q5_K_S GGUF | Qwen3.6-35B-A3B-PARO BF16 (~9× larger active params? no — both are MoE-A3B at ~3B active) |
| Target quant | Q5_K_S (~5 bits/weight) | BF16 (16 bits/weight; ~3× memory traffic per layer) |
| Drafter | 5-layer Qwen3.5-27B-DFlash Q4_K_M | 8-layer z-lab Qwen3.6-35B-A3B-DFlash BF16 |
| Drafter quant | Q4_K_M | BF16 |
| Default `B` | `16` (adaptive 0..16) | `4` in our chain runs (no adaptive yet) |
| Speedup vs AR | `4.40x` median, `4.93x` best | `0.636x` aggregate, `0.911x` best (`class_continuation`) |

The two systems are not directly comparable because BeeLlama runs a **Q5/Q4 GGUF target**
on llama.cpp (~3× lower target memory traffic than our BF16 target) and uses an **adaptive
B up to 16**. The structural choices that generalize are listed in §7.

## 9. References

- `~/beellama.cpp/CHANGELOG.md` (v0.2.0)
- `~/beellama.cpp/README.md` (RTX 3090 benchmark table)
- `~/beellama.cpp/docs/quickstart-qwen36-dflash.md` (recommended launch shape)
- `~/beellama.cpp/docs/beellama-args.md` (full DFlash flag list and defaults)
- `~/beellama.cpp/tools/server/server-adaptive-dm.h` (725-line profit/fringe controllers)
- `~/beellama.cpp/src/models/dflash_draft.cpp` (drafter graph, `dflash_kv_cache`)
- `~/beellama.cpp/src/llama-context.cpp` (`cross_bucket()`, GPU ring, reduced-logits gate)
- `~/beellama.cpp/SD-080-benchmark-notes.txt` (Qwen3.5-27B drafter shape: 5 layers, n_embd=5120)
