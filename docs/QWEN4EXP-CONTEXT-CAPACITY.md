# Qwen4Exp context capacity: the 2051 cap, why it exists, and what raising it costs

Status: public native-context plumbing, real c2 completions/chat startup and
8K retrieval, native-capacity boundary gates complete on Framework gfx1151
(2026-09-06). Section4c records the short
capacity A/B. Native262144 allocation is supported when admitted; this is not
a fresh262144-token inference result. Historical diagnosis below describes the
prior cap; section6 records its replacement.

Audience: whoever owns the Qwen4Exp serving/prefill path next. This exists
because the served default silently caps a 262,144-token model at 2,051 tokens,
and because the number `2051` means two unrelated things in this tree that are
easy to conflate.

## TL;DR

1. `2051` is **not** a tuned threshold and **not** a memory budget. It is an
   algebraic identity from the model's own frozen GGUF constants:
   `qsa_token_budget(2048) + qsa_compression_ratio(4) - 1`. It is the largest
   live context at which QSA top-k selection is provably the identity function,
   i.e. the last point where sparse attention is exactly dense attention.
2. It splits behaviour into two regimes with nothing gradual between them:
   at or below 2051 the sparse machinery is provably inert and results are
   bit-exact dense; at 2052 and above the indexer genuinely chooses which 2,048
   tokens each QSA layer sees. Section 1 tabulates both.
3. The same number is used for two different purposes: as a **model constant**
   (`qsa_dense_equivalent_max_tokens`, correct, must not change) and as the
   **default runner capacity** (`max_sequence_length`, a bring-up default that
   should not have been frozen at the same value). Anyone grepping `2051` will
   hit both. This is the single biggest trap in this area.
4. The old public factory did not forward capacity. It now accepts
   `max_sequence_length` and `resident_capacity`; LLM forwards these only to
   factories explicitly declaring them. Auto defaults to the largest admitted
   native context; explicit requests fail clearly if they do not fit.
5. Raising capacity leaves short requests on the dense branch because dispatch
   keys off *live* count. It is not literally free: measured short decode is
   within0.1% of the small-capacity arm, while prefill loses0.24-0.74%. Full KV
   clearing happens in request reset as well as construction.
6. There is a real and large performance cliff at the boundary, but it is a
   property of the model architecture, not of the default. It is paid by prompt
   length, whatever the capacity is set to.

## 1. Where 2051 comes from

Two frozen GGUF constants, validated on load in
`hipengine/loading/qwen4_exp_gguf.py` (`_geometry_errors`, lines 236-246):

```
qwen4exp.attention.indexer.top_k    = 2048   -> qsa_token_budget
qwen4exp.attention.compress_ratios  = 4      -> qsa_compression_ratio
qsa_block_budget = 2048 // 4 = 512 blocks
```

and the derived property (`qwen4_exp_gguf.py:107-109`):

```python
@property
def qsa_dense_equivalent_max_tokens(self) -> int:
    return self.qsa_token_budget + self.qsa_compression_ratio - 1   # 2051
```

mirrored on the device state as `dense_equivalent_limit`
(`hipengine/runtime/qwen4_exp_runner.py:1082`):

```python
return self.block_budget * self.compression_ratio + self.compression_ratio - 1
```

### Why that expression is the exactness boundary

Read the selection oracle, `kernels/cpu_reference/qwen4_exp.py:663-671`. For a
query at position `p`:

```python
eligible = np.nonzero(starts + ratio - 1 <= query_position)[0]   # complete blocks <= p
ranking  = np.lexsort((starts[eligible], -score[row, eligible]))
chosen   = eligible[ranking[:budget]]                            # top-512
...
if query_position % ratio != ratio - 1:                          # partial current block
    logical_positions.extend(range(tail_start, query_position + 1))   # ALWAYS appended
```

Two properties follow:

- **The tail is unconditional.** The incomplete current block is appended
  without being scored and can never be dropped. `qsa_pool_complete_blocks`
  (`:462-505`) enforces the matching invariant on the other side: only the
  highest block may be incomplete, and it must have no holes.
- **Top-k is a no-op when it is not oversubscribed.** `ranking[:budget]` over
  `len(eligible) <= budget` candidates returns every candidate, regardless of
  score.

So selection discards nothing exactly when the number of complete blocks fits
the budget:

```
floor((p+1) / ratio) <= block_budget
floor((p+1) / 4)     <= 512
p + 1                <= 2051          (2051 // 4 == 512;  2052 // 4 == 513)
```

At live count 2052 the 513th complete block appears, the budget can no longer
cover every block, and the model starts genuinely discarding tokens. That is
Qwen's intended sparse-attention design, not an approximation introduced here.

### The two regimes

The cutoff separates a regime where the sparse machinery is provably inert from
one where the indexer decides what the model sees. Nothing is gradual about it.

| | Regime A: exactly dense | Regime B: indexer-selected |
| --- | --- | --- |
| Live context `L` | `1 <= L <= 2051` | `L >= 2052` |
| Complete blocks, `floor(L/4)` | `<= 512`, fits the budget | `>= 513`, oversubscribed |
| Top-k selection | **identity** — returns every eligible block, scores irrelevant | genuine top-512 by indexer score |
| What actually executes | plain dense GQA attention; the branch at `:2284` is not taken, so index-q projection, scoring, top-k and sparse attend are all skipped | index-q projection + norm/RoPE, score over `floor(L/4)` pooled block keys, stable top-512, expand to positions, sparse GQA |
| Tokens attended per QSA layer | all `L` | `2048 + (L mod 4)`, i.e. 2048-2051 |
| Relation to the dense oracle | bit-exact by construction | an approximation, chosen by the indexer |
| Cost per token | grows with `L` | ~constant in `L`; only scoring grows, at O(L/4) |

The selected budget is fixed, so in regime B the fraction of context each QSA
layer can see falls as `1/L`:

| Live context `L` | Complete blocks | Blocks selected | Tokens attended per QSA layer | Coverage |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 512 | 512 (all) | 2,048 | 100% |
| 2,051 | 512 | 512 (all) | 2,051 | **100%** |
| 2,052 | 513 | 512 | 2,048 | 99.8% |
| 4,096 | 1,024 | 512 | 2,048 | 50.0% |
| 16,384 | 4,096 | 512 | 2,048 | 12.5% |
| 65,536 | 16,384 | 512 | 2,048 | 3.13% |
| 262,144 | 65,536 | 512 | 2,048 | **0.78%** |

Two things that table must not be over-read:

- **It is per QSA layer, not per model.** Only 12 of 48 layers are QSA
  (`attention.compress_ratios = (0,0,0,4) x 12`, so every 4th layer). The other
  36 are GDN, a gated linear-recurrent mixer carrying full-sequence state with
  no selection at all. The model is not looking at 0.78% of a 256K context; its
  *attention* layers are, and its recurrent layers are not. This hybrid split is
  the whole point of the architecture.
- **Selection is per query, per layer, and content-dependent.** Each decoding
  step re-scores and re-selects, and each of the 12 QSA layers selects
  independently. The P10 structural census confirms this behaves as designed:
  at 4K/16K/64K the needle was selected in all 12 layers at every depth, with
  selected spans of 4,096/16,384/65,536 tokens and mean gaps 2.00/8.00/32.02.
  Low coverage is not the same as low recall.

So the practical reading of the cutoff is: **through2051 the dense-equivalent
oracle applies; above2051 full-dense output equality is not the right oracle.**
Native QSA selection, tail inclusion, buffer ownership and declared arithmetic
gates remain binding. Retrieval/task evidence additionally evaluates the model's
content-dependent selection; it does not replace those correctness checks.
That distinction is why the two numerical regimes want different tests, and it
is section 7's organising principle.

**2051 is therefore a correctness boundary, not a performance crossover.** The
performance evidence points the opposite way: crossing it *costs* ~29 ms/token
(section 4). Nobody choosing this number empirically would choose 2051; you
would push it as high as memory allowed, because dense is faster than sparse
everywhere the architecture permits dense.

## 2. The number collision

| Meaning | Value | Where | Change it? |
| --- | --- | --- | --- |
| Model constant: largest exactly-dense live context | 2051 | `models/qwen4_exp.py:23`, `loading/qwen4_exp_gguf.py:107`, `runtime/qwen4_exp_runner.py:1082` | **Never.** Derived from the artifact. |
| Selection-buffer capacity (max positions one query can select) | 2051 | `qwen4_exp_runner.py:5112`, `:5166` | **Never.** 512 blocks x 4 + 3 tail is the true maximum at any context. |
| Default runner/generator sequence capacity | 2051 | `generation/qwen4_exp_gguf.py:59`, `runtime/qwen4_exp_runner.py:5013` | **Yes.** This is the bring-up default under discussion. |

Only the third row is a policy choice. The first two are geometry. A blanket
find-and-replace on `2051` will silently break exactness.

## 3. Why the cap cannot currently be raised

**Historical diagnosis, fixed September6:** the factory now forwards context
and residency and the generator implements `prepare()`. The excerpt below
documents the old behavior; see section6 for the new admission policy.

The registered factory (`generation/qwen4_exp_gguf.py:764-777`) does not accept
or forward `max_sequence_length`:

```python
def make_qwen4_exp_gguf_generator_gfx1151(
    model_path, weight_index, model_plugin, vision_model_path=None
) -> Qwen4ExpGGUFTextGenerator:
    return Qwen4ExpGGUFTextGenerator(
        model_path=model_path, weight_index=weight_index,
        model_plugin=model_plugin, backend="hip_gfx1151",
        vision_model_path=vision_model_path,
    )   # max_sequence_length falls back to the constructor default, 2051
```

registered for both `gguf_q4_k_m` and `gguf_ud_q4_k_xl` (`:780-786`). Every
served request therefore gets a 2051-capacity runner.

`--max-context-tokens` can only clamp downward, because both admission points
compare against the already-constructed runner:

- `Qwen4ExpGGUFTextGenerator.prepare_request_scratch` (`:182`) —
  `raise ValueError("Qwen4Exp serving scratch exceeds sequence capacity")`
- `Qwen4ExpResidentServingRunner.prepare` (`:478`) —
  `raise ValueError("Qwen4Exp serving prepare exceeds sequence capacity")`

and per-request generation rejects the same way at `:229`.

Meanwhile `Qwen4ExpGGUFModel.native_context_length = 262144`
(`models/qwen4_exp.py:22`) is declared and read by **nothing** except
`tests/test_qwen4_exp_model.py:31`. The server's auto path sizes from
`_prepared_context_tokens` / model metadata rather than from this plugin field,
which is why startup asks for more than 2051 and then fails at the scratch
probe until `--max-context-tokens 2051` is passed by hand.

### Load-bearing detail: the capacity parameter already works

`scripts/qwen4exp_row4_state_gate.py:52-55` constructs the generator directly:

```python
generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
    ..., backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=512))
```

and then asserts at `:194,198` that the QSA route engages **iff**
`prompt_tokens > 2051` — a live-count predicate, exercised today. Eleven further
`scripts/qwen4exp_*.py` harnesses construct with their own capacity too, and P10
took this to 64K (section 6). So this is a plumbing gap, not unimplemented
functionality. That materially de-risks the change.

## 4. Cost model

### 4a. Memory — scales with configured capacity, paid up front

Device KV per token, from frozen geometry
(12 QSA layers x 2 KV heads x 256 head_dim x {K,V} x 2 bytes) = **24,576 B =
24 KiB/token**. `Qwen4ExpDenseAttentionState.allocate`
(`qwen4_exp_runner.py:694-711`) eagerly `malloc`s **and `memset`s** the full
key/value cache at `max_positions`, rounded up to 256-token blocks. This is not
paged and not lazy.

Index state is `_qsa_index_state_bytes`
(`loading/qwen4_exp_materialize.py:234-251`), dominated by raw FP32 index keys
(12 x context x 128 x 4).

| context | KV | index state | per sequence | at c2 |
| ---: | ---: | ---: | ---: | ---: |
| 2,051 | 0.05 GiB | 0.02 GiB | 0.06 GiB | 0.12 GiB |
| 16,384 | 0.38 GiB | 0.12 GiB | 0.49 GiB | 0.99 GiB |
| 65,536 | 1.50 GiB | 0.47 GiB | 1.97 GiB | 3.95 GiB |
| 131,072 | 3.00 GiB | 0.95 GiB | 3.95 GiB | 7.90 GiB |
| 262,144 | 6.00 GiB | 1.90 GiB | **7.90 GiB** | **15.79 GiB** |

Two notes. `plan_qwen4_exp_memory_admission`
(`loading/qwen4_exp_materialize.py:185-231`) already computes exactly this,
including weights, staging, runtime state, scratch and reserve, and is already
driven by `scripts/qwen4_exp_memory_plan.py` — it is the right function to
resolve the default against, and it needs no new code. And the PLE table is
`device_resident=False` (host sparse mmap, `:146-155`), so it is correctly
excluded from device bytes despite dominating the on-disk footprint.

Reduction available but not required: `bf16_compressed_index_bytes_per_token`
already exists on the config (`loading/qwen4_exp_gguf.py:129-138`) and is
unused. Moving the index owner off raw FP32 would cut ~1.5 GiB of the 1.9 GiB
at 256K. Out of scope here; noted so it is not rediscovered.

### 4b. Performance — scales with live prompt length, *not* with capacity

This is the part that decides the default. The branch is on live count
(`qwen4_exp_runner.py:2284`):

```python
if not device_position_owned and index_state.count > index_state.dense_equivalent_limit:
```

`dense_equivalent_limit` is a model constant. A 500-token request in a
262,144-capacity runner still takes the dense-equivalent branch and skips the
index-query projection, scoring, top-k and sparse attend entirely. **Raising
capacity does not move short requests onto the sparse path.**

Consequently the cliff is a step, not a slope, and it is paid identically today
at whatever capacity. From already-committed evidence in
`docs/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`:

- P6 boundary profile (`:551-560`): clean wall adds **+29.27 ms/token** at live
  2051 -> 2052, of which ordered sparse attention owns **27.47 ms** and
  score/top-k only **0.92 ms**. Live 2052 -> 4097 is **flat**, confirming fixed
  selected-budget cost rather than context-length growth.
- tg128 benchmark rates (`:536-541`): **14.421 tok/s at p1024 -> 10.416 tok/s at
  p4096** (69.3 -> 96.0 ms/token, +26.6 ms/token), consistent with the boundary
  profile.
- P10 owner profile (`:2140-2152`): clean decode **82.617 / 82.459 ms/token** at
  16K / 64K — flat across a 4x context increase. Complete QSA is 20.834 / 22.132
  ms/token; ordered selected attention 19.642 / 19.584; compressed-key
  score/top-k grows only **1.116 ms (1.35%) at 16K -> 2.490 ms (3.02%) at 64K**.

Note these are three different protocols (boundary clean wall, tg128 benchmark
rate, owner clean decode). Do not divide one by another.

The shape of it: above the boundary the selected set is pinned at 2,048 tokens
plus tail, so attention cost is constant in context length. Only scoring grows,
at O(context/4), and it stays a few percent out to 64K. **2052 costs
approximately what 65,536 costs.**

### 4c. Measured capacity A/B (2026-09-06 UTC, Framework gfx1151)

Clean `2257408cf`, UD-Q4_K_XL/BF16 KV: one shared weight residency, separate
2051- and262144-capacity runners, all eight p512/p1024 category cases, one
warmup per arm and three alternating measurements, tg128. All48 trajectories
are exact and teardown leaves zero allocations. Elapsed13m14.8s.

| Shape | 2051-cap PP / TG | Native-cap PP / TG | PP change | TG change |
| --- | ---: | ---: | ---: | ---: |
| p512 | 159.150 /20.052 | 157.977 /20.039 | -0.737% | -0.066% |
| p1024 | 158.140 /19.357 | 157.764 /19.339 | -0.238% | -0.095% |

No large short-context decode cliff appears, but do not call capacity cost zero.
Native runner setup after weights were resident took0.987s; baseline setup48.05s
includes loading weights and is not a comparable setup interval. Peak tracked
bytes96,552,645,464 include **both** runners, shared weights and MMQ sidecars.
`runner.reset()` clears full KV on every request, so the prefill timing includes
capacity-sensitive clearing. This run does not validate native-length inference
or sparse retrieval; it is the requested short-context capacity gate.
[Evidence](../benchmarks/results/2026-09-06-framework-qwen4exp-context-capacity-ab.json).

## 5. Implications for the in-flight prefill work

Read this section before landing prefill changes; three of these interact
directly.

**5.1 Flash prefill is a short-prompt-only path today.** In
`run_qwen4_exp_qsa_prefill` (`qwen4_exp_runner.py:2611-2623`):

```python
dense_rows = max(0, min(count, index_state.dense_equivalent_limit - start))
qsa_flash = (dense_rows > 0 and head_dim == 256 and ...
             and os.environ.get("HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL", "0") ...)
```

`dense_rows` is clamped by the **model constant**, so it goes to zero for every
chunk whose start is at or beyond 2051. The flash prefill route therefore covers
100% of a p512 or p1024 prompt but only the first ~2051 tokens of a p4096 one,
and a vanishing fraction at 16K+. Any prefill speedup validated at p512/p1024
should not be extrapolated to long prompts without a p4096+ arm, and a p4096
measurement is a blend of two routes rather than one.

**5.2 Prefill score scratch scales with configured capacity.** The one prefill
allocation that is a function of `max_sequence_length` rather than
`prefill_chunk_size` is `Qwen4ExpQSAPrefillMetadata`
(`qwen4_exp_runner.py:5163-5170`):

```python
score_blocks=(self.max_sequence_length + cfg.qsa_compression_ratio - 1) // cfg.qsa_compression_ratio
```

allocated as `rows * score_blocks * 4` bytes (`:828-833`). With
`prefill_chunk_size=512`:

| capacity | score_blocks | prefill score buffer |
| ---: | ---: | ---: |
| 2,051 | 513 | 1.00 MiB |
| 65,536 | 16,384 | 32 MiB |
| 262,144 | 65,536 | **128 MiB** |

A 128x growth, but bounded and affordable. The neighbouring buffers do not
scale that way: `selected_positions` is `rows x 2051 x 8` = 8.01 MiB **fixed**
(it is sized by the model constant, correctly), and `block_tables` reaches only
2 MiB at 256K with `block_size=256`. Anyone retuning prefill scratch should keep
the `selection_capacity` argument bound to `cfg.qsa_dense_equivalent_max_tokens`
and only `score_blocks` bound to capacity.

**5.3 There is a hard incompatibility between device-owned positions and the
sparse path.** `qwen4_exp_runner.py:2269-2271`:

```python
if device_position_owned:
    if position + 1 > index_state.dense_equivalent_limit:
        raise ValueError("device-owned dense QSA capture cannot cross sparse selection")
```

Today `device_position_owned=True` is reached only from
`scripts/qwen4exp_stateful_layer_graph_probe.py:138` and one unit test, never
from the production runner, so it does not block raising the default. But it
does mean **the graph-owned/device-position optimization as currently written
cannot be promoted to a default while long context is enabled.** If the prefill
or graph work is heading toward device-owned positions, that constraint has to
be designed for now rather than discovered at promotion time.

**5.4 Startup cost.** The eager `memset` of 6 GiB of KV per sequence (12 GiB at
c2) happens during construction **and runner.reset() at each request**.
Raising the default therefore affects startup and short-prefill reset cost,
not just construction. The capacity A/B above measures that effect without
changing the clearing contract.

## 6. What already works, and what has to change

**Implemented September6:** the generator resolves native context from artifact/
plugin metadata through `resolve_qwen4_exp_context`, using the existing memory
admission planner plus256-token physical KV rounding. Auto selects the largest
fitting capacity (minimum one compression block); explicit limits never silently
shrink. Default residency is c2 unless a caller configures c1. Admission reserves
4GiB per runner for current scratch and MMQ sidecars, plus4GiB free reserve;
optional vision payload is separately reserved. This conservative policy is not
a byte-exact estimate of every allocation, and current sidecar/scratch growth must
be kept within it or reflected in the policy.

`LLM` forwards optional context/residency only to factories explicitly declaring
those parameters. The generator's `prepare()` reports the admitted capacity or
validates a lower requested limit. Startup creates resident slots within the
admitted count. No QSA model constant, selection budget, kernel arithmetic, or
device-position probe behavior changes. Raw KV clearing remains unchanged.

Real OpenAI-compatible server startup with **no context override**, BF16 and c2,
resolved262144 and returned `VIOLET-7391` from an8192-token archive request.
A262145-token prompt returned HTTP400 `context_length_exceeded`; close freed
all tracked allocations. Peak tracked bytes105,093,140,904 included weights,
both native-capacity runners and sidecars.
[Serving evidence](../benchmarks/results/2026-09-06-framework-qwen4exp-native-context-serving.json).
Clean `6ed342313` followup passes chat retrieval at8154 rendered prompt tokens
and HTTP400 `context_length_exceeded` on oversized chat, with zero final
ownership. At allocated capacity262144, the boundary probe retains live2051
dense-equivalent and2052/4097 indexed-sparse, each three-repeat token/state
exact. Steady windows are allocation-stable;2048 bytes of cross-bucket growth
are recorded and all ownership is released on close.
[Final chat/boundary packet](../benchmarks/results/2026-09-06-framework-qwen4exp-native-context-final.json).
This enables the full native allocation/serving limit subject to memory; it does
not claim a new full256K-length quality/performance qualification. P10's prior
4K/16K/64K evidence remains named separately.

Working and evidenced, do not redo: P10 in the performance campaign
(`:2104-2127`) records natural 4K / 16K / 64K retrieval on the current stacked
production path — `VIOLET-7391` retrieved at every depth, needle selected in all
12 QSA layers, all 2,048 binding CPU-selected positions matched at layer 47,
repeat/rollback isolation passing, zero-allocation teardown. Artifacts:
`benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-natural-retrieval-current.json`,
`...-p10-structural-depth-census.json`,
`...-p10-16k-owner-profile.json`, `...-p10-64k-owner-profile.json`.

Original implementation checklist (items1-3 implemented; MTP remains separate):

1. **Thread the parameter.** Give
   `make_qwen4_exp_gguf_generator_gfx1151` a `max_sequence_length` argument and
   forward it. Check what the generator registry passes factories; if it does
   not currently pass a requested context, that is the actual blocking edit.
2. **Resolve the default from admission, not from a literal.** Default to
   `model_plugin.native_context_length` clamped by
   `plan_qwen4_exp_memory_admission(...)` against real free device bytes and the
   configured resident capacity. Both functions already exist. Do not hardcode
   262144: it must degrade honestly on a smaller box instead of failing at
   startup, which is the current failure mode in reverse.
3. **Give the generator a `prepare`.** `Qwen4ExpGGUFTextGenerator` has
   `prepare_request_scratch` but no `prepare`, so `LLM.prepare()` returns None
   for it (`hipengine/llm.py:741-745`) and the server's `ensure_resident_context`
   cannot negotiate a context downward before the scratch probe raises.
4. **Consider the MTP cap separately.** `generation/qwen4_exp_mtp.py:87-88` and
   `:122` cap the draft at `min(1_024, ...)` independently. P11
   (`:2337-2352`) already records this as blocking 4K MTP admission. Same theme,
   different fix; do not assume step 1 lifts it.

Deliberately **not** proposed: changing any of the three geometry-derived 2051s,
or making the dense/sparse branch configurable. The branch is the model's
semantics.

## 7. Validation that this change touches

Organise this by regime, per section 1. Raising the cap does not weaken any
regime-A guarantee — the exactly-dense path is untouched and its gates stay
binding as-is. What it does is expose regime B to served traffic for the first
time. Regime B is not gated by equality to full dense attention because it is
*designed* to discard tokens. Exact native-QSA selection/tail/ownership checks
and declared arithmetic gates still apply. Add retrieval/task evidence
(P10's needle/selected-position parity is the existing template), not KL-vs-dense
on the full context.

Tests asserting the **model constant** — these must keep passing unchanged, and
their continuing to pass is the guard against conflating the two meanings:

- `tests/test_qwen4_exp_model.py:30` — plugin `qsa_dense_equivalent_max_tokens == 2051`
- `tests/test_qwen4_exp_gguf_config.py:117` — config `== 2_051`
- `tests/test_qwen4_exp_qsa_sparse_hip.py:115` — `selected_count = 2_051`
- `tests/test_qwen4_exp_qsa_hip.py:378` — positions `[2051, 2998, 4095]`
- `tests/test_qwen4exp_qsa_h256_wave.py:120` — `(4, 2051, False, 1)`
- `tests/test_qwen4exp_context_decode_profile.py:117` — boundary probe defaults `[2051, 2052, 4097]`

Likely to need review because they encode a **capacity** assumption:

- `scripts/qwen4exp_layer2_profile_gate.py:735` — `--max-sequence-length` default 2051
- `scripts/qwen4_exp_compare_suite.py:83`, `scripts/qwen4_exp_compare_logits.py:132` — `--context` default 2051
- `tests/test_qwen4exp_perf_gap_report.py:26` — `"2051"` keyed row
- Any server test that asserts startup succeeds without `--max-context-tokens`

Original proposed validation list (per `docs/TESTING.md` tiers):

**September6 completion:** the originally proposed gates below are now backed
by610 server/LLM/model/config tests,17 GPU native-QSA tests (including65,536
pooled blocks versus CPU top-k), the48-trajectory short-capacity A/B, real
c2 completions/chat startup and retrieval, and native-capacity boundary
repeats. Keep their protocol/coverage limits: this is not a fresh256K prompt run.

- Narrow: the focused Qwen4Exp test bundle, which the campaign records at 268
  tests (`:2405-2410`).
- Capacity-sensitive: `scripts/qwen4exp_row4_state_gate.py` — it constructs at
  4352 and asserts the live-count predicate directly, so it is the most
  targeted regression check for this change.
- Boundary: `scripts/qwen4exp_context_decode_profile.py --live-count 2051 2052 4097`
  to confirm the cliff is unmoved by the capacity change.
- Short-context neutrality **(the one genuinely new measurement required)**: an
  A/B of p512/p1024 decode on a 2051-capacity runner versus a large-capacity
  runner. Section 4b argues from the code that this is neutral, but every
  retained short-context number was taken on a 2051-capacity runner, and
  allocation-size cache/page effects are not something to assert from a code
  read. This is the gate for flipping the default. Nothing else in this document
  requires new measurement.
- Serving: startup with no `--max-context-tokens`, plus a request that exceeds
  the resolved context, to confirm the rejection path still reports
  `context_length_exceeded` cleanly rather than failing at startup.

## 8. Product context

The concrete symptom that started this: on tool-eval-bench, 58 of 69 scenarios
were rejected `context_length_exceeded` against the 2051 cap (the suite peaks
around 6.4K tokens/request), producing a 9/100 that is a capacity artifact and
not a behavior score. On the 11 scenarios that fit, safety behavior matched both
llama.cpp lanes exactly. Any published hipEngine number on a suite whose prompts
exceed 2051 tokens is measuring this cap, not the engine.

Note the honest trade being made by raising it: crossing2051 introduces the
model's sparse-route cost. The cited69.3->96.0ms/token example is about1.39x,
not4x, and other protocols must not be divided into it. Raising the cap converts "refuses the request" into "serves it
slowly". That is the right trade for a 256K-context model, but it will move
aggregate throughput numbers on any mixed-length suite, and those rows should be
re-baselined rather than compared across the change.
