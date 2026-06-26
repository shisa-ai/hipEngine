# GGUF MTP llama.cpp Parity Trace and Roadmap

Date: 2026-06-26 (correctness-solved update; original trace 2026-06-25)  
Branch: `mtp-gguf`  
Hardware for all runtime numbers below: **gfx1151 / AMD Radeon 8060S (Ryzen AI Max+ 395)**, not the default W7900. Numbers are single-prompt diagnostics, not retained benchmark rows.  
hipEngine commit used for source links: `98df03ddd00ae682c07e302721343040373e1b55`  
llama.cpp checkout used for source/runtime evidence: `6e9007ae61f4e994c27484759caac6ef2aa32b30`

## Executive summary (2026-06-26)

**Correctness is solved. The remaining gap is pure performance — roughly 1.9x.**

| Milestone | Status |
| --- | --- |
| Target AR first-token parity | ✅ `71093` matches llama.cpp (Qwen3.5 GDN K-head broadcast fix) |
| Target AR 12-token greedy trace | ✅ identical sequence `[71093,12305,198,727,10562,17885,10620,25,1103,8,1411,1103]` |
| Strict B3 draft acceptance | ✅ `2/9` → `9/9`, and `15/15` over 5 cycles (context replay + device MTP KV) |
| F32 router/alpha/beta retention | ✅ landed (registry-dispatched mixed kernels) |

The earlier blocker — hipEngine's target autoregressive stream diverging from
llama.cpp at the first sampled token — is fixed. The root cause was Qwen3.5
linear-attention Gated-Delta-Net K-head mapping: GGML maps value head `v_head` to
key head `v_head % num_k_heads`, while hipEngine inherited grouped `v_head /
repeat`. With the interleaved mapping, target AR and strict B3 acceptance both
match llama.cpp on the merge-sort prompt.

### Performance: current numbers (single-prompt diagnostic, gfx1151)

llama.cpp B3 MTP on the same reasoning-off 12-token trace:
**`eval time = 89.55 tok/s`** (`134.01 ms / 12 tokens`), 100% strict draft
acceptance, from `/tmp/hipengine-llamacpp-mtp-cli-reasoning-off-debug.log:3813`.

hipEngine best diagnostic configs (all `15/15` strict accepts, B3/C5, merge-sort
prompt):

| Configuration | tok/s | vs AR | verify ms/cycle | draft ms/cycle | accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Block verify GEMV prefill + 32k draft cap | 48.1 | 0.80x | ~61–66 | ~17 | 15/15 |
| One-step graph + 32k draft cap | 44.5 | 0.81x | ~72 | ~17 | 15/15 |
| One-step graph, full vocab | 42.3 | 0.77x | ~73 | ~22 | 15/15 |

Gap to llama.cpp: **~48 vs ~90 tok/s ≈ 1.9x slower**, and it is almost entirely
target verification overhead, not acceptance and not draft quality.

### Where the time goes (per B3 cycle)

| Stage | hipEngine | llama.cpp | Gap |
| --- | --- | --- | --- |
| Target verify (4 tokens) | ~64 ms (block GEMV) / ~73 ms (graph) | ~8.9 ms (`dur(g)=26.7 ms / 3 calls`) | 7–8x |
| MTP draft (3 tokens) | ~17 ms (32k cap) / ~22 ms (full vocab) | included in `dur(g)` | ~2x |
| Commit / bookkeeping | ~1.6 ms | negligible | minor |

A synchronized per-layer probe over the first B3 verifier block shows the cost is
in the linear-attention layers: **30 linear-attention layers ≈ 82 ms** vs **10
full-attention layers ≈ 23.5 ms**; snapshot/restore is only ~9 ms / ~0.8 ms.
llama.cpp runs the equivalent layers inside one fused GGML compute graph with
batched ops; hipEngine dispatches each kernel individually from Python.

### Next steps, ordered by impact

1. **Target verifier (the #1 blocker, 7–8x).** Make the 4-row target continuation
   run through a captured HIP graph or C-level dispatch loop instead of ~40
   per-layer Python launches. Priority is a dedicated small-B linear-attention
   layer path — that is where ~82 of the ~106 verifier ms live. Not more
   one-step-graph tuning and not selected/WMMA prefill for tiny verifier blocks.
2. **MTP draft resident path.** Keep all MTP intermediates (embeddings,
   projections, KV, hidden seeds) on device across draft depths; only D2H the
   final top-1 token ID. Chain the B draft steps in one call instead of B separate
   `run_draft()` calls with full alloc/copy per depth. Validate the 32k draft
   vocab cap on the full suite before promoting (saved ~5 ms/cycle here but is
   prompt-sensitive).
3. **Partial-accept rollback is catastrophic (~303 ms for a B5 partial cycle).**
   Track which linear-attention buffers were modified and copy-on-write only
   those, or replay only the accepted prefix instead of full target decodes. Or
   just keep B3 (100% accept on this prompt) and skip B5 until rollback is cheap.
4. **Full-suite validation before any retained speed claim.** Everything above is
   single-prompt merge-sort diagnostics. Need the full
   `mtpbench-code-general-ja.jsonl` category suite, category heldouts, a true
   no-MTP AR baseline from the same protocol, and the draft vocab cap validated
   for non-regressive acceptance across prompts.
5. **Longer-term: match llama.cpp's architecture.** Both target verification and
   MTP drafting run through one optimized GGML compute graph in a single process
   with shared weight memory; hipEngine dispatches each kernel individually from
   Python with generic GEMV/prefill kernels not tuned for B=4. Closing this means
   a C-level dispatch loop or HIP graph capture for the multi-layer forward pass.

The historical trace evidence below is retained as the record of how correctness
parity was reached.

## Source evidence: what llama.cpp does

All llama.cpp source links below point to commit
`6e9007ae61f4e994c27484759caac6ef2aa32b30`.

### 1. Qwen35MoE MTP graph

The Qwen35MoE MTP graph is built as a one-layer decoder graph:
[`src/models/qwen35moe.cpp#L550-L736`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L550-L736).
Important details:

- It requires one NextN/MTP block.
- It chooses `nextn.embed_tokens` when present, otherwise `model.tok_embd`.
- It takes a separate hidden-state input tensor named `mtp_h_input`.
- It calls `build_attn_inp_kv()`, so the MTP block has its own draft-context K/V state.
- It computes:
  1. `h_norm = RMSNorm(h_input, nextn.hnorm)`
  2. `e_norm = RMSNorm(token_embedding, nextn.enorm)`
  3. `concat = [e_norm, h_norm]`
  4. `eh_proj`
  5. attention + gated output projection + residual
  6. MoE/shared-expert FFN + residual
  7. shared-head norm, then LM head fallback to `model.output`.

This graph shape matches our Python/GPU wrapper at a high level.  The gap is in
**state lifecycle and numerical/runtime parity**, not the obvious concat order or
which head/embedding tensors are chosen.

### 2. MTP state maintained by llama.cpp

The MTP speculative implementation stores per-sequence state in
[`common/speculative.cpp#L816-L918`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L816-L918):

- `pending_h`: hidden row used to seed the next MTP draft.
- `verify_h`: hidden rows captured from the target verifier batch.
- `verify_h_rows`: how many verifier hidden rows are available.
- `last_n_drafted`: last draft length, used for recurrent/rollback bookkeeping.

This is the critical lifecycle we only partially approximate today.

### 3. `process()` mirrors target verifier rows into the draft/MTP context

llama.cpp's MTP `process()` is in
[`common/speculative.cpp#L955-L1045`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L955-L1045).
The important behavior:

- It copies target `h_nextn` rows from the target context.
- It builds an MTP batch with token/hidden pairs.
- It calls `llama_decode(ctx_dft, batch)` on the draft/MTP context.
- That decode advances the MTP graph and its K/V state, not just a single isolated
  row.
- It stashes verifier hidden rows in `verify_h` and refreshes `pending_h`.

This is what our old no-context path lacked.  Our new `--mtp-device-kv-cache`
implements a first B1 approximation of the K/V portion, but not the full
llama.cpp process lifecycle or B>1 rollback/transactional semantics.

### 4. `draft()` seeds from `pending_h`, samples from `ctx_dft`, and chains `h_nextn`

llama.cpp's MTP `draft()` is in
[`common/speculative.cpp#L1048-L1168`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1048-L1168):

- It adds the last accepted token `dp.id_last` at `dp.n_past`.
- It overwrites the draft batch embedding with `pending_h`.
- It calls `llama_decode(ctx_dft, batch)`.
- It samples a draft token from the draft/MTP logits.
- It reads `llama_get_embeddings_nextn_ith(ctx_dft, i_batch)` and uses that as
  the hidden seed for the next draft step.
- It repeats up to `n_max`, respecting `p_min`.

This is where llama.cpp gets an actual predictive draft chain.  hipEngine's
`run_draft()` also chains `return_hidden_seed`, but our state before/around that
chain has not matched llama.cpp's `process()`/draft context yet.

### 5. `accept()` chooses the verifier hidden row for the next seed

llama.cpp's MTP `accept()` is in
[`common/speculative.cpp#L1171-L1184`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1171-L1184):

- It chooses `i_h = min(n_accepted, n_rows - 1)`.
- It copies `verify_h[i_h]` into `pending_h`.

This matches our conceptual `pending_hidden_row_index = accepted` logic, but we
must still validate that our captured row is numerically the same row at the same
point in the graph.

### 6. Runtime stats are reported by common speculative stats

The aggregate counters are printed by
[`common/speculative.cpp#L2079-L2103`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L2079-L2103):

- `#gen drafts`
- `#acc drafts`
- `#gen tokens`
- `#acc tokens`
- begin/draft/accept durations

These counters are the cleanest runtime evidence we have without editing the
read-only llama.cpp checkout.

## Source evidence: what hipEngine currently does

All hipEngine source links below point to commit
`98df03ddd00ae682c07e302721343040373e1b55`.

### 1. Acceptance accounting

hipEngine's benchmark implements llama.cpp-style strict acceptance in
[`scripts/gguf_mtp_bench.py#L259-L297`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L259-L297):

- The target samples `[last_token] + accepted_draft_prefix`.
- The first mismatch emits a corrective target token.
- Visible output tokens are accepted draft targets plus the corrective token.

The benchmark also has root/sibling top-K acceptance diagnostics; those are useful
for measuring whether the target is somewhere in the draft distribution, but they
are **not** evidence that the draft chain matches llama.cpp.

### 2. Device-resident MTP KV cache, default off

The new opt-in dense device cache is in
[`hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760),
with the device-to-device write and dense attention read in
[`mtp_nextn.py#L975-L1002`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L975-L1002).

Accepted-row cheap commit is handled via `kv_write_only` in
[`mtp_nextn.py#L880-L930`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L880-L930),
and the benchmark uses it in
[`scripts/gguf_mtp_bench.py#L1126-L1155`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L1126-L1155).

The fixture proving sequential cache writes match two-row dense attention is
[`tests/test_mtp_dense_device_kv_cache.py#L1-L120`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/tests/test_mtp_dense_device_kv_cache.py#L1-L120).

This is useful infrastructure, but it remains default-off because it has not yet
improved same-suite speed/acceptance.

## Runtime trace commands and artifacts

### llama.cpp CLI MTP debug trace

Command:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 12 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --log-file /tmp/hipengine-llamacpp-mtp-cli-debug.log \
  --log-verbosity 5
```

Artifact: `/tmp/hipengine-llamacpp-mtp-cli-debug.log`.

Caveat: `llama-cli --no-conversation` is not supported by this binary.  The
working CLI path is server/chat-style.  The debug trace had `task.n_tokens = 19`.
A `--no-jinja` probe used `task.n_tokens = 17` and still had 100% draft
acceptance, but generation timing collapsed to 0.88 tok/s, so it is not used for
performance comparison.

Aggregate llama.cpp result for the debug trace:

```text
draft acceptance = 1.00000 (8 accepted / 8 generated)
statistics draft-mtp: #calls(b,g,a) = 1 3 3,
  #gen drafts = 3, #acc drafts = 3,
  #gen tokens = 8, #acc tokens = 8,
  dur(b,g,a) = 0.004, 26.710, 0.001 ms
```

Per-draft-call table parsed from the debug log:

| call | history size before draft | drafted | accepted | top-1 draft IDs | corrective / sampled token | new token count |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 19 | 3 | 3 | `[579, 264, 7047]` | 1817 | 23 |
| 2 | 23 | 3 | 3 | `[25, 271, 16]` | 13 | 27 |
| 3 | 27 | 2 | 2 | `[220, 2972, 15771]` | 15771 | 30 |

Interpretation:

- `accepted == drafted` for every MTP call in the trace.
- The verifier call commits `accepted_draft_tokens + 1` visible tokens: 4, 4, and
  3 respectively.
- Visible output / verifier call is therefore `11 / 3 = 3.67`.
- Accepted draft tokens / verifier call is `8 / 3 = 2.67`.

### Target-AR parity trace (new primary blocker)

The cleanest apples-to-apples prompt mode is llama.cpp `--reasoning off`, which
renders the same 21-token text as hipEngine's retained `reasoning='off'` prompt:

```text
<|im_start|>user
Write a Python function that implements merge sort:<|im_end|>
<|im_start|>assistant
<think>

</think>

```

llama.cpp verbose prompt evidence:

```text
common_sampler_init prefill tail:
  248045 <|im_start|>, 74455 assistant, 198 \n,
  248068 <think>, 271 \n\n, 248069 </think>, 271 \n\n
task.n_tokens = 21
next token: 71093 '```'
```

Command/artifact:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 1 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --reasoning off \
  --verbose-prompt \
  --log-file /tmp/hipengine-llamacpp-reasoning-off-verbose-prompt.log \
  --log-verbosity 5
```

hipEngine target traces for the same 21-token prompt:

| hipEngine mode | First token after prefill | Next verifier target | Notes |
| --- | --- | --- | --- |
| retained default (`WMMA prefill + GEMV + graph`) | `760` = `The` | `198` = `\n` | `/tmp/hipengine-mtp-target-parity-off-default.json` |
| no WMMA prefill | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_wmma.json` |
| no WMMA/GEMV/graph/decode-repack | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_fast.json` |
| true token-serial `prefill(..., use_bulk=False)` probe | `1919` = `This` | n/a | top-1 from direct session probe |

None match llama.cpp's `71093` code-fence first token.  Therefore the first
confirmed divergence is **target AR prefill/decode/logit parity**, before MTP
draft acceptance.  The MTP acceptance gap is downstream of this target mismatch.

### hipEngine strict B3 trace

Command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --cycles 3 \
  --draft-n-max 3 \
  --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-b3-strict-trace.json
```

Artifact: `/tmp/hipengine-mtp-b3-strict-trace.json`.

Caveat: the hipEngine benchmark applies the Qwen chat prompt wrapper used by its
GGUF harness and reported `Prompt tokens: 21`; this is close but not byte-for-byte
identical to the llama.cpp CLI trace (`19` chat/server tokens).  The strict B3
numbers are still useful because the acceptance gap is large and consistent with
full-suite behavior.

Metrics:

```text
accept_per_draft     = 0.2222
accepted_per_output  = 0.4000
visible/cycle        = 1.6667
tokens_per_sec       = 33.38
speedup_vs_ar_visible= 0.598x
total_accepted       = 2 / 9 draft tokens
```

Per-cycle table:

| cycle | accepted / drafted | target samples | draft IDs | target rank in draft top-10 | visible output | target verify ms | MTP draft ms |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | 0/3 | `[198]` | `[803, 328, 760]` | `[None]` | 1 | 17.94 | 20.31 |
| 1 | 0/3 | `[17]` | `[760, 21397, 25]` | `[2]` | 1 | 18.00 | 19.51 |
| 2 | 2/3 | `[15, 15, 15]` | `[15, 15, 248046]` | `[1, 1, 2]` | 3 | 53.60 | 20.42 |

Interpretation:

- hipEngine's MTP top-1 is often wrong even when the target is near the top of
  the distribution (`target_rank_in_draft_top10 = 2` in cycles 1 and 2).
- This is exactly why root-top40 raised `accepted_per_output` while strict
  `draft_acceptance` stayed extremely low: the target is often in the top-K but
  not the actual draft token.
- B3 strict verification currently commits only `5/3 = 1.67` visible tokens per
  verifier call, far below llama.cpp's `3.67` in the debug trace.

### hipEngine retained/default and device-KV smoke context

Retained root-top40 B1 smoke artifact: `/tmp/hipengine-mtp-with-attn-smoke.json`

```text
accept_per_draft    = 0.0225
accepted_per_output = 0.4737
visible/cycle       = 1.9
tokens_per_sec      = 46.6
total_accepted      = 9 / 400 candidate-count denominator
```

Device-KV B1 smoke artifact:
`/tmp/hipengine-mtp-device-kv-smoke-fastcommit.json`

```text
accept_per_draft    = 0.0187
accepted_per_output = 0.4286
visible/cycle       = 1.75
tokens_per_sec      = 43.68
total_accepted      = 3 / 160 candidate-count denominator
KV rows             = 7 / 12
commit cost         = ~1.2-1.9 ms per accepted-row KV write
```

The device-KV path is much faster than prior host replay/prefix diagnostics, but
it did not reproduce llama.cpp's high B3 acceptance and remains default-off.

## What llama.cpp is doing that hipEngine is not yet doing

### 0. Target AR parity before speculation

llama.cpp and hipEngine must first agree on the target model's greedy token after
the prompt.  They currently do not.  For the same reasoning-off prompt tail,
llama.cpp picks code fence token `71093`; hipEngine picks `760`, `248069`, or
`1919` depending on prefill path.  This points to a target runtime issue, not an
MTP model-quality issue.

Likely places to investigate in order:

1. Prompt/output-row scheduling: llama.cpp decodes the 21-token prompt as a 17-row
   cached prefix plus a 4-row tail; hipEngine bulk/serial row selection may be
   sampling the wrong hidden row.
2. Qwen3.6 hybrid recurrent/Gated Delta Net state: fastpath toggles change the
   first sampled token, which means recurrent/prefill state is affecting target
   semantics.
3. LM-head/argmax parity: direct token-serial hipEngine top-10 does not contain
   llama.cpp's code fence token, so verify output logits against llama.cpp after
   the prompt.
4. Logit processors/biases: llama.cpp biases EOG tokens to `-inf`; confirm
   hipEngine has equivalent generation-time biasing.  This is unlikely to explain
   `71093` vs `760`, but should be checked.

Until this stage matches, MTP token acceptance is not the primary bug.

### A. Full draft-context lifecycle, not just K/V rows

llama.cpp's `process()` decodes verifier rows through `ctx_dft` and updates all
relevant draft-model state.  For Qwen35MoE MTP this primarily means attention K/V,
but it also means the exact graph scheduling, output IDs, and hidden-row selection
are controlled by the same decode path as `draft()`.

hipEngine now has device K/V row writes, but still drives MTP from a Python wrapper
that repeatedly uploads/downloads intermediates and manually chooses which rows to
commit.  It does not yet have the same transactional draft context abstraction.

**Roadmap item:** add an in-tree `GGUFMTPDraftContext` owning device K/V, position,
pending hidden row, accepted verifier rows, and rollback/commit state.  The
benchmark should call this object rather than open-coding row bookkeeping.

### B. B>1 transactional semantics

llama.cpp B3 drafts can be generated, verified, accepted, and rolled forward while
preserving draft context.  hipEngine's `--mtp-device-kv-cache` intentionally
rejects `--draft-n-max != 1` today because we do not yet have safe rollback for
unaccepted draft rows.

**Roadmap item:** implement draft transaction:

1. Save `kv_len_before_draft`.
2. Append draft rows while generating B tokens.
3. Verify target batch.
4. Roll back unaccepted draft rows.
5. Commit accepted target rows and the corrective pending hidden row exactly like
   llama.cpp's `accept()`.

### C. Numeric parity of MTP logits has not been proven

The largest unexplained delta is that llama.cpp's top-1 MTP tokens are accepted
in the debug trace, while hipEngine's top-1 tokens often miss even when the target
is rank 2.  That could be due to:

- hidden seed captured at the wrong point,
- RoPE position/context count mismatch,
- missing or stale MTP K/V context,
- output ID / row selection mismatch,
- quantized GEMV/layout differences in attention, FFN, or shared head,
- sampler/logit post-processing differences.

**Roadmap item:** create a one-step parity harness that records, for the same
prompt/token position:

- token ID entering MTP,
- `pending_h` checksum/norm,
- K/V cache length,
- MTP top-10 logits/tokens,
- `h_nextn` checksum/norm,
- accepted prefix length.

Without editing the read-only llama.cpp checkout, we can only get aggregate and
some debug candidate logs.  For true tensor parity we need either a temporary
instrumented llama.cpp worktree/copy or a local patch that is not committed to the
reference repo.

### D. hipEngine wrapper overhead is still high

Even when B1 device K/V is active, hipEngine draft time is ~8.5 ms/cycle on the
smoke.  The source-level issue is that the correctness-first Python wrapper still
allocates/copies many intermediates.  The WORKLOG follow-up already identified:

- remove Q/gate D2H split,
- avoid Q6_K temporary H2D uploads in attention,
- keep more MTP intermediates resident,
- move from Python orchestration to one or a few persistent launch wrappers.

**Roadmap item:** after numeric parity, port MTP attention+FFN+head into a real
resident path.  Do not optimize the wrong math first.

### E. Root-topK is not a substitute for draft quality

Root-top40 showed the target is frequently *near* the draft distribution, but the
speculative algorithm commits actual draft tokens.  llama.cpp's debug trace has
true top-1 acceptance.  hipEngine's root-topK acceptance is therefore a diagnostic
for rank quality, not a path to B3/B5 break-even.

**Roadmap item:** keep root-topK as diagnostic only.  Promote only changes that
raise strict top-1 chain acceptance and committed tokens/verifier call.

## What we can adopt from llama.cpp

| llama.cpp behavior | Adopt in hipEngine? | Notes |
| --- | --- | --- |
| `pending_h` / `verify_h` lifecycle | Yes | We already use a similar concept; needs parity checksum tests. |
| Draft context with persistent MTP K/V | Yes | Started with default-off B1 dense device cache; must become transactional and resident. |
| `process()` verifier-row mirroring | Yes | Need a resident `process_verifier_rows()` equivalent. |
| B>1 rollback/commit semantics | Yes | Required before meaningful MTP speedups. |
| `p_min` early stop | Yes, diagnostic first | We already have `--draft-p-min`; tune after top-1 parity. |
| Backend sampling | Maybe | llama.cpp logs backend TOP_K support missing on ROCm in this run; hipEngine top-k is already explicit. |
| Chat/server prompt handling | No as-is | hipEngine benchmark prompt protocol must stay fixed and anti-gaming compliant. |
| Loading full model twice for MTP | No | Must keep hipEngine torch-free/lean and use in-model MTP weights only. |

## Prioritized roadmap to effective MTP

### Phase 0 — target AR parity on one prompt

1. Reproduce llama.cpp's 21-token reasoning-off prompt exactly.
2. Add a hipEngine target-only trace that emits:
   - prompt token IDs,
   - chunking/prefill schedule,
   - final hidden-row index sampled,
   - top-20 target logits after prefill,
   - first generated token.
3. Instrument a temporary llama.cpp copy or use verbose prompt + a small tensor
   dump to get the same target top-20 logits.
4. Fix target parity before changing MTP acceptance logic.

Success criterion: hipEngine target prefill chooses `71093` for the documented
reasoning-off prompt, matching llama.cpp, under the narrowest correctness-first
path.  Then optimize back toward the retained fast path.

**2026-06-25 status:** achieved for both correctness-first and retained fast
paths.  The blocker was Qwen3.5 linear-attention GDN K-head broadcast semantics:
llama.cpp/GGML maps value head `v_head` to key head `v_head % num_k_heads`, while
hipEngine inherited the grouped `v_head / repeat` mapping.  After switching the
GDN decode/prefill kernels and CPU replay oracles to the interleaved mapping, the
same 21-token reasoning-off prompt has `initial_prev_token=71093`.  A follow-up
12-token greedy target trace also matches llama.cpp exactly:
`[71093, 12305, 198, 727, 10562, 17885, 10620, 25, 1103, 8, 1411, 1103]`
(decoded as a Python code fence followed by `def merge_sort(arr: list) -> list`).
The single-prompt B3 smoke improves from
the prior `2/9` accepted drafts / `5` visible output tokens to `7/9` accepted
drafts / `10` visible output tokens.

Evidence command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --prompt-reasoning off --cycles 3 --draft-n-max 3 --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-target-parity-final-c3.json
```

### Phase 1 — exact MTP trace parity on one prompt

1. Add a hipEngine trace mode that emits per-step JSON:
   - prompt token IDs,
   - previous token,
   - position,
   - pending hidden norm/checksum,
   - MTP KV length,
   - MTP top-10 IDs/logits/probs,
   - target samples,
   - accepted prefix length,
   - committed output tokens.
2. Produce a temporary instrumented llama.cpp copy or local patch that emits the
   same fields from `common_speculative_impl_draft_mtp`.
3. Compare the first divergence.
4. Fix math/state mismatches before doing more performance work.

Success criterion: on the same prompt/token positions, hipEngine and llama.cpp
produce the same MTP top-1/top-K tokens for at least the first several draft
steps, or we can explain every difference.

### Phase 2 — B3 transactional device KV

1. Promote the B1 device cache into a draft-context object.
2. Add rollback/commit around B>1 draft rows.
3. Validate with a CPU/synthetic fixture and then a GGUF smoke.
4. Run strict B3, no root-topK, same prompt.

Success criterion: strict B3 `accepted_draft_tokens / generated_draft_tokens`
substantially improves over the old `2/9 = 22.2%` smoke and approaches the
llama.cpp debug trace on the same prompt.

**2026-06-25 status:** achieved for the diagnostic llama.cpp-lifecycle path.  The
missing piece after target parity was the draft model context lifecycle: replay
the shifted prompt rows into a device-resident MTP KV cache, keep the cycle-start
row, roll back rejected speculative rows, and commit accepted rows with
verifier-derived target hidden seeds.  With `--mtp-context-replay`,
`--mtp-device-kv-cache`, `--draft-n-max 3`, and `--root-topk-accept 1`, the same
single-prompt smoke reaches `9/9 = 100%` accepted drafts and `12` visible output
tokens over three verifier calls.

### Phase 3 — full-suite strict acceptance before speed claims

Run `mtpbench-code-general-ja.jsonl` in strict mode and record:

- accepted draft tokens / verifier call,
- visible output tokens / verifier call,
- strict draft acceptance,
- rank histogram for target token in MTP top-K,
- raw tok/s.

Success criterion: committed tokens/verifier call rises enough that speed work is
worthwhile.  If strict acceptance remains low, return to Phase 1.

### Phase 4 — performance optimization only after parity

Once strict acceptance is credible:

- fuse resident MTP attention/FFN/head launches,
- eliminate host-side intermediate copies,
- pre-upload/cache Q6_K weights and scratch buffers,
- replace sequential target verification with a rollback-safe block verifier,
- profile verifier MoE grouping/budgeting to reduce `eta`,
- revisit B2/B3/B5 economics.

**2026-06-25 status:** first draft-side performance wins landed, and a
rollback-safe target continuation block verifier now exists, but performance
parity is still blocked by verifier kernel shape.  Batching accepted-row MTP KV
commit into one `kv_write_only` pass improved the corrected B3 merge-sort smoke
from `41.7` to `42.3 tok/s` (`15/15` strict accepts over five cycles).  A
hot-token draft LM-head cap of `32768` improved the same one-step-graph smoke to
`44.5 tok/s` with unchanged `15/15`, but it is prompt-sensitive and remains
diagnostic until full-suite validation.  The new `--target-block-verify` path
snapshots linear recurrent state, runs the target over `[prev]+drafts` as a
continuation block, records target IDs + FP32 hidden seeds, and restores/replays
the consumed prefix on partial accepts.  Its first version was exact (`15/15`) but
slow on the B3+32k smoke (`37.8 tok/s`, verifier `~90 ms/cycle`) because the
selected/WMMA prefill kernels are the wrong shape for tiny B.  The verifier now
defaults to the GEMV prefill fallback internally (`--no-target-block-wmma-prefill`)
while leaving normal prompt prefill WMMA enabled; that lifts the same B3+32k
smoke to `48.1 tok/s` with unchanged `15/15` and verifier `~61-66 ms/cycle`
(except variance on late cycles).  B5 remains unattractive because a partial
rollback cycle costs hundreds of ms in the generic restore/replay path.  The next
material parity task is a dedicated small-B linear-attention/rollback kernel, not
more one-step graph replay tuning and not selected-prefill for tiny verifier
blocks.

Success criterion: same-protocol full-suite row improves all three: raw weighted
decode tok/s, accepted/output, and strict draft acceptance.

## Bottom line

llama.cpp is not just using a wider candidate set.  It is running a real target
and MTP draft context with verifier-row processing, persistent draft K/V state,
hidden-row handoff, and B>1 accept/rollback semantics.  In the short debug trace
it commits `3.67` visible tokens per verifier call with `100%` strict draft
acceptance.

hipEngine now matches llama.cpp's documented reasoning-off target AR trace and,
with the llama.cpp-style context replay + device MTP KV lifecycle, reaches strict
B3 `9/9` (and `15/15` over five cycles) on the merge-sort smoke.  Correctness
parity is therefore solved.  The remaining gap is purely performance: ~48 vs
~90 tok/s (~1.9x) on gfx1151, and ~7–8x of that lives in target verification of
the 4-row continuation block — specifically the 30 Python-dispatched
linear-attention layers (~82 ms) that llama.cpp runs inside one fused GGML graph.
The highest-ROI next step is a captured-graph / C-level small-B target
continuation path, then a resident MTP draft path, then full-suite validation
against a true no-MTP AR baseline before any retained speed claim.  These remain
single-prompt diagnostics, not benchmark rows.
