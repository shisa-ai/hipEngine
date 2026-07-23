# hipEngine Serving-Latency Optimization Plan

Status: 2026-07-23 (active; S0-S4 complete, S5 pending).

Scope: resident Poolside Laguna S 2.1 Q4_K_M serving on the local Ryzen AI
MAX+ 395 / Radeon 8060S (`gfx1151`) host, starting with exact greedy `c=1`.
The primary metric is **useful-content time to first token (TTFT)**; complete
request latency and loaded-idle memory are secondary metrics.

This is the live punchlist for request-path work outside the model's matrix
prefill kernels. The separate Laguna AR prefill campaign remains authoritative
in [`LAGUNA.md`](LAGUNA.md) and is being executed independently. This campaign
must not edit or retune that agent's selected-expert, source-F16, attention, or
chunk kernels unless a later serving profile establishes a new boundary.

Cross-links:

- [`LAGUNA.md`](LAGUNA.md) — model contract, AR prefill/decode evidence, and
  active AR-O1 through AR-O6 kernel campaign.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence, anti-gaming, timing, and rollup
  requirements.
- [`TESTING.md`](TESTING.md) — RED/GREEN workflow and deterministic gates.
- [`PLAN.md`](PLAN.md) — resident scheduler, batch-shaped ABI, and prefix-cache
  architecture.
- [`REFACTOR.md`](REFACTOR.md) — temporary route/flag removal ledger.
- [`../benchmarks/README.md`](../benchmarks/README.md) — retained performance
  scoreboard, including Qwen GGUF prefix-reuse precedent.

---

## 1. Goal and timing definitions

Optimize the complete resident request path:

```text
HTTP request accepted
  -> validate/auth/session lookup
  -> render chat prompt
  -> encode/count/reasoning/admission
  -> generation queue
  -> acquire/reset resident model state
  -> model prefill + first-token LM head/argmax
  -> stop-safe incremental detokenization
  -> first useful content SSE event
```

Definitions:

- **Useful-content TTFT:** client submission to the first non-empty completion
  text, `reasoning_content`, or tool-call argument/name delta. An OpenAI role-
  only SSE frame is not a token and must not stop the TTFT clock.
- **Backend prefill:** `LagunaGGUFResidentSession.prefill()` start through its
  synchronized first-token argmax. It excludes model loading, tokenizer wall,
  generator session construction, server queueing, and SSE serialization.
- **Queue latency:** request admission to the first scheduler-owned model step.
- **Session preparation:** construction or reset of request-owned KV/scratch
  state after resident weights are available and before prefill begins.
- **End-to-end latency:** client submission through the terminal response or
  `[DONE]`; report blocking and streaming separately.
- **Cold start:** process/model readiness. It remains useful operational data
  but is not credited to resident TTFT.

Every result must state whether it measures direct in-process, FastAPI
in-process, or real localhost Uvicorn/client wall. Never compare unlike timing
scopes as a speed ratio.

---

## 2. Current evidence and latency budget

### 2.1 Model work remains dominant

The retained current-main Laguna route reports **50.389 prefill tok/s**, median
TTFT **1.620 s** on the canonical 68-122-token suite, and **16.384 decode tok/s**.
The current 128-row profile attributes approximately **56.78%** of kernel time
to selected experts and **33.40%** to source-F16 projections. Kernel-span minus
kernel-sum is only about **0.1-0.34%** on retained prefill profiles.

Consequences:

- the other agent's AR-O1/O2 kernel campaign is still the largest idle-request
  TTFT lever;
- graph/host submission fusion is not reopened merely because this serving
  campaign exists; and
- a 10% reduction in the current 1.620-second model-prefill wall is worth about
  162 ms, larger than all currently identified idle host micro-costs combined.

### 2.2 Per-request Laguna session construction is measurable

A local, non-retained diagnostic at current main loaded the exact Q4_K_M model
once from the repacked cache, then constructed and closed six borrowing
`LagunaGGUFResidentSession` instances with cached libraries. The warm five-
sample medians were:

| Stage | Result |
| --- | ---: |
| constructor return | 31.334 ms |
| constructor plus explicit device synchronize | **31.426 ms** |
| explicit synchronize remainder | 0.093 ms |
| close | **5.033 ms** |
| existing `reset_state()` host submission | 0.296 ms median |

Each temporary session added **387,482,684 bytes** (369.53 MiB) across **323
tracked allocations**, then recovered exactly. The owner reported
77,125,390,396 resident bytes, model load took 50.075 s and was excluded from
the session samples, and final tracked ownership returned to zero.

Diagnostic command:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=1 \
  PYTHONPATH=. uv run python -u /tmp/laguna_session_latency_probe.py
```

This is candidate-selection evidence, not a retained performance claim: the
probe script is not yet committed and `reset_state()` was timed as host
submission rather than synchronized request wall. S1 must add a checked-in
harness and measure complete first-token latency before promotion.

### 2.3 Pre-S2 public chat encoded the same prompt repeatedly

The pre-S2 baseline performed prompt-wide token work at multiple boundaries:

1. thinking-budget/context calculation while rendering chat;
2. reasoning-open detection before live chat streaming;
3. context admission validation;
4. model generation tokenization; and
5. another remaining-context count when `max_tokens` is omitted.

Thus the baseline encoded a normal chat prompt at least three times for blocking
and four times for streaming; implicit token budgets could add another pass. A
completion request normally counted once for admission and encoded again for
generation. The retained S2 measurement below observes six complete prompt
encodes per blocking or streaming chat request in its exact isolated scope.

The reconstructed HF encoder's measured median at 4,096 tokens is **3.428 ms**.
Request-local reuse therefore has a directional ceiling of roughly **6.9 ms**
for explicit-budget blocking chat, **10.3 ms** for explicit-budget streaming,
and **13.7 ms** when streaming also recomputes an implicit budget. Short
canonical prompts save less than one millisecond. These are arithmetic bounds
from the measured tokenizer wall, not public HTTP measurements.

### 2.4 Multi-token stop handling can delay visible output

Laguna streaming currently retains `longest_stop_sequence - 1` generated tokens
before releasing any pending text, even when the pending suffix cannot be a
prefix of a configured stop sequence. At 16.384 decode tok/s, each unnecessarily
held token costs approximately **61.0 ms**. A four-token stop can therefore add
up to roughly **183 ms** before first useful content on a nonmatching output.
Default atomic EOT handling does not incur this multi-token hold.

### 2.5 Re-prefilling chat history is the largest repeated-turn opportunity

The current generic chat-session record retains transcript text, not resident
Laguna KV; capabilities truthfully report `resident_kv_commit=false`. Every
follow-up turn re-renders and re-prefills the complete visible transcript.

Directional avoided model wall at current measured rates is approximately:

| Exact reusable prefix | Avoidable Laguna prefill wall |
| ---: | ---: |
| 100 tokens | about 2.0 s at 50.389 tok/s |
| 256 tokens | about 5.1 s at 50.389 tok/s |
| 1,024 tokens | about 22.8 s at the retained 44.855 tok/s 1K rate |

This is an Amdahl estimate, not a retained continuation benchmark. The existing
Qwen GGUF exact p256+s1 precedent proves the architecture but not Laguna's
result: it moved continuation TTFT **249.269 -> 21.188 ms (11.765x)** on a
narrow guaranteed-hit route. Broader Qwen agentic radix experiments were
rejected when hit rate and snapshot cost outweighed reuse. Laguna therefore
starts with explicit stateful-session continuation, not a global default-on
radix cache.

### 2.6 The compatibility bridge makes queue latency unbounded by one token

Laguna currently enters `SubmitPollTextGenerator` through the compatibility
runner. One scheduler decode work item invokes a complete inner generation,
and `LagunaGGUFGenerator._token_steps()` holds the generator lock through
prefill and every output token. A later request can therefore wait for the
prior request's full response rather than one scheduler tick.

At the canonical median and 16.384 decode tok/s, a 32-token response is roughly
1.620 s TTFT plus 31 post-TTFT forwards (about 1.89 s). A request arriving
behind it can inherit about 3.5 s of queueing before its own model prefill. This
is a directional composition, not a measured concurrent Laguna server row.

---

## 3. Ordered task list

Task order is intentional. Finish and commit each validated logical unit before
starting the next. The pi task IDs are session-local coordination aids; the
stable `S*` IDs belong in commits, artifacts, and `WORKLOG.md`.

| ID | Pi task | Candidate | Expected scope | Status |
| --- | ---: | --- | --- | --- |
| S0 | #17 | Document path, definitions, evidence, telemetry, and gates | process only | **complete** |
| S1 | #18 | Pool/reset one generator-owned Laguna resident session | **31.800 ms setup and 34.877 ms direct TTFT saved** | **complete** |
| S2 | #19 | Render/encode once into request-local prepared prompt ownership | **6 -> 1 prompt encodes; 8.17/8.70 ms isolated 4K blocking/streaming TTFT saved** | **complete** |
| S3 | #20 | Prefix-aware stop-safe streaming holdback | **184.536/123.243 ms useful-content delay removed in deterministic nonmatch/failed-prefix lanes** | **complete** |
| S4 | #21 | Exact stateful Laguna KV continuation | **2.347/10.044/21.306 s saved at exact 128/512/1K hits; canonical chat 1.671 -> 0.412 s** | **complete** |
| S5 | #22 | Native scheduler-owned Laguna prefill/decode ticks | seconds under contention; c=1 exact first | pending |
| Q1 | #23 | Audit and port retained serving improvements to Qwen GGUF/PARO | only genuine Qwen deltas; do not duplicate native ownership | queued after S5 |

The separate model-prefill campaign runs in parallel. S1-S5 may consume its
new default kernels after those commits land, but must compare against the
then-current default rather than a stale prefill baseline.

---

## 4. S1 — Persistent/resettable Laguna session

### Implementation

- Move the exact AR `LagunaGGUFResidentSession` from request-local construction
  into `LagunaGGUFGenerator` ownership.
- Lazily create it after immutable resident weights are prepared; reuse it only
  under the existing c=1 lock.
- Reset request state before every new prompt while retaining KV/scratch/RoPE
  allocations and loaded libraries at stable addresses.
- Close the pooled session before shared weights in `LagunaGGUFGenerator.close()`.
- Keep DFlash provider target/drafter/cycle ownership isolated. Do not make a
  DFlash performance claim from an AR session-pool change.
- Failure, deadline, cancellation, empty-generation, and stop exits must leave
  the pooled state reusable or retire/recreate it explicitly before the next
  request.

### RED/GREEN gates

- two sequential requests initialize one session, reset once, produce the same
  IDs/text/telemetry as two fresh sessions, and close exactly once;
- an injected prefill/decode exception cannot leak stale position/KV into the
  next request;
- cancellation and EOT/stop completion leave the next request exact;
- public blocking and streaming retain identical IDs and terminal ownership;
- generator close frees the session and shared weights in safe order;
- the focused HIP test has an explicit no-ROCm skip guard.

### Measurement and telemetry

Add request timing fields with non-overlapping scopes:

- `session_prepare_ms`: constructor plus required synchronization, or reset plus
  stream ordering before prefill;
- `session_prepare_mode`: diagnostic metadata (`create`, `reset`, or
  `recreate_after_error`), not an additive numeric timing field;
- existing `prefill_ms`, `decode_ms`, and `tokenize_ms` retain their current
  meanings.

Use a checked-in cached-build probe. Alternate fresh/reused ordering where
possible, synchronize both arms equivalently, report at least five warm samples,
and require complete ID/state/lifecycle equality. Promotion requires every
paired reused sample to improve session preparation and complete request TTFT
to be non-regressive. Do not claim the current 31.426-ms constructor ceiling as
the realized win until that gate passes.

### Retained result (2026-07-23)

Revision `8ae07d693b6f98d6c44aae90090df6c6d77e8d78` passes the clean detached
hardware gate on the exact Q4_K_M model (`7da520c5...5753f`), gfx1151, BF16 KV,
4K capacity, chunk 128, and the 46-token frozen `no_thinking` Poolside prompt.
One warmup preceded five alternating synchronized samples per mode; model load
was excluded.

| Direct first-token scope | Fresh borrowing session | Pooled reset | Change |
| --- | ---: | ---: | ---: |
| Synchronized session preparation, median | 32.399 ms | **0.598 ms** | **-31.800 ms; 54.14x** |
| Prefill after preparation, median | 930.826 ms | 927.781 ms | -3.045 ms diagnostic |
| Preparation + prefill/argmax TTFT, median | 963.262 ms | **928.384 ms** | **-34.877 ms; -3.621%** |
| Fresh-session close after first token, median | 5.373 ms | deferred to owner close | removed per-request tail |

Every paired setup sample improves (30.723-33.015 ms saved), every first token
is exact ID `5887`, the first-token median is non-regressive, and all tracked
allocations recover to zero. `prepare()` now creates the pool during resident
readiness, so normal server requests enter the reset arm; lazy direct users pay
one initial create and then reset. Telemetry reports `session_prepare_ms` and
`session_prepare_mode`, and poisoned/cancelled/abandoned sessions fail closed to
`recreate_after_error`.

Artifact:
[`2026-07-23-gfx1151-laguna-session-pool.json`](../benchmarks/results/2026-07-23-gfx1151-laguna-session-pool.json).
Exact command:

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=1 \
  /home/lhl/hipEngine-main/.venv/bin/python3 -u scripts/laguna_session_pool_bench.py \
  /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf --backend hip_gfx1151 \
  --context-length 4096 --chunk-size 128 --warmups 1 --repetitions 5 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --repacked-cache /home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1 \
  --model-sha256 7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f \
  --output /tmp/2026-07-23-gfx1151-laguna-session-pool.json
```

---

## 5. S2 — Request-local prepared prompt

### Contract

Introduce a model-neutral, request-local prepared prompt record (name may change)
containing at least:

```text
rendered text, exact token IDs, token count, tokenizer wall, tokenizer identity
```

It is not a process-wide or unbounded cache. Public users may continue to submit
text or raw token IDs. The record exists only from request preparation through
completion/reclaim.

### Required reuse

- chat thinking-budget and context calculations consume the prepared count;
- Poolside reasoning-open detection consumes prepared IDs or a proven exact
  rendered-suffix fast path;
- context admission uses the prepared count;
- generation receives the same IDs without re-encoding;
- usage accounting uses exact prompt IDs/count without decoded-text
  retokenization; and
- raw ID prompts retain `tokenize_ms=0`.

Avoid rendering the same deterministic model prompt twice merely to discover
that the thinking control did not change. Keep generic role/tool-transcript
validation, but separate validation from discarded duplicate string assembly
when tests prove the split exact.

### Gates

- count actual encoder calls: exactly one per text prompt for blocking and live
  streaming, with explicit and omitted `max_tokens`;
- exact rendered bytes, prompt IDs, usage, output IDs, stop/tool/reasoning
  behavior, and context-overflow errors match the old path;
- no tokenizer or prompt cache survives request reclaim;
- measure 128/512/1K/4K text plus the ten-prompt suite; report preprocessing and
  useful-content TTFT separately.

Telemetry should expose `render_ms`, `prompt_encode_ms`, and
`admission_prepare_ms`; `tokenize_ms` may remain as a compatibility alias only
if its ownership is documented and it is not double-counted.

### Retained result (2026-07-23)

Revision `0081d150c08a95423f29fec8fd26779f53c8f730` introduces an
immutable request-local `PreparedPromptInput` with rendered text, exact IDs and
count, tokenizer identity, and render/encode/admission timing. One owner is
shared across duplicate `n` rows and reclaimed with the request. Thinking
budgeting, context admission, Poolside reasoning-open detection, generation,
and usage all consume its exact IDs/count. Text/raw-ID fallback remains public,
and no process-wide tokenizer or prompt cache was added.

The clean comparison runs the same checked-in FastAPI TestClient harness against
baseline `88205779d5f9d69d4393060d89718ae935de7869` and the candidate on the
Ryzen AI MAX+ 395. It uses the exact Laguna Q4_K_M GGUF only for tokenizer
metadata and an immediate deterministic fake model (`ready`, generated ID 1),
so these values isolate rendering, tokenization, admission, usage, HTTP/SSE, and
client parsing. They are **not Laguna model TTFT** and use no GPU.

| Mode | Rendered prompt | Prompt encodes | Prompt encoder wall, median | Useful-content TTFT, median | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| blocking | 128 | 6 -> **1** | 0.585 -> **0.094 ms** | 1.715 -> **1.337 ms** | **-0.377 ms** |
| blocking | 512 | 6 -> **1** | 1.892 -> **0.303 ms** | 2.897 -> **1.924 ms** | **-0.973 ms** |
| blocking | 1,024 | 6 -> **1** | 3.676 -> **0.565 ms** | 4.905 -> **2.691 ms** | **-2.214 ms** |
| blocking | 4,096 | 6 -> **1** | 13.578 -> **2.172 ms** | 15.334 -> **7.167 ms** | **-8.166 ms** |
| streaming | 128 | 6 -> **1** | 0.472 -> **0.099 ms** | 1.373 -> 1.443 ms | +0.069 ms, overlapping noise |
| streaming | 512 | 6 -> **1** | 1.870 -> **0.290 ms** | 2.961 -> **1.593 ms** | **-1.369 ms** |
| streaming | 1,024 | 6 -> **1** | 3.716 -> **0.550 ms** | 5.151 -> **2.360 ms** | **-2.791 ms** |
| streaming | 4,096 | 6 -> **1** | 13.488 -> **2.133 ms** | 15.483 -> **6.786 ms** | **-8.697 ms** |

Across 500 requests per mode over the canonical ten-prompt suite, pooled median
prompt-encoder wall changes **0.357 -> 0.065 ms (-81.93%)** blocking and
**0.340 -> 0.063 ms (-81.39%)** streaming. Useful-content TTFT changes
**1.265 -> 0.998 ms (-21.12%)** and **1.244 -> 0.964 ms (-22.49%)**,
respectively. The isolated 128-token streaming p10-p90 ranges overlap; it is not
a resolved regression and is reported rather than discarded. All exact prompt
usage counts match, the candidate performs one prompt encode in every sample,
and focused generation/server validation reports 125/527 passes.

`tokenize_ms` is now explicitly the compatibility alias of
`prompt_encode_ms`; `render_ms` and `admission_prepare_ms` are separate and not
additive aliases. Artifact:
[`2026-07-23-laguna-prepared-prompt-fastapi.json`](../benchmarks/results/2026-07-23-laguna-prepared-prompt-fastapi.json).
The two raw outputs have SHA-256 `22fa854d...9ad5d768` and
`badd0a26...8946df9`; exact commands are recorded in the artifact.

---

## 6. S3 — Prefix-aware stop-safe streaming

Maintain a pending token suffix and the token-prefix trie (or equivalent
bounded matcher) of configured stop sequences. After each generated token:

1. stop immediately and suppress the exact matched suffix when a complete stop
   sequence is present;
2. retain only the longest pending suffix that is still a prefix of at least
   one possible stop sequence; and
3. emit all earlier pending tokens immediately through the incremental UTF-8
   decoder.

Do not change blocking stop semantics. Atomic EOT/EOS/control-token suppression,
`min_tokens`, `ignore_eos`, overlapping stops, duplicate stops, and caller stop
IDs remain exact.

Tests must cover:

- nonmatching first token with a long stop (first token emits immediately);
- partial then failed prefix (safe prefix flushes at the earliest token);
- exact, overlapping, suffix-contained, and shared-prefix stops;
- a stop sequence split across byte-BPE UTF-8 pieces;
- terminal max-length flush, cancellation, EOT, and blocking/stream equality;
- a deliberately delayed fake decoder proving useful-content TTFT improves by
  the avoided token intervals rather than merely changing chunk count.

No user-selected stop string or fixed token ID may be special-cased.

### Retained result (2026-07-23)

Revision `71f2af038cf5eea88f1997d178d815cfaad15681` precomputes the
bounded set of all proper configured stop-token prefixes. On each nonterminal
step it retains only the longest pending suffix in that set and emits every
preceding token through the existing incremental UTF-8 decoder. Complete stop
matches still terminate and suppress the exact matched suffix before this path;
blocking output semantics are unchanged.

The clean host-only integration probe compares baseline
`a95adcac82d8ae0b018fe1167b5108422afa47a9` with the candidate through the
production Laguna streaming generator. A deterministic fake resident session
replays four exact IDs with 61 ms between decode tokens, approximating the
retained 16.384 tok/s rate. It has two warmups and 20 measured samples per arm.
No GPU, model execution, or tokenizer wall is included; this measures emission
timing, not throughput.

| Four-token workload with four-token stop | Baseline useful TTFT | Prefix-aware useful TTFT | Saved | Complete E2E |
| --- | ---: | ---: | ---: | ---: |
| First token cannot match stop | 184.738 ms | **0.203 ms** | **184.536 ms (99.89%)** | 184.832 -> 184.108 ms |
| First token is a prefix, second disproves it | 184.695 ms | **61.452 ms** | **123.243 ms (66.73%)** | 184.776 -> 184.029 ms |
| Exact stop | no visible content | no visible content | exact suppression retained | 184.767 -> 183.955 ms |

All visible text, generated IDs, and terminal finish details match baseline.
The complete 30-case Laguna generation file covers nonmatch, partial failure,
exact/shared/overlapping/suffix-contained stops, request order, `min_tokens`,
max-length flush, split byte-BPE UTF-8, EOT, cancellation/abandonment, and
blocking/stream reconstruction. Artifact:
[`2026-07-23-laguna-prefix-aware-stop-streaming.json`](../benchmarks/results/2026-07-23-laguna-prefix-aware-stop-streaming.json).
Raw SHA-256 values are `e36a0079...d2c0099f` and
`adfcd691...a78bd32`; exact commands are recorded in the artifact.

---

## 7. S4 — Exact stateful Laguna KV continuation

Start with explicit chat-session ownership where an exact prefix hit is
structural. Do not enable general radix matching by default.

### State contract

- bind one bounded Laguna session/KV owner to a server chat session and exact
  tokenizer/model/revision/context identity;
- record the exact committed token sequence and the point through which model
  state/KV has been processed;
- account for Laguna's generation loop returning a sampled token before that
  final token has necessarily been forwarded into KV;
- append only the newly rendered suffix after verifying exact token-prefix
  equality;
- preserve all 12 global KV families and all 36 SWA rings, absolute positions,
  eviction metadata, and 511/512/513 wrap behavior;
- on any mismatch, unsafe cache action, context truncation/reset, sampling mode
  incompatibility, cancellation ambiguity, or ownership error, fail closed to
  ordinary full prefill;
- cap retained sessions/bytes with explicit LRU/TTL reclamation and truthful
  capability/metrics reporting.

### Gates

- continuation output, complete logits at declared checkpoints, final hidden,
  global/SWA KV metadata and live BF16 rows are exact against full transcript
  prefill;
- final-token pending/processed cases, stop/EOT, tool turns, thinking turns,
  context clear/truncate/new-session, disconnect, timeout, and server shutdown
  pass;
- retained state cannot cross auth principal, session ID, model revision,
  tokenizer identity, or cache-action boundary;
- repeated load/run/evict/close returns tracked and GTT ownership to the expected
  baseline;
- benchmark guaranteed-hit 128/512/1K prefixes plus multi-turn canonical chat,
  reporting avoided tokens, fallback reason, memory, TTFT p50/p95, and exactness.

Only after this lane is positive should Laguna evaluate shared-prefix radix
snapshots across unrelated requests.

### Retained result (2026-07-23)

Production revision `ccc24292cf60c95ec36fc757d5d5ac2f07ca0ba3` retains one
bounded generator-owned KV continuation slot for an explicit chat session. The
opaque key is SHA-256 over auth principal plus session ID; entries expire after
900 seconds and are replaced on the next incompatible request. Reuse requires
an exact processed-token prefix, a nonempty suffix, matching session/KV
position, and a safe transcript commit mode. The retained prefix is exactly
`prompt + generated[:-1]`; the final sampled-but-unprocessed token must be the
first matching continuation suffix token. Every mismatch fails closed to reset
plus full prefill.

Clean measured revision `804e9484f3da0031628805f5bbef62a43badffaa` runs
the production Q4_K_M generator on Radeon 8060S/gfx1151, BF16 KV, 4K capacity,
chunk 128, natural-corpus token prefixes, one warmup, and two measured samples.
Each nine-token continuation is compared with an identical reset/full-prefill
control:

| Exact reusable prefix | Full control wall | Resident reuse wall | Saved | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2,607.195 ms | **260.699 ms** | **2,346.496 ms** | **10.00x** |
| 512 | 10,347.413 ms | **303.268 ms** | **10,044.145 ms** | **34.12x** |
| 1,024 | 21,619.917 ms | **314.255 ms** | **21,305.662 ms** | **68.80x** |

The canonical `code_merge_intervals` rendered-chat gate retains 72 processed
IDs from a 69-token first prompt plus three of four generated IDs. The 89-token
follow-up moves **1,670.692 -> 411.907 ms (4.06x; 1,258.784 ms saved)** and
produces the same next ID `1172`.

For the first measured sample at every synthetic shape and for canonical chat,
the reuse and full-control arms have byte-identical SHA-256 over all
**277,434,816 copied bytes** of 12 global plus 36 SWA K/V payloads and live-span
metadata; session/KV positions are exact at 136/520/1,032 and 88 with no pending
rows. Every next ID agrees, reuse telemetry reports the exact prefix count, and
tracked allocations recover from zero to zero. Model load (49.944 s) is
excluded. Artifact:
[`2026-07-23-gfx1151-laguna-stateful-kv.json`](../benchmarks/results/2026-07-23-gfx1151-laguna-stateful-kv.json),
SHA-256 `b21e26a4...72d7a8`. This is explicit-session reuse only; global radix
matching remains off.

---

## 8. S5 — Native resident scheduler integration

Replace the compatibility runner's whole-generation decode work item with a
Laguna runner implementing scheduler-owned transitions:

- admission reserves a stable request/session slot;
- prefill consumes bounded prompt chunks and commits canonical KV;
- each decode work item advances one target token per ready request;
- generated chunks route by stable request ID through bounded queues;
- cancellation, deadline, EOS/stop, and reclaim occur between model ticks;
- a later request can be admitted and begin prefill without waiting for an
  earlier request's complete decode;
- the initial path remains exact `c=1`; c>N packed execution is a later,
  separately gated throughput extension.

The runner must use the existing batch-shaped scheduler/KV contracts and must
not add model/backend/quant branches to generic engine code.

### Gates

- independent c=1 output/state/KV equivalence for blocking and streaming;
- delayed-arrival test proves request B begins prefill before request A's long
  decode completes;
- `protect_ttft`, `protect_decode`, and `fair` policies report truthful work
  order; no policy is promoted from a single load shape;
- queue p50/p95, useful TTFT p50/p95, ITL p50/p99, end-to-end p50/p95, active
  occupancy, cancellation acknowledgement, and ownership all appear in the
  retained server artifact;
- full pressure/overload/recovery/soak gates pass before changing the package
  default.

---

## 9. Promotion and correctness policy

Every S1-S5 change must satisfy all applicable items:

1. **Exactness:** complete generated IDs for deterministic cases, required
   logits/hidden/KV state, stop/reasoning/tool output, and blocking/stream
   reconstruction match the current default.
2. **No benchmark gaming:** use the full canonical category/heldout suite where
   model output can change; never tune on fixed prompt/token IDs or stop strings.
3. **Torch-free hot path:** no `import torch` in `LLM.generate()` reachability.
4. **Registry and ownership boundaries:** no backend/quant branches in generic
   engine/dispatch; every retained allocation has one explicit owner.
5. **ROCm CI safety:** GPU/HIP tests skip cleanly when HIP is unavailable.
6. **Narrow then broad validation:** run focused tests first. Ask before
   repeating an equivalent benchmark expected to exceed five minutes.
7. **Evidence:** model, quant, workload, hardware, exact command, timing scope,
   warmups/repetitions, result, correctness, memory, and source revision travel
   together.
8. **Rollup:** an accepted performance change updates `WORKLOG.md`, a compact
   result under `benchmarks/results/`, `benchmarks/README.md` and
   `benchmarks/CHANGELOG.md`; rejected measured candidates receive a compact
   rejection artifact when they informed the plan.
9. **Default policy:** exact non-regressive wins become default unless a concrete
   blocker is recorded. Temporary flags/routes get a removal trigger in
   `REFACTOR.md`.

---

## 10. Required timing payload

By S2, a useful streaming response should make the following ownership
reconstructable without overlapping numeric scopes:

```json
{
  "render_ms": 0.0,
  "prompt_encode_ms": 0.0,
  "admission_prepare_ms": 0.0,
  "queue_ms": 0.0,
  "session_prepare_ms": 0.0,
  "prefill_ms": 0.0,
  "first_emit_after_prefill_ms": 0.0,
  "useful_ttft_ms": 0.0,
  "decode_ms": 0.0,
  "request_total_ms": 0.0
}
```

The concrete API may keep server-wall fields separate from backend
`GenerationTelemetry`; the invariant is that field ownership and clocks are
explicit. Role-only SSE must not populate `useful_ttft_ms`. Queue and useful
TTFT must be measured from request/server clocks, while GPU/backend stages use
their documented synchronized host boundaries.

---

## 11. Explicit non-goals

- Do not duplicate the active Laguna AR prefill kernel campaign here.
- Do not credit model load/readiness improvements to resident TTFT.
- Do not enable DFlash automatically or call verifier-derived rows an AR
  baseline.
- Do not enable a global Laguna radix cache before explicit-session reuse is
  exact, bounded, and positive.
- Do not add graph replay solely to remove a sub-percent prefill residual.
- Do not count an SSE role frame, empty delta, or hidden reasoning marker as a
  useful first token.

---

## 12. Qwen follow-up after Laguna S5

Task Q1/#23 is deliberately blocked on S5 so the Laguna campaign finishes one
coherent ownership transition before shared server surfaces move again. Qwen
GGUF and PARO already use native resident model runners; this follow-up is an
audit/port lane, not permission to reimplement working Qwen machinery.

| Serving optimization | Current Qwen status | Follow-up action |
| --- | --- | --- |
| Sidecar-free HF GGUF tokenizer | complete for Qwen GGUF | Retain; no port work. PARO/non-GGUF tokenizers keep their model-native boundary. |
| Request-local prepared prompt | shared server path is active | Verify `render_ms`, `prompt_encode_ms`, and `admission_prepare_ms` survive Qwen native scheduler ownership; currently tokenizer timing has the strongest direct coverage. |
| Resident session pooling | already native in Qwen GGUF/PARO | Do not port Laguna S1 or add a second pool. Confirm readiness/reclaim telemetry only. |
| Prefix-aware multi-token stop streaming | Laguna implementation only | Audit Qwen token/chunk emission for blanket longest-stop holdback. Port the bounded proper-prefix matcher only if the same delay exists, with Qwen blocking/stream exactness tests. |
| Explicit auth-scoped chat KV continuation | Qwen has resident/radix prefix snapshots, not Laguna's one-slot owner | Map explicit chat-session guarantees onto existing Qwen snapshot ownership. Do not duplicate KV storage; require principal/session/model/tokenizer scoping, exact prefix/state gates, bounded reclamation, and fail-closed fallback. |
| Native scheduler-owned prefill/decode ticks | already implemented for Qwen GGUF/PARO | Use Qwen as the architectural precedent for Laguna S5; no Qwen port is expected. Recheck queue/TTFT/ITL telemetry parity after shared S5 changes. |

Qwen promotion evidence must use the relevant Qwen model/quant/hardware and
cannot inherit Laguna's millisecond or state-reuse numbers. Record Qwen baseline
and candidate separately, preserve existing radix default policy, and keep only
exact, same-suite non-regressive changes.
