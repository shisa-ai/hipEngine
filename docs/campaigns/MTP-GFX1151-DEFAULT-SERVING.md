---
status: closed
owns: The acceptance requirements, final repair status, validation tables, and limitations for the gfx1151 default MTP serving campaign.
---

# Task: MTP on by default, end to end, for every model on this gfx1151 host, through `hipengine serve`

## Final repair status — 2026-10-17

The tracked repair tasks and the corrected **four-model** public acceptance
matrix are complete at `af024e122`. Concrete defects found during this work
were repaired and validated. This is not a claim that the original default-on
objective below is fully satisfied: Qwen MoE automatic selection still uses AR,
and Qwen3.8-Flash-Next is not included in this four-model validation packet.
It is also not a full production numerical-quality verdict or a speedup claim.

The lead subsequently authorized publishing `gfx1151-mtp-default`, merging it
into `main`, and pushing `origin/main`. That authorization supersedes the
historical branch-only publication instruction in §6. The original assignment
below is preserved as historical scope, not a statement that every acceptance
criterion passed.

### Repairs completed

| Area | What was fixed or corrected | Status |
| --- | --- | --- |
| Explicit MTP requests | Requests losing MTP intent or taking an unintended route | Fixed |
| Identity-free admission | Native selection depending on artifact/quantization identity rather than implementation capability | Fixed |
| Qwen native execution | Native-route restrictions and sampled graph-capture mismatch | Fixed |
| MoE concurrent verification | Convolution state addressed incorrectly across resident slots | Fixed |
| Diagnostic correctness | AR control replayed output predictions instead of consumed inputs; attribution was misleading | Corrected |
| Gemma generation controls | Thinking budgets, minimum tokens, logprobs, and multi-token stops | Implemented and validated |
| Minimum-token API requests | Missing explicit EOS caused rejection even when the model supplied EOS metadata | Fixed |
| Stop handling | Context-dependent tokenization missed stops; speculative metadata ordering exposed stop text | Fixed |
| Gemma single-token stops | Stop tokens appeared in text and were mislabeled as EOS | Fixed |
| Streaming UTF-8 | Multibyte characters were corrupted; special-token suppression could cross requests | Fixed |
| Literal SSE stops | Stops split across chunks or inside tokens leaked into visible output | Fixed |
| Final stream metadata | Stop handling discarded generated IDs, usage, or MTP attribution | Fixed |
| Evidence and reporting | Concurrency claims, failed captures, and numerical comparisons were mischaracterized | Corrected and archived |

### Four-model public validation

These results cover the recorded `gfx1151`, BF16-KV configurations through
`hipengine serve`, with context capacity 32768, prefix cache off, and up to four
active requests. They do not establish every backend or configuration. C2–C4
means two through four concurrent client submissions; physical batching is
reported separately.

| Model | Explicit MTP / alias | Default route observed | Controls and streaming | Concurrent submissions | Physical MTP batching observed | Long-context completion |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B Q4_K_S | Passed | MTP | Passed after focused repairs | C2–C4 passed | Proposal widths 2, 3, 4 | 30078 prompt tokens; 32 output tokens |
| Qwen3.8-27B Q4_K_M | Passed | MTP | Passed | C2–C4 passed | Proposal widths 2, 3, 4 | 30078 prompt tokens; 32 output tokens |
| Qwen3.6-35B-A3B Q4_K_M | Passed | AR | Passed | C2–C4 passed | Two-row proposal batches in this capture, not physical C4 | 30078 prompt tokens; 32 output tokens |
| Gemma4-26B-A4B Q4_K_XL | Passed | MTP | Passed after focused repairs; whitespace caveat below | C2–C4 passed | Single-request MTP route, not multi-request batching | 30083 prompt tokens; 2 output tokens, then EOS |

Controls covered thinking on/off and hard budgets, minimum output, logprobs,
tokenizable stops, Unicode, seeded sampling, and blocking/SSE comparisons.
Q4_K_S has five original streaming pairs plus six focused control pairs;
thinking SSE was not repeated for that quantization. The other models have ten
original streaming pairs each. Additional literal-stop checks exercised Qwen
Q4_K_M and Gemma through real HTTP requests. Original failed captures remain
unchanged and are linked to focused repair evidence.

### Remaining limitations and unproven claims

| Item | Current state | Meaning |
| --- | --- | --- |
| MoE automatic MTP | Automatic selection still chooses AR (`automatic_mtp_scope_not_promoted`) | Explicit `speculative_mtp` and the alias work; universal default-on MTP is not the delivered state |
| Gemma physical concurrency | MTP route is capped at one request | Concurrent clients work, but this is not physical multi-request MTP batching |
| Literal-only stop efficiency | Output is suppressed, but the backend is drained to retain authoritative accounting | A stop inside an unmatched token does not guarantee immediate compute termination |
| Exact text parity | Blocking chat trims trailing whitespace; SSE preserves it | Gemma's thinking-output pair had identical IDs and reasoning apart from trailing whitespace |
| Production numerical quality | MoE and Gemma production AR/MTP IDs differed on the recorded 256-token prompts | This alone does not prove a defect; a full numerical-quality verdict has not been established |
| Performance | Recorded API and control checks passed after repairs | No blanket MTP speedup claim; older timing rows do not establish performance of the final repaired state |
| Coverage | Representative configurations and targeted regressions were tested | Not exhaustive across all models, contexts, sampling settings, or backends; the original Flash-Next row is not covered by this packet |

### Verification and delivered commits

| Evidence | Result |
| --- | --- |
| Final CPU unit + server API bundle | 14052 passed, 91 skipped |
| Final literal-stop HTTP validation | 16 requests passed; 8 blocking/SSE pairs matched under the existing whitespace policy |
| Corrected public matrix | 131 original request outcomes, linked to 20 focused control-repair checks |
| Documentation, audit, worklog, fixtures, and CPU smoke gates | Passed for the repair units |
| Explicit MTP intent and Gemma controls | `e9a80c4f5` |
| Identity-free Qwen admission | `06c25be33` |
| Packed decode width and physical telemetry | `0ec4aa482` |
| MoE resident-slot state repair | `95b3595e3` |
| Consumed-state attribution correction | `b53353ec5` |
| Native MoE route and sampled graph capture | `2a96d5d46` |
| Qwen streaming UTF-8 repair | `95d80f58b` |
| Minimum-output EOS and streamed token stops | `6b70e99c8` |
| Gemma single-stop repair | `f541f93e6` |
| Corrected matrix | `273c614d4` |
| Literal SSE repair; implementation snapshot for this report | `af024e122` |

Evidence and exact request/server parameters:

- [Corrected public matrix](../../benchmarks/results/2026-10-17-mtp-corrected-public-matrix.json): request summaries, source/raw-capture hashes, original checker outcomes, physical counters, and limitations.
- [Minimum-token and token-stop repairs](../../benchmarks/results/2026-10-17-mtp-public-token-control-repair.json): 12 focused Q4_K_S requests.
- [Gemma single-stop repair](../../benchmarks/results/2026-10-17-gemma-single-stop-repair.json): eight focused requests.
- [Literal SSE stop repair](../../benchmarks/results/2026-10-17-literal-sse-stop-repair.json): 16 requests after the matrix; resolves its recorded literal-SSE leak limitation without changing the original captures.
- [Qwen native state attribution](../../benchmarks/results/2026-10-17-qwen-native-commit-attribution.json) and [Gemma prefix attribution](../../benchmarks/results/2026-10-17-gemma-mtp-prefix-attribution.json): ownership, repeatability, and numerical observations, not a whole-model quality verdict.
- [Final repair worklog](../../worklog/entries/20261017T221108.080636Z-mtp-controls-literal-stream-stops-d6e758.md): exact final CPU command, HTTP checks, and validation history.

## Original assignment

You are working unattended overnight. The lead will review your commits and report in the morning. **No one will answer questions before then.** Don't stop to ask and don't end with "want me to run it?". Every decision you might want to ask about is already made below. If you still hit an unknown, pick the option that gets MTP serving users, write down why, and keep going.

## 1. The goal, stated once

When a user runs `hipengine serve <model>` on this gfx1151 host and sends an ordinary OpenAI request, the server uses MTP speculative decoding by default. This applies to every model on the host that has an MTP/NextN head (built in, or the external head files listed below). It holds for greedy and sampled requests, streaming and non-streaming, short and long context, with no context cap and no HTTP 400s.

That's the whole task. The deliverable is not an analysis, a plan, a "promotion packet", or a list of decisions for the lead. It is working code on the default path, committed, plus the acceptance matrix in §5 filled with measured results.

## 2. Lead decisions (directives, already made — do not reopen)

Per `AGENTS.md` "Decision Authority" these are directives. Docs, gates, calibration envelopes, and `docs/EXECUTION-PROFILES.md` promotion rules do **not** overrule them. If a doc disagrees, update the doc to record this decision.

1. **MTP is the default decode path on gfx1151 for every model with an MTP head.** Requests that don't mention `speculative_mtp` get MTP.
2. **Artifact-scoped automatic admission is removed.** Keying automatic MTP on a specific artifact, SHA, filename, or evidence row is identity gating, which `AGENTS.md` "Never key admission on identity" forbids. Admission keys on the model family's declared capability plus the backend. A Qwen3.8-27B UD-Q4_K_M with no evidence row gets MTP exactly as the measured plain Q4_K_M does.
3. **No promotion packet, quality suite, profiler trace, or "economics" campaign is a prerequisite for turning it on.** Absence of an evidence row is not a reason. Measure *after* it's on (§5) and report.
4. **MoE on gfx1151 is in scope.** Gemma 4 26B-A4B (MoE) already runs MTP on this host, so the kernels execute MoE MTP on gfx1151. Declare the gfx1151 MoE lanes (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next) and run them. "Deliberately undeclared" and "gfx1100-scoped by declaration" are not capability misses; they are declarations you are told to fix.
5. **Sampled requests use MTP** (temperature, top-p, top-k, min-p, penalties, seed), using the existing native sampled-MTP path (`automatic_native_sampled_c1_c4` on Qwen). Extend it to every model here. Silently falling back to AR for sampled requests (`unsupported_sampling_k0`, `mtp_sampling_unsupported`) is the defect to fix, not a refusal to preserve.
6. **Streaming works with MTP.** The error "speculative streaming is not supported by this generator" is a bug. Wire the generator's `stream_speculative_detailed` (or equivalent) into the route the server calls.
7. **Production variants are fine.** You don't need strict cells to engage. Missing strict C1/K3 width-depth entries are not a blocker. Use the production cells. Mark stale strict rows stale in the doc that owns them and move on.
8. **Rows bound to an artifact not on this host** (e.g. plain Q4_K_M, sha `7e78da5d…`) are irrelevant to this task. Don't fetch it, wait for it, or gate on it.
9. **Concurrency:** C1–C4 must work with MTP where batched verification exists (Qwen dense has it). Models with only C1 MTP serialize or run AR for concurrent requests, *with a recorded fallback reason*, and must never error. C5–C8 whole-batch AR is acceptable tonight; record it and don't open a kernel campaign for it.
10. **`"speculative": true` must not 400 on the server's own defaults** (the `kv_scale_dtype` blocker with the default `fp32`). Either it works, or it maps to the MTP route, or it returns a clear 501 naming a real kernel capability miss. A 400 for a configuration the server itself picked is a bug.
11. **Cosmetic, but fix it:** the startup banner must state the MTP route it resolved and the context length. "MTP unavailable (unresolved)" / "Context unknown tokens" on a model that runs MTP is wrong. Top-level `hipengine.speculative_mtp` counters must count.

## 3. What counts as a legitimate "no"

You may leave a cell not working **only** for a reason on the list in `AGENTS.md` "Capability is the only gate", and only with all three of the following in the report:
- the exact error/output you observed when you **ran** it (not a prediction);
- the file/function where the capability is missing;
- the command that will clear it once the missing piece exists.

Not legitimate, and if you write any of these as a reason the task is not done: "not qualified", "no evidence row", "needs promotion packet", "decision needed", "artifact-scoped", "deliberately undeclared", "strict cells absent", "multi-hour campaign — want me to run it?", "out of scope".

A model with genuinely **no MTP head file anywhere on the host** is a legitimate skip. State the path you checked.

## 4. Models on this host (verify heads yourself, first thing)

| Model | Weights | MTP head |
| --- | --- | --- |
| Gemma 4 26B-A4B (MoE) | `/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | `/models/gguf/gemma-4-26B-A4B-it-GGUF/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` |
| Qwen3.8-27B (dense) | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | embedded NextN? check |
| Qwen3.8-27B (dense) | `/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf` | embedded NextN? check |
| Qwen3.6-35B-A3B (MoE) | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | embedded NextN? check |
| Qwen3.8-Flash-Next (MoE) | `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/` | `/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` |

Known state: Gemma 4 MTP runs and measured plain 45.5 vs MTP 69.4 tok/s on this host at budget 4 (`benchmarks/results/2026-10-15-gemma4-gfx1151-mtp-budget4.json`). Qwen3.8-27B UD-Q4_K_M whole-request MTP measured 17.32 vs 9.95 tok/s AR (1.74x) and greedy IDs were byte-identical to AR, 30/30. Both are already runnable. What's missing is the default route.

## 5. Acceptance matrix — the definition of done

For **each** model in §4, start a real `hipengine serve` and send real HTTP requests (curl or a small script). Nothing else counts: no bench harness, no test fixture, no `--diagnostic-plan`, no env var a user wouldn't set. For each request, record the route fields the response or server log reports (selected route, sampler mode, draft depth, acceptance, fallback reason if any).

| # | Request | Pass condition |
| --- | --- | --- |
| A | Greedy, non-streaming, **no `speculative_mtp` field**, ~200-token prompt, 256 out | MTP route ran (no fallback reason); output token-identical to the same request with MTP rolled back to AR |
| B | Same as A with `temperature 0.7, top_p 0.9, top_k 40, seed` | MTP sampled route ran; coherent output; same seed gives the same output twice |
| C | A with `stream: true` | SSE stream completes; concatenated text equals A's text; MTP route ran |
| D | B with `stream: true` | completes; MTP route ran |
| E | Greedy, ~30k-token prompt (or the model's max if smaller) | completes, no 400, MTP route ran |
| F | Thinking/chat template on (where the model has one) | completes, MTP route ran |
| G | 4 concurrent requests (mix of A and B) | all complete with HTTP 200; record per-request route (MTP batched, or serialized/AR with a reason) |
| H | `"speculative": true` | not a 400 on server defaults (§2.10) |
| I | `logprobs`, `stop`, `max_tokens` small | honored on the MTP route |
| J | Startup banner and `/metrics`-style counters | banner shows the resolved MTP route and context; counters non-zero after A |

Also record AR vs MTP decode tok/s for A and B per model: the same server, the same prompt, AR forced by the rollback endpoint or an explicit `speculative_mtp: false`. If MTP is **measured slower** than AR on some cell, it still runs when explicitly requested. Automatic selection may choose AR for that cell only if the measured numbers are written next to the policy declaration. Report it. That is the only performance-based exception, and it requires the measurement.

The greedy-exactness check in A is a **correctness test**, not a gate. If it fails, you have found a bug. Fix it.

## 6. How to work

- Read `AGENTS.md` first and obey it. Its "Product Defaults" section is the spec for this task.
- **Branch:** work on a fresh branch off current `main` (e.g. `gfx1151-mtp-default`). Commit each logical unit as soon as its tests pass, following the commit rules in `AGENTS.md` (explicit `git add <paths>`, no bylines). Push the branch to origin. **Do not push to `main`.** The lead reviews and merges.
- **Tests first** for each behavior change: default routing with no field, sampled admission, streaming route, identity-free admission (a renamed/re-hashed copy of an artifact gets the same route), the `speculative: true` defaults. Don't key tests on model names or SHAs.
- **No backend/quant `if` branches** in engine/dispatch/server code. Extend capability declarations/registry entries (the Qwen MTP declarations in `hipengine/models/qwen35.py` are the pattern; Gemma 4 should get the same declared-capability treatment rather than the generic speculative-provider shim).
- **GPU discipline:** one GPU job at a time on this host. Use the TheRock environment as other gfx1151 runs in this repo do (`HIPENGINE_HIP_ARCH=gfx1151`). Long runs go to background tasks. You have standing approval for every GPU run this task needs. Don't ask.
- **Order:** (1) verify heads; (2) make the default route + identity-free admission work for the two models already known to run (Gemma 4, Qwen3.8-27B UD-Q4_K_M) and pass rows A–J for them; (3) sampled + streaming for all; (4) declare and run the MoE Qwen lanes; (5) concurrency; (6) docs. Commit at each step, so a partial night still leaves reviewable, working increments.
- If a change breaks an existing Qwen gfx1100 MTP test, fix your change. Don't relax the test.
- Record a flag you add in `docs/REFACTOR.md` with its removal condition, and make it default to MTP on.
- Update `docs/API.md` (MTP section) and the relevant model cards to describe the new default behavior. **Don't edit `README.md`.**
- Validation before each commit: the narrowest `pytest` targets, `python3 audit/audit.py check`, `python3 scripts/docs/check_all.py` when docs change.

## 7. Morning report (write it as the final worklog entry and as your final message)

1. **The §5 matrix, filled:** one row per model × request letter. Pass/fail, the route fields observed, AR tok/s, MTP tok/s, acceptance rate.
2. **Commits:** a list with one-line summaries, plus the branch name.
3. **Remaining failures:** each with the three items from §3. Nothing else is allowed in this section.
4. **The exact `hipengine serve` command and one curl** the lead can paste to see MTP serving by default.

Don't include a "decisions needed" section. If you feel you need one, re-read §2.
