---
status: closed
owns: The recorded acceptance requirements and evidence checklist for the gfx1151 default MTP serving campaign.
---

# Task: MTP on by default, end to end, for every model on this gfx1151 host, through `hipengine serve`

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
