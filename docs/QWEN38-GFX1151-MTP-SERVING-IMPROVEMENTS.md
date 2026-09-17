# Qwen3.8 gfx1151 MTP serving improvement review

Review baseline: `b6bbba448`, 2026-09-17. Scope: Qwen3.8-27B GGUF
`Q4_K_M`, BF16 KV, one active request on the zbook Radeon 8060S. This is
an implementation review and proposed work order, not new performance evidence
or authorization to promote an unqualified kernel.

## Findings

The comparison exposes three different problems: a default that prevents MTP
admission, a runtime context limit that prevents full-request MTP, and missing
telemetry that obscures both. Fix those before attributing the gap to draft
acceptance or tuning kernels.

Evidence sources:

- [Comparison artifact](../benchmarks/results/2026-09-17-gfx1151-qwen38-atlas-comparison-mtp-k3.json):
  exact model hash, zbook host/hardware, server/client commands, greedy 128-output
  streaming protocol, 16,384-token session, three measured repeats and text check.
- [Headline artifact](../benchmarks/results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json):
  exact commands, host machine ID, model hash, strict profile/manifest,
  category-suite correctness and timing evidence.
- [Physical-admission decision](../worklog/entries/20260917T094111.437688Z-lhl-mtp-serving-physical-admission-dd97b6.md)
  and [execution-profile contract](EXECUTION-PROFILES.md#29-serving-admission-is-physical-not-shape-scoped).

### The 945-token request appears to switch to AR during decode

`qwen35_gguf_mtp2.py` has two independent limits:

1. Prompt activation rejects `prompt_tokens + 1 >= min(1023, max_sequence_length)`.
   With a sufficiently large session, the largest admitted prompt is **1,021**
   tokens, not 1,022 as the original report stated.
2. Its speculative capability also advertises `max_context_tokens=1023`.
   `speculative/policy.py` computes `context_room = max_context_tokens -
   row.context_tokens - 1`, shrinks candidate depth, and selects AR when no room
   remains (`TARGET_GRAPH_CONTEXT_BUCKET_MISS`). A short admitted prompt can
   therefore outgrow MTP during its output.

The artifact's separate non-streaming probes support that second failure mode:

| Actual prompt | Post-first-token outputs | Cycles + accepted drafts | Outputs not accounted for by MTP cycles |
| ---: | ---: | ---: | ---: |
| 517 | 127 | 55 + 72 = 127 | 0 |
| 945 | 127 | 21 + 56 = 77 | 50 |
| 3530 | 127 | 0 | 127 |

For an ordinary greedy speculative cycle, accepted drafts plus one verifier
output account for emitted tokens. The 945-token row fits the context boundary:
945 + 77 = 1022. About 39% of its post-first-token outputs are unaccounted for
by MTP and are consistent with an AR tail. This is an inference from counters
and source, not a captured per-cycle trace of the timed streaming requests.
Confirm it with per-cycle positions, selected depth, emitted IDs, and AR reasons.

The 517-token row actually has the larger measured speedup (1.533x versus
1.467x), despite lower draft acceptance. Acceptance alone cannot explain these
whole-request rates. `server/api.py` reports `used=true` and
`effective_route=speculative_mtp` if MTP occurred at all; it does not distinguish
this mixed execution from full-request MTP.

### The headline is not a matched regression baseline

The published 20.985 MTP / 11.150 true-AR tok/s, 1.882x result uses all ten
category prompts including heldouts, initial contexts 1–67, natural25 outputs,
strict/K3, one active request, resident capacity four and a 1,024-token session.
The comparison uses different prompts, 128 outputs, a 16,384-token session,
streaming client timing and no explicit execution-profile flag. An evidence key
containing `strict` does not prove the active profile after the physical-only
admission change. Capture the resolved profile and manifest.

The headline identifies host `gfx1151`, machine ID
`55ea6c509d0b49eea8de7094a1023668`, HIP 7.15.26333; the comparison identifies
`zbook` and ROCm 10.0.0 without that machine ID. Physical-host equivalence is
not established by these artifacts. Do not call their absolute-rate difference
a regression. Reproduce both protocols on one identified physical host first.

Atlas is a useful product target, not a drop-in kernel baseline: its NVFP4
weights and chat template differ from hipEngine's GGUF Q4_K_M. Its existing
rows were not rerun in the latest session. Its rates do not establish a speedup
that hipEngine can obtain by removing a guard.

## Prioritized work

### P0 — Make the normal server select a supported candidate budget

- Represent an omitted budget as automatic selection, resolved through
  model/provider capability data and physical identity. For this qualified
  Qwen cell select B3; preserve Laguna's B4-only contract. Preserve explicit
  CLI/environment overrides and explain unsupported values rather than silently
  rewriting them. Do not globally change 4 to 3 or add backend/model branches
  to the engine.
- Report requested/resolved budget, provider, admission and fallback reason at
  startup and through capabilities. Warn when an explicitly enabled MTP server
  cannot admit its configured provider.
- Scope: `server/__main__.py`, server configuration, model/provider capability
  data, `speculative/serving.py`. Existing debt: the candidate-budget entry in
  [REFACTOR.md](REFACTOR.md).
- Acceptance: unit coverage for omitted, CLI and environment budgets, supported
  Qwen B3, Laguna B4, unknown identity and explicit unsupported values; a real
  default-config Qwen request must execute speculative cycles. No unsupported
  route gains admission.
- Expected benefit: fixes out-of-box AR-only behavior. **No speed improvement to
  the reported K3 arm**, which already overrides the bad default.

### P0 — Make partial fallback visible and establish a matched baseline

- Extend request summaries and final streaming usage with MTP-output count,
  AR-output count, selected-depth histogram, fallback reason counts and first
  fallback position. Keep admission, route selection and actual execution
  separate; preserve existing fields for compatibility.
- Surface `specdec2_mtp2_prompt_fallback_reason` instead of reducing every
  backend refusal to `backend_k0_fallback`. Do not treat `used=true` as proof
  of full coverage. Count outputs, not just drafted tokens or acceptance.
- Reuse existing timing fields in `qwen35_gguf.py`: proposal, target, provider
  update, accept, readback, upload and commit. Verify counter ownership so
  streaming events and batch timing owners cannot double-count work.
- Acceptance: extend `test_unit_specdec2_policy.py`,
  `test_unit_qwen35_gguf_mtp2_seam.py`, `test_unit_gguf_mtp_api_gate.py` and
  `test_integration_server_api.py`; cover no MTP, full MTP and mixed MTP/AR,
  short output tails, cancellation and final SSE metadata. Real 945/128
  execution must reconcile every emitted ID with one execution mode.
- Reproduce the headline protocol and comparison protocol on the same host,
  commit, model hash, profile, capacity and toolchain. Record both profiles
  when testing strict versus production; do not mix their denominators.
  Use alternating AR/MTP arm order and the original repeat counts.
- Expected benefit: reliable diagnosis and regression tests, not intrinsic speed.

### P1 — Extend MTP through the whole request, then beyond short prompts

- Replace the adapter's literal 1023 limits with a provider/backend capability
  that covers prompt priming, proposal, target verification, commit/rollback,
  graph buckets and allocated KV/workspaces together. Keep real safety bounds;
  do not restore benchmark prompt/horizon allowlists in serving admission.
- First cover the 945/128 crossing; then 3530/128 and actual 4K/8K prompts with
  128 outputs. Distinguish a declared 16K capacity from a measured 16K request.
- This is **not just changing a constant**. The gfx1151 backend already exposes
  native target limits of 65,544, while the serving adapter still caps 1023.
  `gguf_native_spec_cycle.py` declines graph spans crossing the 1024 attention
  transition or split-workspace boundaries. `qwen35_gguf_runner.py` uses
  `strict_long_rows` to serialize dense attention and FFN rows after the split
  transition. `qwen35_gguf_nextn.py` also has a short exact-chain graph limit.
  Qualify how these paths compose for this model and host before widening.
- Preserve the serial strict verifier as an oracle/fallback. Review a faster
  staged long-row route under the **production numerical contract**, rather
  than rejecting it solely for BF16 differences. Existing historical boundary
  work in [MTP-FIX.md](MTP-FIX.md) is diagnostic context, not certification of
  this Qwen3.8 serving configuration.
- Acceptance: prompt lengths 1021/1022/1023, cycles below/at/above 1024, shifted
  KV pages and page crossings, B1/B2/B3 reject/partial/full acceptance, output
  tails, graph/eager transitions, reset/reuse and cancellation/refill. Check
  exact ownership/rollback plus the selected profile's numerical, determinism,
  isolation and task gates. Prove expected kernel execution with a trace.
- Performance gate: full category suite plus category-heldouts at each target
  shape against true no-MTP AR on this host; report coverage, cycle wall,
  throughput, TTFT and total latency. Functional long MTP that serializes all
  target rows may be slower than AR and is not a performance completion.
- Expected benefit: removes the likely AR tail at 945 and the all-AR 3530 row.
  Magnitude is unmeasured; do not promise Atlas parity or a fixed multiplier.

### P2 — Recover cycle efficiency with profiling, not acceptance tuning

- After coverage is measured, profile the actual serving-selected singleton
  verifier at resident capacity four. Compare with the direct target path using
  identical token IDs and state. Check graph replay versus eager execution,
  target row batching, proposal/head cost, provider catch-up, host/device
  transfers, synchronization and selected-commit work.
- Use existing phase counters to rank candidates, then `rocprofv3` to verify
  kernel families and launch counts. Prebuild kernels and require the cache;
  profile the final child, not a parent harness that launches children.
- Preserve any same-suite non-regressive cycle or transfer improvement even if
  end-to-end rate noise hides it. Record hardware, commands, artifact and gates
  for each retained change; update the benchmark rollup only with measured data.
- Sweep only supported/qualified depths first. Increasing B3 to B4 is not an
  automatic match to Atlas's naming or economics. Any adaptive depth policy
  must use runtime cost/acceptance signals, be validated across all categories
  and heldouts, and never recognize benchmark prompts or candidate token IDs.
- Expected benefit: unknown until the wall-time breakdown identifies the
  dominant cost. The 517-token row's 44.2% acceptance is a workload observation,
  not evidence of a broken draft model.

### P2 — Reduce prompt-priming and serving latency separately

- MTP adds 198.6 ms TTFT at 517 tokens and 353.9 ms at 945 in the comparison.
  Source and timings are consistent with priming work, but the artifact has no
  phase trace proving how much each operation costs.
- Use `scripts/gguf_mtp_prompt_priming_bench.py` and serving timings to isolate
  target prefill, streamed hidden-state handoff, NextN priming, provider/session
  reuse and graph preparation. Check whether larger session workspaces affect
  the short path; do not assume they do.
- Optimize reuse or unnecessary transfers only after attribution. Prefix-cache
  hits currently decline prompt streaming (`prefix_reuse_k0`); supporting MTP
  with prefix reuse requires restoring both target and draft state, not merely
  deleting that check. Keep it a separate scope from cold-prompt performance.
- Acceptance: full outputs and state remain correct through repeat requests,
  cache hit/miss and cancellation; report cold/warm TTFT, decode and total
  completion latency separately. A TTFT-only win must not be described as
  improved steady decode throughput.

## Completion criteria and order

1. Land supported defaults and execution accounting with focused tests.
2. Establish a same-host canonical-suite reproduction and comparison baseline.
   The historical ~21 tok/s / 1.88x is a reference for its own protocol, not a
   mandatory threshold on another physical host or the 128-output workload.
3. Qualify full-request short-to-long transitions and make them economical.
4. Profile and optimize the largest remaining cycle/priming costs.
5. Retain each improvement with the full multi-prompt category suite and
   heldouts, true AR controls, complete output/control checks and applicable
   production numerical/task gates. The first 600 text characters are a useful
   comparison observation, not an implementation-promotion gate.

No runtime defaults or kernels changed during this review. No GPU benchmark
or correctness run was performed. Items above are proposed implementation work;
the only measured rates are those linked in the existing artifacts.
