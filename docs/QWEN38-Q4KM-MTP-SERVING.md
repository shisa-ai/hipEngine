# Qwen3.8 Q4_K_M Exact MTP Serving Campaign

- Status: **S3 complete; S4 automatic product decision next; automatic policy remains K0**
- Started: **2026-08-26**
- Primary host: **Radeon 8060S / `hip_gfx1151`**
- Model: **Qwen3.8-27B `Q4_K_M`, BF16 KV**
- Serving profile at entry: **strict**
- Direct-leaf premise: **exact natural25 B3 at 21.157528 tok/s / 1.809537x true AR**
- Current public policy: **explicit route available; `auto` selects AR**
- Dependencies: [`PLAN.md`](PLAN.md), [`MTP-FIX.md`](MTP-FIX.md),
  [`API.md`](API.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`TESTING.md`](TESTING.md), and [`BENCHMARK.md`](BENCHMARK.md)

## 1. Goal and non-goals

Qualify at most one narrow, honest public serving scope for the exact Qwen3.8
`Q4_K_M` native MTP path. Reuse the existing `LLM`, Generation-2, OpenAI batch,
streaming, cancellation, circuit-breaker, telemetry, session, and resource
owners. Do not create another scheduler or a prompt-conditioned policy.

The direct `qwen36_dense_gguf_suite.py` result is a valid kernel/runtime premise,
not serving evidence: it constructs resident target/provider/verifier objects
directly and excludes prefill from decode timing. Promotion requires the public
`LLM.generate*_detailed()` and OpenAI paths under the same request boundary as
true AR.

Out of scope until a fixed public scope passes:

- stochastic or processed-target MTP;
- tools, structured outputs, logprobs, penalties, forced tokens, or hard
  thinking-budget controls;
- physical C>1 MTP, adaptive K, long-graph expansion, overlap, or DMS;
- transferring Q4_K_S P9 policy, manifests, rates, or compiler observations;
- changing automatic policy from K0 based on direct-leaf speed alone.

## 2. Frozen identity and baseline

The exact model is `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008
bytes, full SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The canonical ten-prompt fixture SHA-256 is
`fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a`.
Direct premise evidence is
[`2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json).

At campaign entry:

- the GGUF generator advertises MTP from actual NextN inventory;
- `LLM.generate_speculative_mtp_detailed()` and the OpenAI explicit route already
  exist;
- server `auto` resolves through a deterministic offline policy and currently
  selects K0/AR;
- `supports_default_mtp` is still a broad dense-model boolean rather than an
  artifact/profile/scope qualification;
- the direct B3 leaf and public dense method share transactional components, but
  do not share a proven timing/serving contract.

## 3. Normative route contract

A candidate serving plan must be immutable and resolved before mutation from
model artifact, backend, quant, execution profile, KV layout, realized group
width, context/page bucket, remaining output horizon, memory fit, and compatible
sampling mode. Prompt text/hash, token IDs, category, heldout identity, task
result, and post-hoc oracle selection are forbidden inputs.

The first possible scope is:

- exact pinned model hash above;
- `hip_gfx1151`, `Q4_K_M`, strict profile, BF16 KV;
- realized C1;
- raw greedy (`greedy_fast`) only;
- B3, natural25, and only context/output buckets mechanically covered by the
  retained transaction/graph fallback contracts;
- explicit route until the complete automatic packet passes.

Every unsupported field or shape selects AR before proposal/target mutation and
reports a stable reason. `enabled` must not turn the current broad dense boolean
into artifact-free default admission.

## 4. S0 — Current public-route baseline

- [x] Freeze clean merged-main source, compiler, model, prompt, environment, and
      GPU ownership.
- [x] Run one excluded cached-build warmup.
- [x] Compare public `LLM.generate_detailed()` with
      `LLM.generate_speculative_mtp_detailed()` over the complete natural25
      suite, with exact generated IDs, acceptance/state/route accounting, and
      zero final ownership.
- [x] Measure operation-complete and decode-owner wall without substituting the
      direct-leaf denominator.
- [x] Exercise one OpenAI explicit completion/chat smoke and verify capabilities,
      generation shape, compact MTP usage, and teardown.

The retained LLM checkpoint is **12.940 vs 9.025 true-AR tok/s (1.4337x)**
operation-complete: 30/30 cells and 90/90 arms are exact, every individual cell
is 1.2995x–1.5515x, every category/heldout slice is positive, and recovery,
post-target candidate D2H, and final ownership are zero. See
[`2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json).
The direct legacy control is 14.149 tok/s but is not a second admissible public
scheduler; Generation-2 remains the serving owner.

The strict C1 OpenAI completion/chat smoke realizes `gguf_specdec2_mtp2` with
one timing owner each, eight cycles each, 32/44 aggregate accepted drafts, exact
25-token usage, correct Prometheus accounting, health after both requests, and
clean shutdown. Its RED capacity-4 control exposed a fail-open summary where a
selected MTP route realized K0 but reported `used=true`; the repair now derives
usage from backend telemetry and reports `backend_k0_fallback`. See
[`2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0-openai.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0-openai.json).

Exit: either a public C1/B3 premise above 1.10x true AR, or a concrete no-go
blocker. No implementation is admitted merely because the direct leaf is fast.

## 5. S1 — Artifact-scoped capability and RED contracts

Only after S0 passes:

- [x] RED/GREEN core: a pinned Q4_K_M/gfx1151/strict/BF16/C1/raw-greedy scope
      resolves an explicit-only candidate plan, while wrong hash/quant/backend/
      profile/manifest/KV/C/context/horizon/budget/sampler/memory resolve K0.
- [x] RED/GREEN core: Q4_K_S cannot transfer and generic dense inventory cannot
      produce an admitted plan.
- [x] RED/GREEN: `auto`, `enabled`, capabilities, response reasons, and rollback
      use the same immutable plan/fingerprint.
- [x] Implement the capability through the model/speculative plugin boundary;
      server/engine dispatch contains no backend/quant admission branch.
- [x] Preserve operator-selected explicit compatibility and strict AR/oracle
      fallbacks independently; broad dense inventory cannot enable a default.

Exit: one typed, fingerprinted pre-mutation serving plan with no default change.
The retained plan fingerprint is
`sha256:1948983ad884f41d9dce4453a3b2ab8f9a357b5005b9a2cabd18556b60084740`.
A real `auto` server reports the exact content-verified artifact scope, keeps the
unflagged request on AR, realizes its explicit twin through
`gguf_specdec2_mtp2` with exact IDs, rejects context 68 before MTP mutation, and
returns the same plan from rollback. Evidence:
[`2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s1.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s1.json).

## 6. S2 — LLM correctness and lifecycle

- [x] Full train/heldout/category exact generated IDs and GPU/CPU acceptance
      agreement where the route exposes both oracles.
- [x] Reject/partial/full commit, following AR, state/KV/cursor/provider/output
      ownership, deterministic repeats, and C1 neighbor isolation.
- [x] Context/page/output-room boundaries and fallback before mutation.
- [x] EOS, stop, cancellation, deadline, injected proposal/target/readback
      failure, circuit breaker, operator rollback, restart, and health-after-fault.
- [x] Stable tracked/HIP memory, repeated clean reopen, pressure/soak, request-page
      drain, provider/claim release, and final zero transient ownership.

The clean-worktree S2 packet covers every natural acceptance count (0/1/2/3),
proposal/target/readback RED→exact-AR recovery with following healthy MTP, 30/30
alternating exact waves, 40 reported requests plus two AR references, logical C8
queue/drain, and byte-exact blocking/stream text and IDs after repairing empty
raw-greedy speculative chunks. Evidence:
[`2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s2.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s2.json).

## 7. S3 — OpenAI serving packet

- [x] Blocking completion/chat and SSE completion/chat reconstruct exact public
      IDs/text and report route, K, acceptance, usage, timing owner, and reason.
- [x] Explicit opt-in, auto K0 fallback, incompatible sampling, thinking hint vs
      hard policy, session/continuation behavior, disconnect, and shutdown.
- [x] Fixed/ragged/delayed admission/refill/retirement and mixed AR/MTP neighbors
      reuse the one C1 scheduler without a hidden physical-width claim.
- [x] Below/near/overload packet records TTFT, ITL, E2E, queue, exact goodput,
      bounded overload, deterministic accepted outputs, and complete drain.

S3 repairs the terminal SSE owner so completion/chat blocking and streams carry
byte-exact text/IDs plus truthful MTP path, fingerprint, acceptance, timing, and
usage. Mixed and delayed pairs admit 2/2; overload admits four and rejects four
with bounded `429 engine_busy`; every accepted load output repeats exactly and
final pending/admitted/active rows are zero. Evidence:
[`2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s3.json`](../benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s3.json).

## 8. S4 — Automatic product qualification

Automatic promotion is considered only if S0-S3 pass and the public
operation-complete route is >1.10x same-host true AR overall, non-regressive in
every category/heldout slice and neighboring AR SLO, exact under all binding
control/ownership gates, and healthy under focused then final soak.

A promoted table may use only the allowed fields in Section 3 and must carry a
canonical fingerprint and evidence link. Every other scope remains K0 with an
exact measured or unqualified reason. The project target remains >1.30x.

## 9. S5 — Closure

Publish retained/no-go code units, compact artifacts, benchmark rollup and
changelog, immutable worklogs, capability/API documentation, refactor/removal
triggers, exact fallback scope, and final teardown evidence. A zero-promotion
closure is valid and preferable to weakening a gate.
