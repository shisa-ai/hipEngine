# Qwen3.8-Flash-Next Strix Halo engine survey

Status: **active comparator refresh prepared 2026-09-05 on the Framework
Desktop / Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`)**. The active overview
tracks only upstream llama.cpp HIP/Vulkan and halo-box HIP/Vulkan. Fresh builds
are ready, but no model benchmark from this host has been promoted yet.

The Framework Desktop is a separate physical benchmark lane from the earlier
power- and heat-limited `zbook`. Do not report old-to-new rate deltas across
these hosts. The 2026-08-31 through 2026-09-01 multi-engine survey remains at
the bottom as historical analysis because its model, fork, accuracy, and
failure-mode comparisons are still useful but no longer define the active
comparator set.

## 1. Active comparator set

All four lanes use the same Qwen3.8-Flash-Next target artifact and will be
measured on the Framework Desktop. A source or backend identity is not a
performance result; the table records only the binaries prepared for the new
comparison.

| Engine | Backend | Source commit | Fresh Release build | Current status |
| --- | --- | --- | --- | --- |
| Upstream llama.cpp | HIP | `4d9176092d00586775af140581bb0b558ddc4389` | `/home/lhl/llama.cpp/llama.cpp-hip/build-hip-release-4d917609/bin/` | Builds and detects `ROCm0`; model benchmark pending |
| Upstream llama.cpp | Vulkan | `4d9176092d00586775af140581bb0b558ddc4389` | `/home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release-4d917609/bin/` | Builds and detects `Vulkan0`; model benchmark pending |
| halo-box | HIP | `b212548e0ddbf0a14e5a1d81b6ffcf8e4d098faf` | `/home/lhl/halo-box-strix-llama/build-hip-release-b212548e/bin/` | Builds and detects `ROCm0`; model benchmark pending |
| halo-box | Vulkan | `b212548e0ddbf0a14e5a1d81b6ffcf8e4d098faf` | `/home/lhl/halo-box-strix-llama/build-vulkan-release-b212548e/bin/` | Builds and detects `Vulkan0`; model benchmark pending |

The HIP builds use the existing ROCm/HIP 7.15 `therock` environment,
`AMDGPU_TARGETS=gfx1151`, HIP graphs, MMQ MFMA, and no virtual memory
management. The Vulkan builds use RADV/Mesa 26.1.6. All three source trees were
clean after the builds. halo-box `master` is distinct from the historical,
unmerged PR #11 head `a7ad7b7f`; active rows must name `b212548e` unless a
separate pinned-PR lane is deliberately added.

## 2. Active speed and reliability topline

No prior `zbook` rate is copied into this table. Fill a row only after all four
fresh binaries complete the same Framework Desktop protocol and the row's
repeatability verdict is known.

| Engine and backend | p512 | p1024 | p4096 | Output repeatability |
| --- | ---: | ---: | ---: | --- |
| Upstream llama.cpp HIP `4d9176092` | Pending | Pending | Pending | Pending |
| Upstream llama.cpp Vulkan `4d9176092` | Pending | Pending | Pending | Pending |
| halo-box HIP `b212548e0` | Pending | Pending | Pending | Pending |
| halo-box Vulkan `b212548e0` | Pending | Pending | Pending | Pending |

Each completed cell will report prompt processing / decode in tokens per second.
A fast row that fails exact repeated greedy output remains a diagnostic and is
not promoted as the active target.

## 3. Framework Desktop comparison protocol

- Physical host: hostname `gfx1151`, machine ID
  `55ea6c509d0b49eea8de7094a1023668`, Framework Desktop, Ryzen AI Max+ 395,
  Radeon 8060S, 128 GB unified memory, kernel `7.1.6-1-cachyos`.
- Target: Unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, four verified GGUF shards,
  BF16 K/V.
- Fixture:
  [`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json),
  SHA-256 `42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`.
- Workload: code, English, Japanese, and mixed Japanese/English at
  p512/p1024/p4096, greedy sampling, disabled prompt reuse, and 128 decode
  transitions.
- Reporting: record source and binary hashes, exact command, runtime and driver
  identity, per-sample rates, output hashes, variance, and teardown status.
  Compare absolute rates only within this Framework Desktop packet.

MTP remains a separate protocol. If measured, every lane must use its stated
sidecar and compare complete messages with its own true autoregressive control;
acceptance alone is not a correctness result.

## 4. Promotion and comparison rules

1. Keep HIP and Vulkan as separate lanes even when they share a source commit.
2. Require exact repeated output for every canonical case before treating a
   rate as a correctness-valid comparator.
3. Do not average a failing or unstable case into a promoted topline.
4. Do not compare Framework Desktop absolute rates with `zbook` absolute rates
   as an old-to-new improvement. Both machines are `gfx1151`, but their power,
   thermal, kernel, and runtime conditions differ.
5. Keep model-quality, alternate-quant, old-fork, and author-reported results in
   the historical section unless they are rerun under the active protocol.

## 5. Next benchmark packet

1. Smoke-load the verified first `UD-Q4_K_XL` shard with each fresh binary and
   record startup behavior.
2. Run the 12-case canonical autoregressive screen for all four lanes under one
   declared Framework Desktop power and cache protocol.
3. Record repeatability before interpreting throughput.
4. Publish one compact artifact containing all four lanes, then replace the
   pending cells in section 2.
5. Add full-logit or MTP packets only as separate comparisons with their own
   correctness gates.

## 6. Historical survey and model analysis

The remainder is the 2026-08-31 through 2026-09-01 `zbook` survey. Terms such
as “current” and “today” below refer to that snapshot, not the active Framework
Desktop builds in sections 1–5.

Historical status: external lanes were surveyed 2026-08-31 and hipEngine
evidence was refreshed 2026-09-01 on `zbook` / Ryzen AI Max+ Pro 395 / Radeon
8060S (`gfx1151`). A source-only review of the Pat1entZ3r0 optimization program
was added 2026-09-01 (historical section H6.7); it was not a surveyed lane and
none of its claims were locally reproduced. The central result was that
static-logit agreement and multi-step reliability are different questions:
several fast forks computed plausible logits for a fixed batch but failed
deterministic autoregressive or MTP output checks.

### H1. Speed topline

All values in the first table use the **same physical host, exact
`UD-Q4_K_XL` weight files, BF16 K/V, exact token arrays, greedy sampling,
disabled prompt reuse, and 128 decode transitions**. Each cell is prompt
processing / decode in tokens per second. These are three-repeat screening
rates, not closure-grade confidence intervals.

| Engine and backend | p512 | p1024 | p4096 | Output repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production `61b1cef1b`, HIP | **83.70 / 14.40** | **83.16 / 14.42** | **69.10 / 10.42** | 12/12 cases |
| Upstream llama.cpp `f1793c1c4`, Vulkan | 240.53 / 22.97 | 259.73 / 20.11 | 266.98 / 18.07 | 12/12 cases |
| Patched upstream `f1793c1c4`, HIP | 239.23 / 17.74 | 301.68 / 16.88 | 294.47 / 14.77 | 12/12 cases; non-stock loader |
| EngramHalo `1423f689`, HIP | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 pass; one p4096 case fails |
| Nathan `ad914eb`, Vulkan | **348.31 / 23.23** | 354.93 / 20.36 | 350.54 / 18.44 | **0/12 cases** |
| apepojken `843d575`, Vulkan | 291.73 / 23.21 | **375.23 / 22.42** | **397.43 / 22.25** | **8/12 cases** |

The fastest raw p1024/p4096 lane is apepojken, but four canonical cases vary
between identical requests. Nathan is fastest at p512 but varies on every
case. Those rates are therefore diagnostics, not correctness-valid targets.
EngramHalo's p4096 rate has the same restriction. The repeatability-valid
short-context ceilings remain patched upstream HIP and upstream Vulkan, plus
EngramHalo only at p512/p1024. Full timings, hashes, and first-difference
positions are in the
[canonical screen](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json),
the [current hipEngine row](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json),
and the
[survey artifact](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json).

Relative to the original hipEngine survey row, current production prefill is
**+1.44%/+2.38%/+1.72%** and decode is **+4.17%/+4.60%/+0.20%** at
p512/p1024/p4096. All 36 current samples are deterministic. Against the
repeatability-valid patched-upstream HIP row, hipEngine now reaches
**81.2%/85.4%/70.5%** of decode throughput and **35.0%/27.6%/23.5%** of
prefill throughput. Against upstream Vulkan, it reaches
**62.7%/71.7%/57.6%** of decode throughput and **34.8%/32.0%/25.9%** of
prefill throughput. Production therefore remains the slowest matched lane,
although its short-context decode gap narrowed.

#### hipEngine P8 graph research—not a matched survey row

| Boundary | Strict runner diagnostic | Strict graph diagnostic | Status |
| --- | ---: | ---: | --- |
| Host PLE publication + embedding + all 48 layers + full logits + device argmax/readback, shallow context | 68.855 ms/step | 61.910 ms/step | **Different processes; 1.112x is not an A/B** |

The graph row preserves the exact changing-token trajectory, full logits, 138
mutable owners, reset→replay, and graph→forced-eager→graph resumption. The
original 194.758-ms eager arm is invalid because it disables the shipped
per-layer MoE cache and uses a script loop. The follow-up 68.855-ms runner row
correctly measures that strict route, but imports 61.910 ms from the separate
graph-probe run. It also hardcodes `ExecutionProfile.STRICT`. Therefore neither
3.15x nor 1.112x is a named-production speedup, and P8 remains admission-
pending until one same-session production graph arm runs. Evidence:
[full transition artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json)
and
[named denominator](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json).

#### Current exact-token profile implications

The matched code-case device sums are hipEngine versus patched llama HIP
**5.972 vs 1.926 s**, **11.196 vs 2.990 s**, and **54.762 vs 10.838 s** at
p512/p1024/p4096. At live 4097, hipEngine QSA attention is **35.587 ms** versus
llama **0.591 ms**; p4096 prefill QSA is **10.229 s** versus **0.526 s**.
Every hipEngine window has 100% role-time coverage, and both family ledgers have
zero generic remainder. The old 41.6% "unattributed" row was incomplete table
coverage, not unknown runtime. Closing the survey gap therefore requires exact
QSA decode plus a prefill MoE+dense/GR+long-QSA portfolio. Evidence:
[current canonical impact profile](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json).

#### MTP speed versus true AR

This table uses the same ten code, English, Japanese, and mixed-language
prompts with 16 generated tokens. Representations differ by row and must not
be compared as an engine race. `Exact` means the complete canonical
reasoning-plus-content message matches that engine's own no-MTP output.

| Engine | Target / K/V / draft | Complete-wall MTP / AR | Acceptance | Exact | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| hipEngine | Q4 / BF16 / Q8, serial verify | 0.955x | 84.28% | **10/10** | Correct but slower; opt-in |
| EngramHalo | Q4 / Q8 / Q8, n-max 4 | 1.128x | 94.55% | **9/10** | Correctness-failing diagnostic |
| Nathan | Q4 / Q8 / Q8, n-max 3 | 1.161x | 95.45% | **8/10** all-four exact | AR and MTP each self-repeat only 9/10; diagnostic |
| apepojken | Q4 / Q8 / Q8, n-max 6 | 1.807x | 92.80% | **9/10** | Correctness-failing, non-counterbalanced diagnostic |

EngramHalo and apepojken both differ on `general_ja_plan` while repeated AR is
stable. Nathan has no fully stable denominator: AR varies on
`general_ja_explain`, MTP varies on `general_en_plan`, and only 8/10 prompts
match across both repeats of both modes. High acceptance therefore does not
prove losslessness. hipEngine is the only locally tested lane that passes all
ten AR-equivalence rows, but its serial verification loses economically.

#### Published speed claims that are not matched rows

| Source/configuration | Published result | Evidence status |
| --- | --- | --- |
| [apepojken Reddit report](https://www.reddit.com/r/StrixHalo/comments/1w0z7n1/qwen38flashnext_on_strix_halo_50_toks_decode_338/), Q3/Q8/MTP | 50.4 tok/s code at 0.963 acceptance; 338 tok/s p32K; 23.3 tok/s prose at 73K | Author-reported; repository and mechanisms verified, Q3 result not locally reproduced |
| [EngramHalo](https://github.com/Aristo94/EngramHalo.cpp), IQ3/IQ4/Q8/MTP | up to 39.3 tok/s shallow; 24.7 at 78K; p4096 up to about 502 | Author-reported and independently directionally reproduced with another quant; not the matched Q4/BF16 protocol |
| [Sleeping Robots Engram test](https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/), AtomicChat 4.27 bpw | 28–38.5 tok/s MTP at working depths; 15.0 at 26K; p26K 395.1 without MTP | Independent same-machine-class cross-check, different host instance and quant |
| Agention ROCmFP4 fork, FP4/Q8 | 27.77 tok/s AR at 512, 24.67 at 32K, 19.70 at 131K; up to 40 with MTP | Different quant, PLE layout, and fork; quant-specific evidence only |
| [Pat1entZ3r0 optimization program](https://github.com/Pat1entZ3r0/strix-qwen-next-flash-optimization) `413c33c`, dense-Q6K / Q8 KV / EasiiX Q8 MTP sidecar, Vulkan | 21–26 tok/s decode at 16K, 14–20 at 128K, up to 469 tok/s shallow prefill, 2.25 s TTFT at 38K | Author-reported; lock files and patch series verified, but the experiment harness and raw data are not committed and nothing is locally reproduced. Headline 2.5–3x compares a tuned stack against an `-ub 512` f16-KV no-MTP baseline. See historical section H6.7 |

### H2. Accuracy and reliability topline

No single column below means “model accuracy.” The columns test distinct
failure modes:

- **Static logits** compare full 248,320-token distributions for 160
  teacher-forced rows from the identical Q4 GGUF.
- **AR repeat** asks whether identical greedy requests produce identical
  129-token continuations.
- **MTP exact** asks whether speculative decoding preserves that engine's own
  AR message.
- **Absolute task score** requires a benchmark with ground truth. Most local
  engine lanes do not have one.

| Engine | Static same-GGUF evidence | Multi-step AR repeat | MTP exact | Absolute local task score |
| --- | --- | --- | --- | --- |
| hipEngine | Strict vs frozen llama: 10/10 top-1, mean/max KL 0.01406/0.04931; production vs strict: 446/450 top-1, mean/max KL 2.79e-4/5.98e-3 | **12/12** canonical cases | **10/10** | No official-suite rerun; 18-prompt production task-validity gate passes |
| Frozen llama.cpp #27742 HIP | 160/160 by definition; duplicate execution 159/160 top-1 | Historical oracle; current upstream HIP is 12/12 | Not measured | No local official-suite score |
| Current upstream HIP | 160/160 vs frozen; mean/max KL 9.27e-10/5.27e-8 | **12/12** | Not measured | No local official-suite score |
| Current upstream Vulkan | 159/160 vs frozen; mean/max KL 0.00337/0.13697; 160/160 on duplicate execution | **12/12** | Not measured | No local official-suite score |
| EngramHalo HIP | 159/160 vs frozen; mean/max KL 9.85e-4/0.01431 | p512/p1024 pass; p4096 fails one case | **9/10** | No full-stack task score; internal QSA PPL A/B only |
| Nathan Vulkan | 159/160 vs frozen; effectively identical to upstream Vulkan | **0/12**; 16 identical prompts produce 16 outputs | **8/10** all-four exact; AR and MTP each self-repeat 9/10 | None |
| apepojken Vulkan | 159/160 vs frozen; 160/160 vs upstream Vulkan, mean/max KL 0.00109/0.01576 | **8/12** | **9/10** | None |

The static packet changes the diagnosis materially:

1. **Nathan's fixed-batch math matches upstream Vulkan.** Nathan versus
   same-revision upstream Vulkan is 160/160 top-1 with mean/max KL
   `2.08e-10/1.93e-8`. Its failure is therefore not a broad static formula
   error. It appears in multi-step autoregressive execution—state, graph reuse,
   or another transition-dependent path.
2. **apepojken's custom delta changes Vulkan arithmetic modestly.** It remains
   160/160 top-1 against upstream Vulkan with mean/max KL
   `0.00109/0.01576`, but its continuation is non-repeatable on four cases and
   its MTP message differs on one prompt.
3. **EngramHalo is close to upstream HIP on static logits.** Mean/max symmetric
   KL are `9.92e-4/0.01516`; its failures occur at the p4096 transition and in
   MTP, not as a broad fixed-batch collapse.
4. The sole frozen-relative top-1 miss is the same low-margin
   `general_en_plan` step for every non-reference backend/fork. Frozen and
   Engram HIP can themselves flip that row between duplicate executions.
   Reporting 159/160 without this control would overstate the difference.

#### Pairwise same-GGUF distributions

Symmetric KL is the mean of KL(A‖B) and KL(B‖A). The packet teacher-forces the
same frozen-HIP greedy sequence, so every engine receives identical tokens.

| Pair | Top-1 | Symmetric KL mean / max | Interpretation |
| --- | ---: | ---: | --- |
| Frozen #27742 HIP ↔ current upstream HIP | 160/160 | ~9.3e-10 / ~5.3e-8 | Effectively identical |
| Frozen HIP ↔ current upstream Vulkan | 159/160 | 0.00325 / 0.11528 | Backend arithmetic difference; upstream Vulkan AR is still repeatable |
| Current upstream HIP ↔ EngramHalo HIP | 159/160 | 0.00099 / 0.01516 | Small fork/backend-path delta |
| Current upstream Vulkan ↔ Nathan Vulkan | **160/160** | ~2.1e-10 / ~1.9e-8 | Effectively identical static path |
| Current upstream Vulkan ↔ apepojken Vulkan | **160/160** | 0.00110 / 0.01620 | Modest custom-fork delta |
| Nathan Vulkan ↔ apepojken Vulkan | **160/160** | 0.00110 / 0.01620 | Same relationship as upstream Vulkan ↔ apepojken |

This is implementation-fidelity evidence, not absolute quality. Obtaining an
absolute BF16 comparison would require either running the approximately 360 GB
official checkpoint on a larger system or producing a reusable official-BF16
logit packet there.

Full 129-token cross-engine equality is much lower: hipEngine matches upstream
Vulkan on 1/12 cases, patched upstream HIP on 0/12, and EngramHalo on 1/12;
patched upstream HIP and EngramHalo match on 1/12. apepojken matches upstream
Vulkan on 3/12 stable continuations. Nathan has no stable full continuation to
compare. These counts are useful arithmetic fingerprints, but compounding
near-tie differences make them too strict to serve as task-accuracy verdicts.

### H3. Testing and promotion guardrails

None of the external forks has a production-correctness promotion contract
comparable to hipEngine's named execution profiles. Upstream llama.cpp has the
broadest general software test matrix, but it does not treat every backend
arithmetic change as a separately calibrated production profile.

| Engine | Committed automated coverage | Actual-model numerical/output coverage | State, MTP, and lifecycle coverage | Promotion guard |
| --- | --- | --- | --- | --- |
| hipEngine | Qwen4Exp CPU formula oracles, per-kernel RED tests, registry/fallback tests, canonical harnesses, c2/server/lifecycle tests | Same-Q4 frozen-llama gate; category+heldout packets; production candidates require 450 rows × 3 repeats with mean/tail/max KL and top-1 by scope | Deterministic repeat/state, graph/eager, restore/replay, physical c2/isolation, cancellation, teardown; MTP must equal true AR on the full prompt suite | **Yes:** named strict/production manifests, calibrated layer scopes, registered strict fallbacks, fail-closed admission |
| Upstream llama.cpp | Broad multi-platform CI, `test-backend-ops`, `test-llama-archs`, save/load and generic recurrent-state tests; #27742 added Qwen4Exp architecture-roundtrip coverage | #27742 development used reduced-model/vLLM and chunking checks; this survey adds same-GGUF and AR checks, but they are not upstream CI gates | Generic state/speculative infrastructure tests; no committed full-size Qwen4Exp MTP equivalence suite found | No profile system; changes are reviewed/merged per upstream tests and evidence, with old paths sometimes retained or env-gated |
| EngramHalo | Inherits upstream tests; adds `test-qsa-gather-ms.cpp` for CPU multi-sequence QSA gather NMSE; exact HEAD has successful cross-build workflows, not a gfx1151 correctness run | Published gather-vs-mask PPL and one 20K top-1 check; local 160-row packet is close to upstream HIP | Documents #25992 multi-slot hazard; no committed MTP AR-equivalence test; local p4096 and MTP checks fail | Env gates and old paths exist, but no mandatory numerical/task/state packet or manifest before a fast path is advertised |
| Nathan | Inherits upstream Qwen4Exp architecture tests and extensive generic Vulkan operator tests; exact `ad914eb` workflow builds/packages HIP and Vulkan but has no test step | Local fixed-batch packet is effectively identical to upstream Vulkan | No committed Qwen4Exp AR repeat/MTP exactness gate found; local 12-case AR fails, and the new short MTP check is only 8/10 exact across repeated AR/MTP modes | Feature env flags permit A/B, but no strict profile, calibrated scope, or fail-closed quality promotion gate |
| apepojken | Inherits llama.cpp tests; the 13-commit custom delta changes no dedicated test file; no code-test action exists at `843d575`; local inherited TOP_K is 445/445 | Commit messages report manual greedy/noise-floor A/Bs; local packet is 160/160 top-1 vs upstream Vulkan | No committed pooled-cache/rollback/MTP end-to-end regression; local AR is 8/12 and MTP 9/10 | Kill switches support manual bisection, but no binding production packet, manifest, strict fallback contract, or CI gate |

Coverage should not be reduced to a test count. Nathan demonstrates why: its
static packet and build workflow look healthy, yet repeated autoregressive
execution fails. EngramHalo demonstrates the converse: one focused QSA test and
PPL A/B are meaningful, but they do not cover MTP or the p4096 state path.
hipEngine's heavier process costs development time, but it explicitly binds
arithmetic, state ownership, determinism, task validity, concurrency, fallback,
and teardown before promotion.

### H4. What the survey supports

- **For correctness-valid AR today:** use upstream llama.cpp HIP/Vulkan or
  hipEngine under its declared profile. Patched upstream HIP is required to
  start this 111 GB artifact on this host and must remain labeled non-stock.
  hipEngine's P8 graph transition is exact research evidence, not yet a public
  production route or a replacement for the matched canonical row.
- **For fastest experimental AR:** apepojken is the strongest raw p1024/p4096
  result measured here, but 4/12 request shapes are nondeterministic. Nathan's
  p512 lead is likewise unusable as a binding target because all 12 cases fail.
- **For external HIP ideas:** EngramHalo is substantially better checked than a
  speed-only reading suggests—its static logits are close to upstream HIP and
  p512/p1024 repeat—but p4096 and MTP still fail explicit gates.
- **For MTP:** none of the fast external rows is retainable under this
  repository's exact-provider contract. The paying paths need their
  transition/verification issue fixed, not an exception to the gate.
- **For absolute model quality:** the available numbers describe the official
  BF16 model or different community quants. They cannot establish that the
  surveyed Q4/Q3 engines retain SWE-bench, GPQA, or coding scores.

### H5. Methodology

#### H5.1 Canonical AR screen

The committed
[`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json)
fixture has SHA-256
`42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`.
It contains exact p512/p1024/p4096 token arrays for code, English, Japanese,
and mixed Japanese/English. Every lane receives the token IDs directly. The
screen uses one warmup, three measured requests, temperature 0, top-k 1,
129 visible output tokens, no prompt cache, `-b 8192 -ub 2048 -t 4`, and warm
OS page-cache state.

The apepojken build is clean commit
`843d5750579a15ed4a42d73eb862855c271021ac`; measured `llama-server` SHA-256
is `fda00d14824075ec37e658c933b73913c84510744c9a364e8b2810186a125f90`.
Its four failing measured cases first diverge at these output indices:

| Case | Unique measured outputs | First differences versus repetition 0 |
| --- | ---: | --- |
| `code-p1024` | 3 | 124, 123 |
| `general_en-p4096` | 2 | no difference, 125 |
| `general_ja-p1024` | 2 | 59, no difference |
| `mixed_ja_en-p4096` | 2 | 19, 19 |

These are generally later than Nathan's identical-prompt control, which often
diverges at indices 1–9, but exact greedy repeatability still fails.

#### H5.2 Full-logit packet

The packet renders the ten canonical prompts with the Qwen chat template and
reasoning disabled. Frozen llama.cpp #27742 HIP generates a 16-token greedy
teacher sequence per prompt. Each engine then evaluates those same sequences
teacher-forced and emits 160 full-vocabulary FP32 rows. Every prompt/teacher
pair is duplicated in the same process to test same-schedule repeatability.

Tested lanes:

- frozen #27742 HIP `bea3b12da`;
- upstream HIP and Vulkan `17252c769`;
- EngramHalo HIP `1423f689` plus its measured loader patches;
- Nathan Vulkan `ad914eb`;
- apepojken Vulkan `843d575`.

The duplicated packet is deliberately stronger than one final-token probe but
smaller than hipEngine's 450-row production-profile gate. Raw logits are local
artifacts; their hashes and per-category statistics are retained in the
[survey JSON](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json).

#### H5.3 MTP equivalence

MTP checks use all ten prompts from
[`mtpbench-code-general-ja.jsonl`](../benchmarks/prompts/mtpbench-code-general-ja.jsonl),
16 output tokens, temperature 0, top-k 1, disabled prompt cache, and canonical
hashes over `reasoning_content` plus `content`. AR and MTP are each repeated to
separate self-instability from AR↔MTP divergence. The apepojken sidecar is
`jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF@69da7334`, 4,135,893,184 bytes,
SHA-256
`713109a7f0dfd5bde305c296b4252daf576aa4c2e380f043f3323aa00dc2cde8`.
It is distinct from EngramHalo's 4,137,429,088-byte sidecar. Nathan uses the
EngramHalo/EasiiX Q8 sidecar at n-max 3; its pinned source explicitly supports
sidecar Qwen4Exp MTP and loaded this artifact successfully.

#### H5.4 What was not tested

- The official 360 GB BF16 checkpoint was not executed on this 128 GB host.
- The published apepojken Q3 headline quant was not downloaded; local matched
  runs use the existing Q4 target.
- No surveyed local lane was rerun on the official SWE-bench/GPQA/LiveCodeBench
  harnesses.
- The external MTP checks are short transition probes, not 70K-session task
  evaluations.
- The canonical speed screen has three repetitions and several noisy rows; it
  is not the five-pair closure protocol in the performance campaign.

### H6. Engine details

#### H6.1 hipEngine

hipEngine production remains the slowest short-context matched lane in this
survey, especially for prefill, but it has the strongest locally retained
correctness packet. Its current canonical row is **83.70/83.16/69.10 pp/s** and
**14.40/14.42/10.42 tg/s**, with all 36 samples deterministic:

- same-Q4 frozen-llama text gate: mean/p95/p99/max KL
  `0.01406/0.04154/0.04776/0.04931`, 10/10 top-1;
- eight predeclared heldouts: mean/max KL `0.00987/0.02874`, 8/8 top-1;
- current production MoE packet: 450 rows × three repeats, mean/max KL
  `2.79e-4/5.98e-3`, 446/450 top-1, all scopes at least 98.67%;
- 18/18 repeat-exact free generation with four task-valid alternatives;
- exact c2/isolation and zero tracked teardown;
- 10/10 AR↔MTP exact, although MTP is 0.955x AR;
- P8 full host-staged graph transition: exact changing-token trajectory, full
  logits, 138 owners, reset/replay, forced-eager/resumption, and clean teardown;
  the 68.855-ms strict runner and 61.910-ms strict graph rows come from separate
  processes, so the derived 1.112x is not a named-production A/B.

The evidence establishes implementation fidelity and profile non-inferiority,
not official benchmark retention. Relevant artifacts are the
[text gate](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-text-bringup.json),
[heldout gate](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-heldout-logits.json),
[production packet](../benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json),
the
[MTP suite](../benchmarks/results/2026-08-28-gfx1151-qwen38-flash-next-mtp-fullsuite-short.json),
the [current canonical row](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json),
and the
[P8 full transition](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json).

#### H6.2 Upstream llama.cpp

Current upstream HIP and Vulkan are repeatable on all 12 canonical AR cases.
Current HIP reproduces the frozen #27742 HIP logit packet nearly exactly.
Vulkan has a larger distribution delta from HIP but remains repeatable; this is
why cross-backend KL or generated-text inequality alone cannot be labeled an
accuracy bug.

Pristine current upstream HIP did not finish loading this 111 GB artifact in
two separate 1,800-second attempts. The numeric HIP row applies the documented
host-buffer and per-buffer-mmap patches and is explicitly non-stock. No
single-patch ablation establishes which patch is individually necessary.

#### H6.3 EngramHalo

EngramHalo's static HIP arithmetic is close to upstream HIP, and p512/p1024
requests repeat exactly. Its two local failures are nevertheless material:

- `code-p4096` alternates from the second generated token;
- MTP differs from stable AR on `general_ja_plan`, producing 9/10 exact rows.

The repository reports Wikitext-2 PPL `4.1466±0.035` for IQ3 and
`4.0430±0.034` for IQ4_XS. Its gathered-QSA A/B reports 4.4601 versus 4.4613
(0.03%). Those are useful quant and mechanism checks, but EngramHalo explicitly
states that it did not run an unpatched-versus-full-patch end-to-end PPL A/B,
and it did not separately measure speculative output identity. Its published
multi-slot warning is also serious: without the #25992 workaround, requests can
receive another slot's response verbatim.

Assessment: credible performance work with partial correctness evidence, not a
fully qualified engine configuration.

#### H6.4 Nathan's Vulkan fork

Nathan's source and release build agree within 1% on synthetic shape rates.
The new logit packet shows that its fixed-batch math is effectively identical
to same-revision upstream Vulkan—160/160 top-1 and mean KL about `2e-10`.
That rules out a broad static Qwen4Exp formula error in this packet.

The autoregressive result is still disqualifying: every canonical case changes
between repeats, and a dedicated control produced 16 unique continuations from
16 identical p1024 prompts. Throughput is stable while tokens are not, so
averaging the fast rates would hide the failure. The new MTP check does not
repair this: complete wall is provisionally 1.161x AR at 95.45% acceptance, but
AR and MTP each self-repeat on only 9/10 prompts and only 8/10 are identical
across both repeats of both modes. The likely fault class is a multi-step
execution/state/graph path; the survey does not localize it further.

Assessment: strong Vulkan static kernels and fast rates, but the measured build
is not suitable for deterministic greedy or MTP use until transition
repeatability is fixed.

#### H6.5 apepojken `qwen4exp-spec-mtp`

The supplied Reddit post is tied to clean commit
`843d5750579a15ed4a42d73eb862855c271021ac`. The custom post-base delta is 13
commits, 23 files, and 1,994 additions / 173 deletions. All 13 commits attribute
Claude/Fable. The repository contains no dedicated test-file addition or
modification for this custom delta, and GitHub shows no branch code-test
workflow—only one dependency-graph run.

That does **not** mean nothing was checked:

- the inherited Vulkan `TOP_K` suite passes 445/445 locally, including
  `k=9999`, ties, and rows through 524,299 columns;
- commit messages record several manual kill-switch A/Bs and greedy comparisons;
- the static packet remains 160/160 top-1 versus upstream Vulkan with a modest
  KL delta.

The missing coverage is exactly where local testing finds failures:

- no committed end-to-end regression test for pooled-QSA state, graph reuse,
  recurrent rollback, MTP AR identity, or the fused epilogs;
- 4/12 canonical AR cases are not repeatable;
- MTP is stable with itself but matches stable AR on only 9/10 prompts.

The source's “noise floor” language is too broad for these outcomes. A
late-token AR flip can be a near tie, but an exact-output claim still fails;
MTP changing a stable target continuation is a provider-correctness failure.
The result does not establish that the reported long sessions were corrupt or
that every optimization is wrong. It establishes that the published evidence
was insufficient to qualify the whole stack.

#### H6.6 Other community configurations

Several useful quality anchors use different checkpoints or quants and cannot
rank the engines above:

- Official Qwen BF16 reports SWE-bench Pro **62.5**, GPQA Diamond **91.7**,
  LiveCodeBench v6 **91.9**, and IFBench **81.3**. These are model-level results,
  not local-GGUF results.
- Cygnal's **abliterated** IQ4XS-NGQ4 artifact reports HumanEval **82.3%** and
  HumanEval+ **78.0%** on a local greedy EvalPlus harness. The checkpoint's
  alignment was intentionally changed, so these scores do not transfer to the
  aligned model.
- Agention's 87.06 GiB ROCmFP4 artifact reports Wikitext-2 PPL
  `4.1062±0.02329` versus a reported unquantized `4.0068±0.02271` (+2.48%).
  This is useful quant evidence, but it also changes tensor format, PLE layout,
  K/V, and fork.
- `cafe-llama.cpp` and other fresh forks remain source leads. No matched local
  rate or correctness packet was available for this survey.

#### H6.7 Pat1entZ3r0 optimization program (source review only, 2026-09-01)

[Pat1entZ3r0/strix-qwen-next-flash-optimization](https://github.com/Pat1entZ3r0/strix-qwen-next-flash-optimization)
is a single-commit (`413c33c`) documentation-plus-patches repository: two
llama.cpp patch lines (`patches/hybrid-04`, a correctness/perf line on master
`c589f0ed` plus #27879; `patches/hybrid-03-mtp`, adding a Qwen4Exp MTP draft
head and recurrent-rollback fixes), model/source lock files with shard
SHA-256s, and prose results/rejected-experiment catalogs claiming 58
controlled experiments on a 128 GB Strix Halo host. Its published rates are
listed in the not-matched-claims table in historical section H1.

Evidence status: **author-reported; nothing locally reproduced.** The
committed `reproduce_final.sh` invokes `scripts/` and `reports/` paths that
do not exist in the repository, so no raw logs, gate captures, or per-rep
data are auditable. Its correctness gate is token parity against
self-captured references at depths 0/4K/16K with a near-tie tolerance
(≤~0.04 nats at fixed positions) — the same gate class that misses the
transition nondeterminism measured in historical section H2, and its wholesale
Nathan-fork evaluation was rejected on depth-decode speed alone with no
correctness finding, even though Nathan fails all 12 canonical AR repeat
cases here. Its "ROCm/HIP is 25–30% slower than Vulkan" claim rests on one
containerized ROCm experiment and conflicts with the matched screen above,
where patched upstream HIP prefill beats Vulkan at p1024/p4096. Treat the
repository as a well-pinned set of leads, not as qualified evidence.

Three items are actionable. **None has been tested in this repository, and we
do not know whether any of them will reproduce, whether EXP-016 actually
fixes the MTP failures measured elsewhere, or whether any of them will help
hipEngine:**

1. **EXP-016 recurrent rollback-ring fix**
   (`patches/hybrid-03-mtp/0006`, derived from apepojken `32af70900` — a later
   commit than the surveyed `843d575`). It claims two ring-buffer bugs in the
   hybrid-memory rollback path — conv-state banks never written beyond group
   0, and stale SSM banks on short batches — producing a ~4.8-nat logit shift
   after draft rejection, and enlarges the spec ring depth to `n_max + 1`.
   This is a plausible explanation for the MTP exactness failures measured in
   the EngramHalo (9/10) and Nathan (8/10) lanes, which use the same EasiiX
   Q8_0 sidecar (4,137,429,088 bytes). Worth doing: (a) audit hipEngine's own
   checkpoint+replay and recurrent-rollback ownership for the same failure
   mode; (b) re-run the historical H5.3 MTP equivalence probe on an external lane
   built from `c589f0ed` + #27879 + EXP-016. Caveat: the patch's own SSM
   clamp is self-described as "best-effort deeper" than exact, so even a
   correct diagnosis may not fully restore MTP output identity.
2. **Depth-conditional draft length** (n-max 2 shallow, n-max 6 at ≥32–64K;
   reported code acceptance decaying from 0.94–0.97 shallow to 0.75–0.88 at
   128K). A candidate input for hipEngine MTP economics. Any adoption
   requires the full mtp-bench category suite plus heldouts under the
   exact-provider contract; the published acceptance figures are code-only
   and author-reported.
3. **Opt-in pooled-key QSA cache** (`LLAMA_QSA_POOL_CACHE=1`; reported
   +38–40% decode at 32–64K with a flat depth slope). The QSA selection it
   produces is explicitly non-bit-exact with no published KL/top-1
   quantification. It is a port candidate for the QSA decode gap identified
   in historical section H1, but only behind the production-profile numerical
   gates, which it may fail.

### H7. Required evidence for a qualified fast fork

A practical minimum is smaller than hipEngine's full production campaign but
larger than a speed screenshot:

1. Pin source, binary, target parts, quant, K/V, sampler, and host identity.
2. Run the 12-case canonical AR screen and require each case to repeat exactly.
3. Run a same-GGUF full-logit packet against an upstream build on the same
   backend; report mean/tail/max KL and top-1 by category.
4. For MTP, compare every full message against true same-protocol AR across the
   complete category plus heldout suite. Acceptance is secondary.
5. Exercise reset, rollback, prompt-cache reuse, session save/restore, and at
   least two slots if multi-slot is advertised.
6. Run one task benchmark appropriate to the quant. Official BF16 scores are
   context, not inherited proof.
7. Keep the speed row only if all binding checks pass. Record failing fast rows
   rather than averaging them away.

### H8. Evidence index

- Compact survey evidence:
  [`2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json`](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json)
- Existing canonical AR matrix:
  [`2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json)
- Current hipEngine canonical AR row:
  [`2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json`](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json)
- Current canonical impact profile:
  [`2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json)
- P8 full host-staged graph transition:
  [`2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json)
- Existing external-fork refresh:
  [`2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json)
- Model/implementation authority:
  [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md)
- Active performance protocol:
  [`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md)
- Official model card: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- apepojken source: [apepojken/llama.cpp](https://github.com/apepojken/llama.cpp)
- apepojken sidecar: [jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/jockevaupptaget/Qwen3.8-Flash-Next-MTP-GGUF)
- EngramHalo source and methodology: [Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp)
- Nathan release: [Nathanw1014/strix-halo-llamacpp v0.7.2](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.7.2)
- Pat1entZ3r0 source review (historical section H6.7): [Pat1entZ3r0/strix-qwen-next-flash-optimization](https://github.com/Pat1entZ3r0/strix-qwen-next-flash-optimization) `413c33c`
