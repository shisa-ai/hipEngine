# AGENTIC-QUALITY2 — ZBook Agent Quality Campaign

- **Status:** complete; AQ0-AQ13 closed on the retained no-go path
- **Approved:** 2026-08-25
- **Execution host:** `zbook`, HP ZBook Ultra G1a, Radeon 8060S / `gfx1151`
- **Primary model:** Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV
- **Comparison models:** Qwen3.8-27B `Q4_K_M` and Ornith-1.5-35B-A3B
  `Q4_K_M`, only after independent loader/template admission
- **Campaign class:** downstream task quality and runtime semantic correctness;
  `performance_claim=false` throughout
- **TaskList:** #40–#53 map to AQ0–AQ13
- **Prior evidence:** [`AGENTIC-OPT.md`](AGENTIC-OPT.md) A6
- **Normative dependencies:** [`PLAN.md`](PLAN.md), [`AGENTIC.md`](AGENTIC.md),
  [`AGENTIC-OPT.md`](AGENTIC-OPT.md), [`SAMPLING.md`](SAMPLING.md),
  [`API.md`](API.md), [`TESTING.md`](TESTING.md), and
  [`BENCHMARK.md`](BENCHMARK.md)

This campaign is independent of [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md).
SPECDEC2 performance remains owned by its stable-hardware lane. AGENTIC-QUALITY2
must not edit speculative kernels, reuse its rates, or delay its commits.

---

## 1. Executive decision

Use the ZBook for quality work whose verdict is insensitive to thermal and
power variation. Do **not** start by implementing a broad grammar engine or by
running generic leaderboard tasks without a product decision. First reproduce
the current live automatic-tool baseline, classify each failure as runtime or
model behavior, freeze disjoint development and heldout task sets with external
or executable oracles, and only then admit at most one model-general runtime
mechanism.

The historical W7900 A6 packet is intentionally sobering: Qwen3.6-35B
`UD-Q4_K_M` completed 10/48 automatic-tool turns; valid call/correct tool was
18/48, external-oracle pass was 16/48, safe patch success was 0/6, and Japanese
complete success was 0/8. That packet ran at commit `878d07a9...` on another
backend/host. Current source has since changed tool constraints, parsing,
serving, sampling, and lifecycle behavior. The old packet nominates questions;
it is neither the ZBook baseline nor an old→new denominator.

The first live row therefore uses the existing v2 suite unchanged on current
main. The expanded suite and any implementation begin only after that row and a
failure taxonomy are durable.

## 2. Objective and completion criteria

The campaign answers four questions:

1. What quality does current hipEngine deliver for automatic tool use on the
   primary ZBook model under real OpenAI-compatible requests?
2. Which failures are caused by model selection/arguments versus tokenizer,
   template, parser, constraint, validation, publication, or lifecycle bugs?
3. Does one model-general runtime mechanism improve development **and** heldout
   task success without weakening OpenAI semantics or fail-safe behavior?
4. Do local Qwen3.8 and Ornith artifacts inherit that mechanism safely, and are
   any cross-model quality differences useful for product selection?

AGENTIC-QUALITY2 closes only when:

- the current v2 baseline is complete and every failure is classified;
- a versioned development/heldout suite and independent oracles are committed
  and mechanically validated before candidate runtime code;
- the primary-model baseline and comparison-model admission/screens are
  recorded from clean source;
- exactly one general mechanism is retained, rejected, or explicitly declared
  no-go from measured evidence;
- endpoint, session, continuation, cancellation, streaming, parser, sampler,
  MTP-blocker, and ownership gates pass for any retained change;
- post-change primary and cross-model quality packets are complete, or a precise
  load/capability blocker is durable;
- retained and rejected evidence is published without a speed claim; and
- local/remote source equality, Worklog2, schema, fixture, link, benchmark-sync,
  and applicable milestone checks pass.

No quality increase is required to call the campaign executed. A valid no-win
or model-quality diagnosis is closure. A partial task matrix, unclassified
failure, inspected-but-unfrozen heldout set, or candidate without complete
semantic gates is not closure.

## 3. Frozen campaign identity

### 3.1 Host and software

AQ0 recorded the planning host as:

| Field | Frozen value / rule |
| --- | --- |
| Host | `zbook`, HP ZBook Ultra G1a 14-inch Mobile Workstation |
| CPU/APU | AMD Ryzen AI MAX+ PRO 395 |
| GPU | Radeon 8060S, `gfx1151`, 40 CUs, unified memory |
| Kernel at AQ0 | `7.2.0-1-cachyos` |
| HIP compiler at AQ0 | HIP `7.15.0-0000000`, AMD clang 23 |
| Python at AQ0 | 3.13.13; runtime minimum still follows `pyproject.toml` |
| amdgpu scheduler | `sched_policy=0` |
| TTM page limit at AQ0 | `32,505,856` pages; record, do not alter |
| Power at AQ0 | STAPM/fast/slow `71/71/60 W`; record, do not alter |
| Source planning base | `16df9926f8f770829027764f5c05cbcfca13b867` |

Power, temperature, and memory remain provenance even though no timing metric is
accepted. Do not alter firmware carve-outs, TTM limits, power, clocks, fan,
IOMMU, or thermal policy. A later source sync records its own exact commit and
invalidates only affected code assumptions, not frozen fixture identities.

### 3.2 Models

| Role | Path | Bytes | Full SHA-256 | Initial status |
| --- | --- | ---: | --- | --- |
| Primary | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,663,387,424 | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | admitted by prior ZBook quality/runtime campaigns; must re-audit current route |
| Dense comparison | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | candidate; independent tokenizer/template/tool admission required |
| MoE comparison | `/models/gguf/Ornith-1.5-35B-A3B-Q4_K_M.gguf` | 21,166,758,080 | `ec50607a13596387b362fecf70aa887b05608bd428f1b6276b6ed4d546647aeb` | candidate; architecture and tool-template admission required |

Each live artifact records GGUF architecture, tokenizer/chat template, special
and EOS tokens, served model identity, quant, effective KV, backend, selected
execution profile/manifest where available, context cap, and capability
manifest. No model inherits another model's template or quality verdict.

### 3.3 Existing v2 inputs

| Input | SHA-256 |
| --- | --- |
| `benchmarks/prompts/agentic-quality-v2.json` | `74903e7225deebcab4427f08eb9a5d4a64015d86c24d134ecbcbde78abb50652` |
| `benchmarks/oracles/agentic-quality-v2.json` | `1252c2b0af2492f24c3d4bce5437e1806b6a771753574ed27af5ecff60f3f2b9` |
| `benchmarks/schemas/agentic-coding-quality-benchmark.schema.json` | `516319b209e0e3eb727229b3fcbce7e4dce76f42f039fb56d08f5f8edb5f4480` |
| `scripts/agentic_coding_quality.py` at AQ0 | `0efb78f0a5d9bf2c12b10360ffc37e98e505948b40d88025be3d610af8581c93` |

AQ2 runs these committed inputs directly. If AQ1 finds a harness correctness
bug, it commits the repair and records old/new hashes before AQ2; it does not
silently mutate fixture wording or oracle answers.

## 4. Scope and non-goals

### In scope

- real localhost `/v1/chat/completions` automatic-tool behavior;
- blocking response quality first; streaming parity where a runtime change
  touches publication or envelope handling;
- tool selection, argument JSON/schema, external result, patch, test, code, and
  instruction-following quality;
- English, Japanese, and mixed Japanese/English families;
- exact response-owned token hashes/counts and deterministic repeats;
- parser/template/constraint/validation/repair/session ownership;
- one evidence-admitted model-general implementation mechanism;
- primary and bounded comparison-model evidence; and
- compact quality artifacts with raw records outside Git when large.

### Not in scope

- tok/s, TTFT, ITL, goodput, or power-efficiency comparisons;
- changing model weights, quantization, prompts, chat template, or system policy
  to fit observed benchmark answers;
- SPECDEC2/MTP acceptance or economics;
- native-sampler performance promotion;
- prefix-cache, routing, scheduler, or concurrency performance tuning;
- autonomous unrestricted repository modification;
- arbitrary generated-code execution on the host;
- a broad grammar engine without measured need;
- OpenAI Responses API, multi-model serving, TP, or model substitution; or
- claims that synthetic or bounded public subsets are a general leaderboard.

Wall time may be recorded only as operator progress/timeout evidence. Quality
artifacts set `performance_claim=false` and must not contain headline latency,
throughput, or speedup fields.

## 5. Quality questions are separate

Do not collapse these into one score:

1. **Runtime implementation correctness:** same intended artifact/context,
   tokenizer/template, parser, schema, and publication behavior; no corruption,
   leakage, or cross-request ownership failure.
2. **Model task quality:** whether the model chooses a suitable tool, arguments,
   patch, code, or instruction response under a valid runtime.
3. **Quantization quality:** BF16-relative distribution drift already belongs to
   `benchmarks/quant/`; downstream task results are additive and do not relabel
   a failed exact-artifact quant gate.
4. **Cross-model product quality:** useful for model selection but not evidence
   that one runtime implementation is more correct.

When a row fails, AQ3 must identify the earliest observable boundary. A parser
failure after a syntactically valid model envelope is runtime failure. A valid,
fully published call selecting the wrong declared tool is model quality. If the
boundary cannot be established, classify it `unresolved`, which blocks a
mechanism decision.

## 6. Failure taxonomy

Every attempted turn receives one primary outcome and optional contributing
causes:

| Primary outcome | Meaning |
| --- | --- |
| `passed` | Valid declared call and external oracle/task succeeds. |
| `no_tool_call` | Auto mode returns no executable call. This can be legal API behavior but fails a tool-required task oracle. |
| `wrong_tool` | Valid call names a declared but incorrect tool. |
| `undeclared_tool` | Envelope names a tool outside the request set. |
| `malformed_envelope` | Tool marker/envelope cannot be parsed safely. |
| `invalid_json` | Selected argument payload is not valid JSON. |
| `schema_violation` | JSON is valid but violates the selected strict schema. |
| `wrong_arguments` | Schema-valid arguments fail the task oracle. |
| `content_alongside_tool_call` | Public content violates the task's tool-only contract. |
| `raw_markup_leak` | Raw reasoning/tool control text reaches public content. |
| `length_exhausted` | Output cap ends before a valid terminal envelope. |
| `parser_mismatch` | Independent parser accepts the envelope but public runtime rejects/misprojects it. |
| `template_or_tokenizer_mismatch` | Rendered tokens/control vocabulary do not satisfy the declared model/template contract. |
| `external_oracle_failure` | Valid selected action executes but task result is wrong. |
| `runtime_error` | HTTP/server/generation failure unrelated to the model answer. |
| `unresolved` | Evidence is insufficient to locate the first bad boundary. |

Contributing fields include selected/expected tool, JSON and schema validity,
independent parser verdict, finish reason/details, generated count/hash, first
invalid token/span where available, repair attempt/outcome, external oracle
kind/result hash, session commit action, and final ownership.

The classifier never rewrites a model failure into a runtime pass. Exact
argument equality remains diagnostic when a semantically equivalent expression
passes an external oracle.

## 7. Anti-overfit and split policy

### 7.1 Development versus heldout

- Existing v2 is a **historical/development diagnostic** because its prompt and
  outcome details are already public in the repository.
- AGENTIC-QUALITY2 v1 creates distinct `development` and `heldout` task IDs.
- All prompts, tools, schemas, oracle cases, source/license records, and hashes
  are committed before AQ8 candidate code.
- AQ6 may report aggregate heldout baseline totals, but candidate selection uses
  development failures and general runtime invariants only. It may not inspect
  heldout token streams to design a fix.
- AQ11 opens heldout detail only after candidate behavior and tests are frozen.
- A retained mechanism must be non-regressive in every family and improve or
  safely neutralize the heldout packet. Development-only improvement is a
  rejection.
- If a heldout infrastructure defect is discovered, freeze a new suite version
  and rerun both baseline and candidate. Never patch an oracle after seeing a
  candidate answer and continue to call it heldout.

### 7.2 Forbidden behavior

Never branch on prompt text/hash, token IDs, fixture/task/category/heldout ID,
expected tool/argument, model answer, generated result hash, or benchmark
position. Never add a prompt-conditioned retry, expected-tool forcing, candidate
rerank, or fixture-specific parser exception.

Allowed mechanism inputs are request-declared semantics: tokenizer/template
capabilities, `tools`, `tool_choice`, strict JSON Schema, response format,
canonical processor state, bounded generic repair count, and explicit public
policy. Model/backend/quant capability selection remains cold and registered,
not a hot-path string branch.

### 7.3 Evidence use

- Repeated deterministic runs prove repeatability, not independent statistical
  sample size.
- Confidence intervals resample independent task/oracle blocks, not duplicate
  runs of the same deterministic row.
- Do not select a model or mechanism from one family or one language.
- No single aggregate can hide a zero/failure in Japanese, patch, code, or
  lifecycle gates.

## 8. Evaluation suite contract

AQ4 evaluates source/license and availability before committing external data.
The intended families are:

| Family | Minimum question | Oracle |
| --- | --- | --- |
| Tool selection | BFCL-style single, multiple, irrelevant, nested, enum, and optional-argument calls | declared tool + schema + executable result |
| Repository read/search | inspect/search committed hermetic files | exact file/search result hash |
| Patch | select or produce a bounded one-region change | apply in isolated copy + expected tests/file hashes |
| Code | HumanEval/MBPP-style pure functions | isolated compile + hidden tests |
| Instruction | IFEval-style count/format/keyword/language constraints | deterministic checker per instruction |
| Japanese/mixed | tool, repository, arithmetic, and instruction rows | same executable/schema checks, not translation-string equality |
| Safety/fail-safe | malformed, ambiguous, undeclared, duplicate, truncated, and content-leak envelopes | exact reject/withhold/finish/session-commit contract |

“BFCL-style”, “HumanEval/MBPP-style”, and “IFEval-style” are not claims that an
official score was reproduced. AQ4 may use an upstream dataset only after
pinning source revision, license, exact split, adapter, and evaluator. Otherwise
commit original bounded tasks and label them `style`, never the upstream name.
Do not vendor a large corpus casually.

### 8.1 Generated-code and patch sandbox

Model output is untrusted. AQ5 must prove the sandbox before any generated code
runs:

- one fresh temporary directory per case;
- no network and no inherited secrets/API keys;
- read-only source fixture plus writable bounded copy;
- strict wall/CPU/memory/file-size/process limits;
- no device files, GPU, home, repository, model, or arbitrary host filesystem
  access;
- fixed interpreter/toolchain identity;
- stdout/stderr/output truncation;
- kill the complete process group on timeout;
- record command, exit/signal, resource limit, and output hashes; and
- fail closed when the required isolation facility is unavailable.

If secure isolation cannot be proven on this host, code rows remain
`blocked_sandbox` rather than being executed with weaker controls.

### 8.2 Minimum suite size

AQ4 chooses the exact size after source/license audit, but closure requires:

- at least four families;
- at least one development and one heldout row per retained family;
- at least four Japanese or mixed-language heldout rows;
- at least four patch/code executable heldout rows combined; and
- enough independent task blocks to report counts by family without treating
  deterministic repeats as new tasks.

The first suite can be bounded. It must be broad enough to falsify a mechanism,
not to claim general coding intelligence.

## 9. Metrics and gates

Report overall, split, family, language, tool/schema class, and oracle kind:

- attempted and terminal turns;
- valid call, declared/correct tool;
- valid JSON and strict-schema pass;
- exact-argument rate as diagnostic;
- external-oracle/task success (**primary quality metric**);
- patch apply/test success;
- code compile/test success;
- instruction-constraint pass;
- no-call, leakage, malformed, truncated, and runtime-error counts;
- repair attempted/succeeded/failed and extra generated tokens;
- deterministic normalized-response equality across repeats;
- response-owned generated token count/hash;
- final request/session/KV/graph/workspace/stream ownership; and
- blocked/unscorable rows with exact reasons.

A runtime mechanism is retainable only when:

1. every endpoint/fail-safe/lifecycle binding gate passes;
2. development external-oracle success improves for the targeted general
   failure class, unless the change is a pure correctness repair whose exact bad
   boundary is eliminated;
3. heldout overall and every family are non-regressive;
4. no new raw markup, unsafe public content, schema bypass, hidden-reasoning
   commit, retry loop, or ownership leak appears;
5. deterministic rows remain deterministic;
6. comparison models either pass unchanged, improve, or fail closed through a
   declared capability rather than a model-name branch; and
7. raw and compact artifacts are complete and provenance-clean.

There is no minimum percentage threshold and no speed gate. A mechanism that
turns invalid calls into valid but wrong actions is not a win; external-oracle
success, not parser acceptance, decides quality.

## 10. Model matrix

### 10.1 Primary row

Qwen3.6-35B is the implementation decision row because it has prior exact
artifact quality, a deployed gfx1151 runtime, the historical A6 comparison, and
an MTP-bearing tool-capable GGUF. Use:

- BF16 KV;
- c1 / max active requests 1;
- cache off;
- native sampler off;
- automatic MTP off;
- `temperature=0`;
- `tool_choice=auto` unless a safety fixture explicitly tests another declared
  choice;
- reasoning off;
- bounded output cap declared by suite; and
- real localhost blocking chat for quality collection.

These settings isolate tool quality from sampling, prefix, concurrency, and
speculation. A separate safety fixture may exercise streaming or fixed tool
choice without entering the primary score.

### 10.2 Comparison rows

AQ7 first audits each model's architecture, chat template, tool control tokens,
EOS behavior, loader route, effective profile, and public capability. It then
runs the identical rendered semantic workload through that model's own admitted
tokenizer/template. Cross-model prompt token IDs need not match; source task,
tool schema, system policy, and oracle do.

A comparison model can be:

- `admitted_complete` — full matrix ran;
- `admitted_partial` — only predeclared subset fits/supports, with no aggregate
  rank;
- `blocked_loader`, `blocked_template`, `blocked_capability`, `blocked_memory`,
  or `blocked_runtime`; or
- `rejected_semantics` when gross template/parser mismatch makes the advertised
  route unsafe.

No fallback silently substitutes Qwen3.6 for a requested comparison model.

#### AQ7 admitted comparison result

Both comparison artifacts are `admitted_complete` at clean source
`1f9c23418`: strict profile, BF16 KV, c1, cache/native sampler/automatic MTP
off, `temperature=0`, reasoning off, and the frozen 192-token cap. Each first
passed the same two-repetition development smoke with exact normalized equality,
10/10 independent policy controls, and zero transient ownership. Only one
model-owning process held `/dev/kfd` at a time.

| Model | Overall | Development | Sealed heldout | Code | Instruction | Repository | Tool selection | Valid calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B `UD-Q4_K_M` (AQ6 reference) | 44/68 (64.71%) | 20/34 | 24/34 | 14/16 | 4/16 | 10/16 | 16/20 | 56/64 |
| Qwen3.8-27B `Q4_K_M` | **50/68 (73.53%)** | **22/34** | **28/34** | 14/16 | **12/16** | 10/16 | 14/20 | **64/64** |
| Ornith-1.5-35B-A3B `Q4_K_M` | 42/68 (61.76%) | 16/34 | 26/34 | 14/16 | 4/16 | 10/16 | 14/20 | 60/64 |

All 34 normalized response pairs match for each model, all model-specific rows
are scored, and neither comparison has a malformed public argument, raw markup
or content leak, truncation, runtime error, blocked sandbox, or final ownership
delta. Qwen3.8's +6 passes versus the primary reference are an instruction
family gain (+8) offset by tool-selection loss (-2). Ornith matches the primary
code/instruction/repository totals but loses two tool-selection observations.
These are same-host, same-suite **product-quality** differences between model
artifacts, not implementation-correctness, quantization, or speed deltas.

Independent metadata admission found:

- Qwen3.8 resolves through the registered `qwen35` dense plugin: 64 AR blocks
  (48 linear/16 full attention), 851 mapped tensors with zero missing,
  unexpected, or shape errors, and one complete trailing GGUF MTP block. Its
  strict manifest is `0e053fd658cacbc72a67b7e7ccf33927502a0fe9379faf14dac30ae6ee2bb65b`.
- Ornith resolves through the registered `qwen35moe` plugin: 40 blocks (30
  linear/10 full attention), 256 experts/top-8, and 733 mapped tensors with zero
  missing, unexpected, or shape errors. No trailing GGUF MTP block is present;
  speculative serving remained explicitly off. Its strict manifest is
  `6f6af11c059dbd02d458405bfa9d70ef0ea4a38daaa47a884c4f52b7a489fc41`.
- Both artifacts have byte-identical Qwen tokenizer vocabulary, merges, and
  token types to the primary, singleton control IDs `248045/248046` for
  `<|im_start|>/<|im_end|>`, `248058/248059` for tool-call open/close, and
  `248068/248069` for think open/close; EOS is `248046`. Embedded GGUF template
  hashes differ, so admission did not infer one model's template from another.
  The current Qwen35 generator exposes the server's generic registered Qwen
  renderer; identical 1,544-token live smoke prompts, exact nested tool calls,
  public capability manifests, and clean full matrices admit that route for
  each exact artifact.

Compact quality-only evidence:
[`Qwen3.8`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq7-qwen38-comparison.json)
and
[`Ornith`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq7-ornith15-comparison.json).
Raw outputs remain local and are not inputs for AQ8 tuning.

## 11. Candidate admission

AQ8 admits **at most one** mechanism after AQ3/AQ6/AQ7. Candidate order is not a
promise:

1. parser/template correctness repair when an independent parser/tokenizer
   oracle proves current runtime mishandles valid model output;
2. complete token-level argument constraint from the request's declared strict
   JSON Schema after a tool envelope begins;
3. one bounded, explicit invalid-tool repair attempt with no commit of the
   invalid output and complete telemetry;
4. bounded patch/diff constraint when patch syntax—not model patch choice—is
   the dominant failure; or
5. no implementation when failures are valid model decisions or no general
   mechanism has a credible heldout benefit.

The declaration names:

- measured development failure count/share and earliest boundary;
- exact request/capability scope;
- semantics of auto, required, specific, and no-tool behavior;
- RED fixture and independent oracle;
- fallback/reject behavior;
- maximum repair count and token/resource budget, if any;
- session/continuation commit policy;
- streaming publication behavior;
- sampler/MTP/native blockers;
- development objective and heldout non-regression gate;
- comparison-model expectation; and
- removal/rejection rule.

A broad grammar engine, model-specific prompt patch, or mechanism without a
measured complete-task premise is no-go.

## 12. Phase plan and punchlist

### AQ0 / Task #40 — campaign ledger

- [x] Freeze host, source, model, and existing v2 identities.
- [x] Separate quality from performance, quantization, and cross-model claims.
- [x] Define split/anti-overfit, failure taxonomy, sandbox, metrics, admission,
      artifacts, stop rules, and closure.
- [x] Link PLAN and AGENTIC status; create immutable worklog.
- [x] Re-read changed docs, validate links/Worklog2/sync/diff, commit, push.

No GPU run.

### AQ1 / Task #41 — current stack audit

- [x] Create a clean campaign worktree from current `origin/main` and record
      exact base.
- [x] Audit server parser, Qwen/Poolside templates, tokenizer constraints,
      structured validation, repair queues, commit policy, collector, schema,
      and capabilities.
- [x] Diff relevant behavior since `878d07a9...`; do not assume all historical
      path commits affect the live row.
- [x] Verify v2 oracle cases independently.
- [x] RED malformed/stale provenance and the discovered exact-argument scoring
      bug.
- [x] Fix only harness/provenance correctness needed for AQ2.
- [x] Run focused benchmark/agentic tests; publish worklog and commit.

#### AQ1 result — current live contract and harness repairs

The clean campaign worktree is `/home/lhl/hipEngine-agentic-quality2`, branch
`agentic-quality2`, starting at AQ0 commit `58d055872...`. The path audit found
36 commits touching the broad server/sampling/tokenization/quality path since
the old A6 source, but most are lifecycle, SPECDEC2, merge, or other-model work.
The quality-relevant current Qwen contract is:

- Qwen uses the generic canonical `<tool_call>` parser, not the later
  Poolside/Laguna model-owned parser.
- With tokenizer support, `tool_choice=auto` constrains a started tool branch to
  one declared tool name plus a canonical envelope and syntactically valid root
  argument object. It still permits a plain-text branch, as OpenAI auto
  semantics require.
- Full declared JSON Schema validation is post-generation. The decode-time
  prefix anchor reaches the first required string key only for required/specific
  single-tool shapes; it does not make automatic multi-tool arguments
  schema-complete.
- Close repair is bounded to a tokenizer-safe marker/object suffix once the
  current prefix is structurally completable. There is no automatic second
  generation/repair request.
- Invalid, undeclared, malformed, schema-violating, content-leaking, and
  required-tool-missing outputs are withheld/fail closed; unsafe session commits
  downgrade to prompt-only or none. Blocking and SSE contract coverage exists.
- Native sampling and speculative MTP remain incompatible with the dynamic tool
  processor surface; AQ2 explicitly disables both.

The old collector was not valid for AQ2 unchanged. AQ1 observed and fixed one
RED scoring defect: broad external-oracle rows still required exact argument
text before consulting the oracle, so semantically equivalent successful
arguments were labeled `wrong_arguments`. External-oracle success now decides
broad task success; exact arguments remain diagnostic. Legacy suites without an
external oracle still require exact arguments.

AQ1 also replaces the hardcoded `gfx1100` provenance label with the selected
backend, binds and hashes the live server capability payload (served model,
backend, tokenizer, tools, and cache checked before generation), records output
cap/repetition count, emits flushed per-turn progress, atomically checkpoints
normalized rows plus local raw responses/prompt IDs, atomically writes final
JSON, and computes normalized deterministic repeat equality while ignoring
random call IDs. The checkpoint supplies the response-owned IDs needed to
reconstruct pre-parser model text during AQ3. AQ2 live startup additionally
proved that Qwen keeps a fixed prepared KV allocation pinned while idle; the
collector records those pre-request refcounted/pinned pages and still requires
strictly zero request-owned deltas at completion rather than mislabeling stable
model-lifetime allocation as a leak.

All 24 committed v2 oracle cases execute independently. Focused quality/oracle,
server-conformance, and harness-trace tests pass; exact commands and counts are
in the AQ1 worklog.

### AQ2 / Task #42 — current v2 baseline

- [x] Start one clean Qwen3.6 server with frozen settings and explicit compiler
      version/cache.
- [x] Run all six v2 workloads twice (24 turns/run).
- [x] Require response-owned IDs, normalized repeat equality, valid artifact,
      and zero final request/session ownership relative to the recorded idle
      persistent-allocation baseline.
- [x] Keep raw records/logs under `/tmp/hipengine-agentic-quality2/<run-tag>/`.
- [x] Publish compact baseline artifact with no performance fields.
- [x] Update campaign/worklog/quality rollup and commit.

#### AQ2 result — qualified current ZBook denominator

The retained run uses commit `fef66d0fd...` and raw directory
`/tmp/hipengine-agentic-quality2/20260825-aq2-v2-qwen36-r2-181806/`.
Its compact artifact is
[`2026-08-26-zbook-agentic-quality2-aq2-baseline.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq2-baseline.json).
It contains no latency, throughput, goodput, speedup, or profiler result.

| Scope | External-oracle success | Valid call | Correct tool | Exact arguments |
| --- | ---: | ---: | ---: | ---: |
| Overall, 48 repeated observations | **38/48 (79.17%)** | **48/48 (100%)** | **46/48 (95.83%)** | **38/48 (79.17%)** |
| Repository | 12/16 (75%) | 16/16 | 14/16 | 12/16 |
| General English | **14/16 (87.5%)** | 16/16 | 16/16 | 14/16 |
| General Japanese | 6/8 (75%) | 8/8 | 8/8 | 6/8 |
| Mixed Japanese/English | 6/8 (75%) | 8/8 | 8/8 | 6/8 |

The 24 independent task blocks are 19/24 successful: all four
`general_en_operations` turns pass; the other five workloads are 3/4. The five
unique failures repeat exactly: four selected the expected tool but failed the
executable result oracle, and one selected the wrong declared tool. There are
no malformed/invalid/schema-invalid/no-call/content-leak/raw-marker/runtime
failures, no repair attempts, and all six patch plus all eight test observations
pass. The 24 normalized repeat pairs are exact after excluding random call IDs;
all 856 generated IDs are response-owned.

Initial model-lifetime ownership is 128 refcounted/128 pinned KV pages and final
request/session/KV/graph/workspace/stream deltas are all zero. Server shutdown
leaves no `/dev/kfd` owner. The first two pre-evidence attempts are excluded: one
found the repaired Qwen detokenization surface, and one populated a missing
AOTriton cache then exposed absolute-versus-baseline ownership accounting. No
partial attempt contributes a quality row.

This is not an old→new comparison with the historical W7900 row: host, backend,
source, constraint stack, and broad-oracle scoring differ. AQ3 owns the five
unique failure classifications; AQ2 admits no implementation mechanism.

### AQ3 / Task #43 — failure taxonomy

- [x] Classify every AQ2 row under Section 6.
- [x] Compare public response, raw generated IDs/text where retained, independent
      parser, tokenizer/template controls, finish details, and oracle execution.
- [x] Record earliest bad boundary and runtime/model/unresolved owner.
- [x] Add classifier tests and a compact taxonomy artifact.
- [x] Name candidate classes by aggregate failure evidence only; no implementation.
- [x] Commit.

#### AQ3 result — five model-quality failures, zero runtime/unresolved

The tested standalone classifier joins all 48 normalized rows one-to-one with
raw checkpoint responses, verifies response-owned IDs, reconstructs pre-parser
text from the exact GGUF tokenizer, and parses the generic Qwen envelope without
calling the server parser. The compact result is
[`2026-08-26-zbook-agentic-quality2-aq3-taxonomy.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq3-taxonomy.json).

| Primary outcome / earliest bad boundary | Independent task blocks | Repeated observations | Owner |
| --- | ---: | ---: | --- |
| `passed` / none | 19 | 38 | none |
| `wrong_arguments` / model-generated arguments | 4 | 8 | model quality |
| `wrong_tool` / model tool selection | 1 | 2 | model quality |
| Runtime implementation failure | **0** | **0** | — |
| Unresolved | **0** | **0** | — |

All 48 prompt token sequences decode→encode exactly. All 48 raw model texts are
single canonical `<tool_call>` envelopes accepted by the independent parser,
and every parsed tool/argument projection equals the public OpenAI response.
All calls are schema-valid, terminate with `tool_calls`, contain empty public
content, use zero repair attempts, and execute their external oracle. Final
request-owned state remains zero; requests are stateless, so there is no session
commit scope.

The four argument failures are concrete model substitutions: query prose for
`ValueError`, a file path for the requested directory, `release_validation` for
`release_check_ja`, and `docs/release.md` for `docs/release_ja.md`. The selection
failure chooses valid declared `read` instead of expected `grep`. Therefore AQ3
nominates only aggregate classes `model_argument_grounding` (4/24) and
`model_tool_selection` (1/24). It admits no implementation. The detokenization
and persistent-ownership defects remain separately recorded pre-evidence
runtime repairs and do not relabel model answers.

### AQ4 / Task #44 — freeze expanded suite

- [x] Audit public dataset source/license/revision and local availability.
- [x] Choose bounded original versus upstream-derived tasks honestly.
- [x] Freeze development/heldout IDs before candidate code.
- [x] Add tool, repository, patch, code, instruction, Japanese/mixed, and
      fail-safe rows under the minimum coverage rule.
- [x] Add external oracle source data without expected answer leakage into
      prompts.
- [x] Record canonical hashes and generation/adaptation process.
- [x] Commit suite/schema/oracle/docs/worklog as one unit.

#### AQ4 result — project-original 17/17 split frozen

[`AGENTIC-QUALITY2-SUITE.md`](AGENTIC-QUALITY2-SUITE.md) is the fixture card.
It pins the exact six input/schema hashes, source/license revisions, creation
method, split IDs, heldout policy, and AQ6 checkpoint plan. BFCL (Apache-2.0),
HumanEval (MIT), MBPP (CC-BY-4.0), and IFEval (Apache-2.0) were audited at exact
revisions; local HumanEval/MBPP copies exist under read-only `/home/lhl/omlx/`,
but no upstream task, solution, assertion, test, or evaluator byte is imported.
The suite is AGPL project-original and claims no official upstream score.

Fake-endpoint validation superseded v1 with v2 before any model or candidate
run: the schema-invalid control now expects public `schema_violation`, and the
reasoning-leak control is genuinely malformed (`</think>` without an opener).
No task prompt, oracle answer, split ID, or result was changed. The frozen v2
suite contains 34 independent one-turn blocks: 17 development and 17 heldout,
with 5/5 tool-selection, 4/4 repository, 4/4 code, and 4/4 instruction
rows. It includes single/nested/enum/optional/multiple/irrelevant automatic-tool
shapes, read/search/patch/test, eight code entry points, eight behavioral
instruction cases, 12 Japanese/mixed heldout blocks, five heldout patch/code
blocks, and ten separate fail-safe controls. Expected result hashes, hidden code
cases, patch replacement text, and reference results remain oracle-only;
machine-readable instruction checks mirror public request constraints but store
no reference prose. Expected code source and exact response prose are not stored.

AQ4 executes no generated code and admits no mechanism. AQ5 must RED-test the
new loader/schema relationships, independently prove every oracle, and establish
or fail closed on the sandbox. Once qualified, AQ6 is 68 repeated model
observations plus controls, estimated at 15–45 minutes after cached startup with
atomic development→heldout→control checkpoints.

### AQ5 / Task #45 — fixture/oracle/sandbox validation

- [x] RED loaders and schemas for duplicate IDs, split overlap, missing language,
      oracle mismatch, expected-answer prompt leakage, and malformed counts.
- [x] RED sandbox network/filesystem/process/resource escapes.
- [x] Prove every committed oracle independently.
- [x] Prove deterministic artifact aggregation and large-raw/compact separation.
- [x] Run focused tests, fixture checker, Ruff/compile, Worklog2/sync/diff.
- [x] Commit.

#### AQ5 result — fixture and generated-code sandbox qualified

The fail-closed loader cross-validates exact file references, 17/17 membership,
family/language/executable cardinalities, one-to-one same-split cases, tool and
argument shapes, code/instruction references, source-use policy, fail-safe
classes, and hidden-answer leakage. It rejects duplicate IDs, overlap, missing
language, malformed counts, unknown/mismatched oracle cases, bad hashes, and
hidden expected material in user-visible prompts. Every **34/34** reference case
and **10/10** independent fail-safe policy controls passes.

The ZBook sandbox is qualified with unprivileged bubblewrap + `prlimit` +
`python -I -S`: a new user/PID/network/session namespace per hidden input; only
read-only `/usr`, `/lib`, `/lib64`, candidate source, generic runner, and that
input are mounted; `/tmp`/`/work` are private; environment is cleared; and
wall/CPU/address-space/file/process/FD/core/output bounds apply. Hidden expected
values remain host-only and never enter the namespace. Probes block network,
`/etc`, home/repository/model paths, `/dev/kfd`, inherited secrets, and child
processes; wall timeout kills the process group; memory/file/output limits fail
closed; valid code passes. Missing bubblewrap returns `blocked_sandbox` without
execution.

The compact qualification artifact is
[`2026-08-26-zbook-agentic-quality2-aq5-validation-v2.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq5-validation-v2.json).
Aggregation is order-independent, checks repeat fingerprints, keeps blocked and
unscorable counts explicit/outside the scored denominator, seals heldout detail
by default, and excludes raw token arrays. AQ5 runs no model or GPU and admits no
candidate mechanism.

### AQ6 / Task #46 — expanded primary baseline

- [x] Run development and heldout Qwen3.6 baseline before candidate code.
- [x] Keep heldout row details sealed from implementation selection; publish
      aggregate baseline.
- [x] Repeat deterministic rows and run fail-safe controls.
- [x] Record all metrics in Section 9 and final zero ownership.
- [x] Publish compact artifact/rollup/worklog and commit.

#### AQ6 result — deterministic expanded baseline

Clean commit `8e228965e` completed all **68/68** observations on the ZBook with
response-owned IDs and no blocked/unscorable row. External-oracle success is
**44/68 (64.71%)**: development **20/34 (58.82%)** and sealed heldout aggregate
**24/34 (70.59%)**. Family totals are code **14/16**, instruction **4/16**,
repository **10/16**, and tool selection **16/20**; language totals are English
**20/40**, Japanese **18/18**, and mixed Japanese/English **6/10**.

All **56/56** published call blocks are JSON-valid, declared, and strict-schema
valid. Valid-call observations are **56/64 (87.50%)**, correct tool/no-tool
selection is **52/68 (76.47%)**, and exact arguments are **22/32 (68.75%,
diagnostic only)**. Patch is **4/4**, code **14/16**, and instruction **4/16**;
multiple-call and repository-test kinds are **0/4** each. There are zero raw
markup/content leaks, malformed public arguments, truncations, runtime errors,
or ownership deltas.

All 34 normalized response pairs match exactly under
`normalized_response_v1`, which hashes response-owned IDs, public content,
calls/arguments, parse errors, and finish state. All **10/10** frozen independent
parser/publication fail-safe controls pass; they are not relabeled as live
endpoint tests, which remain AQ10's responsibility. Per-response repair counters
are unavailable in this baseline and remain explicit nulls rather than inferred
zeroes.

The seven unique development failures repeat exactly and are model-owned: two
no-tool actions, one wrong call count, one wrong tool, and three schema-valid
outputs/actions that fail their external task oracle. No parser, template,
runtime, or unresolved development boundary appears. Heldout model-output detail
remains local and was not inspected for mechanism selection.

The compact quality-only artifact is
[`2026-08-26-zbook-agentic-quality2-aq6-expanded-baseline.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq6-expanded-baseline.json).
A prior whole-APU DMCUB/SMU/MES hang before request prefill retained zero rows;
a cold-boot exact diagnostic and the complete baseline then passed under the
unchanged default SDMA route with a clean kernel journal and post-shutdown KFD.

### AQ7 / Task #47 — comparison models

- [x] Audit/load Qwen3.8 and Ornith independently.
- [x] Run one smoke task before the full matrix.
- [x] Run the frozen suite for each admitted model without changing task wording,
      tools, output cap, or oracle.
- [x] Record exact blocker instead of weakening unsupported paths. Both models
      are `admitted_complete`; no substitution or weakening was required.
- [x] Publish separate artifacts and a cross-model table with no implementation
      or speed inference.
- [x] Commit.

#### AQ7 result — both comparison rows complete

Qwen3.8 completes **50/68 (73.53%)**, development **22/34** and sealed heldout
**28/34**. Ornith completes **42/68 (61.76%)**, development **16/34** and sealed
heldout **26/34**. Both exact 34/34 normalized repeat pairs, 10/10 independent
controls, complete response-owned IDs, clean source/log/kernel-journal health,
and zero final transient ownership pass. No speed fields are retained. Section
10.2 contains the cross-model table and independent architecture/tokenizer/
template admission; separate compact artifacts retain commands, full model
hashes, capability/profile manifests, aggregate quality, and local raw hashes.

AQ6 and AQ7 observed the one-model-owning-process rule. Comparison-model raw
outputs and heldout details remain local and cannot tune AQ8.

### AQ8 / Task #48 — one mechanism declaration

- [x] Join AQ3/AQ6/AQ7 evidence.
- [x] Select at most one Section 11 mechanism or no-go.
- [x] Fill every declaration field before code.
- [x] Prohibit fixture/category/model-conditioned behavior explicitly.
- [x] Add `REFACTOR.md` removal trigger for any temporary flag/path. No entry is
      needed because the declaration adds no runtime path or flag.
- [x] Commit the declaration before implementation.

#### AQ8 decision — retained no-go / no implementation

AQ8 admits **no runtime mechanism**. The Qwen3.6 development denominator has
seven deterministic failed task blocks: two legal no-call choices under auto,
one valid wrong call count, one valid wrong declared tool, and three schema-
valid actions/content that fail an external task oracle. AQ3 independently found
zero parser/projection/runtime/unresolved failures; AQ6 likewise has zero
malformed argument, content/reasoning leak, truncation, runtime, or schema-
invalid published call. All 56 published primary call blocks are declared,
JSON-valid, and strict-schema-valid. Patch is already 4/4.

| Candidate | Measured trigger | AQ8 decision |
| --- | ---: | --- |
| Parser/template correctness repair | 0 independent mismatches | Reject: no runtime RED. |
| Complete token-level argument schema constraint | 0 invalid/schema-violating calls | Reject: cannot choose the intended valid tool/action or external result. |
| Bounded invalid-tool repair | 0 malformed/invalid/truncated/leaked outputs | Reject: retrying valid choices would coerce `auto` semantics. |
| Patch/diff constraint | 4/4 patch observations pass | Reject: no syntax premise. |
| No implementation | 7/7 unique development failures model-owned | **Admit.** |

AQ7 aggregates corroborate model dependence without supplying tuning data:
Qwen3.8 development is 22/34 while Ornith is 16/34 under the same admitted
runtime, and their raw/heldout outputs remain sealed from selection. Qwen3.8's
instruction-family gain especially argues against relabeling a shared runtime
path as defective. The runtime cannot infer benchmark-required tool identity,
call count, arguments, or external result from `tools`, `tool_choice=auto`, and
JSON Schema without guessing task intent.

Existing auto/required/specific/none semantics, strict post-validation,
fail-closed publication/session behavior, streaming behavior, native-sampler/MTP
blockers, and telemetry remain unchanged. Repair count/token budget is zero.
There is no new flag, alternate path, fallback chain, or `REFACTOR.md` debt.
Reopening requires an independent candidate-specific RED: a valid envelope
misparsed/misprojected by the runtime, an argument prefix that violates a
supported strict schema, a generically repairable invalid output, or a patch
syntax failure after intended semantics are independently established.

The compact decision artifact is
[`2026-08-26-zbook-agentic-quality2-aq8-no-go-decision.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-aq8-no-go-decision.json).
Because runtime source is unchanged, AQ9 implementation, AQ10 candidate semantic
qualification, AQ11 post-change primary measurement, and AQ12 transfer are
not-applicable phases on the approved no-go path; redundant GPU reruns cannot
create post-change evidence when there is no change.

### AQ9 / Task #49 — implementation

- [x] Targeted RED is not applicable: AQ8 admitted no mechanism and named exact
      independent RED criteria required to reopen each rejected class.
- [x] Implement no runtime code, prompt, fixture, or model-conditioned behavior.
- [x] Preserve the torch-free hot path and plugin boundaries by zero diff.
- [x] Preserve invalid-output/hidden-reasoning non-commit by zero diff.
- [x] Add no capability/route/repair telemetry because no route exists.
- [x] Keep fail-closed fallback and sampling/MTP behavior unchanged.
- [x] Confirm AQ8 changed no `hipengine/`, `kernels/`, `tests/`, `scripts/`, or
      `docs/REFACTOR.md` path.

**Disposition:** `not_applicable_no_implementation`. There is no AQ9 candidate,
flag, fallback chain, or code commit to qualify.

### AQ10 / Task #50 — semantic qualification

- [x] Candidate unit/fake-endpoint matrix is not applicable.
- [x] Candidate blocking/SSE parity is not applicable.
- [x] Auto/required/specific/none semantics are unchanged by zero runtime diff.
- [x] Malformed/truncated/duplicate/undeclared/schema-invalid behavior is
      unchanged; AQ5/AQ6 controls remain baseline evidence, not relabeled tests.
- [x] Reasoning/content/session/cancellation/continuation behavior is unchanged.
- [x] Sampler/native/MTP capabilities and blockers are unchanged.
- [x] Do not run a candidate real-model smoke when no candidate exists.
- [x] No retain/revert action is needed beyond the committed AQ8 no-go.

**Disposition:** `not_applicable_no_candidate_semantic_qualification`. Existing
semantic evidence remains valid for unchanged source; it is not presented as a
post-change gate.

### AQ11 / Task #51 — post-change primary quality

- [x] Do not rerun unchanged development or heldout inputs as “post-change”.
- [x] Preserve AQ6 as the primary baseline; there is no candidate denominator.
- [x] Preserve heldout details sealed and avoid task-block interval theater.
- [x] Publish no keep/reject quality delta because no runtime delta exists.
- [x] Carry AQ8's no-go artifact and AQ6 raw hashes into closure.

**Disposition:** `not_applicable_no_post_change_primary_run`. A redundant GPU
run cannot create post-change evidence from documentation-only commits.

### AQ12 / Task #52 — cross-model transfer

- [x] No candidate exists to apply to Qwen3.8 or Ornith.
- [x] Do not relabel AQ7 controls as candidate transfer controls.
- [x] Zero runtime diff proves no model-name branch or substitution was added.
- [x] No transfer capability gate or fallback is needed.
- [x] Carry AQ7 aggregate artifacts and AQ8 no-go into closure.

**Disposition:** `not_applicable_no_cross_model_transfer`. Qwen3.8 and Ornith
remain independently admitted on their unchanged routes; raw outputs stay
local and sealed from mechanism selection.

#### AQ9–AQ12 no-go-path result

`git diff 2031efb95..645678fb9 -- hipengine kernels tests scripts docs/REFACTOR.md`
is empty. The only AQ8 paths are the decision artifact,
campaign/benchmark docs, and immutable worklog. Therefore no candidate endpoint,
lifecycle, primary post-change, or transfer execution exists to test. Skipping
those runs follows the focused-repair/no-redundant-rerun rule; it does not claim
they passed for a nonexistent candidate.

### AQ13 / Task #53 — closure

- [x] List retained implementation/default/explicit/unsupported scopes.
- [x] List model-quality outcomes separately from runtime fixes.
- [x] List blocked/rejected/no-go families and reopen triggers.
- [x] Remove temporary paths or update `REFACTOR.md` with exact removal gates.
- [x] Update campaign, AGENTIC/API/SAMPLING/PLAN, benchmark index/changelog, and
      root README only when public-quality wording is justified.
- [x] Run focused-repair-aware milestone validation and schema/link/fixture/
      Worklog2/benchmark-sync checks.
- [x] Commit, merge cleanly, push, and verify local/remote equality.
- [x] Complete the objective audit against Tasks #40–#53 and this checklist.

**Closure:** AQ0–AQ13 are complete with `no_implementation`. The final compact
artifact records retained, rejected, unsupported, and reopen scopes separately
from the three model-product quality rows. No temporary candidate path, flag,
fallback, API/sampling change, or refactor debt exists. The root README stays
unchanged because this bounded synthetic suite is already presented in the
benchmark scoreboard and no runtime feature/default or general leaderboard
claim changed.

## 13. Command book

AQ1 confirms current CLI flags before AQ2. Starting surfaces:

```bash
# Clean identity and compiler cache contract.
git status -sb
git rev-parse HEAD origin/main
hipcc --version > /tmp/agentic-quality2-hipcc-version.txt
sha256sum \
  /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  benchmarks/prompts/agentic-quality-v2.json \
  benchmarks/oracles/agentic-quality-v2.json

# Focused existing contract tests before live work.
python3 -m pytest -q \
  tests/test_agentic_coding_quality.py \
  tests/test_agentic_coding_quality_oracle.py \
  tests/test_agentic_server_conformance.py \
  tests/test_agentic_harness_traces.py

# Existing v2 collector, after one separately started clean server.
python3 scripts/agentic_coding_quality.py \
  --base-url http://127.0.0.1:PORT/v1 \
  --workloads benchmarks/prompts/agentic-quality-v2.json \
  --all-workloads \
  --model Qwen3.6-35B-A3B \
  --model-path /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1151 --target-arch gfx1151 \
  --device-name 'AMD Radeon 8060S Graphics' \
  --quant gguf_q4_k_m --kv-dtype bf16 \
  --compiler-version-file /tmp/agentic-quality2-hipcc-version.txt \
  --require-clean-provenance --concurrency 1 --runs 2 --max-tokens 128 \
  --cache-mode off --timeout-s 600 --idle-timeout-s 60 \
  --records-json /tmp/hipengine-agentic-quality2/RUN/v2-records.json \
  --json /tmp/hipengine-agentic-quality2/RUN/v2-summary.json
```

AQ1 froze the AQ2 server command surface. Use a free port and run from the clean
campaign worktree:

```bash
env -u ROCR_VISIBLE_DEVICES \
  HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1151 \
  GPU_MAX_HW_QUEUES=1 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/agentic-quality2-hipcc-version.txt \
  HIPENGINE_REQUIRE_CACHED_BUILD=1 \
  HIPENGINE_QWEN35_NATIVE_SAMPLER=0 \
  PYTHONPATH=. /home/lhl/hipEngine/.venv/bin/python -m hipengine.server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --served-model-name Qwen3.6-35B-A3B \
  --backend hip_gfx1151 --quant gguf_q4_k_m \
  --execution-profile strict --kv-storage bf16 \
  --max-context-tokens 4096 --max-active-requests 1 \
  --generation-batch-window-ms 0 --prefix-cache off \
  --speculative-mtp-serving off --no-startup-chat-smoke \
  --host 127.0.0.1 --port PORT --log-level info
```

If a required JIT object is absent, perform one untimed startup without
`HIPENGINE_REQUIRE_CACHED_BUILD`, stop it, then restart the exact command above.
Do not let compilation overlap collection. Before the collector starts, hash
`/ready`, `/v1/models`, and `/v1/hipengine/capabilities`; its own capability
preflight now rejects the wrong served model, backend, cache, tokenizer, or tool
contract. Long runs print immediate progress and atomically checkpoint each
turn, including local raw responses and prompt IDs.

## 14. Artifact contract

New committed summaries live under `benchmarks/results/` and record:

- kind/schema/status/date and `performance_claim=false`;
- source commit/branch/worktree cleanliness and excluded unrelated paths;
- host/device/backend/HIP/compiler/kernel/power/TTM identity;
- full model path/bytes/SHA, GGUF architecture, tokenizer/template hash, quant,
  KV, profile/capability identity;
- exact command/environment/server startup settings;
- fixture/oracle/schema/source/license/split hashes;
- attempted/completed/blocked/unscorable counts;
- Section 9 metrics overall and by split/family/language/schema/oracle kind;
- complete failure taxonomy and runtime/model/unresolved ownership;
- deterministic repeat verdict and normalized-response hash;
- raw record/log paths and SHA-256 without committing large token arrays;
- initial persistent KV refcount/pin baseline, final zero request/session
  ownership deltas, and shutdown verdict;
- baseline/candidate relationship, keep/reject/no-go decision, and reason; and
- links/hashes to prerequisite evidence.

Quality summaries must reject latency, tok/s, goodput, speedup, profiler-topline,
or other performance-claim fields. Operator elapsed time may appear only under a
non-scored execution/progress section.

Raw records retain response-owned token IDs, response bodies after secret
redaction, independent parser details, sandbox stdout/stderr, and per-case
oracles under `/tmp/hipengine-agentic-quality2/<run-tag>/`. Raw paths are local;
committed summaries carry hashes.

A schema change is RED-tested. Do not coerce booleans/strings to integer counts,
duplicate task IDs, overlap development/heldout, silently drop failed rows, or
compute rates with blocked/unscorable rows in the denominator without explicit
fields.

## 15. Stop and no-chase rules

Stop or classify before more work when:

- source is dirty in an owned tracked path or diverged from the campaign base;
- another process owns `/dev/kfd`, a model download/hash/build is active, or the
  loaded server does not report the intended model/backend/settings;
- fixture/oracle hashes or split identities differ;
- a comparison model silently resolves another model/template;
- response-owned IDs, finish details, or final ownership are missing;
- generated code lacks the declared sandbox;
- a candidate fails endpoint/session/fail-safe semantics;
- heldout or any required family regresses;
- a proposed fix needs prompt, token, task, category, heldout, expected-answer,
  or model-name conditioning;
- a valid model-quality failure has no runtime-general fix; or
- a mechanism changes performance-sensitive runtime ownership outside this
  campaign.

Do not chase:

- grammar performance, GPU sampling, MTP, prefix reuse, routing, or concurrency;
- a second candidate before the first has a complete keep/reject decision;
- higher output caps merely to let malformed generations ramble into a call;
- system-prompt wording searches;
- post-hoc task removal or oracle relaxation;
- exact prose/token equality as a task-quality metric; or
- cross-host quality deltas presented as arithmetic/runtime equivalence.

A rejected mechanism commits its durable evidence with runtime code removed.
A no-go decision is a valid AQ8–AQ13 path.

## 16. Expected code map

| Path | Campaign role |
| --- | --- |
| `docs/AGENTIC-QUALITY2.md` | This source-of-truth ledger. |
| `docs/AGENTIC-QUALITY2-SUITE.md` | Frozen v1 source/license, split, hash, and execution card. |
| `docs/AGENTIC-OPT.md` | Historical A0-A6 status and pointer to this follow-up. |
| `hipengine/benchmark/agentic_quality.py` | Normalization, classification, aggregation, compact artifact. |
| `scripts/agentic_coding_quality.py` | Live collector; no performance rollups. |
| `benchmarks/prompts/agentic-quality*.json` | Versioned semantic task suites. |
| `benchmarks/oracles/agentic-quality*.json` | Independent hermetic oracles. |
| `benchmarks/sources/agentic-quality2-*.json` | Public-source audit and original-authorship record. |
| `benchmarks/schemas/agentic-quality2-*.schema.json` | Expanded source/suite/oracle contracts. |
| `benchmarks/schemas/agentic-coding-quality-*.schema.json` | Live records/summary contracts. |
| `tests/test_agentic_coding_quality*.py` | Loader/oracle/classifier/artifact RED gates. |
| `tests/test_agentic_server_conformance.py` | Public compatibility and fail-safe contract. |
| `tests/fixtures/agentic_traces/` | Deterministic envelope/replay fixtures. |
| `hipengine/server/` | Only AQ8-admitted parser/constraint/repair behavior; avoid broad `api.py` edits. |
| `hipengine/generation/sampling.py` | Only if the admitted mechanism is processor-owned. |
| `benchmarks/results/` | Compact baseline/candidate/rejected artifacts. |

No normal generation path imports torch. New behavior must be model-general and
capability-driven. If a focused server module extraction is required, it is a
behavior-preserving unit with the complete contract gate; do not combine a
large refactor with the candidate.

## 17. Validation and commit discipline

For every AQ phase:

1. Check/sync source and preserve unrelated files.
2. Read the latest campaign/worklog handoff.
3. Mark the TaskList item `in_progress` before work.
4. Add/observe RED for behavior/schema/math changes.
5. Implement the minimum scoped unit.
6. Run targeted GREEN and affected narrow bundle.
7. Run real GPU only where the phase requires it; state expected duration first.
8. Publish compact evidence/docs/worklog.
9. Run `python3 scripts/worklog.py check`.
10. Run `python3 scripts/sync_benchmark_readme.py --check`.
11. Run applicable JSON Schema, fixture, link, Ruff/compile, and `git diff --check` gates.
12. Stage explicit owned paths; inspect names and full staged diff.
13. Commit immediately before the next phase.
14. Push at safe handoff/milestone points and verify remote equality.
15. Mark task complete and inspect TaskList.

Follow the focused-repair rule after isolated broad failures. Do not rerun a
completed expensive matrix when a focused repair plus existing evidence covers
the change. Do not commit raw model outputs, generated-code sandboxes, logs,
weights, caches, or profiler data.

## 18. Closure handoff

AQ0–AQ13 are complete on the measured no-go path. The authoritative closure
artifact is
[`2026-08-26-zbook-agentic-quality2-campaign-final.json`](../benchmarks/results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
It preserves Qwen3.6/Qwen3.8/Ornith model-product quality separately from
runtime correctness, records zero admitted quality mechanism, and keeps AQ6/AQ7
heldout details and raw outputs local and sealed.

The full milestone run completed 10,510 tests with 10,262 passed, 180 skipped,
4 expected failures, and 67 failures outside the AQ13 closure diff. A focused
rerun of exactly those 67 nodes passed six order-sensitive nodes and reproduced
61 baseline failures; none exercises AGENTIC-QUALITY2 behavior, while every
agentic quality, fixture, live-harness, taxonomy, trace, and server-conformance
test passed in the completed broad run. Per the focused-repair rule, no unrelated
runtime/test expectation was changed and the expensive broad suite was not
repeated. Reopen only under the mechanism-specific criteria in the final
artifact.
