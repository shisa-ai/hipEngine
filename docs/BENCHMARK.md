# hipEngine Benchmark Procedures

Protocols, baselines, and artifact formats for every perf claim hipEngine retains. This doc is the companion to the "Evidence Policy" rule in `AGENTS.md` and `docs/PLAN.md`: when the rule says "record the exact command", it means the commands here.

See `docs/ROOFLINE.md` for the RDNA3 / W7900 hardware model, per-bucket decode analysis, and the "what not to chase" catalog. This doc is the operational layer on top of it. Execution-profile correctness and determinism contracts are normative in [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md).

Human-readable rollup: `benchmarks/README.md` is the compact platform-indexed
topline scoreboard. It records current protocol summaries, evidence status,
artifact pointers, user-visible blockers, refresh entrypoints, and the root
README export block. Exact commands, measured revisions, build environments,
samples, correctness gates, profiler summaries, and optimization deltas belong
in the linked machine-readable artifacts under `benchmarks/results/`.
`benchmarks/HISTORY.md` holds the superseded experiment notebook,
source-lineage targets, and external baselines. `benchmarks/CHANGELOG.md` keeps
reverse-chronological one-line rollup history; substantial implementation
narratives belong in immutable `worklog/entries/`.

## Evidence Policy (restated)

Every retained performance number must carry:

- **Model** (exact path / HF snapshot SHA)
- **Quant** (fp16, w8a16, w8a8-dyn, w4-paro, …)
- **Workload shape** (prompt length, generation length, concurrency, KV policy, warmup)
- **Execution profile** (`strict|production|batch_invariant`), profile-schema
  version, selected/fallback variant-manifest hashes, and whether generated-ID
  equality is binding or diagnostic
- **Hardware** (W7900, ROCm version, `hipcc --version`, driver from `rocminfo`)
- **Exact command** (full shell invocation, reproducible from a clean shell)
- **Result** (prefill tok/s, decode tok/s, VRAM used, peak reserved)
- **Correctness gate** (exact control/ownership in every profile; strict
  exact/parent parity, production calibrated strict-teacher mean/tail/max KL +
  top-1/determinism/isolation/BF16-relative/task gates, or batch-invariant
  metamorphic equality; KL ≤ 0.05 and top-1 ≥ 90% vs `cpu_reference` remains
  the outer kernel floor)

Claims without the declared profile gate are disallowed. A perf win that fails
any binding control, numerical, determinism, category, or task gate is rejected. Raw terminal output is not evidence — retain a compact JSON artifact per the schema at the bottom of this doc.

### GGUF GDN prefill default-selection gate

`SOL-G3` uses `scripts/gguf_gdn_prefill_ab.py` after the exact SOL-G2 matrix is
accepted. The driver loads one resident model/session, runs the production bulk
prefill route at 512 and 4096 prompt tokens, warms each context, and balances
measured `fused -> chain` with `chain -> fused` order. Every measured call must
return the expected exact token, the linked SOL-G2 artifact must cover both
contexts, and performance provenance must have no staged, unstaged, or
untracked files. Median synchronized host wall selects the result: the chain is
promotable only if it wins both contexts. A valid loss rejects chain promotion
and retains fused; it is not a failed benchmark.

```bash
python3 scripts/gguf_gdn_prefill_ab.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1151 --contexts 512,4096 \
  --prompt-token-id 9707 --expected-token-id 9707 \
  --warmups 1 --repetitions 4 --use-wmma-prefill \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --json /tmp/sol-g3-gfx1151-gdn-prefill-ab.json
```

Run from a clean worktree when the shared development tree contains unrelated
untracked artifacts. Model/session construction, JIT preparation, artifact
writing, and correctness-artifact reads are outside measured regions. The
result is a full-prefill wall comparison, while the separate cached-only G2
kernel trace proves the expected exact split kernels executed.

`--baseline-mode` defaults to `fused` so the historical SOL-G3 command and
artifacts remain reproducible. For an incremental candidate layered on an
already-promoted exact route, name that shipped route explicitly (for example,
`--baseline-mode chain_lds32 --candidate-mode chain_lds32_direct`). The linked
exactness matrix may still compare the candidate directly with `fused`, which
remains the trusted byte-exact oracle; the artifact records correctness modes
separately from timing modes. Baseline and candidate must be different.

The same rule applies to `scripts/gguf_gdn_trajectory_gate.py`: fused remains
its default baseline, while an incremental gate names the shipped exact route
with `--baseline-mode`. Natural correctness trajectories and paired decode
walls then use that same explicit pair, and equal modes fail closed.

The 2026-07-11 clean gfx1151 run at `ad773eba` rejects chain promotion:
`1186.842 -> 1248.436 ms` at 512 (+5.19%) and
`10187.300 -> 10870.022 ms` at 4K (+6.70%). Fused remains the selected default.
See `benchmarks/results/2026-07-11-sol-g3-gfx1151-gdn-prefill-interleaved-ab.json`.

## Anti-gaming

A benchmark measures the model/kernels. Tuning a number to the specific inputs
being measured measures nothing and is **INVALID** — it is never a retainable
win, regardless of how the metric moved.

Hard rules:

- **Exact server token denominators.** For hipEngine non-streaming OpenAI
  responses, use `hipengine.token_accounting.total_generated_tokens` and retain
  the per-choice ID rows/counts. `usage.completion_tokens` is authoritative only
  when that exact accounting object is present. Decoded-text re-tokenization is
  recorded only as `retokenized_visible_tokens`; it cannot support a throughput
  or MTP economics claim. Multi-choice requests aggregate every choice.
- **Exact server prompt identity.** Direct and HTTP parity rows use raw token-ID
  prompts, never detokenized text. Retain the committed fixture fingerprint and
  `hipengine.prompt_token_accounting` row hashes/counts. The server hash echo,
  exact usage, and response-owned generated-ID rows are always retained. IDs
  match the direct oracle for strict/batch-invariant and same-manifest/shape
  production parity; production rows with a different physical shape require
  the declared strict-teacher/task gate instead of silent semantic-text parity.
- **Owned server timing.** Retain `timing_scope`, `group_rows`, `timing_owner`,
  and `batch_id` for batch-scoped payloads. Sum choice timing only for an
  explicitly named per-choice-work metric. Deduplicate batch timing by
  `batch_id`, require exactly one owner for each observed batch, and fail the
  artifact if ownership or row metadata is missing/inconsistent. A timing map
  copied to multiple choices contributes once.
- **Exact server shape identity.** Retain `hipengine.generation_shape` v1 for
  every non-streaming hipEngine response. Keep the request-scoped route cap,
  queue-group request/prompt counts, actual backend calls/widths, and verifier
  rows separate. Deduplicate by `queue_group.id`, require all item indices and
  prompt slices exactly once, and sum verifier rows once per queue group. A c8
  client workload capped into two c4 groups is not a width-8 backend/verifier
  result.
- **No input-conditioned shortcuts.** Do not add code that detects the prompt,
  token sequence, candidate-id pattern, logits shape, profile-quality fixture,
  or any fixture-specific signal and changes the output to make a metric look better. Examples that are
  banned: hardcoding token IDs or candidate-pool-prefix reranks to force draft
  "acceptance", special-casing a known fixture's expected tokens, or branching on
  the prompt text. Optimize the drafter/kernel/sampler so it is genuinely better
  on inputs it has never seen.
- **Multi-prompt validation is mandatory for acceptance/quality metrics.**
  Speculative-decode acceptance, sampling quality, and any prompt-sensitive
  metric must be measured on the full multi-prompt **mtp-bench category suite**
  (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, covering `code`,
  `general_en`, `general_ja`, `mixed_ja_en`). A single fixed prompt (e.g. the
  `gguf_mtp_bench.py` `"capital of France?"` default) is a smoke input only and
  its acceptance/quality numbers are **not retainable**.
- **Speculative speedups require a true AR baseline.** The denominator for
  "MTP beats AR" must be a separate no-MTP autoregressive generation path in
  the same benchmark script/protocol, over the same prompts, sampling settings,
  warmup/hermeticity state, and timing window. A `B0`, `off`, or derived AR row
  synthesized from target-verifier timings inside an MTP diagnostic cycle is
  useful economics telemetry, but it is **not** a true AR baseline and cannot be
  used for retained speedup claims or loop keep/revert decisions.
- **Use train + category-heldout splits for optimization loops.** It is valid to
  iterate on a training subset, but every keep/revert decision for acceptance or
  speculative speed must also report full-suite metrics and a heldout subset with
  at least one prompt per category. For
  `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, the default heldout set is
  `code_markdown_table`, `general_en_explain`, `general_ja_explain`, and
  `mixed_ja_en_review`; train is the remaining six prompts. A change is not a win
  if it improves train acceptance while regressing heldout acceptance or true-AR
  speed ratio.
- **Run the committed suite directly.** `scripts/mtp-bench.py` accepts both its
  legacy JSON prompt bundle and the canonical category JSONL. JSONL artifacts
  emit the exact source SHA-256, selected prompt names, category counts, and the
  six-train/four-heldout identity; do not benchmark an untracked converted copy.
- **Greedy selection stays greedy.** Draft/target token selection in benchmark
  harnesses is pure argmax/top-k. The guard test
  `tests/test_gguf_mtp_bench_metrics.py::test_select_topk_tokens_is_pure_argmax_no_prompt_specific_rerank`
  fails if a prompt-specific override is reintroduced; do not weaken it.
- **Cleanup, not just rejection.** When gaming is found, strip the offending code
  and mark every WORKLOG/README/CHANGELOG row and `benchmarks/results/` artifact
  that cited the gamed numbers as `INVALID` (gamed) so they are never reused as a
  baseline. Real, input-agnostic engine wins measured alongside the gaming (e.g.
  draft-compute `ms` reductions) survive on their own evidence.

History: the `mtp-gguf` branch accumulated ~25 hardcoded token-id reranks in
`scripts/gguf_mtp_bench.py::select_topk_tokens` overfit to the France prompt,
inflating "acceptance" with no real drafter improvement; the category suite
exposed it (acceptance collapsed, every MTP budget slower than AR). Those
acceptance rows are INVALID.

### Honest native GGUF-MTP category diagnostics

> **Canonical gate (2026-06-29): use `scripts/gguf_ar_mtp_suite.py`.** It runs the
> true no-MTP AR baseline and the MTP category suite under ONE enforced decode
> config, computes the MTP/AR ratio itself, asserts `apple_to_apple_ok`, and emits
> a single artifact with a `verdict` — and it loads the model once (full suite
> ~3-4 min, not ~40+). **Every GGUF AR/MTP optimization must pass `--scope full`
> before it is retained or made default; microbenches and partials routinely do
> not translate to e2e (dp4a, split-K, rowtile, non-temporal were all isolated
> wins that went flat at e2e).** See `docs/MTP-LLAMACPP-PARITY.md` →
> "Validation protocol — run the suite for EVERY change". The manual two-step
> below is the underlying mechanism; note its `--true-ar-baseline-json` *attach*
> is currently broken (it still demands the #8-retired `graph_replay` AR contract,
> see `docs/REFACTOR.md`), which is why the suite computes the ratio itself.
>
> **Scope: GGUF path only.** `gguf_ar_mtp_suite.py` covers the GGUF Q4_K_M path
> (`/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `Qwen35GGUFResidentSession`). The
> **PARO path** (BF16 / W4-PARO safetensors, e.g.
> `/models/hipengine/Qwen3.6-35B-A3B-PARO-...-MTP-BF16` and the `z-lab` HF
> snapshots) is a **separate MTP/AR codepath** and is **NOT** covered by this
> suite — it has no unified one-command AR-vs-MTP gate yet. PARO-path
> optimizations must be validated e2e with the PARO harnesses
> (`scripts/qwen35_paro_bench.py` for AR, `scripts/mtp_chain_e2e_bench.py` /
> `scripts/mtp_verifier_economics.py` for MTP, `scripts/mtp_verifier_rocprof.py`
> for the verifier trace). A change to kernels shared by both paths must be
> validated on whichever path(s) it touches (ideally both). A unified PARO
> equivalent of `gguf_ar_mtp_suite.py` is a TODO.

Use this protocol before resuming native GGUF-MTP acceptance/speed optimization.
It is the guarded replacement for the old fixed-prompt `gguf_mtp_bench.py`
acceptance loops.

1. **Measure true no-MTP AR first.** Produce a separate AR artifact over the same
   category prompt suite:

   ```bash
   AR_ROOT=/tmp/hipengine-true-ar-category-$(date +%Y%m%d-%H%M%S)
   python3 scripts/gguf_true_ar_category_bench.py \
     --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
     --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
     --decode-tokens 128 \
     --warmup-decode-tokens 1 \
     --compiler-version-file /tmp/hipengine-readme-gfx1151-runs/20260615-040438/hipcc-version-gfx1151.txt \
     --raw-root "$AR_ROOT" \
     --output "$AR_ROOT/true-ar-baseline.json"
   ```

   The artifact must set `schema=1`,
   `kind=hipengine_gguf_true_ar_category_baseline`,
   `status=complete`, `true_autoregressive_path=true`,
   `same_prompt_suite=true`, and `same_timing_protocol=true`; include non-empty
   `commands`; include `repo` code-state provenance; include protocol metadata (`model`, matching `quant`,
   `prompt_file`, `prompt_count`, positive `decode_tokens`, and non-negative
   `warmup_decode_tokens`); include production `timing_protocol` metadata with
   `decode_path=graph_replay`, `graph_replay_decode=true`,
   `graph_steps_per_replay=1`, `decode_repack=true`,
   `effective_decode_repack=true`, `use_gemv_decode=true`,
   `effective_use_gemv_decode=true`, `use_wmma_prefill=true`, and
   `effective_use_wmma_prefill=true`; include top-level `prompt_hashes`; and
   contain one `prompt_metrics[]` row per selected prompt with `prompt_sha256`,
   `finite_final_logits=true`, `output_tokens` matching artifact
   `decode_tokens`, `warmup_decode_tokens` matching artifact
   `warmup_decode_tokens`, and positive `decode_ms`. An eager/raw artifact such
   as `/tmp/hipengine-true-ar-category-fullsuite-d32-20260622-134136/true-ar-baseline.json`
   (`19.67 tok/s`, no production timing metadata/effective decode-repack GEMV)
   is valid only as a diagnostic of that path and is **INVALID** as the
   retained GGUF-MTP speed denominator.

2. **Attach AR to the MTP category matrix.** Run the MTP diagnostic over the same
   prompts/budgets and attach the AR artifact:

   ```bash
   MTP_ROOT=/tmp/hipengine-gguf-mtp-category-$(date +%Y%m%d-%H%M%S)
   python3 scripts/gguf_mtp_category_bench.py \
     --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
     --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
     --budgets 1,2,3,4,5 \
     --cycles 10 \
     --raw-root "$MTP_ROOT/raw" \
     --output "$MTP_ROOT/summary.json" \
     --true-ar-baseline-json "$AR_ROOT/true-ar-baseline.json"
   ```

   The summary must set `schema=1`,
   `kind=hipengine_gguf_mtp_category_matrix`, include non-empty `commands`,
   include per-prompt metadata (`id`, `category`, positive `prompt_chars`, and
   `prompt_sha256`) matching `splits.contract.full_ids` and the default prompt
   fixture text/category/length, include category
   summary metadata whose keys match prompt categories, with a category row for
   each objective budget, count-derived bounded acceptance ratios, positive
   `decode_ms`, finite non-negative speed fields derived from output tokens and
   decode time, prompt counts matching the prompt metadata, and per-category true-AR
   ratios matching the attached true-AR category baselines, and carry attached
   true-AR `true_autoregressive_path=true`, `same_prompt_suite=true`,
   `same_timing_protocol=true`, `artifact_schema`/`artifact_kind` with strict
   integer schema fields, plus self-consistent `protocol` metadata matching
   the MTP `model`, quant family, `prompt_file`, and prompt count, the same
   production `timing_protocol` metadata listed above,
   `true_ar_comparison_available=true`, boolean `performance_claim=false`,
   boolean `speed_claim_eligible=false`, `splits.full`,
   `splits.train`, and `splits.heldout`; prompt fixture rows must use explicit unique non-blank strict
   string IDs (no `name` or line-number fallback), explicit non-blank categories (no `uncategorized` fallback), non-blank prompt text, and explicit supported chat message roles/content before split
   construction; split contract and per-split `prompt_ids` lists must contain strict non-blank strings,
   `splits.contract.heldout_ids` must be the fixed set
   `code_markdown_table,general_en_explain,general_ja_explain,mixed_ja_en_review`,
   `train_ids` must be the default full-minus-heldout complement, and each split
   `prompt_ids` list must match its `splits.contract` counterpart. Markdown tables must label the old
   verifier-derived denominator as `vs verifier off`; same-protocol speed ratios
   appear only in a separate `vs true AR` column. Any future retained summary with
   `speed_claim_eligible=true` must pass the same current-schema checks for the
   MTP summary, attached true-AR artifact identity, and guarded objective
   extraction for every canonical positive `bN` MTP budget row. `performance_claim=true` is invalid
   unless `speed_claim_eligible=true`, and both claim flags must be JSON
   booleans (not truthy strings or integers).

3. **Optimization decisions use all three views.** Report full-suite, train, and
   heldout metrics for every budget:

   - `accepted_per_output` and `draft_acceptance` (finite, bounded to [0, 1]);
   - `decode_tok_s_weighted`;
   - `mtp_vs_true_ar_decode_ratio`.

   Category objective rows are also reported for every present category. A
   train-only gain is not a win. The loop guard is speed-first: full-suite
   `decode_tok_s_weighted` and `mtp_vs_true_ar_decode_ratio` must improve for a
   guarded keep, while full-suite, heldout, and every-category
   `draft_acceptance`, `decode_tok_s_weighted`, and
   `mtp_vs_true_ar_decode_ratio` must not regress. `accepted_per_output` is
   reported as a useful coverage signal, but it is report-only for keep/revert
   decisions; a wider search that raises accepted/output while lowering speed or
   draft efficiency is reward hacking. MTP/true-AR ratio must be computed from
   the attached true-AR split/category baseline rather than `off`/`B0` verifier
   telemetry.

4. **Use the guarded objective CLI for loop metrics.** Future optimize loops
   must consume objective metrics through the harness gate, not ad-hoc JSON paths.
   Objective budget labels are canonical positive `bN` labels (`b1`, `b5`, …);
   the summary `totals` key set must be only `off` plus canonical positive `bN`
   rows. `off`, `b0`, leading-zero labels such as `b01`, bare numeric totals
   keys such as `1`, and other malformed strings are invalid as objective
   budgets:

   ```bash
   python3 scripts/gguf_mtp_category_bench.py \
     --objective-summary-json "$MTP_ROOT/summary.json" \
     --objective-budget b5
   ```

   This CLI rejects verifier-only summaries, partial/smoke prompt suites, and
   artifacts without current schema/kind metadata (schema fields must be strict
   JSON integers, not booleans), explicit true-no-MTP / same-prompt-suite /
   same-timing-protocol true-AR flags, a same-protocol true-AR
   baseline with production timing protocol (`graph_replay` + decode-repack +
   effective GEMV/WMMA), strict attached true-AR source provenance, repo provenance including matching summary / attached true-AR
   `repo_root` and non-null `git_commit`, command provenance, summary prompt/category
   provenance including strict prompt fixture and raw row identity typing,
   default prompt hashes/categories/lengths plus exactly-one prompt text source per fixture row and category budget-row scalar fields, strict JSON-integer token counts and prompt counts
   (not booleans, floats, or strings), strict true-AR protocol / prompt-row count
   fields, attached true-AR total/split/category output counts matching `prompts * protocol.decode_tokens`, repo provenance including strict integer `git_untracked_count` and
   same-repo root/commit checks between the summary and attached true-AR baseline,
   protocol provenance including attached true-AR protocol self-normalization and
   strict summary model/quant/prompt-file/prompt-count matching, strict true-AR `prompt_hashes` / `prompt_metrics`
   non-blank prompt identity/category and hash typing, and true-AR finite-logit
   evidence. CLI `--budgets` values must be unique and cannot contain empty
   comma-separated entries. Build-summary model/prompt/raw-root/cycle arguments, raw MTP
   budget-map keys/row lists, and command provenance lists are checked before
   artifact construction rather than coerced with `str()` / `int()`.
   Summary and attached true-AR category map keys must be strict
   non-empty strings. In-memory prompt rows and raw per-prompt MTP rows are checked with
   unique non-blank strict string prompt IDs/categories/text, non-blank summary prompt metadata IDs/categories, agreeing non-blank raw prompt/category identity fields, and the same strict count and
   timing typing before aggregation; the child raw metrics helper and category aggregator require the same explicit cycle schema with no legacy `accepted`/missing-field fallbacks; raw cycle-list length must match the recorded `cycles` argument; raw per-cycle visible output counts are required and must sum to total output tokens; raw per-cycle generated/accepted draft counts are required and must sum to total draft/accepted counts; raw per-cycle timing keys are required, timing values must be non-negative,
   present `total_cycle_ms` must match the sum of cycle timings, proposed-draft denominators must be positive, accepted counts cannot exceed output tokens or proposed drafts, and present falsy timing values such as `false` or `""` are rejected rather than zero-coerced. Scalar metric fields must be strict JSON
   numbers (not booleans or numeric strings). The returned JSON contains compact `category_metrics` rows
   plus full/train/heldout finite [0, 1] `accepted_per_output` and
   `draft_acceptance`, positive `decode_ms`, finite non-negative
   `decode_tok_s_weighted`, and `mtp_vs_true_ar_decode_ratio`, each with a
   positive prompt count matching the
   split `prompt_ids` length, strict string split `prompt_ids` matching
   `splits.contract`, count-derived split acceptance ratios, and the fixed heldout/train split described above.
   Train+heldout split aggregate counts and `decode_ms` must sum back to the
   full split for both MTP rows and attached true-AR rows, so forged split
   payloads cannot pass by preserving only local ratios.
   The gate also verifies each split and category
   `mtp_vs_true_ar_decode_ratio` against the attached true-AR
   split/category `decode_tok_s_weighted` and matching prompt count / prompt-id
   list length where applicable, checks split acceptance ratios against split
   token counts, checks MTP summary totals against the full split and category
   sums, and checks attached true-AR totals against the full split
   plus category sums. Attached true-AR total, split, and category
   `decode_tok_s_weighted` rows must also equal `1000 * total_output_tokens /
   decode_ms`, so forged denominators cannot pass by adjusting MTP ratios around
   them. MTP summary total, split, and category rows must likewise carry
   positive `decode_ms` and satisfy `decode_tok_s_weighted = 1000 *
   total_output_tokens / decode_ms`, so numerator-side speed forgeries are
   caught before any compare decision.

   When an optimize loop needs a single scalar verify metric, keep the same gates
   and request one split/field or category/field explicitly:

   ```bash
   # Primary speed metric: full-suite same-protocol MTP/true-AR ratio.
   python3 scripts/gguf_mtp_category_bench.py \
     --objective-summary-json "$MTP_ROOT/summary.json" \
     --objective-budget b5 \
     --objective-split full \
     --objective-field mtp_vs_true_ar_decode_ratio

   # Proposal/selector efficiency monitor: full-suite draft acceptance.
   python3 scripts/gguf_mtp_category_bench.py \
     --objective-summary-json "$MTP_ROOT/summary.json" \
     --objective-budget b5 \
     --objective-split full \
     --objective-field draft_acceptance

   # Coverage signal only: accepted/output is not a keep metric by itself.
   python3 scripts/gguf_mtp_category_bench.py \
     --objective-summary-json "$MTP_ROOT/summary.json" \
     --objective-budget b5 \
     --objective-split full \
     --objective-field accepted_per_output

   # Per-category speed monitor: reject aggregate gains that hide category regressions.
   python3 scripts/gguf_mtp_category_bench.py \
     --objective-summary-json "$MTP_ROOT/summary.json" \
     --objective-budget b5 \
     --objective-category code \
     --objective-field mtp_vs_true_ar_decode_ratio
   ```

   `--objective-split` and `--objective-category` are mutually exclusive scalar
   selectors; either selector must be paired with `--objective-field`. Scalar
   mode still calls the guarded objective extractor first; it is not a way
   to read verifier-derived `off`/`B0` telemetry or partial prompt suites. Add
   `--objective-output-json /path/to/objective.json` when a scalar verify command
   should also retain the full guarded full/train/heldout/category objective JSON
   as an artifact; the artifact records `objective_sources`, `objective_command`,
   `objective_cwd`, and output-path provenance, and the command rejects output
   paths that would overwrite the input summary JSON. Any artifact that flips
   `speed_claim_eligible=true` must also pass this guarded
   extractor for every canonical positive `bN` MTP budget before the contract accepts it. Any artifact
   that flips `performance_claim=true` must also flip `speed_claim_eligible=true`
   and pass the same guarded eligibility checks.

   Baseline-vs-candidate comparisons should use the guarded comparator:

   ```bash
   python3 scripts/gguf_mtp_category_bench.py \
     --compare-baseline-summary-json /path/to/baseline-summary.json \
     --compare-candidate-summary-json /path/to/candidate-summary.json \
     --compare-budget b5 \
     --compare-require-pass \
     --compare-require-guarded-improvement \
     --compare-require-draft-acceptance-improvement
   ```

   With `--compare-require-pass`, the command exits non-zero when full, heldout,
   or any category `draft_acceptance`, `decode_tok_s_weighted`, or
   `mtp_vs_true_ar_decode_ratio` regress. `accepted_per_output` deltas are
   retained in `report_only_improvements[]`, but cannot make `guarded_improved`
   true by themselves. Train deltas are reported but are not sufficient for a
   keep decision. The comparator also reports an `improvements[]` list,
   `report_only_improvements[]`, `guarded_improved` (true only when full-suite
   tok/s and true-AR ratio improve), `draft_acceptance_improved`,
   `missing_required_speed_improvements[]`, `decision_state` enum
   (`fail_regressed`, `pass_no_speed_improvement`, or `pass_speed_improved`),
   guarded field/scope lists, and `train_report_only=true` for those same
   guarded full/heldout/category fields. Add
   `--compare-require-guarded-improvement` when an optimize loop must reject
   accepted/output-only or exact no-op candidates in addition to regressions. Add
   `--compare-require-draft-acceptance-improvement` for proposal/selector loops;
   runtime/kernel-only loops may omit that flag but still must not regress draft
   acceptance. The optional `--compare-tolerance` applies a finite non-negative
   absolute tolerance to guarded regressions and improvements; its default is `0.0` for exact
   non-regression. NaN/Inf tolerances are invalid because they can mask
   regressions. Compare-mode JSON also records `comparison_sources` (baseline and
   candidate summary paths plus resolved paths), `comparison_command`, and
   `comparison_cwd`; keep these fields with any loop decision artifact so the
   exact command and compared summaries remain auditable. Use
   `--compare-output-json /path/to/comparison.json` to write the same guarded
   comparison JSON printed to stdout as a durable decision artifact; the command
   rejects output paths that would overwrite either compared summary.

5. **Promotion remains separate from diagnostics.** The category diagnostic is
   not a retained speed claim even when a true-AR baseline is attached. To promote
   a speed row into `benchmarks/README.md`, rerun the retained benchmark protocol
   with hermetic/warm timing (prebuilt kernels, no `hipcc`/clang in the timed
   process), full artifact provenance, correctness gate, and rollup updates. The
   optimization target is MTP/true-AR decode ratio first: diagnostic progress may
   be logged below 1.0× only when the same-protocol ratio and tok/s improve;
   retained speedups require ratio > 1.0×, and the target remains >1.3×. For
   proposal/selector work, full-suite draft acceptance should move toward the
   llama.cpp reference band (~0.50–0.84 depending on budget) rather than being
   traded away for accepted/output; lower accepted/output is acceptable only when
   speed ratio improves and draft acceptance does not regress. No hardcoding.

## Benchmark Output Contract

A benchmark artifact must answer six questions without rereading raw logs:

1. **What ran?** Exact command, model, quant, workload shape, physical host identity, hardware/software context, commit/dirty state. Backend or GPU-architecture equality does not make two hosts comparable: cross-host rows are independent evidence unless one declared same-host protocol measured both sides.
2. **Which numerical contract ran?** Execution profile/schema, selected and strict-fallback variant manifests, teacher source, and whether generated-ID equality is binding or diagnostic.
3. **Did correctness pass?** Exact control/ownership, fixture set, oracle, mean/tail/max KL and top-1 or strict layer tolerance, determinism/isolation, task verdict, exact correctness command(s), and pass/fail status.
4. **How stable is the number?** Warmup count, measured repetitions, per-phase samples, median/p95/min/max/stdev where applicable.
5. **What did the GPU actually execute?** Profiler trace status, expected kernel names, time-share summary, and any profiler blocker. Raw traces stay outside git; compact summaries go in JSON.
6. **Should we keep this number?** Baseline reference, delta, acceptance decision, and rejection/blocker reason if not retained.

For non-streaming server rows, the compact artifact also retains the harness's
validated `generation_shape` rollup: queue-group count and request/prompt rows,
request-scoped route-cap values, flattened actual backend-group widths, maximum
backend width, per-group details, and total verifier rows. Missing or partial
shape metadata makes a new hipEngine server row diagnostic rather than retained.

An A/B chain counts as one measurement per arm, not per run. Each arm must leave its
own artifact plus captured output, and the chain must report the per-arm exit code;
an unconditional `*_DONE` marker is not evidence, because three chains on 2026-08-30
printed one while an arm had died or written nothing at all (`RuntimeError: benchmark
requires tracked-clean source`, two 14-byte logs with no artifact). Use
`scripts/bench_chain.sh TAG OUTDIR "NAME|COMMAND" ...`, which tees each arm to
`OUTDIR/TAG-NAME.log`, derives and validates `$CHAIN_JSON` (`status == "complete"`),
and exits non-zero unless every arm measured. Arms that legitimately emit no artifact
are named with a trailing `!` and are reported as `ok(no-artifact)`.

### Exact-token direct/HTTP gate

Use [`scripts/exact_token_generation.py`](../scripts/exact_token_generation.py)
before comparing PARO or GGUF direct and OpenAI-server results. It defaults to
the committed `fixtures/qwen35_paro/parent_512_32_seed1234.json` 512-ID row and
`max_tokens=128`. Direct mode records the raw prompt and generated-ID oracle;
HTTP mode requires that artifact and fails closed on any input hash/count,
usage, output-row, or generated-ID mismatch.

```bash
uv run python scripts/exact_token_generation.py direct \
  --model-path /models/qwen36-paro --backend hip_gfx1151 --quant w4_paro \
  --json /tmp/direct-p512-d128.json

uv run python scripts/exact_token_generation.py http \
  --url http://127.0.0.1:8000 --model qwen-paro \
  --model-path /models/qwen36-paro --backend hip_gfx1151 --quant w4_paro \
  --oracle /tmp/direct-p512-d128.json --json /tmp/http-p512-d128.json
```

The `hipengine_exact_token_oracle` v1 contract is formalized in
[`benchmarks/schemas/exact-token-oracle.schema.json`](../benchmarks/schemas/exact-token-oracle.schema.json).
The parity artifact is a correctness/identity gate with
`performance_claim=false`; throughput promotion still requires the normal
warmup, repetition, timing-ownership, shape, profiler, and clean-provenance
gates. Exact direct/HTTP generated-ID equality is binding for `strict` and
`batch_invariant`, and for `production` only when both surfaces resolve the same
variant manifest and physical execution shape. A production direct/server shape
change uses exact prompt/control/accounting plus the strict-teacher profile gate;
it must not be rejected solely because a near-tie free-running ID differs.
The current `exact_token_generation.py` implementation still enforces equality;
until its profile-aware adapter lands in P2, use it only for strict/same-manifest
parity and keep shape-different production direct/server rows diagnostic.

### Unified direct/server PARO/GGUF matrix

Use [`scripts/benchmark_matrix.py`](../scripts/benchmark_matrix.py) to join
exact-token rows after the direct and HTTP artifacts exist. A manifest gives
each row a stable case, engine (`paro|gguf`), surface (`direct|server`), and
path variant, plus optional memory and profiler artifact pointers. The report:

- validates every exact prompt hash and response-owned generated hash; requires
  direct/server output equality for strict/batch-invariant and same-manifest/
  shape production cases, otherwise attaches the production profile gate;
- derives total tokens, tok/s, and ms/token from the raw generated-ID rows and
  measured wall instead of accepting a supplied denominator;
- deduplicates batch-scoped timing by `batch_id`, requires exactly one owner and
  the declared number of row copies, and preserves choice/request/client scopes;
- records route cap, queue rows, actual backend widths, verifier rows, and
  execution paths separately;
- attaches memory and profiler summaries by artifact SHA plus RFC 6901 JSON
  pointer; and
- refuses a direct/server rate ratio when timing scopes differ. PARO/GGUF rows
  are side-by-side by default; a cross-engine ratio needs a separately proven
  identical model, quant, math, and timing protocol.

The current matrix validator still enforces output equality within a case. P2
must add profile/manifest and production-gate attachments before shape-different
production rows can become eligible; until then those rows remain diagnostic.

```bash
uv run python scripts/benchmark_matrix.py build \
  --manifest benchmarks/manifests/sol-m1-paro-e5-diagnostic.json \
  --json /tmp/paro-direct-server-matrix.json

uv run python scripts/benchmark_matrix.py validate \
  --json /tmp/paro-direct-server-matrix.json
```

`--run-commands` executes optional row command arrays without a shell before
assembling the report. A failed eligibility gate still writes the artifact and
returns exit 2; use `--allow-ineligible` only for an explicitly diagnostic
matrix. The formal input/output contracts are
[`benchmark-matrix-manifest.schema.json`](../benchmarks/schemas/benchmark-matrix-manifest.schema.json)
and
[`benchmark-matrix.schema.json`](../benchmarks/schemas/benchmark-matrix.schema.json).
The committed diagnostic manifest intentionally reuses the SOL-E5 PARO identity
artifacts; it is not a throughput claim and its direct-call/client-E2E values
are deliberately unratioed.

Allowed artifact statuses:

| Status | Meaning |
| --- | --- |
| `accepted` | Correctness passed, benchmark protocol followed, variance acceptable, and result may be compared later. |
| `rejected_correctness` | Performance may have been measured but correctness failed; number must not be used as a perf claim. |
| `rejected_variance` | Correctness passed but timing was too noisy / contaminated for comparison. |
| `blocked` | Benchmark could not complete (OOM, hang, profiler failure, missing dependency, GPU busy). Record symptom and command. |

A JSON artifact with `status != "accepted"` is still useful evidence, but it is not a retained performance number.

### Canonical artifact provenance

New server, retained PARO, GGUF, and microbenchmark artifacts embed one
top-level `provenance` object produced by
`hipengine.benchmark.provenance.collect_artifact_provenance()`. Its formal
contract is
[`benchmarks/schemas/artifact-provenance.schema.json`](../benchmarks/schemas/artifact-provenance.schema.json):

- `kind="hipengine_artifact_provenance"` and `schema_version=1`;
- for profile-sensitive runs, top-level `execution_profile`,
  `execution_profile_schema`, `variant_manifest_sha256`, and
  `strict_manifest_sha256` fields (the provenance collector will absorb these
  when profile runtime plumbing lands; until then the harness writes them);
- repository root, commit, branch, and separate `staged_dirty`,
  `unstaged_dirty`, `untracked_dirty`, and `untracked_count` fields;
- configured and concrete resolved backend, target architecture, and selected
  device name;
- model path/revision plus a content-derived fingerprint, quant, and KV dtype;
- exact argv and relevant environment, ROCm/HIP compiler identity, build
  profile, timing protocol, warmups/repetitions, and profiler status.

The collector is stdlib-only and torch-free. It hashes model files in full up
to 8 MiB and otherwise hashes deterministic head/middle/tail samples together
with file size. Model directories use a deterministic manifest of relative
paths and per-file fingerprints. Hugging Face `snapshots/<revision>` paths
infer the revision automatically. A missing path is recorded explicitly for
diagnostics; it is not an existing model fingerprint and cannot support a
retained model-performance claim.

The aggregate `dirty` field must equal the OR of the three dirty axes. A new
retained performance row requires all three axes false and, when a model ran,
an existing content fingerprint. Legacy `software`, `repo`, or environment
fields may remain for backward compatibility, but they do not replace the
canonical block. Older artifacts without this block keep their documented
legacy/diagnostic status until rerun.

## Human-readable Rollup

`benchmarks/README.md` is the compact current scoreboard. A reader must be able
to identify the latest eligible row for a platform and protocol without
reconstructing shell history, but the README must not duplicate the experiment
notebook contained in artifacts, the changelog, and worklog entries.

Maintain it with every retained benchmark:

1. Update the top review date.
2. Add or replace the row keyed by platform, GPU, model fingerprint, quant, KV
   type, backend, workload, concurrency, policy, and timing scope. Never append
   a chronological optimization diary below the current row.
3. Keep only the concise protocol scope, evidence classification, memory/timing
   scope where needed for interpretation, and compact artifact links. The
   artifact owns exact revisions, dependencies, commands, repetitions,
   correctness details, samples, profiler data, and optimization deltas.
4. Mention a diagnostic, blocked run, or rejected run only when it removes a
   current row or explains a user-visible limitation. Link one durable artifact
   and a concrete refresh condition; candidate ladders stay out of the README.
5. Keep superseded tables in `benchmarks/HISTORY.md`, compact artifacts, or Git
   history rather than copying them into the live scoreboard.
6. Update the marked public table and run
   `python3 scripts/sync_benchmark_readme.py --write` followed by `--check`.
7. Add a dated entry to `benchmarks/CHANGELOG.md`: model, quant, workload,
   metric `old -> new`, percent delta, reason, and artifact. State explicitly
   when a contract-only change has no metric supersession.
8. Record substantial implementation decisions in a new immutable file under
   `worklog/entries/`; do not turn either benchmark Markdown rollup into a
   substitute worklog.

JSON artifacts remain the durable evidence. The README is an index and current
scoreboard over that evidence, not another evidence store.

## Hardware & Software Context (default)

Unless explicitly stated otherwise, hipEngine benchmarks run on:

- GPU: AMD Radeon Pro W7900 (gfx1100, RDNA3, Navi 31)
- Compute: 96 CUs / 192 SIMD32 / wave32 native
- Memory: 48 GiB GDDR6, 864 GB/s peak bandwidth, 96 MiB Infinity Cache
- Peak throughput (FP16 matrix) 123 TFLOP/s, (INT8 matrix) 123 TOP/s, (FP32 vector) 61.3 TFLOP/s
- Host: `therock` Python 3.12 env; PyTorch `2.11.0+rocm7.13.0` only when the `[torch]` dlpack extra is used
- ROCm: 7.13.x series; HIP runtime `7.13.26162` (verify with `python3 -c "import torch; print(torch.version.hip)"` when torch is installed, otherwise `/opt/rocm/bin/hipcc --version`)

Full spec and roofline derivation: `docs/ROOFLINE.md` §1 (hardware) and §2 (roofline fundamentals).

Capture at the top of every benchmark run:

```bash
rocminfo | grep -E 'Name:|gfx' | head -4
rocm-smi --showmeminfo vram --showuse --showtemp
hipcc --version
python3 -c "import torch; print(torch.__version__, torch.version.hip)" 2>/dev/null || echo "(no torch)"
```

### W7900 hipEngine README rows: use the hermetic TheRock wrapper

For retained W7900 hipEngine PARO/GGUF README rows, run
`scripts/run_w7900_readme_refresh.sh hipengine` or reproduce its `THEROCK_ENV`
`env -i` wrapper exactly. Do not promote numbers from a hand-assembled shell
that merely points at the TheRock Python or a cached compiler-version file.

Known failure mode: a 2026-06-21 direct-shell GGUF Q4_K_M rerun used the right
Python and HIP compiler cache key but inherited the ambient ROCm environment; it
made W7900 GGUF prefill look `~8–23%` slower while decode and token IDs stayed
in-family. The corrected hermetic rerun recovered prefill to within
`~0–5%` of the prior retained row. If a W7900 GGUF result shows "prefill down
hard, decode normal," first rerun through the wrapper before blaming kernels.

The wrapper also captures the TheRock root and compiler version used to key JIT
caches. If reproducing manually, set both the CLI `--compiler-version-file` and
the environment cache-key guard (`HIPENGINE_COMPILER_VERSION_FILE=/tmp/...`; for
HIPCC-specific wrappers also set `HIPENGINE_HIPCC_VERSION_FILE=/tmp/...`). Some
lazy helper kernels are reached below the session constructor and rely on the
environment variable rather than the top-level CLI flag; without it, long runs
can appear to hang in `hipcc --version` probing instead of executing kernels.
Artifact notes may show TheRock HIP `hipMemGetInfo` totals that differ from
`rocm-smi`; use hipEngine tracked/owned allocation peaks for per-session rollups
and keep sampled HIP memory as auxiliary evidence.

### gfx1151 README rows: use the committed UMA-aware wrapper

For retained Radeon 8060S/gfx1151 comparison rows, use
`scripts/run_gfx1151_readme_refresh.sh` from a clean detached worktree. The
wrapper targets native `gfx1151`, uses the hermetic TheRock gfx1151 libraries,
and emits canonical component provenance for PARO, GGUF, llama.cpp HIP, and
llama.cpp Vulkan. Each hipEngine workload uses its own process and right-sized
resident session. PARO retains two discarded warmups plus five measured
resets. The calibrated GGUF lane uses one discarded warmup plus three measured
resets: the 2026-07-13 six-shape audit observed at most 0.132% prefill
stdev/median, and every available first-three median equalled its five-sample
median. Escalate GGUF to five measured runs only for a named variance,
stability, or borderline-decision trigger; lifecycle soaks are separate tests,
not extra performance repetitions. A merge gate rejects incomplete, dirty,
unstable, non-finite, or high-variance components before emitting a numeric
six-shape rollup. If a reproduced external lifecycle blocker prevents one
shape, never carry its stale value or bypass that gate: publish only completed
components plus an explicit blocked cell, a compact causal artifact, and a
fixed-stack rerun condition. Discarded runs warm kernels eagerly. Every measured reset
captures a fresh state-bound graph, excludes capture from decode timing, and
destroys it before the next reset.

On gfx1151, hipEngine applies `GPU_MAX_HW_QUEUES=2` before loading
`libamdhip64`. The 2026-07-26 exact Laguna shared/routed MoE gate supersedes
the one-queue short-context default: seven queue-matched complete-state pairs
win, repeated load/reset/close gates through 4K complete, and cached tracing
proves the secondary stream overlaps caller work. The original one-queue
evidence remains the stability history in
`benchmarks/results/2026-07-15-gfx1151-hip-one-queue-stability-promotion.json`.
Neither policy is a repeated-128K lifecycle guarantee; current production,
router-rollback, and SDMA-disabled gates reproduced the low-power
measured-pass-1 stall even with one queue, so 128K remains blocked. See
`benchmarks/results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json`.
Artifact provenance must capture `GPU_MAX_HW_QUEUES`. Use
`GPU_MAX_HW_QUEUES=1` for the prior single-queue rollback and `=4` only for the
ROCm-default scheduler diagnostic. Never attribute a result across different
queue policies without an explicit process-level comparison. gfx1100 and mixed
recognized architecture sets retain ROCm's queue default.

On gfx1100, hipEngine applies `HSA_SCRATCH_SINGLE_LIMIT=8388608` before loading
`libamdhip64` unless the user already supplied a value. ROCr 7.2.4's upstream
140-MiB threshold is reserved per process/GPU; dispatches above it use
use-once scratch. Cached 4K tracing at the intermediate 32-MiB threshold
identifies one scratch-bearing AOTriton `attn_fwd` family with 16 launches and
16 matching 300-MiB allocation/free pairs. Lowering the final threshold to 8
MiB preserves full-engine behavior and removes the exact 132-MiB difference
from the reserve. Every artifact must capture this variable: use `146800640`
for an upstream-default rollback, do not overwrite an explicit user value, and
do not apply a backend-local default to mixed recognized architectures. See
`benchmarks/results/2026-08-15-gfx1100-rocr-scratch-reserve-retained.json`.

For a bounded 128K stall reproduction, add the default-off persistent prefill
flight recorder to the same production command:

```bash
--prefill-flight-recorder /tmp/gfx1151-128k.flight \
--prefill-flight-recorder-granularity chunk
```

`chunk` is the least-perturbing first pass: host submissions are appended to a
fixed binary mmap at embedding/layer/finalize/sample boundaries, while one tiny
same-stream system-fenced marker retires after reset and each 4K outer chunk
(plus final sample boundaries). It therefore distinguishes the last host-submitted layer
from the last fully retired chunk without calling HIP from the observer. Decode
one snapshot, or watch only cursor changes from a separate process:

```bash
python3 scripts/qwen35_prefill_flight_recorder.py \
  /tmp/gfx1151-128k.flight --entries 8
python3 scripts/qwen35_prefill_flight_recorder.py \
  /tmp/gfx1151-128k.flight --entries 8 --watch-seconds 1
```

On a stalled run, preserve the external watcher output before terminating the
benchmark. The mmap normally remains readable immediately after process exit,
but it is not a crash/reboot durability format. Interpret `last_submitted` as
where the CPU most recently entered submission and `last_completed` as a
same-stream retirement boundary; neither alone proves that the named kernel is
faulty. Escalate to `layer` only after a chunk interval repeats: it adds one
marker dispatch per layer and is substantially more likely to move a
timing-sensitive bug. Recorder-enabled runs are diagnostics, never retained
performance rows. Pair them with the independent `amdgpu_fence_info` sampler so
kernel-ring and KFD-user-queue blind spots remain explicit. The installed
rocprofv3 1.3.2 also exposes `--kfd-trace` (queue, mapping, migration, dropped
events), but run that as a separate traced-incidence experiment: profiler
instrumentation can suppress this timing-sensitive failure, and the traced
process must use prebuilt `require_cached` kernels. The canonical symptom,
control matrix, KFD/MES capture plan, and upstream-report checklist live in
[`DEBUG-GFX1151-STALL.md`](DEBUG-GFX1151-STALL.md).

The APU exposes a 512 MiB visible-VRAM aperture in
`mem_info_vram_{total,used}` but a 120 GiB system-backed allocation domain in
`mem_info_gtt_{total,used}`. Whole-device llama.cpp peak rows on gfx1151 must
therefore sample `gtt`, not `vram`. Keep hipEngine tracked allocator, HIP
phase-sampled, and whole-device GTT scopes explicitly labelled; none may be
silently presented as the other.

## Baselines to Beat

These numbers are measured on the shared `/home/lhl/` workspace and recorded in `~/amd-gpu-tuning/WORKLOG.md`. They are the "must beat" bar for hipEngine on the same hardware. When hipEngine claims a win, the claim is per-column vs the row it beats.

### Qwen3.6-35B-A3B Q8_K_XL on llama.cpp ROCm (current W7900 target)

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 entry.

| Workload | Prefill tok/s | Decode tok/s | VRAM used | Notes |
| --- | --- | --- | --- | --- |
| `llama-bench` native (pp512 / tg128) | 949.89 ± 9.59 | 74.32 ± 0.02 | — | `llama-bench -m Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf -fa 1` |
| Localhost server 4K/4K | 1139.72 | 71.49 | 44.94 GiB | `/completion`, 4096 prompt, `n_predict=4096`, `temperature=0`, `ignore_eos=true`, `cache_prompt=false`, `stream=false` |

Build: `llama.cpp 0f1bb602d (8946)` with ROCm backend, `-fa 1` flash attention.

Decoder roofline: 71.49 tok/s at 4K/4K is ~27.5% of the optimistic GGUF-ratio memory roof (~260 tok/s) for the 3.33 GB active-weight bytes/token estimate. Prefill is ~5.6% of the matrix-compute roof. See `docs/ROOFLINE.md` §5 for Amdahl per-bucket framing.

### Qwen3-0.6B FP16 c=1 shootout (nano-vllm vs mini-sglang, 4K/4K)

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 shootout entry. Reference for the *host architecture* cost we're beating, not the kernel layer.

| Engine | Prefill tok/s | Decode tok/s | KV shape | KV GiB | Notes |
| --- | --- | --- | --- | --- | --- |
| nano-vllm (enforce_eager, ROCm SDPA) | 30,167.12 | 15.33 | `[2,28,1404,256,8,128]` | 38.39 | 267 s wall on 4096 decode tokens |
| mini-sglang (overlap disabled, `torch_sdpa`) | 20,195.46 | 22.58 | `[2,28,1430,256,8,128]` | 39.10 | 183 s wall on 4096 decode tokens |

mini-sglang is 1.47× faster on decode; nano-vllm is 1.49× faster on prefill. Both sit far below the 35B llama.cpp decode baseline despite being 0.6B — the current torch-SDPA paged decode path is the bottleneck.

### llama.cpp MTP external comparison diagnostics

llama.cpp MTP rows are external comparison diagnostics, not accepted hipEngine
performance claims. Use them to answer "what does current llama.cpp do on this
model and prompt mix?" before comparing hipEngine changes.

For cross-engine decode-only tables, follow the
[`Cross-Engine Decode Timing Boundary`](MTP-LLAMACPP-PARITY.md#cross-engine-decode-timing-boundary).
llama.cpp starts `predicted_ms` after sampling the first output while including
that token in `predicted_n`; native `predicted_per_second` is therefore a
self-reported diagnostic, not the cross-engine rate. Request `N+1` outputs and
use the runner's `aggregate_decode_transition_per_second` for `N` timed
transitions. Compare it only with hipEngine complete MTP `cycle_wall_ms`, and
keep client/HTTP wall separate.

Default config:

- Runner: `python3 scripts/llamacpp_mtp_bench.py`
- Config: `benchmarks/configs/llamacpp-mtp-qwen36-27b.json`
- Prompt suite: `benchmarks/prompts/mtpbench-code-general-ja.jsonl`
- Model: `/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`
- Server: `/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server`
- Hardware: W7900/gfx1100
- Server flags: `-ngl 99 -fa on -ctk f16 -ctv f16 -c 8192 --no-cache-prompt`
- MTP flags: `--spec-type draft-mtp --spec-draft-n-max 2`

Run both natural prompts and token-repeat prompts:

```bash
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 8192 \
  --draft-max 2 \
  --protocol both \
  --mode both \
  --output /tmp/llamacpp-mtp-qwen36-27b-diagnostic.json
```

Protocols:

- `natural`: `/v1/chat/completions` over code, English, Japanese, and mixed
  JA/EN prompts, `temperature=0`, `top_k=1`, `max_tokens=512`, `seed=12345`.
- `token-repeat`: `/completion` with explicit prompt token arrays
  `[9707] * {512,4096}`, `n_predict=128`, `ignore_eos=true`.

Artifact status must be `diagnostic_retained` and `performance_claim=false`
unless a future protocol defines a shared correctness gate. Reasons:

- llama.cpp GGUF Q4_K_M and hipEngine PARO w4 are different quantizations.
- MTP can change output hashes at `temperature=0` because target verification
  changes the sampled path and batching shape.
- Repeated-token prompts can produce perfect draft acceptance and overstate
  natural-prompt MTP speedups.

## Standard Workloads

Every new perf number should match one of these shapes unless there's a documented reason not to. Protocol-shape drift is how baselines become uncomparable.

### c=1 short (4K/4K)

Matches the llama.cpp localhost server baseline above.

- Prompt: exact 4096 input token IDs (use `/v1/tokenize` or a fixed token-ID file)
- Generation: `n_predict = 4096`, `ignore_eos = true`, `temperature = 0`
- Concurrency: 1 request, TP = 1
- Warmup: 1 prior request (same shape) discarded
- Report: prefill ms + tok/s, decode ms + tok/s, wall-clock s, VRAM used after run, peak reserved

### c=1 long (16K/256)

For KV-policy and long-context work.

- Prompt: exact 16,384 input token IDs
- Generation: 256 tokens, `temperature = 0`, `ignore_eos = true`
- Concurrency: 1, TP = 1
- Warmup: 1 prior request (same shape) discarded
- Additional report: KV cache shape + bytes, eviction events if KVPolicy ≠ dense

### c=N concurrent (Phase 1+)

Correctness comes before throughput. A c=N benchmark row is not eligible for `accepted` status until all of the following are true:

- `scripts/qwen35_batch_correctness.py --rows N` or the model-level equivalent
  passes exact request/slot/token/position/mask/`KVLiveSpans`/transaction
  ownership and the declared primitive profile tolerances.
- `strict` rows emit generated IDs and declared numerical boundaries equal to N
  independent strict c1 runs. `batch_invariant` rows additionally pass slot,
  neighbor, width, admission, cancellation, and compaction metamorphic equality.
  `production` rows instead compare full logits at identical strict-teacher
  contexts and pass calibrated mean/p95/p99/max KL, top-1 by category/shape/
  transition, same-schedule determinism, same-width neighbor isolation,
  BF16-relative checks where available, and task non-inferiority. Cross-width
  free-running generated-ID equality is recorded as diagnostic.
- The artifact records execution profile/manifests, scheduler occupancy, active
  mask shape, graph bucket key, KV policy, width transitions, fallbacks, and
  whether compaction occurred.
- For continuous batching, include admission/completion timestamps, SLO
  goodput, and per-request p50/p95/p99 latency in addition to aggregate tok/s.

Initial protocol shapes:

| Shape | Purpose | Required correctness command |
| --- | --- | --- |
| `c=2`, prompt 512 / decode 128 | bring-up parity and debugging | primitive/control gate plus the declared strict-ID, production strict-teacher, or batch-invariant metamorphic gate |
| `c=4`, prompt 512 / decode 128 | first scheduler/graph bucket row | same profile-aware gate at rows 4 plus transition coverage |
| `c=8`, prompt 512 / decode 128 | primary early concurrent target | same profile-aware gate at rows 8 plus ragged/isolation/retirement coverage |

GGUF native rows use the model-level equivalent gate rather than the PARO
primitive-only script. Under strict, `tests/test_qwen35_gguf_target_rows.py`
proves full FP32 logits against independent strict c1 at C=2/4/8, including
variable prompt lengths and reclaim/compact/readmit, while
`scripts/qwen35_batch_gguf_diagnostic.py` preserves all generated IDs over the
measured 512/128 and 4K/128 workloads. Under production, those same scenarios
hold strict teacher tokens fixed and emit the profile distribution/isolation
verdict; IDs remain diagnostic. The artifact must additionally carry
native indexed-state, `KVLiveSpans` attention, selected-row MoE, row lm-head,
and row-sampler profiler symbols.

Report both aggregate tok/s and per-request tok/s. Do not compare a c=N aggregate row against c=1 without explicitly showing `aggregate/c1` and `per_request/c1` ratios. SpecDec must be disabled for these rows; SpecDec has a separate acceptance protocol because generated-token equality depends on target verification and KV commit semantics.

### Production OpenAI load/SLO gate

Use [`scripts/gguf_production_load_gate.py`](../scripts/gguf_production_load_gate.py)
for Phase-F4 GGUF serving closure. It starts one prepared model behind a real
localhost Uvicorn socket and must include all of these workload classes in one
artifact: static c1/c8, ragged mixed prompt/output burst, deterministic fixed
arrivals, seeded Poisson arrivals, cancellation plus disconnect and timeout,
queue overload with both exact accepts and `429 engine_busy` rejects, idle
recovery, and a duration/rate-qualified soak.

The command declares queue-p99, TTFT-p95, ITL-p99, and end-to-end-p95 SLOs.
Generated-token goodput counts only response-owned IDs from profile-qualified
completed requests whose own queue, TTFT, every ITL, and end-to-end latency meet
all four thresholds. “Profile-qualified” means exact independent-c1 results for
strict/batch-invariant, or exact control/ownership plus the binding production
strict-teacher/task packet; it never means decoded-text similarity. Decoded
text and `usage.completion_tokens` are not denominators. Each workload reports
p50/p95/p99, outcome/finish-reason counts, occupancy and physical-group
transitions, bounded stream-queue depth, KV/tracked/HIP memory, server-counter
deltas, final ownership, and the declared binding or diagnostic comparison
against independent strict c1 sessions.

Before the retained workload, sweep the declared prefill/decode policy and
prefill-chunk candidates on one frozen mixed-arrival shape. Select the highest
profile-qualified SLO-goodput candidate among rows that pass every binding gate;
use TTFT p95, ITL p99, then smaller chunks only as tie-breaks. Record every candidate, including
failed/neutral rows. A passing workload with dirty source remains diagnostic.

### Coding-agent multi-turn server rows

Use [`benchmarks/prompts/agentic-coding-v1.json`](../benchmarks/prompts/agentic-coding-v1.json)
as the initial synthetic repository/tool-loop suite and
[`scripts/agentic_coding_bench.py`](../scripts/agentic_coding_bench.py) as the
fail-closed A0 record/artifact gate. The workload suite, normalized turn records,
and artifact envelopes are pinned by the three `agentic-coding-*.schema.json`
files under `benchmarks/schemas/`.

A retained live row uses real localhost Uvicorn SSE, the exact committed workload
fingerprint, and concurrency 1/4/8. Report first-token TTFT, complete validated
tool-call-ready latency, ITL, complete turn wall, exact generated tok/s, valid
tool calls/s, prefix hits/reused tokens/cache bytes, sampler and full-vocabulary
D2H state, physical widths, batch timing ownership, and final request/session/KV/
graph/workspace ownership. Do not substitute decoded text, OpenAI usage, or
client concurrency for exact backend token and width evidence.

The A0 gate is model-free and accepts only normalized successful deterministic
tool turns. It rejects undeclared/wrong tools, invalid or schema-mismatched
arguments, raw reasoning/tool markup leakage, missing or non-monotonic timestamps,
exact-token hash mismatch, duplicate request ids, incomplete turn sequences,
ambiguous batch timing ownership, cache activity under `cache_mode=off`, and
leaked final ownership. Build an A0 artifact with:

```bash
python3 scripts/agentic_coding_bench.py \
  --workloads benchmarks/prompts/agentic-coding-v1.json \
  --records /tmp/agentic-coding-records.json \
  --json /tmp/agentic-coding-a0.json
```

A1 uses `scripts/agentic_coding_live.py` against an already-running real server.
It renders exact tokenizer-sized prefixes, builds deterministic prior tool
transcripts, obtains an independent non-streaming c1 exact-token/tool oracle
outside the measured window, and releases the measured SSE requests together.
Retained server goodput uses `active_sse_wave_wall_s`: for each
`(run_id, workload_id, turn_index)` wave, take maximum `response_done_at_s`
minus minimum `submitted_at_s`, then sum the wave walls. This excludes the
independent blocking oracles and tokenizer/control preparation between turns.
The older `workload_wall_s` spans first measured submit through final tool-result
submit and therefore includes those inter-turn controls; its scope is now
explicitly `first_submit_to_last_tool_result_submit_including_inter_turn_control`
and its tok/s is diagnostic only for live A1. Buffered public tool streams report
validated tool-ready latency, not lower-loop TTFT, and still cannot report ITL.
For example:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 scripts/agentic_coding_live.py \
  --base-url http://127.0.0.1:8100/v1 \
  --model Qwen3.6-35B-A3B --backend hip_gfx1100 \
  --workload small_repo --concurrency 1 --runs 1 \
  --cache-mode off --max-tokens 128 \
  --records-json /tmp/agentic-small-c1-records.json \
  --json /tmp/agentic-small-c1-a1.json
```

A validated tool SSE response can be a safely buffered public projection, not
one event per model token. Record it as `token_timing_mode=buffered_public`,
`generated_token_ids_source=matched_nonstreaming_oracle`, and
`sse_exact_ids_observed=false`; report public TTFT/tool-ready latency and withhold
ITL. Oracle/tool equality does not make the measured SSE token IDs observed, so
such a row remains diagnostic and cannot support an exact-token performance
claim. Only response-owned IDs satisfy the exact denominator, and only
`live_exact` one-token events support ITL percentiles. Never spread one buffered
timestamp across token IDs and call the resulting zero intervals ITL.

A1 and later live collectors must produce the normalized input rather than
weakening A0 when backend telemetry is missing. Prefix/routing candidates require
cache-off generated-ID parity and state/KV/refcount gates. Sampled candidates
add fixed-seed repeatability and distribution sanity. Pressure/cancellation and
automatic-tool quality are separate later lanes and must not be mixed into the
first deterministic performance denominator.

A2 prefix decisions use a fail-closed concurrency funnel. Run one complete
warmup and at least three balanced/rotated cache-off versus candidate pairs for
every frozen family at C1 first, using `active_sse_wave_wall_s` goodput and
validated buffered tool-ready latency as the primary metrics. C4/C8 measurement
is authorized only when C1 improves at least one predeclared primary metric,
passes every exactness/lifecycle and variance gate, and does not materially
regress any C1 control. Promotion additionally requires the medium-repository C4
guard. A failed prerequisite produces an explicit no-timing skip artifact; never
infer missing C4/C8 timing from C1 or prior controls. The final decision artifact
must hash the A1 control, C1 pair packet, C4/C8 disposition, and lifecycle packet,
and must keep the candidate non-default when any promotion gate is false.

A3 native-sampler decisions have an earlier fail-closed sampled-tool prerequisite.
Before any measured SSE interval, run the same fixed-seed blocking oracle through
both host and native-eligible `tool_choice=auto` routes on `small_repo` C1. Every
frozen turn must finish with one declared schema-valid tool call, exact response
IDs/accounting, and the advertised sampler/D2H metadata. Separately prove that
specific/required strict-tool forcing, close queues, unsupported processor shapes,
stop/EOS, and bounded logprobs either run natively as advertised or report the
explicit host fallback. A failed blocking oracle stops the C1/C4/C8 timing matrix
and emits a no-timing blocked artifact; do not assign active-SSE/tool-ready rates
to invalid tool output or infer wider-concurrency performance. Native promotion
still requires fixed-seed repeatability, CPU-reference distribution sanity, zero
full-vocabulary D2H on supported rows, one warmup plus three measurements for all
frozen C1/C4/C8 conditions, no C1 regression, and the medium-C4 guard.

The complete active board is [`AGENTIC-OPT.md`](AGENTIC-OPT.md).

### Speculative decode / DFlash rows

DFlash and later MTP rows use `scripts/dflash_speculative_bench.py` as the
schema-normalizing artifact driver. Future native runners should emit one raw row
per `(prompt, draft config)` containing same-session AR and speculative results;
the driver computes the common fields ported from the parent `~/amd-gpu-tuning`
harnesses:

- same-session AR decode tok/s and generated-token sample;
- speculative tok/s, exact equality vs AR, finite AR/draft/verify logits;
- acceptance histograms and cumulative `>=N` rates;
- target-verify rows/output token and verify ETA vs AR per row;
- draft / target-verify / commit split, plus DFlash drafter sub-phase timings
  `draft_context_full_rebuild_seconds`, `draft_context_append_seconds`, and
  `draft_query_seconds` so artifacts distinguish full-context rebuild,
  append-only materialization, and query-only drafter cost;
- draft K/V cache capacity/bytes (`draft_kv_capacity_tokens`, `draft_kv_bytes`);
- scalar/vector device-to-host readback counts, with full-logit readbacks called
  out explicitly;
- graph capture/replay status and bucket key;
- peak memory fields and target/drafter model paths.

The full-model `scripts/dflash_chain_e2e_bench.py` runner attaches the canonical
artifact-provenance block directly, including concrete backend/architecture,
target-model fingerprint, command/environment, and separate staged, unstaged,
and untracked state. Run retained S4 rows from a clean worktree; unrelated
untracked files in the primary checkout must not be hidden or discarded.

Use `fixtures/dflash/stable_prompts.jsonl` for deterministic no-remote prompt
coverage. Its `code_promotion` rows are the first speed-promotion gate;
robustness rows cover general, instruct/prose/math, and multilingual output,
while `synthetic_stress` rows are diagnostic until code rows already beat AR.
Rebuild/validate it with `scripts/dflash_prepare_prompts.py` when the retained
tokenizer snapshot changes.

A speculative row is promotable only when every row is exact/finite, the artifact
contains a true no-MTP autoregressive baseline (not a verifier-derived `off` row),
and aggregate speculative decode is >1.10× that same-protocol AR. The checked-in
`benchmarks/results/2026-05-18-hipengine-dflash-benchmark-contract-diagnostic.json`
is a synthetic schema fixture, not a performance claim.

### PARO c1-c8 strict concurrency matrix

This remains the strict/batch-invariant catalog protocol. Production-profile
c>N uses the strict-teacher gate in the general c=N section above and records
cross-width IDs diagnostically.

Use one raw-token fixture for c1 and c>N; repeated token IDs or detokenized text
are different protocols. The short matrix is prompt 512, 8 warmup decode steps,
128 measured decode steps, greedy sampling, W4 PARO, BF16 KV, and all 40 layers.

- Run c1 with `scripts/qwen35_paro_bench.py --prompt-fixture <fixture>
  --prompt-row 0 --prompt-length 512 --warmup-decode-tokens 8
  --decode-tokens 128 --graph-replay-decode`. Retain at least three fresh-process
  runs and report the median plus exact prompt-ID SHA-256.
- Run c2-c8 with `scripts/qwen35_batch_equality_matrix.py --batch-sizes
  2,3,4,5,6,7,8` and the same fixture/shape. Each row must compare all 137 IDs
  against independent single-request `prefill_native()+step()` sessions.
- A failed native row may keep one timing as a diagnostic, but it cannot enter a
  topline, scaling ratio, routing profile, or profiler-driven optimization queue.
  Classify it explicitly serial until a general algorithm passes the full gate.
- Keep gfx1100 and gfx1151 catalogs separate. A stale W7900 row cannot select a
  gfx1151 route, and vice versa.
- Close lifecycle safety separately with
  `scripts/qwen35_batch_shrinking_correctness.py`. Use one exact ragged prompt
  vector, leave a non-edge physical slot alive through c1, retire one row by
  EOS, and cancel enough rows to exercise front, middle, and tail holes without
  compaction. Compare every generated ID plus SHA-256 of all persistent linear
  Conv/GDN state and live full-attention K/V prefixes at each retirement
  boundary. A per-segment ragged fallback is correctness evidence only, not a
  c>N throughput claim.

The current gfx1151 catalog is
`benchmarks/results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json`:
c1 is retained; every c2-c8 native candidate fails at generated index 2 and
production uses width-1 sessions. The matching lifecycle gate is
`benchmarks/results/2026-07-11-sol-p2-gfx1151-paro-ragged-lifecycle.json`:
ragged c8-to-c1 EOS/cancel transitions pass all token/state/KV rows on the
production true-c1 route; `performance_claim=false`.

### OPTIMAL MoE/PARO parity rows

For the Qwen3.5-35B-A3B-PARO exercise, first keep source-lineage parent rows and hipEngine attempts as separate artifacts:

- Parent/source-lineage rows use `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` with the 23 base flags from `~/amd-gpu-tuning/docs/OPTIMAL.md` and `--decode-use-step-graph-replay`.
- Initial parity shapes are `512/128` and `4K/128`; later add `1K/128`, `32K/128`, and `128K/128` after the port path is stable.
- Parent rows can be `accepted` source-lineage artifacts when finite logits and graph/eager validation pass. They are comparison targets, not hipEngine measurements.
- hipEngine rows stay `blocked` until `LLM.generate()`, `w4_paro` loading/layout, Qwen3.5 model plugin, required kernels, and graph replay exist.
- When hipEngine runs, compare against the matching parent artifact and require the same post-run quality gates plus hipEngine's KL/top-1 gate.

Current local parent artifacts:

- `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`
- `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`
- `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json`

### Microbenchmark (single kernel)

For kernel-local claims (port parity, fusion wins):

- Warmup: 50 iterations
- Measure: 200 iterations.
- Report for each measured metric: samples count, median, p95, min, max, and stdev.
- Report profiler fields: kernel name, grid size, workgroup size, duration, `VGPR_Count`, `Scratch_Size`, and `LDS_Block_Size` from `rocprofv3 --kernel-trace`. If the CSV has `Start_Timestamp` / `End_Timestamp` instead of `DurationNs`, compute `DurationNs = End_Timestamp - Start_Timestamp` in the compact summary/artifact.

Kernel-local wins that do not translate to ≥ 1% E2E impact on the c=1 short workload are recorded but not defended — see `docs/ROOFLINE.md` §11 "What Not To Chase" (~100 iterations on a 19%-of-time kernel while 76.9% sat untouched is the canonical anti-pattern).

## Measurement Statistics

Every accepted benchmark artifact records timing as **samples**, not just one number.

Minimum for E2E workloads:

- `warmup_runs`: normally `1` for full workload shapes.
- `measured_runs`: normally `3` for expensive E2E benchmarks unless cost is prohibitive; if fewer, explain in `notes`.
- For each phase (`prefill`, `decode`, `wall`): sample list plus `median`, `p95`, `min`, `max`, `stdev`.
- For memory: pre-run idle and post-run usage in the applicable amdgpu domain
  (`vram` on discrete GPUs, `gtt` on UMA APUs), peak allocator reservation when
  available, KV cache bytes/shape, and the exact measurement scope.

Minimum for microbenchmarks:

- `warmup_iters`: normally `50`.
- `measured_iters`: normally `200`.
- Duration stats in nanoseconds and, when meaningful, derived throughput.

Variance guard: if stdev is >5% of median for E2E or >10% for a microbenchmark, mark the artifact `rejected_variance` unless the variance is understood and documented.

## Correctness Gate

Declare the execution profile first. Two granularities are required for every
new/ported kernel; production routes add the profile-wide dynamic and task gate
before a perf claim is accepted.

### Layer-level (`kernels/cpu_reference/` oracle)

```bash
uv run pytest tests/test_<family>_correctness.py -q
```

For each registered `(backend, layer, quant, variant)` tuple, run the same fixture input through the HIP kernel and the CPU-reference implementation. Assert the outer floor:

- Mean KL divergence ≤ 0.05 over the fixture set
- Top-1 logit agreement ≥ 90%

Strict variants additionally meet their declared exact/parent-parity boundary.
Production T1/T2 variants retain a registered strict fallback and pass the
calibrated strict-teacher mean/p95/p99/max KL, top-1, determinism/isolation,
BF16-relative, and task gates in `EXECUTION-PROFILES.md`. The outer floor alone
cannot promote a production default.

### End-to-end (fixed-prompt smoke)

```bash
uv run python scripts/smoke.py --model Qwen3-0.6B --prompt fixtures/smoke_prompts.jsonl \
  --reference outputs/cpu_reference/Qwen3-0.6B.logits.npy
```

Runs the full `LLM.generate()` path on a fixed prompt set, saves logits, and
diffs against the archived CPU-reference logits. This supplies the same outer
KL ≤ 0.05 / top-1 ≥ 90% gate. Strict uses its exact fixture contract;
production additionally runs the complete multi-category strict-teacher,
dynamic ownership/isolation, deterministic-repeat, and task-quality packet.

### P9 qwen35moe GGUF WMMA+GEMV decode gate

For P9.A3/P9.B7-style qwen35moe GGUF benchmark rows that enable the P8 WMMA bulk-prefill opt-in and/or the P9 decode GEMV opt-in, run the resident 512/128 contract before reporting throughput:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
PYTHONPATH=. python3 scripts/qwen35_gguf_p9_e2e_correctness.py \
  --fixture tests/fixtures/gguf/qwen36_35b_a3b_q4km_p9_e2e.json \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --json benchmarks/results/<date>-qwen36-35b-a3b-q4km-p9-e2-correctness.json
```

The fixture compares a candidate launched with `HIPENGINE_GGUF_WMMA_PREFILL=1` + `HIPENGINE_GGUF_GEMV_DECODE=1` against the legacy row-GEMV path (`0`/`0`) over the prefill sample plus 128 eager decode logits rows. Do not infer fastpath use from requested flags alone: artifacts must record `fastpath_safety` and requested vs effective flags, and only rows with `effective_use_wmma_prefill=true` / `effective_use_gemv_decode=true` can be used as WMMA/GEMV performance evidence. A passing gate with `effective_* = false` is a correctness fallback only. Acceptance is mean KL ≤ 0.05, top-1 agreement ≥ 90%, finite final logits, and deterministic candidate tail token IDs across three runs. A failed gate makes any dependent throughput row `rejected_correctness`; do not promote it to the rollup.

Fixtures (prompts + reference logits) are tiny (< 10 MB) and *are* committed under `fixtures/`. They are not "benchmark outputs" and do not count against the never-commit rule.

## Post-run Quality Gates

After every E2E benchmark attempt, extract and record these fields before presenting throughput:

1. **Correctness / sanity**
   - `finite_prefill_logits` must be `true` when the benchmark emits it. `false` or `null` means the run is NaN-corrupted or incomplete; mark it `rejected_correctness` or `blocked`.
   - Graph replay validation, when active, must pass (`decode_step_graph_validation=true` or equivalent).
   - For same-prompt A/B comparisons, `generated_sample` equality is binding for
     strict/batch-invariant and diagnostic for production. Production arithmetic
     changes require matched strict-teacher logit and task-quality evidence;
     neither stochastic labeling nor semantic text similarity can replace it.
2. **Performance**
   - Report `prefill_tok_s`, `decode_tok_s`, and total `wall_seconds` with units.
   - If a run is warm-started, say so; do not compare warm-start to cold-start without labeling it.
3. **Memory**
   - Report `allocated_after_load_gib` when available and peak allocated/reserved bytes as GiB.
   - Flag any run above the 24 GiB PARO usability gate separately from W7900-only diagnostic rows.
4. **Presentation**
   - For multiple configs, use a compact table containing correctness, prefill/decode, wall time, and memory in one view.
   - Include external baselines (llama.cpp HIP/Vulkan, parent `docs/OPTIMAL.md`, or previous hipEngine artifact) when the shape has a known comparable row.

Throughput without these fields is not a retained benchmark number.

## Microbenchmark & rocprofv3

For any port-parity or fusion-win claim, capture a kernel trace. Dumps go under `/tmp/hipengine-profile/` (gitignored). Keep only the compact JSON artifact (below) per run.

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- \
  uv run python scripts/smoke.py --model Qwen3-0.6B --workload c1-short
```

Profile the leaf workload, not a benchmark wrapper that spawns children. In particular, do not put `rocprofv3` around the MTP prompt-suite/economics parent harness; use `scripts/mtp_verifier_rocprof.py` or pre-warm and profile the final `mtp_chain_e2e_smoke.py` child directly.

For the correct GGUF eager-decode baseline and layer-family Amdahl audit, use
`scripts/gguf_decode_rocprof.py`. Its parent warm-builds outside the profiler,
then requires cached builds for every timed child. The retained baseline is one
resident session, one discarded full run, and four measured full runs at
`[9707] * 512` / 128 eager steps. The profiler uses one synchronized ROCTX range
per eager step and accepts only kernels fully contained in those ranges, so
model load, prefill, and warmup cannot enter the Amdahl denominator. Every child
records every generated ID and fails on the first non-`9707` token. SOL-G4 also
requires clean worktrees at direct-parent revisions `74b11dbc` and `4499fb13`;
the same graph-off, repacked p8/d32 protocol identifies the first performance-
changing revision without comparing graph output to correct eager output.

For the state-bound GGUF decode-graph gate, use
`scripts/gguf_decode_graph_g5.py` against the production
`Qwen35GGUFResidentSession.capture_decode_graph()` API. The retained protocol
uses the same `[9707] * 512` / 128-transition workload, one warmup, four rotating
same-session repetitions, and charges capture/instantiate/destroy to every
candidate window. Every graph launch must match eager byte-for-byte for the
generated token, FP32 hidden seed, all resident Conv/GDN states, and all live
BF16 K/V rows. Third-and-later launches are mandatory; replay-only timing is
diagnostic, and per-token recapture is a separately timed rejection control.

For the in-tree retained-PM4 P6 transport comparison, use
`scripts/pm4_graph_bench.py` on gfx1100. The focused baseline is one loaded GGUF
Q4_K_M session, `[9707] * 512`, 128 transitions, one discarded warmup, and five
measured rounds. Capture one stable-pointer graph generation for each selected
`hipgraph|aql|pm4` transport, rotate transport order every round, reconstruct
exact state with reset/prefill/rearm before every window, and report host API
call wall, synchronized replay wall, capture-inclusive wall, queue/provenance
ledgers, final token, recurrent/KV hash, final-logit hash, and context teardown.
Native API call wall includes the required stream drain and finite native wait;
HIP call wall is asynchronous issue time, so synchronized replay is the primary
cross-transport metric. The harness remains `performance_claim=false` until
natural prompt/category and heldout gates satisfy `docs/PM4.md` promotion policy.
It never performs submit-plus-queue-recreate stress.

For the SOL-G6 replacement-residency gate, run a clean persistent-session
`scripts/qwen35_gguf_bench.py` p512/d128 row with the production graph selected,
then compact it with `scripts/gguf_residency_g6.py`. Snapshot the graph live,
after graph close, and after session close. The retained census must classify
resident weights by raw/replacement/dense layout; name BF16/INT8 KV payload and
scales, recurrent state, decode scratch, prefill/session buffers, and graph
residency; audit source tensors for simultaneous raw+replacement layouts and
optional sidecars; and check owned plus tracked bytes against 24 GiB. A memory
gate does not create a speed claim: link an accepted exact same-path performance
artifact by SHA-256 and set `performance_claim=false` unless timing is repeated
under the full performance protocol.

Post-process the CSV to rank kernels by total `DurationNs`. Audit-first discipline (time share → occupancy → iters-per-thread → VGPR) lives in `~/amd-gpu-tuning/AGENTS.md`.

## Artifact Format

Every benchmark attempt writes one JSON file under `benchmarks/results/<date>-<tag>.json`. The JSON is committed when it is small and useful. Raw `rocprofv3` CSVs, terminal logs, large logits, and model outputs are not committed.

Schema `2` is the benchmark-output contract. The profile fields below are the
P2 schema-extension target; until the evaluator/schema validators implement
them, a new profile-sensitive run remains diagnostic rather than silently
claiming legacy-schema acceptance.

```json
{
  "schema": 2,
  "status": "accepted",
  "timestamp": "2026-05-12T18:30:00+09:00",
  "run_tag": "qwen06-c1-short-baseline",
  "summary": "Qwen3-0.6B fp16 c1-short baseline",
  "execution_profile": "strict",
  "execution_profile_schema": 1,
  "variant_manifest_sha256": "<sha256>",
  "strict_manifest_sha256": "<sha256>",
  "arithmetic_class": "T0",
  "generated_id_equality": {"binding": true, "diagnostic": {}},
  "provenance": {
    "kind": "hipengine_artifact_provenance",
    "schema_version": 1,
    "collected_at": "2026-07-11T12:00:00+00:00",
    "repo_root": "/home/lhl/hipEngine-main",
    "hipengine_commit": "<sha>",
    "git_branch": "main",
    "staged_dirty": false,
    "unstaged_dirty": false,
    "untracked_dirty": false,
    "untracked_count": 0,
    "dirty": false,
    "configured_backend": "auto",
    "resolved_backend": "hip_gfx1100",
    "target_arch": "gfx1100",
    "device_name": "AMD Radeon Pro W7900",
    "model_path": "/models/Qwen3-0.6B",
    "model_revision": "<immutable revision>",
    "model_fingerprint": {
      "algorithm": "sha256-directory-manifest-v1",
      "value": "<sha256>",
      "size_bytes": 123,
      "sampled_bytes": 123,
      "exists": true,
      "path_type": "directory",
      "file_count": 1
    },
    "quant": "fp16",
    "kv_dtype": "bf16",
    "command": ["python3", "scripts/bench.py", "--shape", "c1-short"],
    "environment": {"HIP_VISIBLE_DEVICES": "0"},
    "rocm_version": "7.13.x",
    "hipcc_version": "<hipcc --version>",
    "build_profile": "release",
    "timing_protocol": "median-of-3-after-1-warmup",
    "warmups": 1,
    "repetitions": 3,
    "profiler": {"status": "captured"}
  },
  "hardware": {
    "gpu": "AMD Radeon Pro W7900",
    "arch": "gfx1100",
    "cus": 96,
    "vram_total_bytes": 48301604864,
    "pre_run_vram_used_bytes": 27930624,
    "post_run_vram_used_bytes": 43307237376
  },
  "software": {
    "rocm_hip": "7.13.26162",
    "hipcc_version": "<from hipcc --version>",
    "python": "3.12.x",
    "torch_rocm": "2.11.0+rocm7.13.0 or null",
    "hipengine_commit": "<sha>",
    "hipengine_dirty": false
  },
  "workload": {
    "shape": "c1-short",
    "model": "Qwen3-0.6B",
    "model_path": "/home/lhl/gpu-tuning/models/Qwen3-0.6B",
    "model_revision": "<hf snapshot or git/ref>",
    "quant": "fp16",
    "prompt_tokens": 4096,
    "gen_tokens": 4096,
    "concurrency": 1,
    "kv_policy": "dense_paged",
    "warmup_runs": 1,
    "measured_runs": 3
  },
  "commands": {
    "environment": ["rocminfo | grep -E 'Name:|gfx' | head -4", "hipcc --version"],
    "correctness": ["python3 scripts/check_fixtures.py"],
    "benchmark": "python3 scripts/bench.py --shape c1-short --model Qwen3-0.6B --quant fp16",
    "profiler": "rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- ..."
  },
  "correctness": {
    "passed": true,
    "oracle": "cpu_reference",
    "fixtures": "tests/fixtures/qwen3-0.6b-smoke/",
    "kl_mean": 0.0,
    "kl_p95": 0.0,
    "kl_p99": 0.0,
    "kl_max": 0.0,
    "top1_agreement": 1.0,
    "category_top1_min": 1.0,
    "control_semantics_passed": true,
    "same_schedule_deterministic": true,
    "isolation_passed": true,
    "task_quality_passed": true,
    "layer_fixture_max_abs": 0.0003,
    "command_exit_code": 0
  },
  "measurements": {
    "prefill_ms": {
      "samples": [135.4, 135.8, 136.1],
      "median": 135.8,
      "p95": 136.1,
      "min": 135.4,
      "max": 136.1,
      "stdev": 0.29
    },
    "decode_tok_s": {
      "samples": [15.2, 15.3, 15.4],
      "median": 15.3,
      "p95": 15.4,
      "min": 15.2,
      "max": 15.4,
      "stdev": 0.08
    }
  },
  "memory": {
    "kv_shape": [2, 28, 1404, 256, 8, 128],
    "kv_bytes": 41221619712,
    "allocator_reserved_peak_bytes": 42859495424
  },
  "profiler": {
    "status": "captured",
    "raw_trace_path": "/tmp/hipengine-profile/results.csv",
    "expected_kernels_present": true,
    "top_kernels": [
      {
        "name": "qwen35_paged_full_attn_decode_splitk",
        "total_duration_ns": 123456789,
        "time_share": 0.42,
        "grid_size": 4096,
        "workgroup_size": 256,
        "vgpr_count": 80,
        "scratch_size": 0,
        "lds_block_size": 0
      }
    ],
    "notes": "raw_trace_path is not committed"
  },
  "baseline": {
    "name": "llama.cpp Qwen3.6-35B-A3B UD-Q8_K_XL 4K/4K",
    "source": "~/amd-gpu-tuning/WORKLOG.md 2026-04-28",
    "decode_tok_s": 71.49,
    "prefill_tok_s": 1139.72
  },
  "comparison": {
    "decode_delta_pct": -78.6,
    "prefill_delta_pct": 2547.0
  },
  "decision": {
    "accepted": true,
    "reason": "correctness passed and variance below threshold"
  },
  "notes": "baseline; no kernels ported yet, engine runs on cpu_reference backend"
}
```

If a benchmark is blocked or rejected, keep the same schema but set `status` and `decision.accepted=false`, then fill `decision.reason`, the exact failing command, and any symptom fields (`oom_bytes`, `signal`, `exception`, `profiler_status`, etc.).

Fields marked with `<...>` are filled at runtime by the applicable benchmark
harness. The legacy `software.hipengine_commit` +
`software.hipengine_dirty` pair may remain in older schemas, but new claim
eligibility uses the canonical provenance block and all three dirty axes.

## Playbook: Running a Benchmark

Minimum sequence for a retained number:

1. **Contract snapshot.** Declare execution profile/schema, arithmetic class,
   selected/strict-fallback manifests, teacher source, and whether ID equality
   is binding or diagnostic.
2. **Environment snapshot.** Capture `rocminfo`, `rocm-smi`, `hipcc --version` output into the JSON artifact.
3. **Context clear.** `rocm-smi` shows VRAM near idle; no other jobs on the GPU.
4. **Warmup run.** One full workload-shape pass, discarded.
5. **Measurement.** Run the workload; `torch.cuda.synchronize()` around prefill and decode phases when torch is in play; `hipStreamSynchronize` on the default stream otherwise.
6. **Correctness.** Run the layer-level outer gate plus the declared strict,
   production, or batch-invariant whole-path gate. A failing binding gate kills
   the number — do not publish.
7. **Artifact + rollup.** Emit the JSON under `benchmarks/results/`, update `benchmarks/README.md`, and add a short entry to `benchmarks/CHANGELOG.md`.
8. **Log.** Create a unique immutable worklog entry with `python3 scripts/worklog.py new`, then summarize the number, delta vs prior baseline, and anomalies (high VGPR, scratch, unexpected kernel in trace). Validate it with `python3 scripts/worklog.py check` and commit the entry/artifact/rollup/changelog with the code change, or as its own `perf:` unit otherwise.

If the number contradicts the roofline prediction by > 2×, stop and re-audit before publishing. Overperformance usually means a measurement bug; underperformance usually means a pathology worth naming.

## Failure as Evidence

A benchmark that failed for a specific reason (OOM at shape X, hang on ROCm version Y, crash on concurrency Z) is still evidence and should be recorded in an immutable worklog entry with the same rigor: exact command, exact symptom, workload shape, and hardware context. "We tried this path and it doesn't work yet" keeps us from wasting time on the same path later.
