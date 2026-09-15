# Journey Remaining Experiments

Resumed September 15, 2026 JST, from `394da3289`. This is the execution
inventory for [the journey](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md) and
[the runtime review](QWEN4EXP-WILKIN-RUNTIME-REVIEW.md). Completed historical
experiments remain evidence; an unchanged rejection is not rerun. A changed
composition or implementation is a new candidate, not a rewritten verdict.

Fixed lane: Framework machine `55ea6c509d0b49eea8de7094a1023668`, gfx1151,
Flash-Next UD-Q4_K_XL revision `8bdc666649440e9bdc97e16f3f75782c98478ff5`,
BF16 KV, existing Q8 draft. No global ROCm replacement or quant substitution.
The restored baseline is guarded Q8 down plus quad/ordered QSA. All retained
improvements require the applicable unchanged production quality gates.

## Inventory

| ID | Concrete experiment | State / prerequisite |
| --- | --- | --- |
| R01 | Existing tiled GDN/DPP suffix on corrected production | Numerical/state pass; targeted EOS task fails predeclared no-new-material-error rule; not promoted ([evidence](../benchmarks/results/2026-09-15-journey-gdn-restoration/README.md)) |
| R02 | Multi-column GDN on corrected production | Short numerical/state gate passes; depth, complete code task and perf pending ([evidence](../benchmarks/results/2026-09-15-journey-gdn-multi-restoration/README.md)) |
| R03 | Serial-prefix GDN instruction/reuse changes | Current ISA and complete-owner census; no blind layer widening |
| R04 | Dense Q8 iu8 independently | Count actual projection dispatch; short and canonical depth gates |
| R05 | GR iu8 up independently | Isolated780-row failure; identical-input replay excludes split epilogue in sampled roles and points to activation reconstruction; correction pending ([replay](../benchmarks/results/2026-09-15-journey-gr-operands/README.md)) |
| R06 | GR iu8 down independently | Isolated780-row failure; sampled projection error substantially exceeds reconstruction-only estimate; compensated accumulation candidate pending ([replay](../benchmarks/results/2026-09-15-journey-gr-operands/README.md)) |
| R07 | Dense MMQ numerical correction | Existing corrected-down + MMQ depth failure is binding; localize operands before rewriting |
| R08 | Approximate MoE grouped and Q4 iu8 restoration | Separate families, real routing/repair counts, then composition |
| R09 | Decode DP4A restoration | Full category/heldout numerical/task gates and true AR cost |
| R10 | Real expert populations and repair telemetry | Capture updated baseline before ranking new matrix tiles |
| R11 | Chunk scratch model and larger chunks | 2048 numerics/native-c2 allocation pass;4K PP/TG +2.32/+1.22%, small short costs; boundary/isolation/active-task pending;4096 scratch under-accounted by2.16GB ([evidence](../benchmarks/results/2026-09-14-journey-chunk-admission/README.md)) |
| R12 | Other mixed-quant matrix geometry/dequant-on-load | Actual Q4_K/Q5_K/Q5_1 roles and whole-owner cost, not IQ4_NL transfer |
| R13 | PLE copy elision / sorted dedup | Existing one-pair screen mixed; isolate gather/scatter savings on unique and duplicate rows |
| R14 | Persistent pread / direct-I/O and row cache | Existing cold gain/warm loss; cache-state policy and new-row workload required |
| R15 | PLE asynchronous staging / UMA ring | Delayed consumer, wrap, cancel, c2 and graph generations before timing |
| R16 | Remaining HC gate/mix and BF16 mirror fusions | Last-reader and publication census, registered unfused fallback |
| R17 | Remaining conv/gather/mean/norm fusions | Exclude existing bulk ports; complete owner plus lifetime tests |
| R18 | QSA indexer/packing/attention | Full BF16 selected-position owner at 2051/2052, 4096/4097 and tails |
| R19 | Multi-CTA QSA threshold + stable compaction | 513-65536 pooled blocks; ties, tails, finite domain, extra scratch/launch cost |
| R20 | MoE top10 and vocabulary top1 | Distinct tie/output contracts; measure complete owners before port |
| R21 | Host dispatch and backend refresh overhead | Stable-generation cache adopted; PP +3.29/+1.59/+1.88%, all complete requests improve; TG lower, explicitly scoped ([evidence](../benchmarks/results/2026-09-14-journey-backend-cache/README.md)) |
| R22 | Current graph/submission census | Separate prefill/decode, updates, cold capture and exposed wall |
| R23 | Private runtime ordinary AQL versus current AQL | Exact pinned runtime source, compatible private build, loaded DSO identities |
| R24 | Private retained PM4 versus its ordinary AQL | Engagement logging without activity tracing, dependency/scratch/in-flight/c2/lifecycle gates |
| R25 | Collapsed batch merge separately | Require actual eligible batches; do not conflate with PM4 |
| R26 | Native gfx1151 transport alternative | Separate encoder qualification only if private route leaves a justified gap; never remove ASIC guard |
| R27 | Batched MTP target verification | Rejection-depth, accepted-prefix and target/draft rollback before economics |
| R28 | MTP at 4K and beyond | Allocation/state gates, full category/heldout true-AR comparison; automatic MTP stays off |
| R29 | External compatibility gaps | Inspect known HIP gather fault / BF16 graph restrictions before changed arm; no replay of unchanged faults |
| R30 | Final stack ablations and comparison | Five counterbalanced pairs, same-artifact quality-valid comparator, current manifests and cleanup |

## Execution Rules

- Work serially on GPU and keep CPU tests/compiler work outside timing.
  Read-only source work may proceed during correctness-only runs.
- Commit tested harness/fixture changes before source-pinned GPU runs.
- Use the 18-prompt short suite and canonical depth matrix for arithmetic
  gates. Any differing free output receives the predeclared paired task
  review; do not quietly substitute exact-ID rejection or loosen the policy.
- Keep the complete-EOS evidence gap visible. Do not call short trajectories
  a long-form quality certificate.
- Record each item as adopted, measured-negative, or blocked with a concrete
  prerequisite and artifact. An untested row is open, never complete.
- Re-rank from current owner costs after each adoption. Do not multiply
  historical gains or attribute all disabled families to Q8.

## Gate Engagement

The 18 natural category/heldout task prompts in the R01 capture are only
39-71 tokens. Dense Q8 iu8 and GR iu8 up/down require more than 256 prefill
rows, so that natural suite cannot exercise those optimizations, even with
long free completions. R04-R06 need counted canonical-depth execution and
task prompts whose prefill actually enters the candidate. An unchanged
short-task pass is an inactive-path control, not active-path qualification.

R01/R02 restoration declarations are T2 relative to the strict fallback:
the column-warp recurrence changes reduction association.
Earlier captures preserve their original T1 metadata for provenance. This
classification correction changes no numerical limit, recorded result or
task verdict; both classes require the full production gate.

R04's generic `Q8_IU8_WMM` switch can also cover Q8 GR-down through
`launch_gguf_linear` when the dedicated `GR_IU8_DOWN` switch is off. A CPU
selector test confirms the K10240/N320 route activates above256 rows.
Therefore R04 and R06 are not disjoint merely because their flag names differ.
Count actual kernel calls by shape/role; label generic coverage honestly or
use an explicit, counted role exclusion when independent attribution needs it.
