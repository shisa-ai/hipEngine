# AGENTIC-QUALITY2 v1 Fixture Card

- **Status:** frozen before candidate code
- **Campaign phase:** AQ4
- **Suite ID:** `agentic-quality2-v1`
- **License:** project-original `AGPL-3.0-only`
- **Official upstream score reproduced:** no
- **Upstream task/solution/test bytes imported:** no

This card records the immutable input boundary for AGENTIC-QUALITY2 AQ5–AQ13.
The suite is intentionally bounded. It measures automatic tool selection,
argument grounding, hermetic repository work, Python function behavior,
instruction constraints, Japanese/mixed-language behavior, and fail-safe
semantics. It does not claim an official BFCL, HumanEval, MBPP, or IFEval score.

## Frozen files

| File | Exact SHA-256 |
| --- | --- |
| `benchmarks/prompts/agentic-quality2-v1.json` | `631519bee6acf04097341d31d46e2e0c8793b7b1e8c688265c80d9bd0f181df9` |
| `benchmarks/oracles/agentic-quality2-v1.json` | `f633f4f84e0dfd31053af0688278b3d172934db2eccf58cbedf55c4073ec585f` |
| `benchmarks/sources/agentic-quality2-v1-sources.json` | `eacab37e9ace592e326a17052e154adc167191cb971f9d62b9c476b8cdb64e35` |
| `benchmarks/schemas/agentic-quality2-suite.schema.json` | `43ef716613de843bbaa209c28ed9d5a93b59a29f757cd1856adfb8b21d5e3f0f` |
| `benchmarks/schemas/agentic-quality2-oracles.schema.json` | `3ef2248c13313a71f2bfe48d84fb389c85066612c075143d63d50aae8581d491` |
| `benchmarks/schemas/agentic-quality2-sources.schema.json` | `1387383df17ec1c463825ff4f4475238a589848145a5f04210df6af8fb9a8fad` |

Any fixture change after this freeze creates a new suite version and reruns both
baseline and candidate. Do not update these hashes in place after observing a
candidate answer.

## Public-source audit and selection

The source manifest pins the audited revisions and license evidence:

| Conceptual reference | Pinned revision | License | ZBook-local availability | Use in v1 |
| --- | --- | --- | --- | --- |
| Berkeley Function Calling Leaderboard | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | Apache-2.0 | absent | Conceptual function-selection shapes only |
| OpenAI HumanEval | `6d43fb980f9fee3c892a914eda09951f772ad10d` | MIT | read-only copy under `/home/lhl/omlx/` | Conceptual pure-function execution only |
| Google MBPP | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC-BY-4.0 | read-only copy under `/home/lhl/omlx/` | Conceptual basic-function execution only |
| Google IFEval | `e6890f85757dd84e27ca6df2dd30651dafad28e0` | Apache-2.0 | absent | Conceptual deterministic instruction checks only |

The campaign deliberately selected **project-original bounded style tasks**.
No upstream prompt, solution, assertion, expected value, or evaluator byte was
copied or adapted. The synthetic telemetry repository, tool descriptions,
Japanese/mixed prompts, code specifications, hidden cases, instruction checks,
and fail-safe controls were authored for this campaign. The fixture was
serialized as UTF-8 JSON with two-space indentation, sorted keys, unescaped
Unicode, and one trailing newline; hashes cover exact file bytes.

## Split and coverage

Development and heldout IDs are explicitly listed in the suite and are
disjoint. Each workload has one independent turn, preventing growing-history
coupling from converting one failure into several task blocks.

| Family | Development | Heldout | Total |
| --- | ---: | ---: | ---: |
| Tool selection / argument shapes | 5 | 5 | 10 |
| Repository read/search/patch/test | 4 | 4 | 8 |
| Python code behavior | 4 | 4 | 8 |
| Instruction constraints | 4 | 4 | 8 |
| **Total** | **17** | **17** | **34** |

Additional binding coverage:

- 12 heldout workloads are Japanese or mixed Japanese/English;
- five heldout workloads are patch or generated-code executable rows;
- development and heldout each contain single/nested/enum/optional/multiple and
  irrelevant/no-tool automatic-selection shapes;
- code has eight distinct entry points with at least four hidden cases each;
- instruction rows use behavioral checkers rather than exact prose;
- ten separate fail-safe controls cover malformed, truncated, duplicate,
  undeclared, schema-invalid, content/reasoning leakage, required-tool missing,
  ambiguous required selection, and `tool_choice=none` violation;
- repository patch/test state is synthetic and hermetic; and
- prompts do not receive expected result hashes, hidden code tests, patch
  replacement text, reference result objects, expected code, or reference
  response prose; machine-readable instruction checks only mirror constraints
  that are explicitly part of the user request.

The heldout **inputs and evaluator definitions** are committed now so the split
cannot move after results. AQ6 may publish heldout aggregate counts only.
Candidate selection uses development failures and general runtime invariants; it
must not inspect heldout model-output/token detail until candidate behavior and
tests are frozen.

## Tool and oracle model

Nine strict tools cover file read/search, lookup, arithmetic, nested record
transformation, exact patching, fixture tests, code submission, and response
submission. `tool_choice=auto` remains normative. The suite explicitly admits
one- and multi-call expected outcomes plus irrelevant requests where no tool is
the correct automatic behavior.

The separate oracle fixture owns reference actions, synthetic file state, patch
regions, file-hash suites, code entry points and hidden inputs/outputs,
machine-readable instruction checks, expected canonical result hashes, and
fail-safe outputs. Expected code source and exact instruction response prose are
intentionally not stored; instruction checks mirror only the public constraints. AQ5 must implement and RED-test a loader/evaluator that proves every
reference result and rejects split overlap, duplicate IDs, malformed counts,
missing language, oracle mismatch, and prompt leakage.

Generated code remains **data only** at AQ4. It must not execute until AQ5 proves
the sandbox contract in `docs/AGENTIC-QUALITY2.md`: no network, secrets, device
files, GPU, home/repository/model access, or orphan process; strict resource and
output limits; fresh directory; complete process-group kill; fail closed when
isolation is unavailable.

## Frozen collection settings

- real localhost blocking OpenAI chat;
- automatic tool choice;
- temperature `0.0`, thinking disabled;
- maximum 192 generated tokens;
- two deterministic repetitions;
- response-owned generated IDs;
- quality-only (`performance_claim=false`);
- checkpoint after every turn; and
- exact zero final request/session ownership relative to idle persistent state.

## AQ6 execution estimate and checkpoints

After AQ5 validates loader/oracle/sandbox behavior, the primary expanded
baseline executes 68 model observations (34 task blocks × two repetitions) plus
bounded fail-safe endpoint controls. On the ZBook, budget **15–45 minutes after
the cached model startup**. The assigned campaign authorizes this run.

Collection stages are:

1. clean source/model/capability/suite hash preflight;
2. all development rows, checkpointed per observation;
3. heldout rows, checkpointed per observation with model-output details kept
   local and only aggregate split/family totals committed;
4. deterministic fail-safe controls;
5. independent oracle and sandbox result join; and
6. final ownership/shutdown/hash validation.

A failure preserves the latest atomic checkpoint. Do not silently drop a row,
weaken isolation, change output cap/tool wording, or rerun only a favorable
split.
