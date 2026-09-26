---
status: current
owns: Qwen3.8 reasoning-trace compression research, external training references, and proposed low-cost post-training experiments.
---
# Qwen3.8 training: shorter thinking traces

## Recommendation and scope

Target **less unnecessary reasoning, not less reasoning regardless of necessity**.
Start with native reasoning-effort controls and compact prompting. The first
training comparison should use a shared dataset of verified, compact
self-generated or self-pruned traces: low-rank adaptation (LoRA) with supervised
fine-tuning (SFT), versus direct preference optimization (DPO) with an auxiliary
SFT loss. Preserve a long-reasoning mode and evaluate difficult-task regressions
separately. Consider on-policy distillation (OPD) for capability recovery before
paying for a larger reinforcement learning (RL) run.

This is a research reference and proposed experiment design, not an implemented
hipEngine training pipeline, a runtime admission policy, or a local benchmark
result. Qwen3.8-27B and Qwen3.8-Flash-Next are separate targets; recipes and
measurements do not transfer automatically between them. No architecture or
runtime default changes are proposed here.

### Source status

- **Primary cards inspected:** the two UkisAI pages linked below, accessed
  2026-09-26. Their contents establish what the authors disclose, not independent
  reproduction of training or evaluation. Links point to mutable pages; pin
  revisions before an experiment.
- **Imported research:** the user-supplied external deep-research synthesis.
  Its citation markers numbered 0–33 arrived without a bibliography or resolvable
  URLs. The study descriptions and numbers in the imported-research sections
  are retained as research leads, **not independently checked findings**. Missing
  citations have not been replaced with guessed links. Recover the papers and
  exact versions before relying on these numbers in a decision or publication.
- **Proposals:** the dataset construction, experiment order, evaluation design,
  and recommendations are working hypotheses, not a published matched ranking
  of training methods or costs.

## UkisAI sources

### Agent-training dataset

[Qwen3.8-27B-multi-turn-agent-sft](https://huggingface.co/datasets/ukisai/Qwen3.8-27B-multi-turn-agent-sft)
provides the starting corpus. The inspected dataset card describes approximately
15,200 agent traces, generated with Qwen3.8-27B in FP16 using the Terminus-2
harness, based on
[OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT).
It covers terminal, coding, and software-engineering tasks, including `nl2bash`
and `InferredBugs`. The viewer exposes conversations and provenance fields such
as task, episode, run ID, trial name, model, and agent.

The card describes SFT warm-start use and further post-training or RL. That is
not a claim that every trace is correct, compact, or suitable for direct
training. Swift's authors explicitly say they resample the data and convert it
into RL environments rather than use it unchanged.

Before constructing a training set:

1. Pin the dataset revision, inspect its schema and licenses, and retain source
   task and trajectory identifiers.
2. Check success labels, verifier availability, thinking/tool boundaries, and
   whether the underlying environment can be reconstructed. Replay agent data
   only in an isolated environment; dataset shell commands are not instructions
   to the training host.
3. Split by underlying task or task family before sampling episodes, trials,
   or edited variants. Related trajectories must not leak across train and eval.
4. Verify outcomes rather than equating a fluent final response with success.
   Separate tool observations from model-generated text in loss masking.
5. Keep original trajectories alongside compact alternatives and record how
   each edit was verified. Do not assume the public corpus alone reproduces Swift.

### Swift 1.5 Flash-Next training disclosure

Source: [Swift1.5-Qwen3.8-Flash-Next model card](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next),
“Training approach.” The author statement supplied for this reference also
appears on the inspected card:

> We made Swift Flash Next efficient by figuring out which tokens were linked to pathological overthinking and penalizing them without "attacking" the reasoning length directly then regained the accuracy with RL and OPD, leading to "compressed" token usage while maintaining accuracy.
>
> Swift 1.5 produces shorter reasoning traces. In our testing, we also observe fewer overthinking errors.
>
> This release also features our previously mentioned post-training methods adapted specifically for coding and long-horizon agent work such as personal agents, terminal use and software engineering.
>
> Our training data is viewable here: https://huggingface.co/datasets/ukisai/Qwen3.8-27B-multi-turn-agent-sft albeit it is not used out of the box, but rather re-sampled, turned into proper RL environments etc.

This discloses a token-targeted intervention followed by RL and OPD, not a
reproducible objective. The card does not specify the token-selection algorithm,
penalty formula, teacher, loss weights, sampling transformations, or compute
budget. Do not translate this into “ban `Wait`” or “apply a length penalty”; neither
is the disclosed recipe. The “maintaining accuracy” wording is an author claim,
not a guarantee for every task.

### What the Flash-Next card's evaluation does and does not show

These are **author-reported, externally unverified** values from the card's
“Evaluation” section, not hipEngine measurements. Both checkpoints are BF16;
the card describes xhigh thinking, MTP disabled, temperature 1.0, top-p 0.95,
top-k 20, and five seeds for seeded question benchmarks. The table selects rows
that illustrate different trade-offs; it is not a pooled result.

| Benchmark | Base score → Swift 1.5 | Mean tokens, base → Swift 1.5 | Token definition / cap |
| --- | --- | --- | --- |
| GPQA-Diamond | 89.80% → 89.60% | 17,683 → 7,823 | Thinking; output cap 100,000 |
| IFBench | 73.20% → 70.13% | 8,310 → 4,411 | Thinking; output cap 81,920 |
| AIME 2026 | 98.67% → 96.67% | 23,015 → 15,806 | Thinking; output cap 250,000 |
| LiveCodeBench v6 | 88.40% → 90.39% | 17,833 → 9,849 | Thinking; output cap 100,000 |
| Terminal-Bench 2.1 | 67.64% → 69.66% | 40,591 → 45,428 | **Total generated output**, not thinking alone |

The headline 63.4% is the GPQA **median thinking-token reduction**. Its mean
reduction is 55.8%; neither is a workload-wide savings estimate. Terminal-Bench
reports a 17.9% median reduction but an **11.9% mean increase** in generated
output. Shorter typical traces can coexist with a more expensive tail. AIME
and IFBench also illustrate why an aggregate accuracy headline is not a
per-domain guarantee.

Terminal-Bench uses five attempts per task, Harbor 0.20.0 / Terminus-2 2.0.0,
a pinned 89-task dataset, interleaved thinking, top-p 1, a 131,072-token server
context, a 3,600-second LLM call timeout, and native task limits. Swift used
concurrency 8 and a context-recovery fix applied during the run. The card warns
that default task timeouts make these Flash-Next results unlike its Swift 1.5
27B comparison. These details prevent treating the cards as a shared leaderboard.
No exact reproduction command, physical host identity, or independent evaluation
artifact was established for this reference; no local latency claim follows.

The same card reports GPQA score losses of 0.20 percentage points at xhigh,
2.62 at medium, and 2.42 at low, comparing base and tune at the same effort.
Compression must therefore be checked across the effort range, not only at one
setting. The card documents `xhigh`, `medium`, and `low`; whether each setting
and thinking-disabled mode works through a chosen serving template must be
confirmed separately, not inferred as a hipEngine API guarantee.

The card's licensing section describes revenue-linked commercial terms for the
Swift contribution and separate base-model obligations. Review the actual
licenses for the chosen revision, dataset, and intended use; this summary is
not legal advice or a license grant.

## Imported research: distinguish interventions and models

Everything attributed to the external synthesis in the following research
sections has the **not independently checked** status defined above. The
numbers preserve the supplied comparison context; they are not local targets.

### Released tunes are not interchangeable evidence

| Project in the supplied research | Disclosed approach / interpretation | Reported result and limitation |
| --- | --- | --- |
| Swift-Qwen3.8-27B, not Swift 1.5 Flash-Next | Tokens associated with overthinking are penalized; a transfer component comes from ThinkingCap-Qwen3.6-27B. Exact objective and token selection were not identified. | GPQA mean thinking −41.0%, score 88.38 → 88.28; the 58.3% headline is its largest median reduction. AIME 2026 score 98.67 → 94.00 for 26.7% fewer mean thinking tokens. Not lossless. |
| ThinkingCap-Qwen3.8-27B | Efficiency training; insufficient disclosure to label it SFT, DPO, or a specific RL method. | Macro-averaged percentage reduction 37.2%, aggregate accuracy 86.65 → 85.79; AIME 98.13 → 94.27. Terminal-Bench thinking −10.7%, total generated tokens −7.5%. |
| Signal-3.8-27B | Self-distillation from direct-answer prompting; only `lm_head.weight` changes. | General prompts: median answer −57%, median thinking −52%; coding: answer −11%, thinking −26%. Small evaluation does not establish hard-math or agent preservation. The synthesis says a September 13 checkpoint changed the trade-off but measurements described the first release. |
| Ornith 1.5 | Joint task-generation, scaffold, and solution RL on top of broader pretraining and post-training work. | Not a clean brevity-only intervention or evidence of a cheap compression recipe. |

ThinkingCap's reported macro-average token counts, 15,735 → 12,144, imply
about 22.8% reduction by ratio of means, not 37.2%. A mean of benchmark
percentage reductions answers a different question. Neither should substitute
for the traffic-weighted cost of the deployment.

### Four different ways to shorten reasoning

| Intervention | What changes | Failure to watch for |
| --- | --- | --- |
| Compress language | Sentences become equations, fragments, or structured notes. | Lost qualifiers, dependencies, or computational steps. |
| Prune redundancy | Remove duplicate derivations and resolved branches. | A useful correction or prerequisite looks repetitive. |
| Improve strategy | Replace an inefficient search with a shorter algorithm. | Requires capability learning, not only a style change. |
| Stop earlier | End when further thinking is unlikely to help. | Premature confidence prevents recovery. |

### No-training baselines

Test supported native effort settings and thinking-disabled mode, then a
compact-scratchpad instruction. This proposed prompt is not a published optimum:

> Use compact working notes: equations, constraints, intermediate results, and necessary checks. Avoid restating the problem or repeating completed checks. Revisit a conclusion when there is a specific unresolved inconsistency.

Prefer structured notes to maximally broken grammar. For example:

```text
Constraints: n ≥ 1; duplicates allowed.
Candidate: sort, then scan.
Issue: sorting destroys required order.
Fix: hash set + stable scan.
Check: empty input, duplicates, negative values.
```

Preserve operators, units, negation, variable bindings, counterexamples, and why
a failed approach failed. Count the target tokenizer's tokens, not characters.

The supplied research identifies these training-free leads:

- **Chain of Draft:** short reasoning steps reportedly use as little as 7.6%
  of conventional chain-of-thought (CoT) tokens in tested cases. This is not a
  Qwen3.8 hard-task result.
- **CAVEWOMAN (2026):** output compression reportedly reduces costs roughly
  1.4–2.4×, but input compression can provoke longer output and higher total
  cost. Short visible answers do not prove less hidden reasoning, and correct
  final labels do not prove semantic preservation.
- **DEER — Dynamic Early Exit for Reasoning:** confidence-based intermediate
  answer probes reportedly change DeepSeek-R1-Distill-Qwen-7B MATH accuracy
  87.4 → 89.8 while tokens fall 3,858 → 2,143; the synthesis also reports AIME
  improvement. Count probe overhead and calibrate out of distribution. A hard
  cap is an emergency limit, not a learned stopping strategy.

## Imported research: SFT and preference training

Neither “SFT cannot fix overthinking” nor “DPO always beats SFT” follows from
these studies. Target construction can matter as much as the loss.

### Compact self-training and controllable adapters

**Self-Training Elicits Concise Reasoning in LLMs** uses concise demonstrations
to elicit short self-generated solutions, selects correct ones, and trains on
them. The supplied synthesis reports about 30% average length reduction with
average accuracy preserved across five model families on GSM8K and MATH.
Self-generated traces preserved performance better than some human/GPT-4o
traces in that setup. Eliciting compact candidates first is more attractive
than blindly generating hundreds of long candidates to find one short trace.

**CoT-Valve** learns a controllable parameter direction with LoRA and long/short
reasoning data. The reported QwQ-32B GSM8K comparison is 741 → 225 tokens and
95.07 → 94.92 accuracy, using rank-2 LoRA and two H100 80GB GPUs. Hardware count
is not a total cost estimate. The useful design idea is an adjustable
compression level, with validation over its full range.

### LCPO: a useful preference-objective comparison

The supplied synthesis reports this DeepSeek-R1-Distill-Qwen-7B MATH-500 table
for LCPO, a length-oriented preference method:

| Method | Accuracy | Mean generated tokens | Reduction |
| --- | ---: | ---: | ---: |
| Original | 92.20 | 4,223 | — |
| SFT | 87.00 | 3,467 | 17.9% |
| DPO | 90.40 | 2,601 | 38.4% |
| LCPO | 92.00 | 1,813 | 57.1% |

LCPO chooses shortest/longest trajectories on reliably solved problems. Its
reported 400-pair, 50-step run is not an equal-budget comparison with the
350-step comparison checkpoints. About 22,000 trajectories were generated
before selection: 400 pairs is not 400 generations. Include it as an experiment,
not as proof that it dominates well-curated SFT.

### Self-guided pruning plus an SFT anchor

**Your Reasoning Model Knows What Counts: Self-Guided Chain-of-Thought Pruning
for Efficient Reasoning** (SGP-CoT; described by the supplied synthesis as
ACL 2026) scores semantic segments for answer contribution and local coherence.
It trains on pruned/original pairs with DPO plus an auxiliary SFT loss, using
approximately 1,500 pairs and LoRA.

Reported DeepSeek-R1-Distill-Qwen-7B AIME results are 52.3 → 52.7 accuracy and
10,400 → 6,965 tokens. The SFT-only ablation is 40.3 accuracy / 17,317 tokens;
removing the SFT component gives 49.7 / 8,333. The combination matters in that
setup, but transfer to Qwen3.8 is an experiment. Reported pruning impact
analysis uses under 10 H800 GPU-hours; approximately 200 GPU-hours describes
the broader study, not one mandatory fine-tune.

For our comparison, use both preference directions:

- **Short correct > long correct** when the extra material is redundant.
- **Long correct > short incorrect** so length cannot predict the label alone.

Do not make every long trajectory a rejected response. Preserve difficult
examples whose successful solution needs more computation.

## Imported research: what to prune, and what to keep

**TokenSkip** uses an importance model based on LLMLingua-2 and trains the
student to generate compressed reasoning conditioned on compression level.
The supplied Qwen2.5-14B/GSM8K result is 313 → 181 tokens with less than
0.4 percentage points accuracy loss. Training the model to generate fewer
tokens can save decoding work; deleting tokens after generation cannot recover
that work, though it can reduce later context processing.

Token salience is not a sufficient deletion rule:

- **NoWait — Wait, We Don't Need to “Wait”!** reportedly suppresses selected
  thinking expressions at inference and reduces tokens roughly 27–51% on its
  tested models and benchmarks. This is a cheap experiment, not a universal
  banned-word list.
- **Demystifying Reasoning Dynamics with Mutual Information** associates words
  such as “Hmm,” “Wait,” and “Therefore” with information peaks. **Thought
  Anchors** uses counterfactual continuations and interventions to identify
  important sentences, often involving planning or backtracking. A surface word
  can precede either a redundant recheck or an essential correction.
- **Let's Think Dot by Dot** finds useful computation in apparently meaningless
  filler on controlled algorithmic tasks. Semantic redundancy does not prove
  computational redundancy, nor does that study prove Qwen verbosity necessary.
- **R1-Compress** combines within-chunk compression and search across chunks;
  **Prune-on-Logic** separates core reasoning from self-verification. The
  synthesis treats both as leads for structure-aware compression.

Proposed evidence ordering: deletion followed by successful student continuation
is more persuasive than model-estimated answer contribution, which is more
persuasive than generic linguistic salience. Attention weights, entropy, or
discourse markers alone are not enough to justify removing a step.

### Editing procedure

1. Identify constraints, intermediate facts, chosen strategy, failed branches,
   corrections, dependencies, and final checks.
2. Remove duplicate derivations and repeated narration. Collapse a failed branch
   into its useful conclusion, such as “sorting rejected because order must be
   preserved,” rather than deleting the reason for the change.
3. Validate the answer with tests, an exact verifier, or an independent check;
   label the verification strength rather than treating all checks as equal.
4. Validate continuation from before the edited region. Confirm that the student
   can use the compact prefix and finish without facts leaked from later steps.
5. Retain productive error recovery and long successful solutions as controls.

An editor who knows the final answer can create a concise explanation containing
an unsupported leap. A correct explanation is not necessarily an executable
reasoning trajectory for the student.

## Imported research: teacher distillation and capability recovery

### Offline and on-policy distillation

**s1: Simple test-time scaling** reportedly learns from 1,000 selected problems
and teacher traces. Its budget-forcing result raises AIME performance 50% → 57%.
This supports small-data distillation while warning that additional thinking
can still help.

**Small Models Struggle to Learn from Strong Reasoners** reports that students
around 3B parameters and below do not consistently benefit from strong teachers'
long traces; mixing lengths or matching teacher capability can help. This is
a capacity-mismatch warning, not evidence against a 27B student's benefiting
from a stronger teacher.

Proposed teacher use: send unresolved or high-value examples to the stronger
model, have it discover or repair the solution, express the solution compactly,
then validate student continuation. Do not pay to rewrite the entire corpus
before establishing that the targets are learnable. Capability transfer and
brevity transfer are separate objectives.

Offline distillation trains on teacher trajectories. OPD instead supplies
teacher-distribution feedback on states visited by the student. Teacher-forced
scoring can evaluate a student sequence in parallel rather than autoregressively
generate a replacement, although actual cost depends on the implementation.

The supplied synthesis gives this Qwen3-8B comparison, **not a brevity result**:

| Stage | AIME 2024 | GPQA Diamond | Reported GPU-hours |
| --- | ---: | ---: | ---: |
| After off-policy distillation | 55.0 | 55.6 | — |
| Additional RL | 67.6 | 61.3 | 17,920 |
| Additional OPD | 74.4 | 63.3 | 1,800 |

This suggests testing OPD for recovery before expensive sparse-reward RL. The
reported incremental GPU time is not all-in teacher creation, data generation,
or a hardware-normalized cost comparison. Simple token-distribution matching
is easiest with compatible tokenizers and accessible teacher log probabilities.
It requires more infrastructure than offline SFT/DPO.

### Repair and compression are different labels

**Step-DPO** compares correct and incorrect next reasoning steps after a shared
prefix. The synthesis reports Qwen2-7B MATH 53.0 → 58.6 and GSM8K 85.5 → 87.9
using about 10,000 preference pairs. This is capability improvement, not direct
brevity evidence. Use the principle to distinguish:

- **Correction examples:** which inference is wrong, and what replaces it.
- **Compression examples:** which already-correct material is unnecessary.

Do not label an entire long trace bad when one early error caused the wandering.

## Imported research: when RL is worth the cost

RL is a candidate when the student rarely produces a compact correct strategy,
leaving selection and editing without enough valid targets.

**ShorterBetter** uses a per-problem target based on the shortest correct sample
in a rollout group, rather than a universal length. The synthesis reports
roughly 50–80% reductions while retaining performance in its experiments; those
are not Qwen3.8 guarantees.

**The Art of Efficient Reasoning: Data, Reward, and Optimization** distinguishes
early length adaptation from later reasoning refinement. Shortening may precede
accuracy recovery; easier prompts can provide useful positive feedback;
aggressive short-budget training can damage long-budget capability. Mishandled
negative examples can teach a “short-is-correct” shortcut.

Use a bounded efficiency incentive subordinate to correctness, preserve long
successful trajectories, and evaluate the full budget–accuracy curve. A large
universal token penalty is not the default experiment.

## Proposed experiment sequence

These are proposals, not committed sample counts, approved training budgets,
or implementation tasks.

### 1. Establish matched baselines

Compare the base at supported effort settings, the base with compact prompting,
and relevant released tunes. Match chat templates, quantization, sampling,
output caps, task timeouts, and tool harnesses. Document unavoidable differences.
First isolate model quality from speculative decoding and serving changes;
measure the intended deployment configuration separately afterward.

Evaluate both realistic caps and a generous-budget reference. Completing inside
a cap that truncates the base is a useful deployment benefit, but does not prove
preserved unconstrained capability. Do not pool BF16 and quantized comparisons.

### 2. Build one shared, deliberately varied dataset

Start with a few thousand candidate prompts and disjoint evaluation tasks.
Generate a small number of ordinary and compact candidates per prompt. Use
semantic pruning or a stronger teacher selectively when cheaper generation
fails. Include easy and difficult tasks, successful recovery, tool episodes,
and tasks where longer reasoning is necessary. Include Japanese and
mixed-language tasks if they are part of deployment; English math results do
not establish multilingual transfer.

Record at least:

| Field group | Required experimental record |
| --- | --- |
| Provenance | Dataset/model/tokenizer revisions, task family, split, original trajectory ID, generation settings, chat template |
| Outcome | Correctness, verifier identity/version, verification strength, task/environment completion |
| Content | Original/compact pair, thinking/answer/tool boundaries, target-tokenizer lengths, edit method, continuation-check outcome |
| Coverage | Domain, difficulty, language, recovery behavior, whether long reasoning was necessary |
| Preference | Chosen/rejected rationale; distinguish correctness, correction, and redundant-length preferences |
| Cost | Generation, editing, replay/verification, teacher scoring, training, evaluation |

### 3. Run a small ablation matrix

| Experiment | Purpose | Main caveat |
| --- | --- | --- |
| Compact prompting | No-training baseline | May be inconsistent or lose accuracy. |
| Head-only self-distillation | Test output/style selection as a cause | A frozen backbone does not guarantee preserved behavior; forward computation still costs. |
| LoRA SFT on verified compact traces | Cheapest serious training baseline to test | The student must be able to execute the target. |
| LoRA DPO + SFT anchor, same curated data | Learn to prefer omission of redundant work while reinforcing good targets | Both trajectories and reference scoring add cost; long rejected traces can be expensive. |
| LCPO-style preference objective | Test an alternative length-oriented objective | Match data and budgets instead of comparing cross-paper headlines. |
| Budget-conditioned or adapter-strength-controlled variant | Preserve an adjustable quality/compute trade-off | Validate the full operating range, including long mode. |
| OPD, then targeted RL if needed | Repair capability gaps not fixed offline | Rollouts, teacher scoring, and infrastructure add cost. |

For SFT versus DPO, derive targets and pairs from the same verified prompt pool,
record training tokens and total project cost, and avoid selecting checkpoints
on the final holdout. SFT anchors, pair construction, and budget controls should
be explicit ablations rather than hidden recipe differences.

### 4. Judge complete tasks

The proposed selection objective is:

\[
\min_\theta \mathbb{E}[C_{\text{complete task}}]
\quad\text{subject to}\quad
A_d(\theta) \ge A_d(\text{base}) - \epsilon_d.
\]

Here \(d\) is a domain or difficulty slice, \(A_d\) is its success metric,
and \(\epsilon_d\) is an explicitly accepted regression allowance. Choose
allowances before automated selection; this equation is an evaluation proposal,
not a guarantee from any training loss. Easy-task gains must not conceal an
unacceptable regression on hard code, mathematics, or recovery.

Measure:

- **Quality:** task success, difficult-subset success, instruction adherence,
  tool correctness, error recovery, and long-budget capability.
- **Compute:** thinking and answer tokens separately, total generated tokens,
  tool calls, repeated context processing, probes, retries, and completed tasks.
- **Serving:** median and tail latency, throughput at actual concurrency, and
  cost per successful task, including failed attempts.

For the same evaluated workload, generated-token savings are:

\[
1 - \frac{\sum_i T_{\text{tuned},i}}{\sum_i T_{\text{base},i}}.
\]

This is not an unweighted mean of benchmark percentage improvements. If sampled
traffic differs from production, weight both sums by the declared traffic mix.
Report mean, median, and tails; lengths on correct and incorrect outputs; and
truncation and timeout rates. A model that abandons difficult tasks quickly can
look efficient in token-only averages.

Repeat sampling where needed and estimate uncertainty over independent tasks,
not only seeds. Repeated attempts on the same small task set measure sampling
variance, not an equally large set of independent reasoning problems.

### 5. Count the full cost

\[
C_{\text{project}} = C_{\text{generation}} + C_{\text{editing/verification}}
+ C_{\text{training}} + C_{\text{evaluation}} + C_{\text{engineering}}.
\]

Small training sets can hide expensive selection rollouts. DPO avoids online
exploration during training but processes both sides of a pair. Head-only
training still pays backbone forward cost. OPD's reported incremental GPU time
does not include every upstream expense. Record licenses and estimate how many
successful production tasks repay the experiment only after measuring actual
cost reduction and quality trade-offs.

## Open questions and next research work

- Recover the original bibliography for imported research, prioritizing
  SGP-CoT, LCPO, concise self-training, CoT-Valve, and the Qwen OPD comparison.
  Check paper versions, loss definitions, datasets, exclusions, budgets, and
  whether reported ablations share an evaluation protocol.
- Obtain Swift's token-selection and penalty details if published, along with
  RL/OPD teachers, objectives, environment conversion, and stage-wise ablations.
  The inspected card does not establish which stage causes each improvement.
- Audit the public agent corpus for success labels, replayability, licensing,
  contamination, and useful compact alternatives before treating it as a
  ready-made preference dataset.
- Select the first student explicitly: 27B versus Flash-Next, precision,
  tokenizer, context budget, and training hardware. Do not infer a feasible
  training budget from inference memory requirements.
- Preserve long mode and productive correction. The target is a shorter
  executable reasoning process, not merely a shorter explanation of an answer
  already known to the editor.

For local serving and architecture context, see [PLAN.md](../PLAN.md),
[API.md](../API.md), and the
[Flash-Next implementation campaign](../campaigns/QWEN3.8-FLASH-NEXT.md).
Any future hipEngine performance claim needs its own measurements under
[OPTIMIZATION.md](../OPTIMIZATION.md); these external research notes establish
no runtime capability restriction.
