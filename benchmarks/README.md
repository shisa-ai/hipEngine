# hipEngine Topline Benchmarks

Last updated: **2026-09-07 UTC**

QSA CPU-floor diagnostic:policy17 minimum2->4GHz accepted but early
decode1.169/1.589s and~1.5GHz/~600MHz feedback persist. Original
minimum/affinity restored;no production policy change.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-cpu-floor.json).

ROCr MWAITX-request diagnostic:parent/candidate tg128 wall8.496/8.937s,
early CPU-frequency split persists. Exact state;runtime hashes retained,
actual MWAITX engagement unproven. No production environment change.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-mwaitx.json).

HIP wait-policy QSA probe:spin was already1;yield1->2 accepted but early
CPU recovery penalty persists. Parent/candidate tg128 wall8.496/8.925s
(spin),8.509/8.940s (yield),exact state. No production flag change.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-wait-policy.json).

CPU17 affinity isolation does not fix QSA transition:parent/candidate
early decode1.167/1.573s,exact state,early~1.5GHz/~600MHz.
Original affinity restored;no production pinning or clock changes.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-phase-affinity.json).

QSA transition frequency sampling:after candidate prefill,early decode
remains near600MHz versus parent~1.5GHz;both later~5.1GHz.
Wall1.170->1.567s,sampling3.5-8ms. No clock changes;policy/firmware
trigger still unproven and candidate remains default-off.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-phase-frequency.json).

QSA CPU-counter isolation:early decode wall1.169->1.581s despite
~3.91B instructions and1.72-1.75B cycles in both arms,100% counter
running. Lower effective CPU rate is plausible;clock cause unproven.
No production change.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-phase-perf.json).

QSA transition CPU accounting:early decode wall1.170->1.574s follows
thread CPU1.165->1.563s;no GC or major faults. Off-CPU waiting does
not explain this run. Instructions/cycles next;production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-phase-cpu.json).

QSA early-transition trace:same29440 kernels,parent/candidate GPU
busy981.8/976.1ms but wall1203.1/1569.3ms. Extra time is outside
kernel execution;CPU/submission investigation next. Profiler changes
queue ring;no production or throughput claim.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-phase-trace.json).

QSA phase isolation:decode flag held0,preceding candidate prefill changes
code/mixed tg128 wall8.478->8.925s /8.482->8.936s,exact tokens/state.
Extra~0.45s concentrates in first16 steps;steady tail converges.
Candidate remains default-off;hardware cause not established.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-prefill-phase.json).

Paired-head flag-only decode isolation:fixed parent roots,code/mixed4096,
4 balanced pairs of16 steps,wall+0.055%/+0.019%,exact tokens/state,
zero paired-head calls. Does not explain candidate-prefill phase effects;
production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-decode-flag.json).

Paired-head QSA staged model run:p4096 PP186.922->193.577 (+3.560%),
TG13.662->12.771 (-6.526%);72 exact,first24 samples preserved.
Two long-request means regress;candidate retained default-off pending
decode/phase investigation. No production change.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-head-pair-model.json).

Paired-head QSA admission:eight full-logit/state/KV cases exact,
all four p4096 categories,24 sparse calls each,zero decode/short calls.
Both binders0;no production change,staged throughput gate pending.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-head-pair-state.json).

H256 QSA paired-head kernel candidate:synthetic2051-selected attention
512/1024 rows2.544x/2.610x,30 pairs exact,23 tests. VGPR40->72,
no scratch/LDS. No runtime/default change or whole-model speedup claim.
[Evidence](results/2026-09-07-framework-qwen4exp-qsa-head-pair.json).

Raw-vector square64 rejected against production128x64:actual Q512/1024
screen0.959-0.985x,80 exact pairs,27 tests. Candidate removed,
production unchanged;historical zbook evidence remains separate.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-square64-rejected.json).

Prepacked MMQ token64 rejected:qkv/SSM512/1024 screen0.931-0.969x,
80 exact pairs,27 tests. Lower VGPR did not improve timing.
Prepacked128 and promoted raw-Q64 defaults unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-prepacked-token64-rejected.json).

Raw-Q token64 promoted:full12-case PP190.315->191.353 (+0.545%)/
202.324->203.168 (+0.417%)/186.490->186.895 (+0.217%),72 exact.
All PP and11 request means improve;Japanese4096 request loss0.183%
explicitly retained under prefill-first policy. External rates unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-token64-production.json).

Raw-Q token64 admission:five chunk1024 full-logit/state/KV cases exact,
12/48 calls,zero decode/final owners. Prepacked/GR unchanged,both
binders0;no production change,throughput gate pending.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-token64-state.json).

Raw MMQ token64 candidate:large Q1024 layer3/7 operation-complete
screens1.109x/1.138x,200 pairs exact,23 tests. VGPR184->160,no scratch.
Mixed short-Q/long-GR results retained;no runtime/default change.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-token64.json).

Fresh combined-default95df2c3a9/chunk1024 versus pinned halo-box b212548e0:
HE PP190.94/201.82/186.03,TG19.79/19.34/14.93;
Vulkan PP342.90/393.68/420.95,TG25.60/25.38/24.55.
108 trajectories repeat within engines;sequential screen,not statistical
closure. All engines have cases above2% CV;HIP max PP CV13.71%.
[Evidence](results/2026-09-07-framework-qwen4exp-current-default-baselines.json).

MMQ bank-first loop rejected:nearflat512/small positive1024 medians,
but mixed arm-order evidence.80 pairs exact,22 tests;VGPR184->192.
Candidate removed,production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-bank-first-rejected.json).

MMQ scale-hoist rejected:actual qkv/Q projection screen0.319-0.338x
at512/1024 rows,80 exact pairs,22 tests. VGPR184->256,
scratch0->1132B;candidate removed,production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-mmq-scale-cache-rejected.json).

One-pair screening calibration:two training/two held-out Framework
chunk1024 packets give empirical first-pair error factors1.0352 PP/
1.0340 request. Held-outs stay within envelope but do not establish
all-case early promotion. Diagnostic only; no confidence guarantee.
[Evidence](results/2026-09-07-framework-qwen4exp-screen-calibration.json).

Q5_1 row16 publication rejected:two-bank512/1024 screen0.429x/0.412x,
40 exact pairs,9 tests;no spills but larger LDS. Candidate removed,
production row8 unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-q51-row16-rejected.json).

GDN wave normalization promoted on actual-model owner saving:
serial p4096~487->467ms across four categories,378 calls/756 measured
pairs exact,all call means faster. Full-suite throughput remains mixed
nearzero; no new headline rate. Strict/tiled paths unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-gdn-wave-norm-owner.json).

GDN wave-norm full-model qualification:72 exact trajectories,
PP+0.010%/+0.139%/-0.044%,mixed request-case means. No headline win
established; kernel retained default-off pending actual-model owner timing.
[Evidence](results/2026-09-07-framework-qwen4exp-gdn-wave-norm-model.json).

GDN wave-normalization admission:five chunk1024 full-logit/state/KV cases
exact,21/84 serial-prefix calls,zero decode;18 CPU tests. Both binders0.
Corrected harness count36->21 reflects existing tiled suffix,not widening.
[Evidence](results/2026-09-07-framework-qwen4exp-gdn-wave-norm-state.json).

GDN exact wave-tail normalization candidate:Hk16/Hv48/D128 synthetic
512/1024 tokens2.812->2.699ms /5.574->5.358ms (~4% kernel gain),
60 measured pairs exact. Scratch24->36B; no runtime/default change.
[Evidence](results/2026-09-07-framework-qwen4exp-gdn-wave-norm.json).

Q4 two-block prefetch rejected:actual layer3 gate/up+SiLU screen
0.910x/0.909x at512/1024 tokens,40 pairs exact;14 tests.
VGPR88->112,no scratch; candidate removed,production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-q4-block-pair-rejected.json).

Post-Q5_1-row-publication family refresh,clean e7024541b/chunk1024:
p4096 four-category FFN11.231->10.281s (-8.46%),total22.196->21.227s
(-4.36% snapshot). Linear4.523s,GR3.414s,QSA2.068s,GDN0.826s.
12 phases fully attributed; Vulkan profiles reused,not fresh throughput.
[Evidence](results/2026-09-07-framework-qwen4exp-post-q51-row-publish-family.json).

Q5_1 per-row publication promoted:chunk1024 full12-case PP512/1024/4096
181.824->189.634 (+4.296%)/191.750->200.824 (+4.732%)/
177.070->184.724 (+4.323%).72 exact,all prefill/request averages improve;
total request1.024648x,TG-0.012%/-0.048%/-0.201% explicit.
[Production evidence](results/2026-09-07-framework-qwen4exp-q51-row-publish-production.json).

Q5_1 per-row publication admission:five chunk1024 full-logit/state/KV
cases exact,25/100 prefill calls,zero decode/final owners;16 CPU tests.
Historical default-off admission; production result above supersedes it.
[State evidence](results/2026-09-07-framework-qwen4exp-q51-row-publish-state.json).

Q5_1 per-row publication kernel candidate:two-bank512/1024 screen
1.413x/1.523x; captured mixed512 routing1.498x.60 pairs exact,
22 tests pass,scratch36->0B. No default change; model throughput gate pending.
[Evidence](results/2026-09-07-framework-qwen4exp-q51-row-publish.json).

Q5_1 first-wave reduction rejected before model admission:
actual-weight paired-down screen0.809x/0.798x at512/1024 tokens,
all40 pairs exact.21 tests pass; production unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-q51-wave-tail-rejected.json).

Q8 paired-load candidate rejected and removed:full12-case PP
+0.433%/+1.664%/+1.865%, but one prefill/nine request cases regress;
p4096 TG -14.822%,total request0.975227x.72 trajectories exact.
Production unchanged; decode slowdown cause unproven.
[Evidence](results/2026-09-07-framework-qwen4exp-q8-prefetch2-rejected.json).

Historical Q8 paired-load gate admission:five chunk1024 full-logit/state/full-KV
cases exact,36/144 prefill calls,zero decode calls/final owners.
Subsequent throughput rejection above supersedes this admission.
[State evidence](results/2026-09-07-framework-qwen4exp-q8-prefetch2-state.json).

Historical exact Q8 paired-load kernel candidate, layer0/4 gates at1024 rows:
10.909->7.992ms /10.893->7.998ms; independent layer20 pressure screen
11.078->8.096ms. All180 pairs exact, VGPR72->80/no scratch.
K640 shared-down loses; subsequent model gate rejects the candidate.
[Evidence](results/2026-09-07-framework-qwen4exp-q8-prefetch2.json).

Post-Q8-register chunk1024 family refresh at cleancc3d48a70:
four-category p4096 FFN11.356->11.231s, linear4.515s, GR3.429s,
QSA2.071s, GDN0.835s; total22.281->22.196s (-0.38% snapshot).
All12 phases fully attributed; Vulkan profiles explicitly reused.
Not a new external throughput comparison or single-change A/B.
[Family evidence](results/2026-09-07-framework-qwen4exp-post-q8-register-family.json).

Q8 down register reuse promoted at chunk1024: full12-case PP512/1024/4096
181.424->182.969 (+0.851%)/191.973->193.319 (+0.701%)/
177.663->178.812 (+0.647%).72 exact trajectories, all12 prefill/request
averages improve; total request1.004743x, weakest only1.000040x.
Max PP/TG CV1.181%/3.356%; no causal decode claim.
[Production evidence](results/2026-09-07-framework-qwen4exp-q8-down-register-production.json).

Q8 down register-weight kernel candidate:compact37.373->32.127ms,
mapped37.228->30.808ms,40 exact pairs/both orders positive.
18 kernel tests pass,VGPR24->32/no scratch. Default-off admission also
passes30 CPU tests and five chunk1024 full-logit/state/KV cases exactly;
compact/mapped engagement verified, zero decode calls/final owners.
Historical admission; subsequent production result above supersedes pending A/B.
[State evidence](results/2026-09-07-framework-qwen4exp-q8-down-register-state.json).
[Evidence](results/2026-09-07-framework-qwen4exp-q8-down-register.json).

Router shuffle rejected after full12-case A/B:PP+0.357%/+0.168%/+0.253%
but one prefill/four request cases lose.72 exact trajectories;candidate
removed,production unchanged. No favorable rerun.
[Evidence](results/2026-09-07-framework-qwen4exp-router-shuffle-rejected.json).

Router shuffle model admission at chunk1024:five cases exact for full
logits/routing/state/KV;48/192 prefill calls,zero decode/final owners.
Both binders0;canonical throughput gate remains.
[Evidence](results/2026-09-07-framework-qwen4exp-router-shuffle-state.json).

Exact router shuffle-tail kernel candidate:layer0/27 at1024 rows
3.043->2.635ms /3.051->2.648ms,80 pairs exact,both orders positive.
16 tests pass,resources unchanged. Model gates pending,default unchanged.
[Evidence](results/2026-09-07-framework-qwen4exp-router-shuffle.json).

Chunk1024 full family refresh at `fc947658a`:p4096 FFN12.698->11.356s,
GR3.755->3.404s;QSA1.922->2.070s. Snapshot includes mapped-Q8 and
chunk promotions,not single-change attribution. All12 phases pass
coverage/state/lifecycle;Vulkan profile explicitly reused.
[Family evidence](results/2026-09-07-framework-qwen4exp-post-chunk1024-family.json).

Qualified gfx1151 UD-Q4_K_XL production now defaults to chunk1024:
full-suite long prefill+7.87%/+8.18%,72 exact trajectories and full
state/KV/native-c2/public8K gates pass. p4096 decode-1.72% and short-work
fluctuations retained explicitly;no new external comparator row.
[Promotion evidence](results/2026-09-07-framework-qwen4exp-chunk1024-promotion.json).

Chunk1024 native-context/c2 allocation passes:two262144-context runners,
106.87GB tracked,1.46GB scratch margin before reserve,zero final owners.
Allocation only,not native-length inference;scoped promotion recorded above.
[Evidence](results/2026-09-07-framework-qwen4exp-chunk1024-native-c2-memory.json).

Full-category chunk512->1024 screen:PP1024176.71->190.61 (+7.87%),
PP4096162.80->176.11 (+8.18%);p512-0.053% neutral-work control.
72 exact trajectories;all long requests faster,but p4096 decode-1.72%
and small p512 request losses remain explicit. Subsequent scoped promotion
and native/c2 qualification are recorded above.
[Evidence](results/2026-09-07-framework-qwen4exp-chunk1024-full.json).

Chunk1024 state gate:all12 cases exact for full logits/four decode steps/
recurrent state/full KV;coverage verified,zero final owners. Shared larger
allocation;native/c2 memory and full-category throughput gates remain.
[Evidence](results/2026-09-06-framework-qwen4exp-chunk1024-state.json).

Framework code-only chunk512->1024 screen:PP1024179.23->190.61
(+6.35%),PP4096168.06->177.26 (+5.48%);p512 neutral.
18 trajectories exact,zero final owners;shared larger-capacity allocation.
Default unchanged pending full categories/state/KV/memory gates.
[Evidence](results/2026-09-06-framework-qwen4exp-chunk1024-code.json).

| Framework UD-Q4_K_XL / BF16 KV mapped Q8 down | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 178.770 -> 181.065 (+1.284%) | 175.601 -> 178.108 (+1.427%) | 161.575 -> 164.150 (+1.594%) |

All72 trajectories exact,every prefill/request case improves,total1.00962x.
No new allocation;TG-0.059%/+0.031%/+0.239% incidental.
Max PP CV3.027%;not a new external comparator row.
[Production evidence](results/2026-09-06-framework-qwen4exp-q8-mapped-down-production.json).

Earlier mapped Q8 down admission:five full-logit/state/KV cases exact,
1/8 enabled prefill calls,zero decode/final owners. Current-call map
ownership explicit;both binders0 at admission,subsequent A/B above passes.
[State evidence](results/2026-09-06-framework-qwen4exp-q8-mapped-down-state.json).

Mapped Q8 down screen:layer2 weights with borrowed code/mixed layer0
counts80.728->37.969ms /80.566->37.420ms (2.126x/2.153x),40 pairs
exact. Reuses existing kernel and map ABI;model gates subsequently passed.
[Evidence](results/2026-09-06-framework-qwen4exp-q8-mapped-down.json).

Post-raw-MMQ-vector full owner refresh at `eee086e25`:p4096 four-category
linear4.762->4.556s,GR4.254->3.755s (-4.34%/-11.73% snapshot changes).
FFN12.698s remains largest;all12 phases have100% attribution and clean
teardown. Vulkan profile explicitly reused,not a new throughput comparison.
[Family evidence](results/2026-09-06-framework-qwen4exp-post-raw-vector-family.json).

| Framework UD-Q4_K_XL / BF16 KV raw MMQ vector staging | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 173.204 -> 179.037 (+3.367%) | 170.840 -> 175.707 (+2.849%) | 157.211 -> 161.879 (+2.969%) |

All72 trajectories exact,every prefill/request case improves,total1.01730x.
TG-0.354%/-0.064%/-0.214% retained,no new allocation;max PP CV2.929%.
Not a new external comparator row.
[Production evidence](results/2026-09-06-framework-qwen4exp-mmq-raw-vector-production.json).

Earlier raw MMQ vector model admission passes five full-logit/state/KV cases:
242/1936 enabled prefill calls,zero decode/final owners. Both binders0 at
admission;subsequent canonical gate above passes.
[State evidence](results/2026-09-06-framework-qwen4exp-mmq-raw-vector-state.json).

Raw MMQ vector kernel screen:GR/query/output512 complete chains
1.891->1.371 /6.452->5.293 /2.891->2.229ms,all120 pairs exact.
25 tests pass,resources unchanged,no sidecar. Model gates subsequently passed.
[Evidence](results/2026-09-06-framework-qwen4exp-mmq-raw-vector.json).

Q4 cooperative residual staging rejected:61.357ms versus exact17.403ms;
no-unroll59.887ms and LDS-padding57.152ms still lose. Reference-exact,
not strict-exact;cooperative code removed,production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-cooperative-rejected.json).

Q4 two-plane WMMA diagnostic:gate/up BF16 agreement89.66%->99.65%,
post-SiLU82.46%->99.37% on one actual-weight fixture,but all tested
widths lose to exact pair2. T2 reference only;no model/default change.
[Numerical and negative timing evidence](results/2026-09-06-framework-qwen4exp-q4-residual-wmma-reference.json).

Q4 paired-input layout rejected:packing+gate/up+SiLU17.470->22.938ms,
exact and both orders slower. Consumer itself regresses despite VGPR88->72;
packer costs only0.184ms. Candidate removed,production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-input-pair-rejected.json).

Q4 wave-metadata scalarization rejected:VGPR88->80 but actual gate/up+
SiLU17.478->17.490ms,order-sensitive and exact. Candidate removed;
production unchanged,no full-model A/B spent.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-wave-meta-rejected.json).

Full post-Q5_1-register owner refresh at clean `932af5889`:four-category
p4096 FFN13.442->12.813s (-4.68% snapshot delta),linear4.762s,GR4.254s.
All12 phases pass100% attribution,matched roots and clean teardown.
Vulkan profile reused explicitly;not a new throughput comparison.
[Family evidence](results/2026-09-06-framework-qwen4exp-post-q51-register-family.json).

| Framework UD-Q4_K_XL / BF16 KV Q5_1 register reuse | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 168.518 -> 173.392 (+2.892%) | 165.910 -> 170.780 (+2.936%) | 153.381 -> 157.547 (+2.716%) |

All72 trajectories exact;all prefill/request cases improve,total1.01790x.
Aggregate TG+0.212%/+0.012%/+0.434% incidental;private scratch36B
explicit,tracked peak unchanged. Not a new external comparator row.
[Production evidence](results/2026-09-06-framework-qwen4exp-q51-register-cache-production.json).

Earlier Q5_1 register-cache model admission:five full-logit/state/KV cases exact,
25/200 enabled prefill calls,zero decode/final owners. Both binders0 at
admission;subsequent canonical gate above passes.
[State evidence](results/2026-09-06-framework-qwen4exp-q51-register-cache-state.json).

Earlier Q5_1 K640 register reuse kernel screen: captured code/mixed512
two-bank projections34.450->28.195ms /33.756->27.819ms
(1.222x/1.213x),all40 pairs exact.17 GPU tests;private scratch36B
and96 VGPR explicit. Subsequent full-model gates passed.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-register-cache.json).

Q5_1 decoded-LDS cache rejected: actual captured-routing two-bank projection
34.449->58.987ms (+71.23% time),exact and both orders slower. Candidate
removed;production unchanged,no model A/B spent.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-weight-cache-rejected.json).

| Framework UD-Q4_K_XL / BF16 KV fresh combined-default screen | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine production | 168.58 / 19.64 | 166.86 / 19.01 | 155.89 / 14.89 |
| halo-box Vulkan target | 346.78 / 26.12 | 397.38 / 25.69 | 421.63 / 24.83 |
| halo-box HIP diagnostic | 307.38 / 21.45 | 393.31 / 20.95 | 355.00 / 19.18 |

Clean `5104604e1`,same pinned halo-box;all108 trajectories repeat and
teardown clean. Sequential screening,not statistical closure. Per-case
SD/CV/range/drift retained;max HE PP/TG CV2.78%/4.62%.
[Current baseline/variance evidence](results/2026-09-06-framework-qwen4exp-post-vec4-baselines.json).

Full post-MMQ-vector owner refresh at clean `1e89361d5`: p4096 four-category
non-GR linear5.363->4.770s (-11.07% snapshot delta),FFN13.442s,GR4.268s.
All12 phases have100% coverage,matched decode roots and zero final owners.
Vulkan profile explicitly reused;not a new throughput comparison.
[Family evidence](results/2026-09-06-framework-qwen4exp-post-vec4-family.json).

| Framework UD-Q4_K_XL / BF16 KV MMQ vector staging | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 164.712 -> 168.040 (+2.021%) | 162.186 -> 165.711 (+2.173%) | 150.246 -> 153.335 (+2.056%) |

All72 trajectories exact;every prefill/request case improves,total1.01205x.
TG-0.180%/-0.038%/-0.349% retained;no new memory. Existing prepacked
MMQ scope only;not a fresh Vulkan comparison.
[Production evidence](results/2026-09-06-framework-qwen4exp-mmq-vec4-production.json).

Earlier MMQ vector staging model admission passed five exact full-logit/state/KV
cases:72/576 prefill calls,zero decode/final owners. Both binders were0;
the subsequent canonical12-case gate above passes and promotes it.
[State evidence](results/2026-09-06-framework-qwen4exp-mmq-vec4-state.json).

Earlier MMQ vector activation staging kernel screen: actual QKV/SSM complete chains
at512 rows improve5.196->4.022ms (1.292x)/2.857->2.156ms (1.325x).
All80 pairs exact, both orders positive;25 tests pass. Same VGPR/LDS/scratch.
Subsequent model gates passed; promotion evidence is above.
[Evidence](results/2026-09-06-framework-qwen4exp-mmq-activation-vec4.json).

Q5_K bundle rejected after the complete12-case A/B: all72 trajectories exact,
but five prefill and six request-wall cases regress. Candidate removed;
retained production defaults unchanged.

| Framework UD-Q4_K_XL / BF16 KV rejected Q5_K bundle | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> rejected candidate | 165.556 -> 165.781 (+0.136%) | 163.883 -> 163.842 (-0.025%) | 151.007 -> 151.462 (+0.301%) |

Mixed case signs and drift preclude retention; this is not a new topline.
[Rejection evidence](results/2026-09-06-framework-qwen4exp-q5k-bundle-rejected.json).

Q4 row4/output4 activation-reuse retile rejected as a blanket replacement:
captured layer3/mixed0.992x versus layer0/code1.008x,exact. Candidate removed;
retained production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-row4-pair4-rejected.json).

Full six-case/twelve-phase owner refresh at `c075a6692` is complete.
Four-category p4096 FFN13.437s,linear5.363s,GR4.251s;all phase ownership/
decode-root/state/lifecycle gates pass. Earlier Vulkan is explicitly reused.
[Generated full-family evidence](results/2026-09-06-framework-qwen4exp-post-fold-pair-family.json).

| Framework UD-Q4_K_XL / BF16 KV folded Q5_1 pair | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 161.799 -> 164.954 (+1.950%) | 160.478 -> 162.953 (+1.543%) | 147.876 -> 150.680 (+1.896%) |

All72 trajectories exact;every prefill/request case improves. Aggregate
TG-0.359%/-0.318%/-0.140% retained;no new persistent allocations.
[Production evidence](results/2026-09-06-framework-qwen4exp-q51-fold-pair-production.json).

Earlier folded Q5_1 pair admission passed five exact full-logit/state/KV
cases:25/200 prefill calls,zero decode calls/leaks. Promotion evidence is above.
[State evidence](results/2026-09-06-framework-qwen4exp-q51-fold-pair-state.json).

Folded Q5_1 pair-reduction candidate: captured code/mixed512 two-bank screens
40.002->34.404ms (1.163x)/39.123->33.729ms (1.160x),exact.
Small-row order regression excludes that scope;model gates pending.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-fold-pair.json).

Post-Q8-bundle code4096 attribution confirms32 bundled calls/1.208s.
FFN14.104s,linear5.360s,GR4.249s remain dominant; code-only diagnostic
against reused Vulkan,not a new full-category or causal timing comparison.
[Owners](results/2026-09-06-framework-qwen4exp-post-q8-bundle-family.json).

| Framework UD-Q4_K_XL / BF16 KV Q8 bundled reduction | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 160.762 -> 162.095 (+0.829%) | 158.892 -> 159.958 (+0.671%) | 147.110 -> 148.240 (+0.768%) |

All72 trajectories exact; every case improves prefill/request wall. Aggregate
decode-0.041%/-0.011%/-0.085% retained explicitly; no new allocations.
[Production evidence](results/2026-09-06-framework-qwen4exp-q8-down-bundle-production.json).

Earlier Q8 bundle admission passed five exact full-logit/state/KV cases;
4/32 prefill calls,zero decode calls/leaks. Promotion evidence is above.
[State evidence](results/2026-09-06-framework-qwen4exp-q8-down-bundle-state.json).

Q8 down bundled-reduction kernel candidate:43.003->37.343ms (1.152x) on
layer4 actual weights/counts; second weight bank confirms1.153x with borrowed
routing. Exact; runtime defaults unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q8-down-bundle.json).

Post-fold128 code4096 attribution verifies200 candidate calls; FFN14.266s,
linear5.350s and GR4.237s remain dominant. Code-only, reused Vulkan,
diagnostic instruments; the full-category snapshot is separately labeled.
[Owners](results/2026-09-06-framework-qwen4exp-post-fold128-family.json).

| Framework UD-Q4_K_XL / BF16 KV Q5_1 fold128 promotion | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 159.222 -> 160.363 (+0.716%) | 156.767 -> 158.300 (+0.978%) | 145.546 -> 146.878 (+0.915%) |

All72 trajectories exact; all12 prefill/request walls improve. Tiny per-case
decode decreases preserved; no intrinsic decode gain or added allocation.
[Production evidence](results/2026-09-06-framework-qwen4exp-q51-fold128-production.json).

Earlier Q5_1 fold128 admission passed five full logits/state/KV cases:
25/200 calls in enabled prefill, zero decode calls/leaks; promotion is above.
[State evidence](results/2026-09-06-framework-qwen4exp-q51-fold128-state.json).

Q5_1 fold128 kernel candidate: actual two-bank down with captured code/mixed
routing41.501->40.236ms (1.031x)/40.624->39.219ms (1.036x), exact.
Runtime defaults unchanged; model admission pending.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-fold128.json).

Q5_1 simultaneous pair reduction rejected: exact but0.661x at512 tokens
on two actual down banks. Production remains unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-dual-reduce-rejected.json).

Post-Q8-row4 code4096 attribution confirms32 production calls. FFN14.679s,
linear5.370s, GR4.254s remain dominant; this is a code-only diagnostic refresh
against reused Vulkan, not new throughput or the full-category overview.
[Joined owners](results/2026-09-06-framework-qwen4exp-post-q8-down-family.json).

| Framework UD-Q4_K_XL / BF16 KV Q8-down row4 promotion | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Same-residency parent -> production | 158.216 -> 159.516 (+0.822%) | 155.729 -> 156.718 (+0.635%) | 144.452 -> 145.470 (+0.705%) |

All72 trajectories exact; every case improves prefill and request wall.
Small per-case decode decreases remain in the packet; no intrinsic decode
gain claimed. No additional sidecars, zero final allocations.
[Production evidence](results/2026-09-06-framework-qwen4exp-q8-down-row4-production.json).

Q8 down row4 default-off model gate: five cases pass full logits/state/KV
and four decode steps exactly; candidate runs only in prefill, final owners0.
Clean12-case throughput A/B remains pending.
[State evidence](results/2026-09-06-framework-qwen4exp-q8-down-row4-state.json).

| Framework Q8 down row4 kernel candidate | Parent -> candidate | Gate |
| --- | --- | --- |
| Layer4 weights / layer4 code4096 counts | 47.805 -> 43.175ms (1.107x) | Exact |
| Layer30 weights / borrowed layer4 mixed4096 counts | 47.254 -> 41.546ms (1.137x) | Exact; not layer30 routing |

Kernel candidate only; full-model gates and production promotion pending.
[Evidence](results/2026-09-06-framework-qwen4exp-q8-down-row4.json).

Q4 fixed K2560/N640 screen: exact but near-flat1.003x/1.004x on real
layer3/0 routing, with candidate-first order reversals. Removed, no promotion.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-shape-rejected.json).

| Framework current HIP graph census, UD-Q4_K_XL/BF16 KV | Captured graph activity | Diagnostic result |
| --- | --- | --- |
| code4096 prefill / live4097 decode | 0 prefill launches;48 decode graphs/625 nodes | Prefill337ms non-kernel residual; decode1.35-1.43ms intra-graph gaps |

No PM4 speedup measured; prefilling remains kernel-dominated. Decode token/state
repeats and teardown pass.
[Census](results/2026-09-06-framework-qwen4exp-wilkin-graph-census.json).

Q4 packed LDS cache rejected: actual layer3/captured routing0.646x, exact;
VGPR88->128 and LDS4608->12800 bytes. Production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-lds-cache-rejected.json).

Q4 packed-nibble cache also rejected:0.404x parent throughput; volatile
metadata variant0.136x. Both exact, both256 VGPR; latter spills1208 bytes.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-packed-cache-rejected.json).

Q4 full decoded-weight register cache rejected: real layer3/captured mixed4096
routing gives0.666x parent throughput, exact; VGPR88->168 with no spills.
[Evidence](results/2026-09-06-framework-qwen4exp-q4-weight-cache-rejected.json).

Opposite GR RB2/H4 column reuse also rejected:512-row attention1.006x,
FFN0.996x, exact but order-dependent. No runtime change.
[Evidence](results/2026-09-06-framework-qwen4exp-gr-col-reuse-rejected.json).

GR constant-accumulator row reuse rejected: real attention/FFN up at512
rows is0.847x/0.841x parent throughput, exact. Candidate removed.
[Evidence](results/2026-09-06-framework-qwen4exp-gr-row-reuse-rejected.json).

| Framework UD-Q4_K_XL / BF16 KV routing diagnostic | Coverage | Gate |
| --- | --- | --- |
| Real Q4/Q5_1 pair boundaries | code512 + four4096 categories;1683 boundaries | Three-arm logits/state exact; zero final allocations |

Q4 p4096 medians9-12 active rows;88.7-91.7% of rows are in experts with>8
rows. No new throughput claim; captured-count replay still uses synthetic
activations. [Evidence](results/2026-09-06-framework-qwen4exp-real-routing.json).

Qwen4Exp tool/grammar development checks: short15 scenarios93/100;
structured-output6 scenarios83/100. Model-choice failures remain; not a
full69-case qualification or throughput comparison.
[Evidence](results/2026-09-06-framework-qwen4exp-tools-grammar.json).

Native-capacity short-context check (2051->262144 allocated tokens):
p512/p1024 decode-0.066%/-0.095%, prefill-0.737%/-0.238%;48/48 trajectories
exact. Public serving now resolves native262144 on Framework through memory
admission; c2 completions/chat8K retrieval and over-limit rejection pass.
No new256K-length inference claim.
[Capacity A/B](results/2026-09-06-framework-qwen4exp-context-capacity-ab.json).
[Serving/boundary gates](results/2026-09-06-framework-qwen4exp-native-context-final.json).

| Framework UD-Q4_K_XL / BF16 KV incremental promotion | p512 PP | p1024 PP | p4096 PP |
| --- | ---: | ---: | ---: |
| Q8 MMQ prepack, same-residency parent -> production | 156.707 -> 157.748 (+0.664%) | 154.396 -> 155.375 (+0.634%) | 143.903 -> 144.736 (+0.579%) |

All72 trajectories exact; all12 prefill cases improve. Mixed512 request wall
loses0.14%;11 others improve. Extra memory1.67GiB, preparation0.107s.
No decode-kernel or new external parity claim.
[Production evidence](results/2026-09-06-framework-qwen4exp-mmq-prepack-production.json).

GR-specific MMQ64x128 rejected: real layer0/4 rows512 complete-chain speedups
0.722x/0.720x, exact. Production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-mmq64n128-rejected.json).

Direct Q5_1 metadata also rejected (0.910x/0.921x at64/512 tokens);
production unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-direct-meta-rejected.json).

Q5_1 pair2 wave-tail screen rejected (0.949x/0.974x at64/512 tokens);
production and retained rates unchanged.
[Evidence](results/2026-09-06-framework-qwen4exp-q51-wave-reduce-rejected.json).

| Incremental Framework UD-Q4_K_XL / BF16 KV result | p512 PP | p1024 PP | p4096 PP | Gate |
| --- | ---: | ---: | ---: | --- |
| GR wave-scale production, same-residency parent -> candidate | 156.017 -> 156.711 (+0.445%) | 153.748 -> 154.285 (+0.350%) | 143.419 -> 143.908 (+0.341%) | 72/72 exact; all12 request walls improve |

[GR production evidence](results/2026-09-06-framework-qwen4exp-gr-wave-production.json).
Not a new external baseline or intrinsic decode-speed claim.
The [post-MMQ family refresh](results/2026-09-06-framework-qwen4exp-post-mmq-family.json)
at clean `ef63870f9` passes all12 phase captures with100% coverage. P4096
mean FFN14.433s, non-GR linear5.367s and complete GR4.282s remain the main
prefill costs. Earlier Vulkan captures are explicitly reused; this is
diagnostic attribution, not an additional A/B speedup.

This file is the current benchmark scoreboard. It intentionally contains only
current user-facing results, compact protocol/status notes, and links to the
authoritative evidence. It is not an optimization journal.

## Root README performance summary

The root README exports this compact retained summary verbatim.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2852.100** | **115.804** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2763.590** | **94.603** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **875.364** | **28.681** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | 4,096 input tokens; prompt processing only | **440.893** | — |

#### Multiple requests

Each value is the total tokens per second across all active requests:

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (engine) | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (server) | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — MTP-3 | **60.929 tok/s** | **2.0684x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **122.67 tok/s** | **1.2679x** |

### RX 7900 XTX (`gfx1100`) — Qwen3.8-27B `Q4_K_M` prefill

| Workload | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **959.4** | 965.0 | -0.6% | 865.7 | +10.8% |
| 1K | **999.7** | 979.5 | +2.1% | 832.6 | +20.1% |
| 4K | **981.8** | 945.7 | +3.8% | 836.5 | +17.4% |

#### Dedicated-server context
| KV route | Server shape | Measured context | Peak / headroom |
| --- | --- | ---: | ---: |
| BF16 default | c1 operational | **32K** | 21.869 / 2.115 GiB |
| Pure INT8 explicit | c1 repeated natural soak | **112K** | 23.323 / 0.661 GiB |
| Pure INT8 explicit | c1 one-request physical ceiling | **126K** | 23.963 / 0.022 GiB |

#### Decode / MTP

| Metric | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR decode 512 | **34.06** | 32.86 | +3.6% | 13.39 | 2.54x |
| AR decode 1K | **34.91** | 32.75 | +6.6% | 13.38 | 2.61x |
| AR decode 4K | **31.79** | 32.41 | -1.9% | 13.31 | 2.39x |
| MTP natural | **62.44 B3** | 44.33 B2 | +40.9% | 73.33 B2 | -14.8% |

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (GEMV lib hoist) | sync'd eager, per-token | — | **38.9** |
| Qwen3.8-27B Dense GGUF `Q4_K_S` | 512 input tokens, 128 output tokens | **396.091** | **13.069** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |
#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.8-27B Dense GGUF `Q4_K_S` — MTP-3 | **23.853 tok/s** | **1.7845x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — public C1 MTP-3 automatic scope | **12.940 tok/s** | **1.4337x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary.
llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR; hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three draft tokens. The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
<!-- END TOPLINE:README_HIGHLIGHTS -->

## Current default notes

Strix Halo Qwen3.8 `Q4_K_M` automatic MTP is restricted to its verified
strict/BF16/C1/B3/raw-greedy key; other scopes use K0/AR.
[`Serving closure`](results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s5-closure.json).
Qwen3.8 `Q4_K_S` defaults to FP16 recurrent state with FP32 rollback; its exact
DMS sidecar remains default-off pending serving gates. [`DMS`](../docs/DMS.md).

Agentic quality is quality-only: Qwen3.8-27B `Q4_K_M` scores **50/68 (73.53%)**
with 64/64 valid calls; no runtime mechanism is retained.
[`Final`](results/2026-08-26-zbook-agentic-quality2-campaign-final.json).
Generation-2 automatic serving remains K0: gfx1151 P9 is exact 540/540 but c2/c4
are 0.6975x/0.5843x AR; gfx1100 exact speculative cells remain behind direct.
[`Closure`](results/2026-08-26-gfx1151-specdec2-perf-campaign-closure.json) ·
[`Recovery`](../docs/MTP-CONCURRENCY2-RECOVERY.md).

## Where detailed evidence lives

Use result artifacts for commands/samples/profilers,
[`CHANGELOG.md`](CHANGELOG.md) for rollups, [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)
for protocols, and [`worklog/entries/`](../worklog/entries/) for decisions.

## Benchmark harness catalog

Compare only matching harness scopes. ✓ marks a reported axis; blanks are
unmeasured. **AR/MTP/Prefill/Decode/Mem/Conc** mean true autoregressive,
speculative, prompt, generation, memory, and concurrency respectively. Use the
hermetic target-architecture wrapper; see `docs/BENCHMARK.md`.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen4exp_canonical_ar_bench.py` | Exact-token Qwen4Exp cross-engine p512/p1024/p4096 prefill plus context-conditioned tg128, output hashes, and comparison artifact | ✓ | | ✓ | ✓ | | | `hipengine --model-root <model>` or `llamacpp --server-bin <binary> --model <part1>` |
| `qwen4exp_profile_gap.py` | Exact-fixture prefill ROCTX roles, launch/copy/allocation census, lifecycle, and separate selected-expert telemetry | | | ✓ | | ✓ | | `--mode prefill --case-id code-p512 --profile --role-markers` |
| `qwen4exp_context_decode_profile.py` | Restored exact live-context transition roles, complete mutable-state hashes, per-bucket lifecycle, and allocation census | ✓ | | | ✓ | ✓ | | `--live-count 513 1025 4097 --repetitions 3 --profile --role-markers` |
| `qwen4exp_llamacpp_exact_profile.py` | Exact-token pinned llama.cpp prefill and cached single-transition decode under direct rocprof, selected by monotonic bounds | ✓ | | ✓ | ✓ | | | `--case-id code-p512 --case-id code-p1024 --case-id code-p4096` |
| `qwen4exp_mtp_head_profile.py` | Isolated Qwen4Exp MTP full-Q8 draft/head/D2H timing and selected-head Amdahl ceiling | | ✓ | | ✓ | ✓ | | `--output <json>` |
| `qwen35_gguf_bench.py` | GGUF c=1 AR prefill/decode, fresh resident session per run, HIP-graph decode | ✓ | | ✓ | ✓ | ✓ | | `--model <model> --prompt-length 512 --decode-tokens 128` |
| `gguf_true_ar_category_bench.py` | True no-MTP AR baseline over the mtp-bench category suite (the legitimate MTP speed denominator) | ✓ | | ✓ | ✓ | | | `--model <model> --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| `gguf_mtp_category_bench.py` | MTP category matrix over budgets 1..8 with guarded objective extraction; attach a true-AR baseline for ratios | | ✓ | | ✓ | | | `--budgets 1,3,5 --objective-budget b5` |
| `gguf_mtp_long_context_gate.py` | Eager-native MTP correctness vs serial-exact teacher across context/page/budget/acceptance boundaries; optional real host-proposal AR-ID gate (no speed claim) | ✓ | ✓ | | | | | `--cycle-ends 1016-1032,4K --candidate-budgets 1,2,3 --fail-on-fail` |
| `gguf_ar_mtp_suite.py` | One-command AR-vs-MTP decode ratio over the category suite under one enforced decode config | ✓ | ✓ | | ✓ | | | `--scope partial --output <json>` |
| `specdec2_perf_bridge.py` | Current-source Generation-2 true AR vs staged SPECDEC2 plus C1 direct control; complete/decode timing, ownership stages, physical C/K, exact IDs, and ROCTX leaf mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | `--backend hip_gfx1151 --concurrency 1 --budgets 1,2,3 ...` then separate `--concurrency 2,4 --budgets 2 ...` |
| `qwen35_batch_retained_bench.py` | **PARO-path** compact c>N batch decode; aggregate + per-request tok/s, equality vs c1, optional MTP draft depth | ✓ | ✓ | | ✓ | ✓ | ✓ | `--batch-size 8 --decode-tokens 128` |
| `qwen35_batch_gguf_diagnostic.py` | GGUF c>N generated-token **correctness** equality vs independent c1 (no throughput claim) | ✓ | | | | | ✓ | `--rows 8 --execute` |
| `server_f1_concurrency_bench.py` | Matched gfx1151 F1 HTTP concurrency through c32; profile-aware throughput, SLOs, routes, control, and memory | ✓ | | | ✓ | ✓ | ✓ | `--engine hipengine --model <model> --concurrencies 1,2,4,8,17,32` |
| `gguf_concurrency_baseline.py` | GGUF c1 + explicit serial c2/c4 timing controls (Phase-A route baseline) | ✓ | | ✓ | ✓ | | ✓ | `--model <model> --concurrencies 1,2,4` |
| `mtp-bench.py` | llama.cpp-compatible MTP prompt-suite benchmark (server economics); can wrap hipEngine verifier economics | ✓ | ✓ | | ✓ | | | `--mode hipengine-current` |
| `exact_token_generation.py` | Direct/HTTP generated-token identity gate (correctness, not throughput) | ✓ | ✓ | | | | | `direct --model-path ...` then `http --oracle ...` |
| `benchmark_matrix.py` | Join exact-token direct/server rows into a validated matrix report | ✓ | ✓ | | | | | `build --manifest ...` |

Keep this catalog synchronized whenever a harness gains a measured axis.

## Evidence status

| Status | Meaning | Eligible for a current numeric table? |
| --- | --- | --- |
| **Retained** | Correctness, provenance, repetition, and protocol gates passed for the named scope. | Yes. |
| **Current snapshot** | Clean current-production measurement used to describe the shipped route, but not itself a new optimization claim. | Yes, with that label. |
| **Diagnostic** | Useful attribution or comparison with a known limitation. | No; keep it in its artifact/changelog unless it explains a current blocker. |
| **Stale / superseded** | A newer route, dependency, or evidence contract replaced it. | No. |
| **Blocked / rejected** | The protocol could not complete or the candidate failed a gate. | No numeric topline. |

A row is scoped by platform, model/quant/KV, workload, concurrency, policy, and
timing window. A newer diagnostic never replaces a retained row.

## Current Generation-2 qualification

W7900 Qwen3.6-35B-A3B `UD-Q4_K_M`, BF16 KV, p128/d8, token-budget
scheduling, and same-loaded-server c1 oracles:

| Logical concurrency | 1 | 4 | 8 | 17 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggregate HTTP tok/s | **27.443** | **43.337** | **46.158** | **45.797** | **44.320** |
| Exact rows | 1/1 | 4/4 | 8/8 | 17/17 | 32/32 |

The canonical W7900 packet retains physical c1/c2/c4/c8 and logical c1-c32:
all nine fixed/ragged/load/cancel/overload/recovery/soak workloads pass 210/210
correctness-accounted rows, bounded overload, complete admission/reclaim, and
zero final ownership or tracked-memory delta. Exact Qwen3.8 physical c1-c8 and
its planar-Q6 row8 kernel are also retained; detailed rows remain in the
benchmark changelog and result artifacts.

On Radeon 8060S/gfx1151, the final Qwen3.8 `Q4_K_S` package retains queue2,
exact physical c1-c8/logical c1-c32 mechanics, packed prefill, direct resident
state, Q4 row8 two-wave, and scoped Q5 col8. The 130-row width, 2,100-request
load, context/graph/prefix/pressure, and lifecycle packets pass. Product closure
remains blocked at c32: **10.590 tok/s**, **18.617 s TTFT p95**, **2.125 s ITL
p99**, **24.171 s E2E p95**, and **0/3 SLO runs**; C2 64K and heavy-load SLOs
also remain blocked. [`gfx1151 campaign final`](results/2026-08-24-gfx1151-qwen38-concurrency2-campaign-final.json).

## Qwen3.8-Flash-Next implementation-first status

Fresh combined-default Framework `gfx1151` baseline, UD-Q4_K_XL/BF16 KV,
four categories, p512/p1024/p4096 + tg128:

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine | 153.96 / 19.38 | 152.30 / 18.75 | 142.03 / 14.42 |
| halo-box Vulkan | 316.28 / 25.27 | 391.68 / 25.30 | 425.72 / 24.51 |
| halo-box HIP | 282.76 / 21.08 | 368.33 / 20.56 | 351.08 / 18.83 |

Weighted tok/s,36 samples per engine, all outputs repeat, clean teardown.
All lanes exceed2% per-case CV somewhere; these are sequential screening
comparisons, not statistical parity.
[Frozen baseline evidence](results/2026-09-05-framework-qwen4exp-refreshed-baselines.json).

The completed same-host family alignment covers six prompts and both phases.
It identifies FFN/linear/GR as the largest prefill gaps and QSA as the largest
decode gap; HIP kernel sums and Vulkan query intervals remain diagnostic.
[Generated family evidence](results/2026-09-05-framework-qwen4exp-family-alignment.json).
The post-Q8 current-HE refresh keeps that ranking, with linear5.551s,
GR4.362s and FFN14.445s at p4096; the Vulkan profile is explicitly reused,
not remeasured. [Updated family packet](results/2026-09-05-framework-qwen4exp-post-q8-family.json).

Standalone GR up+sigmoid+mean screen, actual weights, rows512, no memory
preconditioning:

| Weight | Parent (ms) | Candidate (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Layer0 attention up | 3.338 | 3.234 | 1.032x |
| Layer0 FFN up | 3.331 | 3.207 | 1.039x |
| Layer4 attention up | 3.320 | 3.204 | 1.036x |
| Layer4 FFN up | 3.313 | 3.209 | 1.032x |

F32 gate/mixed bits exact,20 tests pass, both order strata positive at512;
small-row order reversals are disclosed. Model admission pending.
[GR screen](results/2026-09-06-framework-qwen4exp-gr-wave-scale.json).

Latest retained Q8 wave-scale production A/B, normal model execution:

| Prompt | Parent prefill | Wave-scale prefill | Gain |
| --- | ---: | ---: | ---: |
| 512 | 154.29 | 155.45 | +0.75% |
| 1024 | 151.67 | 153.43 | +1.17% |
| 4096 | 141.58 | 143.41 | +1.29% |

Tok/s on Framework, UD-Q4_K_XL/BF16 KV. All72 trajectories exact; all12 cases
improve prefill and complete-request wall, with zero final allocations.
Decode drift remains: no intrinsic decode improvement or new external
parity claim. Full A/B33m53s, without synthetic memory preconditioning.
[Q8 production](results/2026-09-05-framework-qwen4exp-q8-wave-scale-production.json).
The earlier microbench's order reversal remains recorded in its
[separate evidence](results/2026-09-05-framework-qwen4exp-q8-wave-scale.json).

The larger Q4 row-batch screen is rejected: exact RB16/32 variants lose
against RB8 in all12 final actual-weight, synthetic-routing cells across
tokens512/1024/2048. Production is unchanged.
[Screen evidence](results/2026-09-05-framework-qwen4exp-q4-rowbatch-rejected.json).

The preceding Q4 output-pair retention remains part of the baseline:

| Shape | Parent prefill | Q4 pair prefill | Gain | Decode before -> after |
| --- | ---: | ---: | ---: | ---: |
| p512 | 145.21 | **153.78** | **+5.90%** | 19.385 -> 19.383 |
| p1024 | 142.82 | **151.21** | **+5.87%** | 18.556 -> 18.512 |
| p4096 | 134.01 | **141.30** | **+5.44%** | 13.113 -> 12.899 |

All rates are tok/s on Framework `gfx1151`, UD-Q4_K_XL/BF16 KV, four
categories and tg128. This is a same-residency incremental A/B with all72
trajectories exact, all12 prefill cases positive, and complete-request wall
speedups of1.67-3.90% per case. Decode losses of0.23%/1.63% at p1024/p4096
are retained under the owner's prefill-first direction and remain open work.
Both arms drift; this is not a stable absolute decode comparison or external
parity claim. Full A/B elapsed34m53s; teardown is zero.
[Full timing and promotion evidence](results/2026-09-05-framework-qwen4exp-q4-pair-production.json).

The preceding serial-GDN admission improved prefill9.50-10.55% and remains
part of this baseline. Its separate decode tradeoff is preserved in its
[GDN evidence](results/2026-09-05-framework-qwen4exp-gdn-register-production.json).

The preceding Q5_1 pair2 admission improved prefill3.73-4.17% and remains
part of this measured baseline.
[Pair2 evidence](results/2026-09-05-framework-qwen4exp-q51-pair-production.json).

The prior promoted combination remeasured at121.14 pp/s on the code-p4096
diagnostic (not an all-category refresh). The original standalone Q5_1 output-
pair candidate reduces two actual down banks at tokens512 from about54.5
to43.5 ms (1.253x), exact outputs; the full model result is above.
[Combined profile and Q5_1 screen](results/2026-09-05-framework-qwen4exp-q51-pair.json).

Production retains exact page256 QSA and bundled-Q4 prefill under the
2026-09-05 owner decision to take the prefill gains and optimize decode next.
Strict keeps the prior owners. Separate component A/B measurements are:

| Component | Shape | Prefill before -> after (tok/s) | Gain | Decode change |
| --- | --- | ---: | ---: | ---: |
| Bundled Q4 | p512 | 123.34 -> 127.62 | +3.47% | -0.23% |
| Bundled Q4 | p1024 | 121.47 -> 125.33 | +3.18% | -0.06% |
| Bundled Q4 | p4096 | 97.34 -> 99.72 | +2.45% | -0.64% |
| H256 wave QSA | p4096 | 97.83 -> 116.70 | +19.29% | -0.48% |

These are separate same-residency component runs, not additive gains or
combined throughput. The measured hot-decode tradeoffs remain open work.
[Promotion and evidence](results/2026-09-05-framework-qwen4exp-prefill-promotion.json).

The original standalone exact Q4 gate/up bundled publication screen reduces the actual-weight
gate/up+SiLU boundary at tokens512 from30.39 to24.97 ms (1.217x), all pairs
exact. Timing variability and whole-model admission remain open; this is
not a production gain.
[Q4 bundle](results/2026-09-05-framework-qwen4exp-q4-bundle.json).

The current same-host comparator screen ran on the Framework Desktop (physical
host `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`, Ryzen AI Max+
395 / Radeon 8060S) with the verified four-part Unsloth `UD-Q4_K_XL` artifact,
BF16 K/V, one warmup, and three measured requests per canonical case:

| Engine | p512 pp/tg128 | p1024 pp/tg128 | p4096 pp/tg128 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production + exact Q5_K row4, HIP (later same-residency A/B) | **122.57 / 19.97** | **121.47 / 19.25** | **97.85 / 15.29** | **12/12**, cross-arm exact |
| hipEngine production `c0cfdc3ef`, HIP | 118.44 / 19.92 | 117.79 / 19.22 | 95.14 / 15.21 | **12/12** |
| Upstream llama.cpp `4d9176092`, HIP | 283.85 / 21.06 | 367.97 / 20.79 | 395.02 / 19.63 | **11/12** |
| Upstream llama.cpp `4d9176092`, Vulkan | 230.35 / 24.94 | 305.47 / 24.59 | 357.44 / 23.53 | **11/12** |
| halo-box master `b212548e0`, HIP | 265.69 / 21.02 | 368.90 / 20.50 | 356.62 / 18.63 | **12/12** |
| halo-box master `b212548e0`, Vulkan | **298.97 / 24.92** | **369.72 / 24.52** | **402.46 / 23.47** | **12/12** |

The two upstream lanes vary on `mixed_ja_en-p4096`. Every external lane also
exceeds 2% maximum per-case coefficient of variation on at least one metric,
so this is a screening result, not a frozen closure target. Do not compare
these rates as old-to-new deltas against `zbook`. Exact commands, binary and
model hashes, per-sample rates, and output hashes are in the
[Framework comparator packet](results/2026-09-05-framework-gfx1151-qwen38-flash-next-current-comparators.json).

The row4 row is a later same-host internal A/B, not a paired rerun against
the external lanes. Its own parent rates are 118.92/117.72/95.42 pp/s:
prefill improves 3.07%/3.19%/2.55%, all 72 measured trajectories are exact,
max per-case prefill CV is 0.23%, and teardown is clean.
[Production evidence](results/2026-09-05-framework-qwen4exp-row4-production.json).

The original H256 sparse-attention standalone screen at
24Q/2KV/D256 and selected stride2051, rows512 attention measures
171.93→28.02 ms (6.14x), with exact parent output bits. This leaf ratio is
not a whole-model gain.
[QSA candidate](results/2026-09-05-framework-qwen4exp-qsa-h256-wave.json).

Its full-suite internal A/B measures p4096 prefill 97.83→116.70 tok/s
(+19.29%), but decode 15.27→15.20 (-0.48%). All 72 trajectories are exact;
the candidate initially stayed default-off pending focused decode followup. An English
128-step probe also matches full logits and complete K/V, but is not a
replacement for the original timing protocol.
[Full-suite audit](results/2026-09-05-framework-qwen4exp-qsa-fullsuite-audit.json).

The isolated English rerun has flat decode, but its subset initially reversed
the original arm order. That diagnostic does not waive the full-suite finding;
the harness now preserves original fixture indices for focused comparisons.
[Subset-order audit](results/2026-09-05-framework-qwen4exp-qsa-subset-order-audit.json).

The corrected-order three-case rerun still finds mixed-language decode
down 0.64%; English/Japanese are effectively flat and all 18 trajectories
are exact. A 2880 MHz clock snapshot during the affected section is a
lead, not a causal diagnosis. The later owner decision accepts the measured tradeoff.
[Corrected followup](results/2026-09-05-framework-qwen4exp-qsa-corrected-followup.json).

The standalone page256-addressing sibling reduces rows512 attention from
28.13→27.56 ms versus the generic wave candidate, with both arms exact.
This does not establish a power benefit or clear the whole-model decode gate.
[Page256 screen](results/2026-09-05-framework-qwen4exp-qsa-page256.json).

External phase telemetry reproduces page256's late mixed decode loss and
records lower after-arm clocks (about 2880–2882 versus 2898 MHz). It is
correlation, not yet a causal proof or promotion.
[Phase clocks](results/2026-09-05-framework-qwen4exp-qsa-phase-clocks.json).

A separate mixed p4096 control holds every sampled phase at2700 MHz:
prefill93.01→112.25 tok/s (+20.69%), decode14.800→14.798 (−0.017%), exact
outputs. This supports an operating-point effect but is not promotion at
the original setting; 2900/2900 high policy was restored.
[Fixed-clock control](results/2026-09-05-framework-qwen4exp-qsa-fixed-clock-control.json).

The subsequent Framework code-case owner diagnostic at `cf9c55920` has 100%
role coverage and traced/unprofiled final-logit equality:

| Diagnostic | code-p512 | code-p4096 |
| --- | ---: | ---: |
| Unprofiled prefill wall (s) | 4.306 | 43.046 |
| Routed MoE device time (s) | 2.402 | 19.024 |
| QSA device time (s) | 0.042 | 8.983 |
| GDN device time (s) | 0.489 | 3.906 |

No optimization or new paired comparator verdict is claimed.
[Owner refresh](results/2026-09-05-framework-gfx1151-qwen38-flash-next-owner-refresh.json).

Framework standalone Q5_K grouped-row4 gate/up, including its device map,
screens at 30.022→24.391 ms (64 rows) and 239.491→107.911 ms (512 rows),
with bit-exact paired outputs. This is a development-tree primitive result,
not a runtime promotion or whole-model gain.
[Candidate evidence](results/2026-09-05-framework-qwen4exp-q5k-grouped-row4.json).

Earlier implementation-calibration evidence was collected on physical host
`zbook` (Ryzen AI Max+ Pro 395 / Radeon 8060S, `gfx1151`). The pinned artifact
runs through public `LLM.generate()` under the strict c1/greedy text scope.
Frozen same-artifact llama.cpp PR #27742 full logits over all 10 canonical
code/general-English/general-Japanese/mixed prompts measured:

| Artifact | Context scope | Mean / p95 / p99 / max KL ↓ | Top-1 | Tracked peak / after close |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | real ≤2,051-token canonical text gate | **0.01406 / 0.04154 / 0.04776 / 0.04931** | **10/10** | 82.718 GB / **0 B** |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | predeclared eight category heldouts, matched BF16 K/V | **0.00987 / 0.02331 / 0.02766 / 0.02874** | **8/8** | same residency / **0 B** |

The pinned 111.335-GB/four-hash artifact owns one 28.800-GB sparse-mmap PLE
table and 82.523 GB hot weights. Exact batching passes 687/687 rows; the strict
prefill default chunk is now 512 (PLE staging capacity plumbed to the chunk;
previously silently capped at 256): same-session counterbalanced sweeps give
p508 **8.458→8.270 s (-2.22%)** and p1012 **17.062→16.751 s (-1.82%)** with
identical logits SHAs, and natural 16K improves to **341.177 s / 47.989 tok/s**
with the full gate passing (prior chunk-256 steady rows were p508 58.466 and
p1006 55.046 tok/s).

The earlier `zbook` exact-token screen fed all engines the same four category
prompts at p512/p1024/p4096 and measured 128 decode transitions after each
prefix:

| Engine | p512 pp/tg128 | p1024 pp/tg128 | p4096 pp/tg128 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine current production (PF-5 GDN tile-16 after arm) | **89.87 / 14.81** | **88.97 / 14.77** | **72.93 / 12.39** | 12/12 deterministic; one-residency cross-mode exact; 12/12 prefill-positive |
| hipEngine current production (PF-1/PF-3 after arm) | **89.34 / 14.84** | **88.54 / 14.79** | **72.58 / 12.40** | 12/12 deterministic; one-residency cross-mode exact |
| hipEngine pre-PF baseline (`37d59564…`, HB-1 retained arm) | 83.37 / 14.32 | 82.91 / 14.27 | 69.20 / 12.18 | 12/12 deterministic; cross-arm exact; historical |
| Upstream Vulkan `f1793c1c4`, queue/repack/fit-off | 200.01 / 24.39 | 241.84 / 21.33 | 266.58 / 18.98 | 12/12 exact; noisy p512/p1024 rows |
| Patched-upstream HIP `f1793c1c4` | 235.89 / 17.75 | 306.51 / 16.99 | 283.73 / 14.89 | 12/12 deterministic; cross-arm exact; non-stock loader |
| Halo-box base `6c84c7d5` + loader patches | 223.89 / 17.66 | 308.28 / 16.88 | 301.62 / 14.90 | 12/12 deterministic; cross-arm exact; short-shape drift |
| Halo-box PR11 `a7ad7b7f` + loader patches, fresh matched-BF16 screen | **240.11 / 18.04** | **324.64 / 17.24** | **349.49 / 15.10** | 12/12 deterministic; p512/p1024 prefill unstable; p4096 stable |
| EngramHalo HIP `1423f689` | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 exact; p4096 fails |
| Nathan Vulkan `ad914eb`, queue/repack/fit-off | 360.23 / 24.34 | 357.61 / 21.10 | 351.85 / 19.01 | diagnostic: 0/12 exact |
| apepojken Vulkan `843d575` | 291.73 / 23.21 | 375.23 / 22.42 | 397.43 / 22.25 | diagnostic: 8/12 exact |

Nathan produced 16 different outputs from 16 identical-prompt requests;
apepojken varies on four canonical cases; EngramHalo varies on one p4096 case.
Their affected rates remain diagnostics rather than correctness-valid targets.
Pristine upstream HIP did not finish loading in two 1,800-second attempts, so the
measured patched-upstream lane is explicitly non-stock. The fresh halo-box
screen exactly matches the HB-1 BF16 configuration and binary; its maximum
per-case prefill CV is **10.0%/9.7%/1.26%** at p512/p1024/p4096, so only the
p4096 row satisfies the ≤2% stability rule. The previous retained halo-box arm
was 246.55/343.48/354.21 pp/s; the fresh −2.61%/−5.49%/−1.33% shift confirms
that short-shape absolute comparisons need counterbalanced thermal pairs.

This remains a screening refresh, not section-6 closure: five same-thermal
competitor pairs and 4K MTP remain open. Current production's maximum per-case
CV in the one-residency packet is **1.64% prefill / 1.07% decode**. The Vulkan
rows were refreshed with their entitled graphics queue, repack, and fit-off
configuration; upstream p512 prefill/decode and p1024 prefill also remain too
noisy to freeze the closure target.
[`PF-1/PF-3 production refresh`](results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json),
[`halo-box HB-1 comparison`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb1.json),
[`current P12 packet`](results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-packet.json),
[`generated report`](results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-report.md),
[`canonical AR screening`](results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json),
[`entitled Vulkan refresh`](results/2026-09-02-gfx1151-qwen38-flash-next-entitled-vulkan-canonical-refresh.json).

Frozen halo-box HB-base/PR11 exact profiles confirm that the PR activates its
Q4/Q8 MMQ retunes, prompt top-10 compaction, weighted top-10 sum, 32-warp GDN,
and selected elementwise/recurrent specializations on the binding Q4 payload.
Routed-compact/J48/J64, shared-mul-add, and Q8-KV attention paths are inactive.
The largest isolated trace change is p4096 GDN core kernel sum
**2,139.377→647.976 ms**; it is diagnostic pending HB-3 operation-complete
matched pairs. [`halo-box HB-2 census`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb2.json).
HB-3 stops before timing: pinned operation harnesses pass their exposed cases,
but **0/5 active families** currently share an identical cross-engine fixture,
dtype/layout contract, and operation boundary. No mechanism ratio or candidate is
reported. [`halo-box HB-3 blocker`](results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json).

The 2026-09-04 remediation resolves the PF-1/PF-3 review blocker. One committed
harness kept a single generator resident and toggled both exact routes in ABBA
orders reversed across adjacent cases. Weighted prefill improves
**86.62→89.34 (+3.13%)**, **85.88→88.54 (+3.09%)**, and
**70.80→72.58 tok/s (+2.51%)** at p512/p1024/p4096; all 12 cases improve and
all 72 measured trajectories are exact across modes. Decode changes
−0.15%/+0.13%/−0.07%. Production again selects PF-3 Q5_1 M1 and PF-1 grouped
Q8_0 down; strict retains the preceding registered owners. PF-4's fused-combine
whole-model rejection remains provisional, and the Q4_K M1/PF-5 w32 kernel
losses remain valid.
[`Production refresh`](results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json),
[`Review plan`](../worklog/entries/20260904T100046.831998Z-lhl-qwen4exp-halo-box-campaign-review-511155.md).

The 2026-09-05 PF-5 closure promotes the exact GDN token-tile-16 prefill owner
(binding Hk=16/Hv=48/D=128) as the production default inside the colwarps gate
after a fail-closed, engagement-verified one-residency A/B: weighted prefill
**89.435→89.873 (+0.49%)**, **88.553→88.966 (+0.47%)**, and
**72.661→72.929 tok/s (+0.37%)** at p512/p1024/p4096 with all 12 cases
non-negative, 72/72 cross-mode exact outputs, and per-case prefill CV ≤1.6%.
The binding-shape leaf wins 23.3%/29.1%/35.3% at rows 16/64/512, bit-exact in
outputs and final FP32 state. The columnwarp parent stays registered for
non-envelope shapes and the `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0` opt-out;
serial strict remains the registered fallback. The first same-day A/B was
invalidated as a no-op (the runner bypassed the replaced registry key, so both
arms ran the parent) and is superseded. Scaling the loop's 0.3037 screening
baseline by the measured code-only geomean infers ~0.3045; the frozen-halo
screening metric itself is refreshed at the next stable-clock closure verify.
[`PF-5 tile-16 promotion A/B`](results/2026-09-05-gfx1151-qwen38-flash-next-pf5-gdn-tiled16-whole-model-ab.json),
[`Hv48 correction`](../worklog/entries/20260905T022638.146026Z-lhl-qwen4exp-pf5-gdn-tiled16-hv48-correction-c178a9.md).

The frozen p508 role/API profile still puts hipEngine versus llama HIP device
kernels at **5.959 vs 1.625 s (3.67×)** and decode at **48.63 vs 38.90
ms/output (1.25×)**. The main p508 owners are MoE **3.161 s** (layers 0–26:
**2.526 s**), GDN **634.94 ms**, and QSA **110.49 ms**. The largest single miss
is layer-2 Q5_K gate/up at **301.47 vs 15.38 ms**. Decode submits **1,195
direct kernels plus 48 MoE graphs/token**; 625 additional rows/token are
graph-expanded nodes. A strict, layer-local stateful graph diagnostic now
captures one complete 34-kernel GDN+MoE physical layer: output and all request
state owners remain exact through four replays, while synchronized layer wall
falls **4.051→1.258 ms (3.22x)**. A chained layers-0..2 rung is likewise exact
and contracts **9.801→3.896 ms (2.52x)**. A fixed-position layers-0..3 mixed
GDN/QSA diagnostic remains device-state/output exact at **12.160→4.955 ms
(2.45x)**. Its advancing-position successor passes positions 8–11 across
position/context, K/V, QSA index, GDN state, and output at **13.882→4.974 ms
(2.79x)**. An eight-layer successor adds active PLE and a second QSA owner and
remains exact at **26.739→10.112 ms (2.64x)**. The all-physical-layer rung
covers active PLE, all 48 layers, all 12 QSA owners, and 136 device-state owners
at **154.346→57.900 ms (2.67x)** without reproducing third-replay corruption.
The complete host-staged transition then adds generated-token PLE publication,
embedding, final full-vocabulary head, device argmax, and token feedback: the
changing-token trajectory and 138 owners are exact; reset→replay and forced-
eager→graph resumption also pass. The probe-local 194.758-ms eager arm disables
the shipped per-layer MoE cache and adds a script loop, so its 3.15x ratio is
retired. The follow-up strict runner measures **68.855 ms** with shipped MoE
graphs and **143.989 ms** without them, but imports the **61.910-ms** graph row
from the separate probe process. Therefore the derived **1.112x/6.945 ms is not
a named-production A/B** and P8 remains admission-pending. Device argmax versus
host full-logit D2H differs by at most 0.35 ms with a sign flip; steady
allocation growth and teardown pass.
[`named denominator`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json),
[`stateful layer graph`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-stateful-layer-graph.json),
[`three-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-gdn-segment3-graph.json),
[`mixed fixed-position segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-mixed-segment4-graph.json),
[`advancing mixed segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-mixed-segment4-graph.json),
[`eight-layer segment`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-segment8-graph.json),
[`all 48 physical layers`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-all48-graph.json),
[`full host-staged transition`](results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json).

The current exact-token impact packet fully attributes p512/p1024/p4096 and
live-513/1025/4097. hipEngine versus patched llama HIP device sums are
**5.972/11.196/54.762 s vs 1.926/2.990/10.838 s** prefill and
**51.062/53.262/87.732 ms vs 41.009/41.110/44.005 ms** decode. At p4096,
QSA alone is **10.229 s vs 0.526 s** prefill and **35.587 vs 0.591 ms** decode.
Every hipEngine role window is 100% attributed, both family ledgers have zero
generic remainder, exact decode state/lifecycle gates pass, and median active
experts are 333/327/325 of 512. The old 41.6% remainder was incomplete table
coverage. The patched llama CSVs flushed and hashed, but rocprof required forced
exit after flush, so comparator subwindows remain diagnostic attribution.
[`canonical impact profile`](results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json).

The first retained impact-ranked unit replaces the serialized H256 p4096 QSA
attention owner with an exact three-pass path: parallel incumbent-order QK
scores, one global selected-order online-softmax coefficient recurrence, and
independent output-column weighted-V recurrences. Across four canonical p4096
categories and 12 counterbalanced tg128 pairs, complete decode improves
**93.912→80.061 ms/token (1.173x)**; every pair wins, the aggregate 95% ratio
interval is **1.170–1.176**, and full logits/IDs remain exact. The named trace
records all three expected kernels, reduces the QSA role **36.304→20.913
ms/token**, attributes 100% of device time, allocates nothing in the measured
windows, and tears down to zero. This retained row does not replace the
five-pair section-6 closure baseline.
[`exact ordered QSA decode`](results/2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json).

A durable isolated-route recheck reopens the layer-2 grouped-WMMA candidate:
the p508 trace cuts layer-2 MoE **371.10→88.13 ms (4.21×)** and Q5_K gate/up
**279.86→16.66 ms**. Same-process p508 improves **90.25→95.06 tok/s
(+5.34%)**; all 20 category-balanced p512 pairs improve, with per-category
means **+4.83% to +5.20%** and every five-pair 95% CI above 1.0. Each route is
repeat-exact and keeps the same final top-1 token, but full logits differ. The
complete 450-row gate then **rejects** the T2 candidate: overall mean/p95/max KL
`5.03e-4/2.65e-3/0.01238` and 446/450 top-1 pass, as do every category,
repeat/state, and lifecycle checks, but the binding prefill-last/prefill-to-c1
mean KL is **0.001179 > 0.001**. The route remains default-off; c2 and depth
promotion gates were not run because they cannot compensate for this failure.
[`P1 layer-2 rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json).

The fresh P2 split keeps current production default-off for that candidate and
profiles layers 0–26 at **2.366 s**: exact Q4/Q5_K gate/up **1.200 s**, exact
Q5_1/Q8 down **1.152 s**, and activation plus routing/shared tails only
**13.25 ms**. Layers 3–26 alone retain **1.849 s**; active experts span 166–298
with median 9 rows per active expert. The next exact/T1 work therefore targets
multi-row weight reuse/output tiling in both projection halves, not the <0.6%
tail. Telemetry was collected separately and its D2H wall is excluded.
[`P2 early-MoE profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p2-early-moe-profile.json).

The P3 split names another **1.670 s** of primary p508 roles outside routed MoE:
GR projection/read **709.32 ms**, Q8 `attn_qkv+attn_gate` **532.36 ms**, router
**181.91 ms**, `ssm_out` **137.84 ms**, and shared projections **121.61 ms**.
The first operation-complete target is the 36-layer qkv+gate boundary; it must
preserve current qkv-MMQ and exact-gate arithmetic or qualify a declared T1
pair, with both singleton routes retained as fallbacks. The first extension—Q8
MMQ on the omitted K2560/N6144 gate—wins **1.0352x** p508 and passes all
numerical scopes, but is rejected because candidate state repeat 1 differs from
repeats 2–3 on the first prompt. Ignoring the first same-schedule run as warmup
is not a valid production rule; exact coltile remains default. The next P3
subunit fuses GR sigmoid materialization with gated mean for rows <=256. It
removes one launch per GR read and improves clean counterbalanced
p508+128-step decode **14.162→15.111 tok/s (1.0670x, 95% CI
1.0543–1.0797)**. The complete T0 gate
is exact: **450/450 logits, 18/18 state/task prompts, three repeats, and clean
teardown**. Rows >256 remain unfused after a rows508 primitive loss. The
multirow F32 router projection also reuses each weight row across four prompt
rows while preserving dense arithmetic: clean p508 improves **89.689→91.121
tok/s (1.0160x, 95% CI 1.0143–1.0177)**, with 450/450 logits and 18/18 state/task
prompts exact. c1 remains on the dense owner. The rows>256 GR-up composite also
preserves the exact Q8 reduction while emitting sigmoid gates and branch mean:
clean p508 improves **91.158→91.600 tok/s (1.00484x)** and code-p1024
**88.754→89.239 tok/s (1.00547x)**, with 450/450 logits and 18/18 state/task
prompts exact. P4 also promotes the exact fixed256/precomputed-offset/vector2
QSA dense owner: the real primitive improves **6.846→2.485 ms (2.755x)**,
clean p508 **91.529→92.442 tok/s**, and code-p1024 **89.150→90.634 tok/s**, with
the complete exact/state/task gate passing. P5 moves normal greedy top-1 to the
device: Python-visible D2H falls from **993,280 to 8 bytes/token (124,160x)**
with 450/450 logits, 18/18 generated task sequences, compact state, physical-c2
outputs, and lifecycle exact. Resident-token chaining and normal-AR hidden-copy
elision then reduce the ledger from **28 to 26 blocking copies/token** while
preserving 12 async copies. The p508+128-step wall ratio is neutral at
**1.00343x (95% CI 0.98776–1.01909)**; this is a transfer-boundary retention,
not a wall-speed claim. The current P12 canonical p512/p1024/p4096 production snapshot is
**83.35/82.93/69.20 pp/s** and **14.18/14.16/12.16 tg/s**, all 36 measured
samples deterministic with zero teardown. Named strict is
**61.05/60.32/52.56 pp/s** and **13.52/13.43/9.47 tg/s**; its three-repeat
variance prevents a closure-rate claim.
[`P3 Q8-gate rejection`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-rejected.json).
[`P3 fused GR`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json).
[`P3 F32 router tile4`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json).
[`P3 GR up+sigmoid+mean`](results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json).
[`P4 QSA dense fixed256`](results/2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json).
[`P5 device argmax`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json).
[`P5 current canonical AR`](results/2026-08-31-gfx1151-qwen38-flash-next-p5-current-canonical-ar.json).

P6 localizes the long-context cliff to indexed QSA activation. Identical
transition medians at live counts 2,051/2,052/4,097 are **66.61/95.88/96.02
ms**. The boundary adds **30.77 ms** of profiled kernel time; sparse attention
alone adds **27.47 ms**, while score/top-k adds **0.92 ms**. The nearly flat
2,052→4,097 result points to the fixed ~2K selected-attention budget rather than
continued context growth. [`P6 context profile`](results/2026-08-31-gfx1151-qwen38-flash-next-p6-context-transition-profile.json).

A same-weight external-fork refresh built EngramHalo HIP `1423f689` and
Nathan Vulkan `ad914eb` locally. BF16-KV p508/p1012/tg32 shape rows are
**296.12/362.72/17.62** and **413.04/396.25/23.85 tok/s**; Nathan's local
build agrees with its v0.7.2 payload within 1%. These are historical
`llama-bench` shape diagnostics, not exact-prompt or source-only A/B rows. Nathan
lazy-on/off averages **413.04/329.23 p508 (1.255x)** but converges by p1012;
an Engram MTP diagnostic is 1.128x complete-wall at 94.55% acceptance but only
**9/10** AR-message exact, so it is not a valid speed target.
[`external fork refresh`](results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json).

The cross-engine survey adds a 160-row, full-vocabulary, same-GGUF packet.
Current upstream HIP is 160/160 top-1 and effectively identical to frozen
#27742 HIP. EngramHalo is 159/160 with mean/max KL **9.85e-4/0.01431**.
Upstream Vulkan, Nathan, and apepojken are each 159/160 versus frozen HIP, but
Nathan is effectively identical to upstream Vulkan (160/160, mean KL about
**2e-10**) and apepojken remains 160/160 versus upstream Vulkan at mean/max KL
**0.00109/0.01576**. This localizes Nathan's failure to multi-step execution
rather than broad static math. Short Q8-KV apepojken MTP is **1.807x**
complete-wall at 92.8% acceptance but only **9/10** AR-message exact, matching
EngramHalo's failing prompt. Nathan MTP is provisionally **1.161x** at 95.45%
acceptance, but AR and MTP each self-repeat only 9/10 and just **8/10** prompts
match across both repeats of both modes. All affected speed rows are invalid as
targets. The survey also compares upstream/fork test coverage and
absolute-quality evidence.
[`Strix Halo survey artifact`](results/2026-08-31-gfx1151-qwen38-flash-next-strix-halo-survey.json).

The previous GDN decode-all claim is **invalid**: its selector was unreachable,
so the packet compared the strict owner to itself; the 16.2 tok/s helper also
used all-layer DP4A rather than admitted safe43. Wiring the actual candidate
costs **6.832 ms/token plus a 0.117-ms tail**, versus **2.454 ms/token** for the
retained GDN owner, and lowers full decode. Commit `15a436766` clears the dead
binder route. Prefill colwarps 27–47 remains certified. Current
production/strict manifests are `9e27fec0…` / `42509601…`; omitted routes stay
strict.
The certified
compact-WMMA MoE suffix (layers 27–47: Q4_K dual gate/up + Q5_1 down on the
f16-WMMA matrix-core kernels, tile 16×16; replaces the ds4-MMQ suffixes and
strict owners on those layers; `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL=1`)
passes the complete 450-row/three-repeat packet at KL
mean/p95/p99/max `2.79e-4/1.53e-3/3.49e-3/5.98e-3`, **446/450 top-1** (all
scopes ≥ 98.67%), exact repeat/state, 18/18 repeat-exact free generation
(4 task-valid divergences), exact c2 with zero teardown, and improves paired
p508/p1012 **6.572→6.287 s (-4.34%, 80.82 tok/s)** /
**13.398→12.694 s (-5.26%, 79.73 tok/s)**. The layer-27 boundary is the
maximal envelope-admissible suffix (full-layer WMMA screens at mean 5.9e-3).
The certified GDN column-warp suffix (llama gated_delta_net layout, layers
27–47; 4.58× per launch, −17.1%/−15.7% paired p508/p1012, supersedes
peer-GDN) and the iu8-WMMA gate/up suffix (layers 35–47 within the WMMA-MoE27
route; exact Q4_K q values + 3 residual activation planes + min-offset
ds-trick) passes the complete packet at KL mean/p95/p99/max
`2.62e-4/2.20e-3/4.34e-3/5.52e-3`, **446/450 top-1**, zero scope failures,
exact repeat/state, 18/18 repeat-exact free generation (15/18 strict-exact),
exact c2 with zero teardown, and improves paired p508/p1012
**7.430→6.650 s (-10.5%)** / **15.260→13.469 s (-11.7%)** over the f16
production stack under matched conditions; the binder selects it via
`HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL=1` (layers 35–47). Current natural 16K improves **946.999→341.177 s (-63.96%, 47.989 tok/s; chunk-512 gate**
re-passed with retrieval/oracle/transactional/teardown exact) with every
binding control exact; 64K historical evidence is retained but not rerun because
47.989<100 tok/s. 262K is capacity-only (91.126 GB tracked), not inference. Q8 MTP is exact on 10/10
prompts but remains opt-in at **0.955x AR**. A PR-#303 feasibility diagnostic
measures its 675.4-MB full Q8_0 draft head at **3.153 ms (41.3% of a 7.639-ms
draft step)**, but even a free head projects only **0.964x AR** on the retained
suite; target verification and host draft outputs remain first. [`hot-head diagnostic`](results/2026-09-01-gfx1151-qwen38-flash-next-mtp-hot-head-feasibility.json).
<=1K image/video/PNG chat and
request-owned c2 blocking/SSE pass with zero teardown; packed c-aware speed,
remote media, multimodal SSE, and 128K+/262K inference are not claimed.
Evidence: [`gap`](results/2026-08-28-gfx1151-qwen38-flash-next-llamacpp-matched-baseline.json) · [`MoE graph`](results/2026-08-29-gfx1151-qwen38-flash-next-exact-moe-graph-decode.json) · [`production`](results/2026-08-29-gfx1151-qwen38-flash-next-moe27-q8-32-production.json) · [`chunk512`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-chunk512.json) · [`Q8 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q8-mmq-prefill-production.json) · [`Q5_1 MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q5-1-mmq-suffix32-production.json) · [`Q4_K MMQ`](results/2026-08-29-gfx1151-qwen38-flash-next-q4-k-mmq-suffix35-production.json) · [`MMQ+DP4A stack`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-prefill-dp4a43-stack.json) · [`profile manifest`](results/2026-08-29-gfx1151-qwen38-flash-next-production-mmq-profile-manifest.json) · [`peer GDN`](results/2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json) · [`final campaign`](results/2026-08-29-gfx1151-qwen38-flash-next-prefill-mmq-campaign-final.json) · [`master re-baseline`](results/2026-08-29-gfx1151-qwen38-flash-next-llamacpp-master-rebaseline.json) · [`WMMA MoE27`](results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json) · [`iu8 gate35`](results/2026-08-30-gfx1151-qwen38-flash-next-iu8-wmma-gate35-production.json) · [`GDN colwarps27`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps27-production.json) · [`QSA flash (key-parallel 35-47)`](results/2026-08-30-gfx1151-qwen38-flash-next-qsa-flash31-production.json) · [`fresh full profile`](results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json) · [`invalid GDN decode correction`](results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

## Current Qwen3.6-35B quantization quality

The current gate scores 90 full-vocabulary BF16-teacher positions across all ten
code/English/Japanese/mixed prompts. Every row uses the exact local artifacts
and identical teacher contexts; no historical or unmatched-artifact rows are
mixed into this table. This is a cross-runtime distribution gate, not
held-out-corpus PPL.

| Exact local artifact | Size / BPW | Evidence scope | Mean KL vs BF16 ↓ | Top-1 agreement ↑ | Status |
| --- | ---: | --- | ---: | ---: | --- |
| GGUF `UD-Q4_K_M` | 21.107 GiB / 5.180 | exact-artifact, ROCmFPX HIP | **0.013713** | 92.222% | Matched-runtime quality baseline |
| ROCmFP4 STRIX_LEAN | **17.739 GiB / 4.354** | exact-artifact, ROCmFPX HIP | 0.045984 | **97.778%** | Quality-traded: KL/category margin fails |
| PARO full8192 packed | 19.068 GiB / 4.680 | exact-artifact, hipEngine HIP | 0.027038 | 92.222% | Quality-traded; runtime-correct and deterministic |

ROCmFP4 is 15.96% smaller than local Q4_K_M and retains more BF16 greedy
argmaxes, but fails the paired KL/category margin. PARO is runtime-correct and
deterministic after the packed-layout repair, yet remains quality-traded versus
hipEngine Q4_K_M. See the [`quality artifact`](results/2026-08-16-zbook-qwen36-quant-quality.json)
and [`protocol`](quant/README.md).

Current package decisions are compactly separated by execution profile:

- Packed PARO retains exact SiLU+down-rotation (**1.371x leaf, 69 fewer c8/L4
  launches**) with neutral aggregate wall; unsafe math is rejected.
- ZBook strict c1 retains the exact cooperative router (**30.438 -> 33.219
  tok/s, 18/18 wins**). Physical c4/c8 retain exact Q8T16 rowtiling while c2
  remains direct.
- The combined c1/cN package is exact over **1,050/1,050 rows** and remains the
  implementation default, but is not a public `production` profile: the
  60-second server soak completed 87/120 requests and rejected 33 as overloaded.

Evidence: [`PARO boundary`](results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp3-silu-rotate-retained.json),
[`c1 router`](results/2026-08-16-zbook-qwen36-c1-router-retained.json),
[`c4/c8 rowtile`](results/2026-08-16-gfx1151-q8t16-batch-route-retained.json),
[`package decision`](results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json), and the
[`ROCmFPX transfer report`](quant/ROCMFPX-TRANSFER.md).

Current Qwen3.5-0.8B gfx1151 remains **Vulkan parity blocked** while the exact
D08-X package is retained: the final gate is **1794/1800 top-1, max KL
0.005930**, with **72/72** graph trajectories exact. [`Campaign`](../docs/QWEN35-08B-GFX1151-VULKAN-PARITY.md).

## Current single-request scoreboards

### Radeon Pro W7900: Qwen3.6-35B-A3B

The repaired-runtime publication uses two warmups and five measured resets per
right-sized session. `Peak` is hipEngine tracked allocator high-water.

| Workload | PARO prefill | PARO decode | PARO peak | GGUF prefill | GGUF decode | GGUF peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | **2852.100** | **115.804** | **18.144 GiB** | 2763.590 | 94.603 | 21.073 GiB |
| 1K/128 | 2965.063 | **103.113** | **18.367 GiB** | **3198.957** | 99.728 | 21.133 GiB |
| 4K/128 | 2927.519 | **106.020** | **19.161 GiB** | **3177.565** | 101.917 | 21.468 GiB |
| 32K/128 | 2085.511 | **92.422** | **19.851 GiB** | **2154.871** | 89.432 | 22.060 GiB |
| 64K/128 | 1559.680 | **79.098** | **20.344 GiB** | **1600.734** | 78.021 | 22.736 GiB |
| 128K/128 | 1049.467 | 61.804 | **21.881 GiB** | **1058.075** | **63.177** | 24.088 GiB |

PARO leads short-context generation and memory; GGUF leads prefill from 1K and
128K generation. IDs, variance gates, and clean provenance pass. Evidence:
[`PARO sweep`](results/2026-08-23-w7900-current-default-hipengine-paro-packed-5run.json), [`GGUF sweep`](results/2026-08-23-w7900-current-default-hipengine-gguf-q4km-5run.json).

### Radeon Pro W7900: Qwen3.6-27B Dense GGUF

This `Q4_K_M`/BF16-KV snapshot uses one warmup and three measured resident
resets per shape with state-bound PM4 graph decode.

| Workload | Prefill | Decode | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | **875.364 tok/s** | **28.681 tok/s** | 15.587 GiB |
| 1K/128 | **911.658 tok/s** | **29.383 tok/s** | 15.681 GiB |
| 4K/128 | **878.721 tok/s** | **26.747 tok/s** | 16.204 GiB |

All nine IDs are stable/finite; prefill/decode CV is at most 0.733%/0.475%, and
the ten-prompt gate is exact. [`Current-default evidence`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json). Qwen3.8 details remain in the XTX tables above.

### Radeon 8060S: Qwen3.8-27B Dense GGUF retained campaign state

Qwen3.8 uses `Q4_K_S` with BF16 K/V. The campaign is closed at merged commit
`20e5106da`; the Q5 source-F16 prefill retention (2026-08-17) raises 512/1K/4K
prefill on gfx1151 via the byte-identical K_M-derived Q5T16 recurrent-output
route (counterbalanced +4.51%/+3.02% at 512/1K, +2.95% at 4K with a
capacity-conditional scratch cap that keeps 8K+ memory flat). Prefill and true
AR beat both clean llama backends at every working shape, exact native B3 beats
the correctness-valid llama HIP row, and process GTT stays below the lower
valid llama row at 512/1K/8K+ (4K peak grows a fixed +2.30 GiB to enable the
4K source-F16 win).

| Shape | Clean prefill | Clean AR | Retained process GTT | Lower valid llama GTT |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **396.091** | **13.069** | **15.275 GiB** | 15.785 GiB |
| 1K/128 | **387.648** | **12.894** | **15.710 GiB** | 15.816 GiB |
| 4K/128 | **380.305** | **13.038** | **17.863 GiB** | 16.004 GiB |

Exact native B3 is **23.85263 tok/s / 1.7845x AR** with all ten prompt
trajectories and GPU/CPU acceptance decisions exact; retained process GTT is
**15.899 GiB** versus valid llama HIP's **16.358 GiB**. Natural true AR is
**13.36641 tok/s** versus same-file llama Q4_K_S HIP/Vulkan at
**5.53853/7.51888 tok/s**. Rejected aliases and direct file mapping remain
recorded—not discarded—in the linked evidence.
Evidence: [`clean Q4_K_S`](results/2026-08-16-gfx1151-qwen38-27b-q4ks-clean-publication.json),
[`exact B3`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json),
[`memory package`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-memory-parity-retained.json),
[`G6 closure`](results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json), and the
[`campaign plan`](../docs/QWEN38-27B-GFX1151-CAMPAIGN.md).

### Radeon 8060S: Qwen3.6-35B-A3B GGUF

This is the latest clean, exact one-queue production snapshot. The artifact is a
campaign completion gate, not a claim that its final step improved every row.

| Workload | Prefill | Decode | Tracked peak | Whole-device GTT peak |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **1369.489 tok/s** | **54.330 tok/s** | 20.566 GiB | 21.000 GiB |
| 4K/128 | **1430.215 tok/s** | **54.798 tok/s** | 20.951 GiB | 21.499 GiB |
| 32K/128 | **1144.713 tok/s** | **46.405 tok/s** | 21.597 GiB | 22.152 GiB |
| 64K/128 | **936.218 tok/s** | **40.180 tok/s** | 22.336 GiB | 22.890 GiB |
| 128K/128 | — | — | — | — |

Repeated 128K remains blocked by the documented later-pass lifecycle stall; no
numeric 128K row is carried forward. Evidence:
[`SH14-C1 completion gate`](results/2026-08-06-gfx1151-gguf-sh14-c1-cumulative-completion-gate.json).

### Laguna S 2.1

| Platform / format | Workload | Prefill | Decode | Evidence |
| --- | --- | ---: | ---: | --- |
| W7900 / `UD-Q2_K_XL` | 4096 prompt, prefill only | **440.893 tok/s** | — | [`H8B production`](results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json) |
| Radeon 8060S / `Q4_K_M` | 512/128 | **654.249 tok/s** | **23.221 tok/s** | [`prefill production`](results/2026-07-27-gfx1151-laguna-attention-packed-query-producer-candidate.json), [`decode production`](results/2026-08-01-gfx1151-laguna-registry-resolution-cache-retained.json) |

The W7900 Laguna decode campaign and rejected H7/H8 ladders are implementation
history, not scoreboard content; follow the production artifact, changelog, and
[`docs/LAGUNA-PARITY-STATUS.md`](../docs/LAGUNA-PARITY-STATUS.md).

Explicit gfx1151 Laguna DFlash remains non-default and uses the tile1 target
verifier. The attempted tile4 transfer was trajectory-identical to tile1 but
failed the shared full-suite true-AR gate and did not improve complete E2E wall;
see the [`tile4 rejection`](results/2026-08-20-gfx1151-laguna-dflash-iq3-tile4-rejected.json).

## Current concurrency scoreboards

All values are aggregate generated tokens per second. Direct rows time the
resident model path; server rows include the named OpenAI serving protocol and
must not be compared as the same timing scope.

### W7900 Qwen3.6-35B-A3B GGUF `UD-Q4_K_M`

| Interface | c1 | c2 | c4 | c8 | c9 | c13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct engine | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| OpenAI SSE | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

Direct c1 uses HIP graph; the admitted c2/c4/c8 rows use the exact scoped PM4
transport. Server c9/c13 are declared grouped execution, not native widths.
All 189 server request rows and 24,192 generated IDs pass the exact gate.
Evidence: [`context-scoped C8 server refresh`](results/2026-08-08-gfx1100-context-scoped-c8-server-refresh.json).

### Maple-Preview 2-bit on Radeon 8060S

| Interface | c1 | c2 | c4 | c8 | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Public engine generation64 | **123.131** | **165.697** | **202.038** | **214.788** | Admission, prefill, generation, reclaim |
| Fixed helper decode64 | — | **250.481** | **346.365** | **428.063** | Decode helper only; excludes public scheduling |

Evidence: [`public P4`](results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json)
and [`D1 helper`](results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json).

## Current speculative decode scoreboards

| Platform / model | Contract | True AR | MTP | MTP / AR | Status and evidence |
| --- | --- | ---: | ---: | ---: | --- |
| W7900 / Qwen3.6-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 29.457 | **60.929** | **2.0684x** | Current clean snapshot; all ten prompts, greedy outputs, and GPU/CPU acceptance agree. The ratio replaces stale historical denominators. [`artifact`](results/2026-08-23-w7900-qwen36-27b-current-default-publication.json) |
| RX 7900 XTX / Qwen3.8-27B Dense `Q4_K_M` | Exact/default natural25 B3 | 35.287 | **62.440** | **1.7695x** | Clean idle-card correction; exact greedy and GPU/CPU acceptance, retained fusion improves matched AR 3.764% and B3 0.439% with every category non-regressive. [`artifact`](results/2026-08-15-qwen38-27b-xtx-clean-idle-performance-correction.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Exact natural25 B3 | 11.692 | **21.158** | **1.8095x** | Clean current-main direct-leaf snapshot; all ten prompts and 30 MTP comparisons are exact, GPU/CPU acceptance agrees, and cached profiling confirms the qualified scalar-C1 and native Q4 rows4/2 owners. [`artifact`](results/2026-08-26-gfx1151-qwen38-current-main-ar-mtp.json) |
| Radeon 8060S / Qwen3.8-27B Dense `Q4_K_M` | Public LLM strict/BF16 C1 natural25 B3, artifact-scoped automatic | 9.025 | **12.940** | **1.4337x** | Complete request through terminal reclaim: 30/30 exact cells, every category/heldout positive and every cell 1.2995x–1.5515x. Exact hash/profile/BF16/C1/B3/context1-67/natural25 auto-promotes after lifecycle/SSE/load qualification; every other scope is K0. [`artifact`](results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s4-auto.json) |
| W7900 / Qwen3.6-35B-A3B packed PARO W4A16+MTP BF16 | Production/default B1 fast, raw D24 | 110.830 | **115.770** | **1.0446x** | Exact `720/720`; complete 10-prompt numerical/repeat/task/state gate passes. Fast improves strict MTP 10.33% overall and every category. [`artifact`](results/2026-08-24-w7900-paro-fast-d24-3run-default.json) |
| W7900 / Qwen3.6-35B-A3B `UD-Q4_K_M` | `llama-compat` MTP-2 natural suite | 96.75 | **122.67** | **1.2679x** | Retained explicit opt-in; accuracy-traded versus normal AR. [`artifact`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json) |
| Radeon 8060S / Qwen3.6-35B-A3B `UD-Q4_K_M` | `llama-compat` MTP-2 natural suite | 56.09 | **80.10** | **1.4282x** | Retained explicit opt-in; accuracy-traded versus normal AR. [`artifact`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json) |

MTP ratios always use a true no-MTP AR path from the same protocol. Verifier
`off`/`B0` diagnostics are not speedup denominators. The full category suite,
heldouts, and anti-gaming rules are mandatory; see
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md#anti-gaming).

## Maple-Preview retained backend comparison

These are same-model retained rows, but CUDA and HIP run on different hardware.

| Platform | Workload | Current throughput | Exactness / scope | Artifact |
| --- | --- | ---: | --- | --- |
| Radeon 8060S | Native prefill 128/320/512 | **750.854 / 741.890 / 754.458 tok/s** | 18/18 states, 90/90 positions, KL 0 | [`P4`](results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json) |
| Radeon 8060S | c1 natural+heldout continuation | **153.201 tok/s** | 18 prompts, 1,152 timing pairs, exact state/head | [`D0`](results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json) |
| RTX PRO 6000 Blackwell | Native prefill 128/320/512 | **1953.820 / 1852.124 / 1917.492 tok/s** | 18/18 states, 90/90 positions, KL 0 | [`CUDA prefill`](results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json) |
| RTX PRO 6000 Blackwell | c1 natural+heldout continuation | **402.361 tok/s** | 1,152/1,152 paired wins; 1,296/1,296 positions exact | [`CUDA split-K`](results/2026-08-09-cuda-sm120a-maple-splitk-global-decode-retained.json) |

CUDA resident batching and serving are not claimed by these c1 rows.

## Reading the tables

Workloads use `prompt_tokens/decode_tokens`. Compare only matching timing,
model/quant/KV, concurrency, and memory scopes; bold identifies the reported
row, not a universal leader.

## Maintenance contract

1. Replace the current row for a protocol tuple; do not append an optimization
   diary beneath it.
2. Put exact commands, samples, deltas, profiler data, correctness details, and
   candidate decisions in the compact JSON artifact.
3. Put the one-line old-to-new transition in [`CHANGELOG.md`](CHANGELOG.md) and
   substantial implementation decisions in a new immutable worklog entry.
4. Mention a blocked/rejected run here only when it removes a current numeric
   row or defines a user-visible limitation. Link one artifact and one rerun
   condition; keep candidate ladders out of the scoreboard.
5. Keep superseded tables in [`HISTORY.md`](HISTORY.md), artifacts, or Git
   history rather than copying them forward (`git show 6a8d38ae70b9e2c4244df10d8621db83da6c8112:benchmarks/README.md`).
6. Update `Last updated`, then synchronize the public block:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
git diff --check
```

The full evidence and artifact requirements remain authoritative in
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
