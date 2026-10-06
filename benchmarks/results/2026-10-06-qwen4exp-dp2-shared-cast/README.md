# DP-2 natural two-consumer cast screen

Sharing one F16 staging cast benefits the measured p1024/p4096 paired
operations. It is a diagnostic result, not a production implementation or
model-throughput result. Keep the screen and proceed to a bounded native
ownership contract. Do not admit p512 from this packet.

## Results

Both six-case runs use the same unchanged consumers and source revision.
The final run adds canonical provenance and byte-level staging checks; it
is not an optimization over the first run. Absolute shared-expert times
vary substantially between runs, so both sample sets are preserved.

| Run / measured rows | Two-cast sum of owner wall medians ms | One-cast ms | Duration change |
| --- | ---: | ---: | ---: |
| Initial, all measured rows | 159.161 | 151.947 | −4.533% |
| Final, all measured rows | 138.414 | 130.648 | −5.611% |
| Final, p1024 | 8.860 | 8.535 | −3.671% |
| Final, all four p4096 categories | 124.047 | 116.711 | −5.914% |

These sums price only the sampled layer13/24/46 pairs. They are not an
engine duration and must not be scaled to a prefill rate or all layers.
Final QKV+gate owner wall ratios span 1.00219–1.03730x; shared gate+up
spans 0.98270–1.35687x. Final p1024/p4096 owner medians and both order
strata are positive. There are two limitations:

- The initial p1024 layer24 shared pair has a two-casts-first wall ratio
  of 0.98803x despite a positive pooled median. A focused 20-pair recheck
  gives pooled 1.03189x and order strata 1.05170/1.01519x; the final
  six-case run also passes this owner. The original loss is preserved.
- Final p512 layer13 shared pair is slower overall (0.98270x), with a
  0.94305x two-casts-first wall stratum. Layer46 QKV+gate has a 0.99156x
  stratum. Initial p512 is positive, but that is not repeatable admission
  evidence. Keep p512 parent pending separate qualification.

## Protocol and checks

- Host `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
  HIP 7.15.26333 from therock10 staging, performance CPU governor/high GPU.
- Qwen3.8-Flash-Next UD-Q4_K_XL revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, BF16 KV, named production
  profile, chunk4096/context4352. Model/fixture/manifest identities are in
  [artifact.json](artifact.json) and [final.json](final.json).
- Natural cases: code p512/p1024/p4096, plus English, Japanese and mixed
  p4096. At each of layers13/24/46, QKV+gate and shared gate+up run one
  warm pair plus six measured pairs with alternating arm order. Each
  six-case run has 36 owner captures and 504 replay operations.
- The normal production pair runs first. Replay uses disjoint diagnostic
  F16 scratch and outputs, the same source, weights and resolved native
  F16-input GEMMs. Baseline is `cast,A,cast,B`; candidate is `cast,A,B`.
  No cache, production loan, arithmetic or default changes.
- Host wall charges cast plus both GEMMs, event records and final event
  synchronization. HIP-event intervals are recorded separately. Allocation,
  validation copies, dispatch selection and surrounding model work are
  outside timing; validation transfers can perturb caches.
- Every replay output matches native output bytes and is finite. Final F16
  staging bytes match CPU conversion. Source and raw Q8 weight hashes are
  unchanged, per-replay allocation counts return to their prior level, and
  final allocations/bytes are zero. The earlier artifacts use F16 value
  equality rather than a byte hash; they do not establish signed-zero bits.
- Final canonical provenance records staged/unstaged clean and untracked
  files separately, including the hashed diagnostic script. This is not a
  tracked-clean production-promotion measurement. The unrelated untracked
  workspace probe is not imported or executed by this harness.
- 104 focused CPU/provenance/registry/ownership tests pass, as do fixture
  and CPU smokes, bytecode compilation and test-tier checks. No new kernel
  is introduced. Independent review could not launch because its configured
  model resolves to multiple providers; the parent performed source review.

## Commands

```bash
P=/home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$P/_rocm_sdk_devel/lib:$P/_rocm_sdk_core/lib:$P/_rocm_sdk_libraries/lib
export PATH=$P/_rocm_sdk_devel/bin:$P/_rocm_sdk_core/bin:$PATH
export HIPENGINE_HIP_ARCH=gfx1151
uv run python scripts/qwen4exp_shared_f16_cast_owner.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --layer 13 --layer 24 --layer 46 --pairs 6 \
  --output /tmp/qwen4exp-dp2-final-six-case.json
```

The initial command differs only in output path
`/tmp/qwen4exp-dp2-six-case.json` and uses the pre-provenance harness hash
in [initial.json](initial.json). The focused command uses
`--case-id code-p1024 --layer 24 --pairs 20 --output
/tmp/qwen4exp-dp2-p1024-recheck.json` in place of the three layers/six pairs;
its complete samples are in [focused_p1024.json](focused_p1024.json).

## Native qualification required

Start with exactly1024/4096 rows and the existing production layer scope,
not interpolated shapes or prompt-conditioned branches. The runner must open
an ordered, immediate two-consumer scope over one bounded session workspace.
Bind source/count/runtime/stream/workspace and consumer identities; invalidate
on mismatches, competing dispatch, overwrite, nesting, exception or context
exit. Do not turn pointer equality into a persistent activation cache or weaken
DP-5's separately qualified single-use contract.

Require fake-runtime RED/GREEN ownership tests, native full-logit/Conv/GDN/
full-KV identity, determinism/isolation/fallback/teardown and exact launch
engagement, then tracked-clean same-host canonical/heldout qualification and
counterbalanced engine A/B. Strict, Q4_K_M, other backends and unmeasured
rows keep parent. The historical7.478ms half-cast estimate is not a measured
engine saving and does not provide an admission threshold.
