# Qwen4Exp checked-index GDN production qualification

On 2026-10-06, Radeon 8060S `gfx1151` host
`55ea6c509d0b49eea8de7094a1023668` measured a separately registered D128
GDN wave-norm sibling with checked int32 element indexing. It preserves the
FP32 recurrence, state layout and norm tree. The parent uses int64 indexing.
This screen uses Qwen3.8-Flash-Next UD-Q4_K_XL geometry, synthetic F32
inputs and nonzero FP32 recurrent state; no model weights are loaded.

## Complete-operation result

HIP events cover the complete recurrence, normalization and sigmoid output
gate in one launch. State reset, synchronization and result downloads are
outside the timer for both arms. Three warmup pairs precede 21 alternating
parent/candidate pairs. All measured output/state pairs are bit-identical.

| Tokens | Parent median ms | Candidate median ms | Speedup | Speedup by order |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.102472 | 0.100227 | 1.0224x | 1.0396x / 1.0206x |
| 64 | 0.333825 | 0.324287 | 1.0294x | 1.0347x / 1.0226x |
| 512 | 2.709651 | 2.594705 | 1.0443x | 1.0500x / 1.0321x |
| 1024 | 5.371735 | 5.155940 | 1.0419x | 1.0460x / 1.0361x |
| 4096 | 21.318647 | 20.451752 | 1.0424x | 1.0432x / 1.0408x |

The one-token sample is order-sensitive (1.1408x / 0.9793x) and supports
no speed claim. It is retained in the JSON. Production does not select this
prefill route for one row. No model throughput or decode claim follows from
this operation screen.

## Resources and correctness

The actual cached code object reports:

| Resource | Wide-index parent | Narrow-index sibling |
| --- | ---: | ---: |
| Static VGPR | 256 | 256 |
| SGPR | 51 | 44 |
| Private bytes/lane | 36 | 28 |
| VGPR spills | 8 | 6 |
| Static scratch instructions | 11 | 6 |
| Dynamic LDS bytes | 2560 | 2560 |

A cached-only `rocprofv3 --kernel-trace` child records 14 launches per arm at
512/4096 rows, with scratch sizes 36/28 bytes and plausible durations
(about 2.6/2.6 ms and 21.3/20.5 ms). Timing claims use the unprofiled
measurements, not the profiler samples.

The focused GPU fixtures cover tokens 1/17/64/512/1024/4096, exact repeated
output/state and split-at-7 replay, zero Q/K heads, nonzero initial state,
read-only input guards, output/state guards, cross-head state isolation and
raw-ABI overflow rejection. CPU-reference output/state tolerance and the
KL≤0.05/top-1≥90% outer floor pass on 1/17/64 rows. Python admission tests
cover invalid dimensions, huge products and the accepted int32 boundaries.
The element counts for QKV and recurrent state must each fit signed int32;
that also bounds output/gate/alpha/beta offsets. The raw-pointer ABI retains
int64 host dimensions and rejects unsafe shapes before launch.

The gfx1151 UD-Q4_K_XL production profile selects the candidate only at
Hk16/Hv48/D128 and 2–4096 rows. Other quant/profile scopes do not select it;
the wide-index parent stays outside the bound and as explicit rollback.
Strict primitives remain registered. The full-model qualification and
engine A/B below support the exact owner saving, not an all-case request
speedup.

## Natural-model owner result

The same host's production UD-Q4_K_XL model generated inputs for 216 GDN
calls across six canonical cases at chunk4096. Each call restores the same
starting state outside timing, runs one warmup pair and two counterbalanced
measured pairs, then hashes complete output and recurrent state after each
arm. All pairs are exact; both measured pair totals improve in every case.
The current UD arithmetic recovery uses this serial leaf on all36 GDN layers,
not the historical21-layer prefix. No allocation remains after close.

| Case | Parent owner ms | Candidate owner ms | Speedup |
| --- | ---: | ---: | ---: |
| code-p512 | 99.625 | 95.758 | 1.0404x |
| code-p1024 | 199.528 | 191.747 | 1.0406x |
| code-p4096 | 817.974 | 796.693 | 1.0267x |
| general_en-p4096 | 815.047 | 787.595 | 1.0349x |
| general_ja-p4096 | 819.483 | 788.585 | 1.0392x |
| mixed_ja_en-p4096 | 820.201 | 785.555 | 1.0441x |

These are sums of per-call paired mean HIP-event intervals, not request walls.
`code-p4096` has pair ratios1.0087x/1.0452x; both are positive, but the spread
is explicit. [natural-owner.json](natural-owner.json) records all call timings,
model fingerprint, parent manifest and teardown. The named parent profile
is intercepted by a diagnostic registry hook; its manifest does not certify
the candidate. Full model-state/KV qualification also passes below.

```bash
uv run --no-sync python scripts/qwen4exp_gdn_owner_probe.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --narrow-index --prefill-chunk-size 4096 \
  --output /tmp/qwen4exp-gdn-narrow-20261006/natural-owner.json
```

## Full-model state qualification

All12 canonical cases and8 category-heldouts pass at BF16 KV/chunk4096,
maximum sequence4352 and16 decode steps. Each case runs parent, three
candidate repeats, then parent. All five runs share full-vocabulary prefill/
step-logit hashes, prefill state, post-decode state/layout/metadata and full
KV hashes. This is100 runs/1700 full-vocabulary rows including repeats.
Candidate engagement is36 calls/run on all60 candidate runs:2160 prefill
calls total and zero decode calls. Finite state and leak-free close pass.

[model-state.json](model-state.json) contains the compact audit and original
artifact hash. This is candidate-vs-admitted-production parity for unchanged
arithmetic, not a new strict-teacher numerical calibration or task-quality
score. This packet was measured against the pre-promotion parent manifest;
it is not itself evidence that the binder selects the candidate.

```bash
uv run --no-sync python scripts/qwen4exp_row4_state_gate.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --route-package gdn-narrow --prefill-chunk-size 4096 --all-cases \
  --include-heldouts --decode-steps 16 --full-kv \
  --output /tmp/qwen4exp-gdn-narrow-20261006/model-state.json
```

## Tracked-clean engine A/B

Clean source `7a1dc7b22` runs one process and one model residency with all
12 canonical cases, one warmup per arm/case and three counterbalanced
measurements per arm. Prefill uses chunk4096; each request includes the first
greedy output plus 128 decode transitions. All 72 trajectories are exact and
deterministic, with 36 candidate prefill calls per after-run (1296 total),
zero candidate decode calls and no allocations after close. The diagnostic
hook's manifest hash names the parent, not the candidate.

| Prompt tokens | Parent weighted prefill tok/s | Candidate weighted prefill tok/s | Change |
| ---: | ---: | ---: | ---: |
| 512 | 291.658 | 292.093 | +0.149% |
| 1024 | 317.351 | 318.314 | +0.303% |
| 4096 | 324.199 | 324.936 | +0.227% |

Both order strata improve prefill: before-first 1.002110x, after-first
1.002559x, 18 pairs each. Eleven case prefill aggregates improve; Japanese
p512 is -0.00794%, with its parent/candidate CV 0.629%/0.621%. Its three
pair ratios are 1.004042x/0.996559x/0.999174x. Those measurements are not
removed or replaced by a favorable rerun.

Complete request wall improves +0.108% overall but loses three case
aggregates: English p1024 -0.00529%, mixed p512 -0.54646%, mixed p1024
-0.12246%. Request order strata are 0.999679x/1.002448x. Maximum within-case
decode CV is 2.628%; the candidate does not change the decode owner. All 72
samples have zero process-read-byte and major-fault deltas. Minor fault
deltas, including 21 samples at 574, are retained in the artifact; no cause
is asserted. The measured first-to-last span is recorded separately from
loading/warmup wall.

Promotion retains the independent exact natural-input owner saving under
`AGENTS.md`'s sub-window policy. It does not assert all-case request
non-regression or a decode-speed gain. [engine-ab.json](engine-ab.json)
contains clean-source provenance, raw hash, every compact measured sample,
order strata, CV limits and all case/shape/category summaries.

```bash
uv run --no-sync python scripts/qwen4exp_halo_box_campaign_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --require-cached-build \
  --route-package gdn-narrow --prefill-chunk-size 4096 \
  --warmups-per-mode 1 --repetitions-per-mode 3 \
  --output /tmp/qwen4exp-gdn-narrow-20261006/engine-ab.json
```

## Runtime selection and rollback

The UD production manifest names disjoint serial scopes at 2–4096 and
above4096 rows. Only the bounded scope selects
`qwen4exp_sigmoid_wave_norm_narrow_prefill`, with
`qwen4exp_sigmoid_strict_prefill` as its declared strict fallback. The runner
also checks Hk16/Hv48/D128 and registration before launch; an unavailable
narrow key falls back to wide-index, then the register parent. One-row decode
never enters this selector. Existing int32 product checks in Python and the
int64 raw host ABI remain mandatory even within the qualified runtime bound.
Inactive peer/tiled layer-scope declarations are removed from the UD manifests;
strict arithmetic is unchanged and Q4_K_M's plans are unchanged.

For bisection, set
`HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT=qwen4exp_sigmoid_wave_norm_prefill`
before profile registration/generator construction. This resolves a parent
manifest and the binder publishes that same selection. Setting it after
binding is a diagnostic override, not a named manifest. Owner/state/A/B
hooks pin the parent after binding so the two-arm comparison still engages
correctly on a promoted tree. The temporary rollback/hook cleanup is tracked
in `docs/REFACTOR.md`.

A separate native-binder smoke records the actual production/strict manifests
and uses logging-only wrappers around the registered leaves. It compares
native/parent/native/parent at code512/1024/4096, Japanese512 and a Japanese
category-heldout with16 decode steps, full logits, prefill/post-decode state
and full KV hashes. All five cases pass, with 360 native narrow calls,
360 parent wide calls, zero decode calls and leak-free teardown. Production/
strict manifests are `77fad810048c7fc4d4ecce0ba3ae7858aaec081f9777b19726e2252b41cc2cfc` /
`f58c7110da72e8176d109085608939d128ba7f79fed5065ce2e71b424680639b`.
[native-smoke.json](native-smoke.json) records raw/script/source-diff hashes,
engagement and teardown. This is a routing/correctness smoke, not another
performance measurement.

```bash
uv run --no-sync python scripts/qwen4exp_gdn_narrow_native_smoke.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output /tmp/qwen4exp-gdn-narrow-20261006/native-smoke.json
```

## Reproduction

The synthetic operation screen used base `c4619579b` with tracked candidate
edits; `f4c75ac62` commits that kernel. Natural-input and full-state qualification
used tracked harness edits at `f4c75ac62`; `7a1dc7b22` commits that harness.
The engine A/B ran on clean `7a1dc7b22`. Production-selection edits and the
native smoke were then validated in the working tree; `native-smoke.json`
explicitly records dirty source because it verifies the new binder.
`screen.json` records source/build hashes, compiler version, exact commands,
all timing samples and compact trace/resource evidence. No new external
kernel was ported; this is an in-tree specialization of the current parent.

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
unset HIPENGINE_REQUIRE_CACHED_BUILD
uv run --no-sync python -c 'from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import build_qwen4_exp_gdn; print(build_qwen4_exp_gdn(load=False))'
export HIPENGINE_REQUIRE_CACHED_BUILD=1
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_gdn_narrow.py \
  tests/test_unit_qwen4exp_gdn_narrow.py tests/test_gpu_qwen4exp_gdn_register.py \
  tests/test_gpu_qwen4exp_gdn_wave_norm.py
uv run --no-sync python scripts/qwen4exp_gdn_register_micro.py \
  --narrow-index --tokens 1 17 64 512 1024 4096 --repeats 21 \
  --output /tmp/qwen4exp-gdn-narrow-20261006/screen.json
rocprofv3 --kernel-trace --output-format csv --output-file narrow \
  --output-directory /tmp/qwen4exp-gdn-narrow-20261006/trace -- \
  uv run --no-sync python scripts/qwen4exp_gdn_register_micro.py \
    --narrow-index --tokens 512 4096 --repeats 3 \
    --output /tmp/qwen4exp-gdn-narrow-20261006/trace.json
```

The compiler-version file must be produced by the same installed `hipcc`
before cached-only profiling. Raw traces and code objects stay in `/tmp`;
[screen.json](screen.json) is the committed compact evidence.
