# Bounded padded HC-down production route

On 2026-10-06 UTC, Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`, measured the padded dense-wide F16-input
Q8_0 HC-down leaf on Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV, chunk4096.
The production profile selects it only at K10240/N320 and 512–4096 rows,
within the existing dense-wide layers 13–47 scope. Decode, strict, Q4_K_M,
peer backends, other shapes and an absent/undersized staging workspace do
not acquire this leaf.

## Same-host engine comparison

Tracked-clean source `1d9601d1b` ran one process and one model residency,
all 12 canonical cases, one warmup per arm and three counterbalanced
measurements per arm. Only the qualified parent leaf was substituted in
the candidate arm; named arithmetic and chunk were otherwise unchanged.
Each measured request charges synchronized prefill with first output plus
128 decode transitions and the final device synchronization. Model loading,
construction and final generator close are outside request timing; close is
separately checked for leaks.

| Prompt tokens | Parent prefill tok/s | Padded prefill tok/s | Change | Decode change |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 291.581 | 294.602 | +1.036% | +0.018% |
| 1024 | 317.528 | 319.415 | +0.594% | −0.089% |
| 4096 | 324.275 | 325.719 | +0.445% | −0.200% |

Prefill improves in all 12 cases and both order strata (1.00502x/1.00560x).
Total measured client wall is 493.851329→493.049429 seconds (−0.162%).
Ten request cases improve; Japanese p4096 is 0.214% slower and mixed p512
is 0.279% slower. There is no all-case request-wall or decode-speed claim.
This small request-level result is supported by the independently qualified
natural complete-GEMM saving of 1.312–1.361x across six cases/420 calls,
with F16 tail private scratch 92→0 bytes/lane and spills 58→0.

All 72 measured output trajectories match exactly and are deterministic.
Each candidate arm records 70 admitted calls (2,520 measured calls total),
with 0 in parent arms. Measured request I/O and page-fault deltas are zero;
teardown reports 0 active allocations and 0 current bytes.

Exact command (ROCm environment and cached compiler identity are prerequisites):

```bash
uv run --no-sync python scripts/qwen4exp_halo_box_campaign_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --prefill-chunk-size 4096 --warmups-per-mode 1 --repetitions-per-mode 3 \
  --route-package dense-wide-padded \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --require-cached-build --output /tmp/qwen4exp-dense-tail-20261006/engine-ab.json
```

[`engine-ab.json`](engine-ab.json) preserves every measured sample, identity,
command, timing and output digest. Untimed warmup payloads, duplicate arm
aggregates and per-sample output-ID arrays are omitted; the raw capture's
SHA256 is recorded. The compact summary is [`artifact.json`](artifact.json).

## Full-model state qualification

The gate covers 12 canonical cases and 8 category-heldout fixtures, each
with parent/candidate/candidate/candidate/parent arms. The kernel is
unchanged from the operation screen; the production wiring adds only
bounded admission after successful F16 staging.

| Gate | Result |
| --- | --- |
| Cases | 20/20 |
| Candidate calls | 4,200; 70 per enabled arm/case |
| Candidate decode calls | 0 |
| Full prefill logits | Exact across all arms |
| Post-decode logits/state | Exact across 16 transitions |
| Prefill state/layout and full KV hashes | Exact across all arms |
| Output token IDs | Exact across all arms |
| Teardown | 0 active allocations and 0 current bytes |

Short heldouts repeat/truncate to 512 token IDs only to exercise state and
ownership. They are not task-quality scores. Fixture/suite hashes,
manifest identities, commands and per-arm digests are in
[`full-state.json`](full-state.json). Source capture `9c668e702` was dirty
while the harness was being committed in `035338ed6`; the original source
record is preserved, not rewritten. State evidence is separate from the
tracked-clean performance capture.

## Selection and rollback

The named UD production manifest declares the bounded padded leaf with
`coltile8_rowbatch4_f32_f32_out` as registered strict fallback. The runtime
consults gfx1151 registry-axis capability metadata only after the existing
F16 workspace fit check. Missing candidate registration keeps the F16
parent. F32-input padded siblings remain diagnostic, not selected.

Before profile-plan registration, set
`HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT=dense_wide256_f16in_f32_f32_out`
to create a parent rollback manifest. The binder publishes its manifest's
value and overwrites conflicting pre-binder values; diagnostic overrides
must be applied after binding. Strict disables the existing dense-wide route.
The rollback has a distinct manifest hash.

[`native-smoke.json`](native-smoke.json) passes five cases with
native/parent/native/parent arms: code512/1024/4096, Japanese512 and a
Japanese category-heldout state fixture. Each native arm records 70 padded
calls (700 total), each parent arm 70 parent calls, and all arms record 0
prefill-leaf calls during decode. Full logits, state, full KV and output IDs
are exact; teardown is clean. Wrappers only count actual dispatch and never
substitute a leaf. Production manifest
`a3baefbeb6237a4a24b4751c8f46f5c0cfa4c5ef166a289eb7a5fbe124dfdcb0`
and strict manifest
`13a2117a83e113b2a9afc18ba1f8c762b40b802c74b6b2ab00f7a4b6a2253d27`
are persisted in the native artifact. The run has an honest dirty-source
record plus source-diff/script hashes because wiring was not yet committed.

[Operation screen, CPU-reference fixtures and cached trace](../2026-10-06-qwen4exp-dense-wide-padded-screen/README.md).
