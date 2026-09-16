# Layer-scoped Q8_0 dense prefill route A/B

What the wide-row Q8_0 prefill kernel is worth on a real prefill, measured
against the two routes it can replace at layers 16-47.

**Status: route diagnostic, not a topline rate row.** The wide arm is opt-in
(`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=1`); nothing here is promoted. One
category, three prompt lengths, one host, prefill only: no decode, no MTP,
no other category, and no claim outside this protocol.

## What it provides

- Against the **current production default** (the certified f16 WMMA route at 16-47): **5.4% less prefill wall at 1K** (4.389 -> 4.151 s) and **5.4% less at 4K** (18.208 -> 17.217 s).
- At 512 prompt tokens the wide and default routes are indistinguishable here: 1.09% apart, inside the default arm's own 2.0% repetition spread.
- Against the **exact coltile chain** (the pre-2026-09-17 default), the wide route is **22.0% / 24.3% / 24.0%** faster at 512/1K/4K. The certified WMMA route it now takes precedence over
  already holds 21.1% / 20.0% / 19.6% of that, so the wide kernel is worth the difference between those two lines, not the whole gap.

## Prefill wall by arm

Median of three interleaved repetitions per arm; lower is better.

```
                   exact    wmma-16-47    wide-16-47   (current default)
code-p512         2.822s        2.225s        2.201s
code-p1024        5.487s        4.389s        4.151s
code-p4096       22.648s       18.208s       17.217s

code-p512  512 prompt tokens
  exact chain   2.822s  ██████████████████████████████████████████████
  wmma-16-47     2.225s  ████████████████████████████████████
  wide-16-47     2.201s  ████████████████████████████████████

code-p1024  1024 prompt tokens
  exact chain   5.487s  ██████████████████████████████████████████████
  wmma-16-47     4.389s  █████████████████████████████████████
  wide-16-47     4.151s  ███████████████████████████████████

code-p4096  4096 prompt tokens
  exact chain  22.648s  ██████████████████████████████████████████████
  wmma-16-47    18.208s  █████████████████████████████████████
  wide-16-47    17.217s  ███████████████████████████████████

```

## Deltas

| Case | exact | wmma-16-47 (default) | wide-16-47 | wide vs default | wide vs exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 2.822 s | 2.225 s | 2.201 s | **-1.09%** (0.024 s) | **-22.00%** (0.621 s) |
| code-p1024 | 5.487 s | 4.389 s | 4.151 s | **-5.41%** (0.237 s) | **-24.35%** (1.336 s) |
| code-p4096 | 22.648 s | 18.208 s | 17.217 s | **-5.44%** (0.991 s) | **-23.98%** (5.430 s) |

Prompt-processing rate implied by those walls (prompt tokens / median wall,
this protocol only - not comparable to any other harness's PP column):

| Case | exact | wmma-16-47 | wide-16-47 |
| --- | ---: | ---: | ---: |
| code-p512 | 181.4 | 230.1 | 232.6 |
| code-p1024 | 186.6 | 233.3 | 246.7 |
| code-p4096 | 180.9 | 224.9 | 237.9 |

## Reproducibility

A second process, same host, same command, reproduced every ratio:

| Case | wide vs default, run 2 | wide vs default, run 1 | wide vs exact, run 2 | wide vs exact, run 1 |
| --- | ---: | ---: | ---: | ---: |
| code-p512 | -1.09% | -1.17% | -22.00% | -22.00% |
| code-p1024 | -5.41% | -5.51% | -24.35% | -24.28% |
| code-p4096 | -5.44% | -5.42% | -23.98% | -23.95% |

Per-arm repetition spread (max-min over median) stays at or below
2.01%, so every delta above is
outside its own arm's repeat spread except the 512-token wide-vs-default gap.

## What the run proves about the route, not just the clock

The three arms ran the same 7191 launches over the same roles and shapes;
only the family serving the layer-16-47 dense roles changes:

| Arm | Dense Q8_0 prefill family at 16-47 | Exact coltile chain | Other | Total | Distinct shapes |
| --- | --- | ---: | ---: | ---: | ---: |
| exact chain (all layers) | — | 7164 | 27 | 7191 | 21 |
| wmma-16-47 | `gguf_q8_0_wmma_prefill_f32_f32_out` x4752 | 2412 | 27 | 7191 | 37 |
| wide-16-47 | `gguf_q8_0_dense_wide256_f32_f32_out` x4752 | 2412 | 27 | 7191 | 37 |

The substitution accounts for itself: the exact chain's launch count falls
7164 -> 2412, and the 4752 launches that disappear are exactly the 4752 the wide
family takes over. The remaining 2412 exact-chain launches and the 27
`pack8_gemv` launches (decode-shaped work inside the prefill pass) are identical
in every arm, so nothing else in the model moved.

Arithmetic, in the same runs:

- The wide arm is **bit-identical to the certified WMMA arm** on all three
  cases, in both runs: `logits_sha256` matches per case, not per class.
- Both differ from the exact chain, as expected: the WMMA and wide kernels
  are f16-operand arithmetic and the exact chain is not.
- Every arm and both runs sample the same token (`248068`) on all three cases.
- Per-case digests are identical across the two processes, so the comparison
  is deterministic on this host.

## Protocol

- Command: `qwen4exp_dense_route_ab.py --repetitions 3 --output <scratch>/route-ab/run2.json` (recorded verbatim in `artifact.json`)
- Host: AMD Radeon 8060S Graphics (`gfx1151`), host name `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`, HIP version: 7.15.26333-0000000
- Model: `/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`, quant `gguf_ud_q4_k_xl`, KV `bf16`, fingerprint `fb1f2fbf73d588c9` (111334654784 bytes, 4 files)
- Source: `7e8247a657c386f3382f4418f022875355e7d2c9` on `qwen3.8-flash-next`; tracked tree clean
  (the two untracked paths at run time were this harness and an unrelated
  `docs/superpowers/` directory)
- Production profile manifest `35e57360e648bcc8` (strict `1335b8244237ffb0`), `fell_back_to_strict=False`
- Fixture: `qwen4exp_canonical_ar_p512_p1024_p4096.json`, sha256 `42b562bd8e9644be`
- round-robin, arm order rotated per repetition; 1 warmup per arm per case after its selector flip; chunk 1024
- Statistic: median of per-repetition prefill wall; timer perf_counter around runner.prefill + device_synchronize
- Selectors are flipped **post-binder** in one process with one model load, so
  the three arms share residency, clock state and allocator history.

Arms:

| Arm | `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` | `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` | `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS` |
| --- | --- | --- | --- |
| exact | `(unset)` | `0` | `(unset)` |
| wmma-16-47 | `16..47 (32 layers)` | `0` | `(unset)` |
| wide-16-47 | `16..47 (32 layers)` | `1` | `16..47 (32 layers)` |

The literal selector strings are recorded in `artifact.json` under
`protocol.arms`.

## Limits

- One category (`code`) at three prompt lengths. Four-category weighting is not
  measured here; the route swap is per-shape work, but this protocol does not
  establish the other categories' deltas.
- Prefill only. Decode is untouched by the swap (the Q8 dense prefill selector
  is prefill-scoped), and MTP is not measured.
- The wide arm's 12-case production numerical gate has not been run. Its
  bit-identity with the WMMA arm here is evidence that it inherits that arm's
  certified arithmetic on these three cases; it is not a substitute for the
  12-case envelope.
- The kernel-level packet result (2.570 ms against the WMMA route's 4.401 ms on
  `layers.8.attn_qkv`, rows=1024) is a single-shape measurement. The end-to-end
  saving here is smaller because the layer-16-47 dense roles are only part of a
  prefill; that share is inferred from these two measurements, not profiled.
- Same physical host as the campaign's other `gfx1151` rows (machine id
  `55ea6c509d0b49eea8de7094a1023668`); the deltas are a same-host, same-session
  comparison, not a cross-host rate.

## Files

- `artifact.json` - the run whose numbers are tabulated above, with provenance.
- `repeat-run1.json` - the first process, same command, used for the
  reproducibility table.

## Reproduction

```bash
uv run python scripts/qwen4exp_dense_route_ab.py --repetitions 3 \
  --output benchmarks/results/<dir>/artifact.json
```
