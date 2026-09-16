# Wide-row Q8_0 dense prefill at layers 16-47: production numerical gate

The wide-row f16 dense Q8_0 prefill route (`dense_wide256_f32_f32_out`) was
gated against the exact coltile chain on the Qwen4Exp UD-Q4_K_XL canonical
exact-token fixture: 12 cases (4 categories at 512/1024/4096), 128 decode
transitions each, production profile, chunk 1024, three candidate repeat runs,
one warmup per arm after the selector flip.

| Measure | Value | Limit | Headroom |
| --- | ---: | ---: | ---: |
| Rows | 1548 | | |
| Mean KL | **3.80e-4** | 1e-3 | 2.6x |
| p95 KL | **1.76e-3** | 5e-3 | 2.8x |
| p99 KL | **5.42e-3** | 2e-2 | 3.7x |
| Max KL | **1.53e-2** | 5e-2 | 3.3x |
| Top-1 agreement | **1538/1548 = 0.99354** | 0.99 | |
| Verdict | **pass** | | |

`measurement_valid: true`, no qualification blockers, no scope failures, three
identical trajectory hashes, and all ten top-1 misses inside the 154-row
flip-eligible set with none outside it.

## The envelope is bit-identical to the certified WMMA route

This run reproduces
[`../2026-09-16-q8-wmma-dense-prefill-layers-gate/artifact-layers16-47-4cat.json`](../2026-09-16-q8-wmma-dense-prefill-layers-gate/artifact-layers16-47-4cat.json)
exactly, not approximately:

- `strict_logits_sha256` `550bb9b832bc7ba2…` and `candidate_logits_sha256`
  `65370710c0b9c40f…` match that artifact's values byte for byte.
- Every numerical field of the quality summary is identical. The only difference
  anywhere in the two artifacts is the route name embedded in `scenario_id`.

So the wide kernel is a retiling of the already-certified f16 arithmetic, and
the certified 16-47 envelope is inherited rather than re-derived. That claim was
a prediction from one case
([`../2026-09-17-q8-dense-route-ab/README.md`](../2026-09-17-q8-dense-route-ab/README.md));
it is now a 12-case, 1548-row result.

## Performance half

This gate records no timing (`timing_protocol: none_full_logits_only_v1`). The
route's measured end-to-end value is in the route A/B: on the `code` category,
median prefill wall 4.389 -> 4.151 s at 1K and 18.208 -> 17.217 s at 4K
(**-5.4%**) against the WMMA default, and 22.0/24.3/24.0% against the exact
chain, with the two dense routes indistinguishable at 512.

## Protocol and provenance

- Command: `scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route
  dense_wide --layers 16-47 --decode-steps 128 --repeat-runs 3
  --prefill-chunk-size 1024`
- Route surface: post-binder `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` +
  `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS=16..47`, with
  `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` **cleared in both arms** so the
  comparison carries the wide kernel's arithmetic only. Teacher: the exact
  coltile chain, `coltile8_rowbatch4_f32_f32_out`.
- Source `1f06cd5e3`, `execution_affecting_dirty: false`.
- Host: AMD Radeon 8060S Graphics (`gfx1151`), machine
  `55ea6c509d0b49eea8de7094a1023668`; production manifest
  `35e57360e648bcc8`, strict `1335b8244237ffb0`.

## What this does not say

It is not a task-quality or long-form-factual certificate, and it does not
certify any scope other than 16-47 or any quant other than `gguf_ud_q4_k_xl`.
`performance_claim` is false by construction: the artifact carries no timing.
