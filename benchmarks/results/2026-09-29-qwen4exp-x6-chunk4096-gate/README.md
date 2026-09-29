# X-6: the canonical numerical gate at chunk 4096 reproduces chunk 1024 bit-identically

X-6's requirement is *"Re-run the numerical gate at each new chunk size"*, and the existing
certification recorded `prefill_chunk_size: 1024` with scope *"this route/layer set,
manifest and chunk size only"*. RT-1's promotion of chunk 4096 is blocked on exactly this.

**Re-run at chunk 4096. It reproduces the chunk-1024 certification bit-identically, and the
chunk-4096 qualification now exists for this route.**

## The comparison

Same protocol (`canonical_12_case_d128_three_repeat_numerical_v1`), same route (`wmma`,
layers 16-47, `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` cleared in both arms), same 12 canonical
cases, same 128 decode steps, same 3 candidate repeats. **Only the chunk size differs.**

| field | chunk 1024 (cert) | chunk 4096 (this run) | same |
|---|---|---|---|
| `strict_logits_sha256` | `550bb9b832bc7ba2...` | `550bb9b832bc7ba2...` | **YES** |
| `candidate_logits_sha256` | `65370710c0b9c40f...` | `65370710c0b9c40f...` | **YES** |
| `quality.summary.rows` | 1548 | 1548 | YES |
| `kl_mean` | 3.795524059877289e-4 | 3.795524059877289e-4 | **YES** |
| `kl_p95` | 1.761932752783211e-3 | 1.761932752783211e-3 | **YES** |
| `kl_p99` | 5.424248305527542e-3 | 5.424248305527542e-3 | **YES** |
| `kl_max` | 1.5304399916084921e-2 | 1.5304399916084921e-2 | **YES** |
| `top1_agreement` | 0.9935400516795866 | 0.9935400516795866 | **YES** |
| `repeat_determinism` | 3 runs, passed | 3 runs, passed | YES |
| `trajectory_sha256_by_prompt` | 12 prompts | 12 prompts | **YES** |
| `status` | complete | complete | YES |
| `measurement_valid` | true | true | YES |
| `canonical_numerical_qualification_claim` | true | true | YES |
| **`prefill_chunk_size`** | **1024** | **4096** | *differs by design* |

**Every compared field is identical, including all five KL metrics to the last digit and all
12 trajectory hashes.** The only intended difference is the recorded chunk size.

## Why it is bit-identical, not merely "within tolerance"

This was predicted. The chunk-invariance packet
(`benchmarks/results/2026-09-29-qwen4exp-chunk-invariance-class/`) showed that chunks 1024,
2048 and 4096 publish **identical `logits_sha256` on all 12 canonical cases**. If the logits
are bit-identical, every KL metric computed from them must be identical too - and it is.
The gate run is therefore both a qualification and an independent confirmation of the
invariance finding at the gate level.

## What this does and does not establish

**Establishes:** the wmma 16-47 route's canonical numerical qualification holds at
**chunk 4096**, bit-identically to chunk 1024. X-6's "re-run at each new chunk size"
requirement is **satisfied for this route**.

**Does not establish:**

- `profile_qualification_claim` is still **false**. X-6's other certificates are untouched:
  category-heldout numerical evidence, strict-teacher and BF16-relative evidence, exact
  control/isolation/graph/lifecycle evidence, task-quality evidence, and candidate route
  engagement/repair-oracle coverage.
- **Other routes have their own gates.** This qualifies the wmma 16-47 dense-prefill route,
  not the engine's chunk-4096 default. A blanket chunk flip would need every affected route
  re-gated.
- **RT-3's blocker is separate and still open**: short-request costs historically kept 1024
  as the default, so the right form is a shape-based policy rather than a blanket flip.
- No performance claim. This run is a qualification, not a timing measurement.

## Limits

- One model, one quant, one host (`gfx1151`), one execution profile, one route/layer set.
- The equality is of the **hashed published logits** as the harness hashes them, plus the
  metrics derived from them; it is not a claim about intermediate tensors.
- A single run at chunk 4096, not a distribution. The determinism check covers 3 candidate
  repeats within this run.
- The bit-identical result is specific to the {1024, 2048, 4096} invariance class. **Chunk
  256 is outside that class** and its logits differ on all 12 cases, so this result must not
  be extrapolated to it.
