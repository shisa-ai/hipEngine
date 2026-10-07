# MT-1 verifier-body boundary: the rows>1 prefill body is not c1-exact

MT-1's remaining work was recorded as "composing the q8_0 rowbatch into a verifier
body pass and measuring the COMPLETE pass". This packet composes the rows>1 body
and measures it against the serial c1 decode oracle, on synthetic layer fixtures
and on the real target. The result is a **concrete blocker**: the rows>1
production dispatch does not reproduce the c1 arithmetic per row, so the composed
body is not a width-invariant verifier.

## The composition

The strict decode QSA mixer is rows==1-only
(`run_qwen4_exp_dense_qsa_token_mixer` rejects widths 2..8). The rows>1 body that
exists is the one the prefill chunk uses: `run_qwen4_exp_qsa_prefill_layer` (QSA),
`run_qwen4_exp_gdn_layer` (GDN, shared with decode) and `run_qwen4_exp_ple`.
`scripts/qwen4exp_mt1_batched_body_probe.py` composes those into a non-resetting
verifier body and scores all rows with one batched head, then compares row by row
against `verify_target_block_serial_exact`.

## Synthetic layer fixtures: bit-exact

`scripts/qwen4exp_mt1_verifier_body_probe.py`, one measurement per process:

| primitive | rows | result |
|---|---:|---|
| strict decode QSA mixer | 2..8 | **rejects every width** (`requires rows == 1`) |
| rows>1 QSA attention (`qsa_prefill_token_mixer`) | 6 | output, KV cache, index raw keys **bit-exact** vs serial |
| full QSA layer (`qsa_prefill_layer`) | 4 | output, KV key/value **bit-exact** vs serial |
| GDN recurrent layer (`gdn_layer`) | 3 | output, conv state, recurrent state **bit-exact** vs serial |
| MoE | 3 | bit-exact vs serial (`test_qwen4_exp_runner_moe_matches_reduced_topk_shared_cpu_oracle[3-False]`) |
| q8_0 dense rowbatch | 2..8 | bit-exact width-invariant (`test_q8_0_coltile8_rowbatch4_is_width_invariant_to_serial_c1`) |

So the **layer structure** is not the blocker, and the q8_0 rowbatch is not the
whole story.

## Real target: not c1-exact

`scripts/qwen4exp_mt1_batched_body_probe.py` on Qwen3.8-Flash-Next UD-Q4_K_XL,
production profile, toy token chain `1..rows`:

| rows | top-1 agreement | hidden max abs delta | logits max abs delta | row-0 hidden delta |
|---:|---:|---:|---:|---:|
| 2 | 1.0 | 0.3125 | 1.0818 | 0.1055 |
| 4 | 1.0 | 0.3125 | 1.0818 | 0.1055 |
| 6 | 1.0 | 0.5176 | 1.0818 | 0.1055 |
| 8 | 1.0 | 0.5176 | 1.0818 | 0.1055 |

Top-1 matches on this toy chain, but the hidden rows and logits diverge, and
**row 0 already diverges** (hidden 0.1055, logits 0.5148). Row 0 of a batched body
should equal one serial step; it does not, which localizes the divergence to the
**rows>1 production dispatch**, not to cross-row accumulation. The prefill chunk
wraps the rows>1 pass in `raw_k_prefill_rowbatch_session`, the q8 MMQ prefill
context and the wide-F16 activation context, so the model's non-q8_0 dense
projections (Q4_K/Q5_K/Q6_K) and the batched attention/MoE run prefill-shaped
kernels whose per-output reduction differs from the c1 decode GEMV. The q8_0
rowbatch is width-invariant; the other quant paths are not.

## Disposition

- The composed body is **diagnostic only**; it is not in the engine and not wired
  into the provider. The provider still runs the serial verifier.
- MT-1's remaining unit is therefore **not wiring but arithmetic**: the rows>1
  dense/MoE/attention dispatch must be made to reproduce the c1 per-output
  reduction per row, at every width 2..8, before a batched body can carry the
  exact accepted prefix.
- The `verify_target_block_batched_body` method was prototyped, measured, and
  **removed** from `hipengine/runtime/qwen4_exp_runner.py`: shipping a verifier
  body that returns divergent hidden/logits is a correctness footgun.

## Limits

- The probe runs each synthetic primitive in its own process. A combined run
  reached the second measurement and aborted with `free(): invalid size` in
  teardown; whether that is a probe-setup bug or a real kernel heap issue is
  **not established** and is recorded rather than hidden.
- The real-target comparison uses one toy token chain (`1..rows`) on one model,
  one quant, one host. Top-1 agreement is not the MT-1 gate; it is reported only
  to show the divergence is in the logits/state, not (here) the argmax.
- No end-to-end MTP speedup is claimed. The composition is not promoted.
