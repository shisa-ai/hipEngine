# Qwen4Exp MTP context admission

Measured on the Framework gfx1151 host (`55ea6c509d0b49eea8de7094a1023668`) with Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV, production profile, and the Q8_0 MTP sidecar.

## Full category suite

The 10-prompt `mtpbench-code-general-ja.jsonl` suite ran for 64 generated tokens at candidate budget 3. True AR and MTP were measured in one resident process with one warmup per arm and batched target verification.

| Arm | tok/s | MTP/AR | acceptance | identical prompts |
| --- | ---: | ---: | ---: | ---: |
| true AR | 22.63 | — | — | — |
| Qwen4Exp MTP | 28.93 | **1.279x** | 56.1% | 8/10 |

Category speedups were code 1.558x, general_en 1.246x, general_ja 1.172x, and mixed_ja_en 1.035x. The non-identical rows are diagnostic verifier arithmetic differences; the provider completed all rows with deterministic transaction handling.

## Boundary-crossing context

A 3,433-token prompt was run with `max_sequence_length=4096`, budget 3, and 16 generated tokens through the public `Qwen4ExpGGUFTextGenerator.generate_speculative_mtp_detailed` API after provider attachment. MTP accepted all 11 proposed draft tokens, matched AR for all 16 output tokens, and completed four verification cycles. It measured 1.893 tok/s MTP versus 4.939 tok/s AR (`0.383x`), so larger context support is functional but not a performance win at this prompt length.

The long-context result is a capability/rollback validation, not a retained speedup claim.

Artifacts: `full-suite.json`, `long-context.json`, `long-context-public.json`.
