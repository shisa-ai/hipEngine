# Dense Q8 checked/full-row natural and model-state qualification

The separately registered checked-int32/full-four-row Q8 DPP sibling passed the
natural-input and model-state gates on Radeon 8060S/gfx1151. It remains
**diagnostic-only**: no production selector, profile manifest or default was
changed, and a complete-engine A/B is still required before promotion.

## Measurement identity

- Host: Radeon 8060S, gfx1151, machine-id `55ea6c509d0b49eea8de7094a1023668`
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL, model fingerprint
  `fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53`
- Production profile manifest: `be32e7193f30c4c4a65f5cbaf99b62033fd6458edd295fc2d7806e734001469b`
- Strict manifest: `1fcbf992271251220cc4707eb8d24d28506cd3714550756cf2e32f387d879670`
- Fixture: `42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`
- Tracked-clean source: `6b3f5253e` for natural owner timing and `40365a64c` for the
  state gate; the state harness itself is committed at `40365a64c`.

## Natural-input complete-owner screen

The diagnostic hook replayed the current DPP parent and candidate on identical
live model activations and resident raw Q8 weights. It covered six canonical
cases: `code-p512`, `code-p1024`, `code-p4096`, and category-balanced
`general_en-p4096`, `general_ja-p4096`, `mixed_ja_en-p4096`. Six declared dense
geometries were engaged at each row size: K/N 2560/10240, 2560/6144,
6144/2560, 2560/640, 640/2560 and 10240/320.

There were **594 candidate captures** (99 per case), four counterbalanced
synchronized HIP-event samples per arm per capture, and 60 out-of-scope calls
that correctly stayed on the parent.

| Case | Candidate/parent complete-owner ratio |
| --- | ---: |
| code-p512 | 1.454x |
| code-p1024 | 1.458x |
| code-p4096 | 1.466x |
| general_en-p4096 | 1.466x |
| general_ja-p4096 | 1.467x |
| mixed_ja_en-p4096 | 1.466x |

The arithmetic and geometric means are both approximately **1.463x**. Every
capture compares the complete output tensor bit-for-bit, verifies finite
values, and confirms that the natural activation and resident raw-weight bytes
are unchanged. Teardown reports zero active allocations and zero current
bytes.

This is complete-owner operation timing, not request throughput. The source
artifact is [`natural-owner.json`](natural-owner.json), SHA256
`24831f141d04c4669f0e45b9bc17e8cbfc3f48335434966069ebfcbe2f51eb54`.

## Full model logits/state/KV gate

The candidate was published through a diagnostic registry substitution only for
the six declared Q8 geometries and rows 512/1024/4096. Each of 12 canonical
category/shape cases ran parent/candidate/candidate/candidate/parent with 16
autoregressive decode steps and full KV hashing.

- **3,564 candidate calls**, 297 per case (99 in each candidate arm), and
  2,976 parent calls.
- Full prefill logits exact across all five arms.
- All 16-step decode-logit arrays exact across all five arms.
- Recurrent state and layout summaries exact.
- Full key/value cache digests exact.
- Token IDs and candidate repeats exact.
- Finite state and zero active allocations/current bytes after close.

The full-state artifact is [`full-state.json`](full-state.json), SHA256
`19837f1edd1786c153ca0d3d745772a68382c144a3f196a06c13ff0db3ac98ee`.

## Decision and next gate

Natural-input owner timing and complete model-state ownership are qualified for
the declared diagnostic domain. The parent remains the strict fallback for
unsupported shapes, incomplete row groups, unsafe byte ranges and all normal
production dispatch. The candidate is not promoted because the evidence above
charges individual projection owners and state correctness, not complete client
request wall time. A tracked-clean same-residency engine A/B over the canonical
suite is the next gate; only a non-regressive complete request result can
support a bounded production decision.

## Reproduce

```bash
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
uv run --no-sync python scripts/qwen4exp_dense_dpp_checked_full_rows_natural_owner.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --repetitions 4 \
  --output /tmp/qwen4exp-dense-dpp-checked-natural-canonical-20261007.json

uv run --no-sync python scripts/qwen4exp_dense_dpp_checked_full_rows_state_gate.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --decode-steps 16 --full-kv \
  --output /tmp/qwen4exp-dense-dpp-checked-state-canonical-20261007.json
```
