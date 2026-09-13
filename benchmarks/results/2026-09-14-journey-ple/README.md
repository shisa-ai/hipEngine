# PLE Mapping Advice: Cold-Cache Improvement

Measured September 14, 2026 JST on Framework `gfx1151`, machine ID
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/40CU.
Qwen3.8-Flash-Next UD-Q4_K_XL, unchanged four-shard fingerprint, BF16 KV,
chunk1024, one request, 128 post-first-output AR transitions.

The gfx1151 production profile now applies mapping-only `MADV_RANDOM` to
the sparse PLE table. This prevents sequential readahead on sparse lookups.
It does not change row values, arithmetic, quantization or selected kernels.
Strict uses normal advice; `HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS=normal`
is the diagnostic rollback. Cold remaps preserve the policy, and explicit
full-table warming temporarily uses sequential advice.

## Complete-Model Cold Results

Same-residency before/after, all four categories at each length, one warmup
per arm/case and three counterbalanced pairs. PLE file range only is evicted
before each sample; no global cache flush.

| Shape | Normal PP | Random PP | PP ratio | Normal TG | Random TG | TG ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 84.161 | 260.484 | 3.095x | 13.388 | 19.560 | 1.461x |
| 1K/128 | 134.330 | 295.260 | 2.198x | 13.053 | 18.659 | 1.429x |
| 4K/128 | 225.864 | 290.598 | 1.287x | 13.544 | 18.643 | 1.376x |

Rates are token/time-weighted tok/s. All 12 cases improve in both phases.
All 72 measured samples match generated IDs, final logits and state
fingerprints; teardown reports zero tracked bytes and allocations.

## Warm Controls And Mechanism

The full warm matrix also passes all72 samples exactly. Mean paired PP ratio
is0.999191 (approximate95% interval0.997588..1.000794): no warm speed claim.
A small apparent Japanese4K regression (-0.22%) was not reproduced by five
fresh pairs:1.000004, interval0.998342..1.001665, exact outputs/state.
Both runs remain in the artifact; late slowdown in both arms is not erased.

The isolated cold code-p512 gather/staging screen measured5.476526s normal
versus0.266083s random, exact, with process reads8,134,029,312 versus10,055,680
bytes per arm. This is CPU-owner/process-I/O evidence, not a request rate or
GPU bandwidth counter. Warm all-unique and repeated-row controls are neutral.

The incumbent production numerical failure documented in
[the baseline packet](../2026-09-14-journey-production-baseline/README.md)
is unchanged. This exact-preserving I/O improvement does not certify that
profile's strict-teacher envelope. No threshold was changed.

## Reproduce

Use the existing TheRock runtime/library paths recorded in the baseline
packet, `GPU_MAX_HW_QUEUES=2`, `HIPENGINE_HIP_ARCH=gfx1151`:

```bash
PYTHONPATH=. .venv/bin/python scripts/qwen4exp_ple_gather_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
  --method mmap_random --cache-mode cold --output /tmp/ple-cold.json
```

Repeat with `--cache-mode warm` for the warm control. The targeted replication
adds `--pairs 5 --case-id general_ja-p4096`. Each harness holds the physical
host's advisory benchmark lock and fails on output/state mismatch.

`artifact.json` includes all six input hashes, commands, source/script
identity, host/model identity, paired samples and scope labels. Its assembler
checks completed status, paired cells, exactness and teardown. No other PLE
algorithm (deduplication, pread, persistent workers or overlap) is promoted.
