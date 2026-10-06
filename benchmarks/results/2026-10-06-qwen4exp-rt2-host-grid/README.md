# RT-2 charged host-grid screen

Host-sized launch grids do not improve this Q4/Q5_1 operation screen: 28 of
30 owner medians are slower and the sum of owner medians moves
557.760 → 559.006 ms (+0.223%, static/host ratio 0.99777x). Neither of the two
positive medians wins both order strata. Keep the current static-grid owners.

## Scope and correctness

- Physical host `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
  therock10 staging HIP 7.15.26333.
- Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV, named production profile,
  chunk4096, max sequence4352. Model, fixture and selected/strict manifest
  hashes are in [artifact.json](artifact.json).
- Six natural canonical inputs: code p512/p1024/p4096 plus English,
  Japanese and mixed p1024. Layers0/24/47 supply18 Q4 gate/up and12 Q5_1
  down owner cases. Each runs one warm pair and six measured pairs in
  alternating order,420 replay calls total.
- Both arms rebuild the device tile map and run the same risk counter reset,
  unchanged iu8 owner, exact repair and Q4 SiLU. Host sizing downloads the
  map's8-byte total inside synchronized wall/HIP-event timing. Allocation,
  validation copies and input preparation are outside timing.
- Repaired output, SiLU, risk count/membership and tile maps match at every
  repeat. Compact activation/start bytes are preserved and allocations return
  to their prior level after each replay. Final teardown has zero allocations
  and zero bytes. Six Q8 down observations already use the exact host-sized
  total, so no Q8 grid reduction is proposed.

| Diagnostic | Result |
| --- | ---: |
| Tile-grid reduction | 8.03–34.75% |
| Owner static/host wall ratio | 0.99176–1.00399x |
| Q4 sum of owner medians, static → host | 368.687 → 369.509 ms |
| Q5_1 sum of owner medians, static → host | 189.073 → 189.497 ms |
| Q5_1 layer0 code4K, static-first / host-first | 0.99756 / 1.00893x |
| Q4 layer47 code4K, static-first / host-first | 0.99995 / 1.00105x |

## Command

```bash
P=/home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$P/_rocm_sdk_devel/lib:$P/_rocm_sdk_core/lib:$P/_rocm_sdk_libraries/lib
export PATH=$P/_rocm_sdk_devel/bin:$P/_rocm_sdk_core/bin:$PATH
export HIPENGINE_HIP_ARCH=gfx1151
uv run python scripts/qwen4exp_rt2_host_grid.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --layer 0 --layer 24 --layer 47 --pairs 6 \
  --output /tmp/qwen4exp-rt2-six-case.json
```

The runtime/kernel tree was tracked-clean at `7c97170fc`; the diagnostic
script was untracked and its SHA-256 is recorded. Libraries are cached-only
and hashed. No kernel body, arithmetic, runtime default or model quantization
changed. The retained artifact includes every measured wall/event sample.

## Limits and reopening

This prices tile-map/reset/iu8-risk/repair/Q4 SiLU, not full MoE or engine
wall. Count/prefix/scatter-gather and router/down/combine are shared inputs
or excluded work. It does not test overlapping readback behind independent
shared-expert work. The summed medians are a diagnostic aggregate, not a
model prefill duration or speedup. No full production numerical/task
qualification follows from parent replay parity.

Reject the serialized exact-total readback candidate. Reopen RT-2 only for a
distinct overlapped-readback or device-sized dispatch mechanism with charged
complete-operation evidence and exact ownership. No runtime flag is added.
