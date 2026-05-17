# Lessons Learned

This file records hipEngine-specific debugging lessons that are likely to recur.
Keep entries compact, evidence-backed, and actionable. Parent-workspace kernel
R&D notes still belong in `~/amd-gpu-tuning/LESSONS-LEARNED.md`; this file is
for issues observed while integrating stable kernels into hipEngine runtime,
state, and gates.

## 2026-05-15 — Native prefill flakiness can hide in full-attention prefill softmax

### Symptom

After native compact/single-request prefill was enabled and grouped MoE library
loading was fixed, the parent 512/32 fixture became repeat-flaky:

- some runs matched serial resident prefill + decode;
- failing runs diverged after several decode tokens, often producing
  `[1739, 220, 16, 15, 15, 4, 220, 16, ...]` instead of the expected
  `[1739, 220, 16, 15, 15, 15, 15, 15, ...]`;
- failing full-logit gates showed `max_kl≈8.6-9.0` and top-1 agreement around
  `0.485`;
- `HIP_LAUNCH_BLOCKING=1` did not eliminate the flake.

### What did *not* cause it

Targeted probes ruled out several tempting explanations:

- session close/free ordering after removing accidental compiler delays;
- grouped MoE preload/on-demand behavior;
- c=1 MoE vs grouped compact MoE;
- linear-attention state update;
- full-attention KV append content;
- decode state after prefill.

### Localization method

Use targeted, state-family probes rather than guessing:

1. Bisect by `max_layers`; the first pass/fail hidden divergence appeared at
   layer 3, the first full-attention layer.
2. Compare pass/fail runs at that layer. Hidden input, Q/K/V/gate tensors, and
   appended BF16 KV cache were identical.
3. Re-launch `prefill_full_attention_gqa_gate_fp16` twice on identical inputs in
   the same session. The old wrapper produced different `gated_attn` outputs
   (`repeat max abs` roughly `0.05-0.39`).

That localized the nondeterminism to the full-attention prefill softmax kernel
launch, not to runtime state or MoE.

### Fix

`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` now launches
single-request and varlen prefill GQA gate kernels with a 64-thread block instead
of the old 256-thread block. The wrapper also allocates shared scratch as:

```cpp
max_context_len + threads
```

rather than `max_context_len * 2`, because the kernel needs one score slot per
context token plus one reduction slot per thread. The old formula could
under-allocate short varlen/compact rows.

Commit: `4f252cf kernel: stabilize native prefill attention`.

### Validation evidence

Commands run for the retained fix:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py scripts/qwen35_native_prefill_fixture_gate.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
for i in $(seq 1 5); do python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/fixture-final-det-$i.json; done
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/packed-det-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/prefill-det-final-512.json
```

Results:

- native fixture gate passed 5/5 repeats;
- max KL stayed around `0.00553-0.00570`;
- top-1 agreement was `100%`;
- compact prompt8 gates still passed for c=2/4/8;
- 512/128 prefill measured `479.755 tok/s`, essentially flat vs the post-preload
  `482.057 tok/s` baseline.

### Checklist for similar bugs

When a native prefill correctness failure is flaky rather than consistently
wrong:

- compare repeated native runs, not only native vs serial;
- checkpoint final-row hidden after each layer to find the first divergent layer;
- at that layer, separately compare layer input, projected Q/K/V/gate, KV cache,
  attention output, MoE input, and MoE output;
- re-launch suspect kernels on identical inputs in the same session;
- verify shared/LDS sizing against both long rows and short varlen/compact rows;
- do not retain throughput improvements until the repeat fixture gate is stable.
