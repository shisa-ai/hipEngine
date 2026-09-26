# Qwen4Exp bounded F16 activation staging

On 2026-09-26, F16-input staging improved all 12 canonical prefill cases by
3.28–3.70% against the F32-input production owner. Both arms use the same
wide-route layer scope, 16–47. The runner now enables staging by default;
`HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT=0` restores the F32-input owner.

## Protocol and results

- Physical host: `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`, AMD gfx1151.
- Model: Qwen3.8 Flash Next / `qwen4_exp_gguf`, UD-Q4_K_XL; BF16 KV.
- Source measured: `daad2dbacdcc0dc9beaf94132a8b10f39ab10932`, clean tracked tree.
- Warm PLE; 1024-token chunks; four categories (`code`, `general_en`,
  `general_ja`, `mixed_ja_en`) at 512, 1024 and 4096 tokens.
- One resident model; five route arms; one warmup per arm/case; three measured
  repeats with rotated arm order. Wall time includes prefill and synchronization.
- Compare `wide-f32in` with `wide-f16in` in [ab.json](ab.json), **not** with
  its historical diagnostic `exact` arm. Rates below average the four per-case
  median rates, rather than pooling tokens/time.

| Tokens | F32 input tok/s | F16 input tok/s | Improvement |
| --- | ---: | ---: | ---: |
| 512 | 235.5036 | 243.4926 | 3.3923% |
| 1024 | 250.2970 | 259.2435 | 3.5743% |
| 4096 | 241.4542 | 249.6595 | 3.3983% |

All 12 final-logit hashes match. The standalone code-p512 census changed all
792 wide launches from `hipengine_gguf_q8_0_dense_wide256_f32_f32_out` to
`hipengine_gguf_q8_0_dense_wide256_f16in_f32_f32_out` over three repeats;
total recorded quantized matmul launches stayed 1203. This census does not
count every kernel (including casts). The full A/B artifact records engagement
across all cases.

## Correctness and lifetime

Seven GPU hoist fixture tests pass. [state.json](state.json) records a separate
production-parent comparison: each of 12 cases runs F32 input, three F16-input
repetitions, then F32 input again, with 64 freely generated decode steps each.
All 60 trajectories match full logits, token IDs, recurrent state, full KV
storage and live index state at prefill and decode completion. Staging survives
reset without reallocation; runner close releases it and leaves zero active
allocations. The artifact replaces detailed state payloads with their SHA256
and preserves the equality verdicts; the [executed script](state-gate.py)
reproduces the unabridged report. Peak allocated device bytes: 84,440,816,904.

This is an arithmetic-preserving change relative to the **current production
parent**. It does not close the broader strict-teacher, BF16-relative, heldout
or task-quality gaps documented in `docs/QWEN4EXP-STATUS.md`. No decode or MTP
performance claim is made. Larger chunks, context depths, and other hardware
need their own measurements. The performance harness stores only the final
repeat's logit hash; repeated equality is established by the separate state run.

## Reproduction

The local `/tmp/run-gpu.sh` wrapper selects the installed gfx1151 ROCm Python
and build environment. Its content is captured in [gpu-wrapper.sh](gpu-wrapper.sh).
Do not interpret the wrapper as a portable installation procedure. The A/B
command is stored in `ab.json.command`; run it with
`HIPENGINE_QWEN4_EXP_PLE_WARM=1` **inside** the wrapper's Python process because
the wrapper resets some environment settings. Enumerate all 12 `--case-id`
arguments recorded in that command. The harness sets its census recorder.

Executed A/B launcher:

```python
import os, sys, runpy, json
os.environ['HIPENGINE_QWEN4_EXP_PLE_WARM'] = '1'
cases = json.load(open('benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json'))['cases']
sys.argv = ['scripts/qwen4exp_dense_route_ab.py', '--model-root',
 '/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL',
 '--repetitions', '3', '--warmups', '1', '--prefill-chunk-size', '1024',
 '--output', '/tmp/qwen4exp-f16-integration-20260926/route-ab-12cases.json']
for case in cases:
    sys.argv.extend(['--case-id', case['id']])
runpy.run_path(sys.argv[0], run_name='__main__')
```

For state reproduction, create `/tmp/qwen4exp-f16-integration-20260926`, copy
`state-gate.py` there, then run:

```sh
/tmp/run-gpu.sh /tmp/qwen4exp-f16-integration-20260926/state-gate.py --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
/tmp/run-gpu.sh -m pytest -o addopts= -q tests/test_gpu_qwen4exp_dense_wide_f16_activation_hoist.py
```

The diagnostic script has fixed model/output paths; its trailing model argument
also makes its workload visible to the other worktree's process guard. Ensure
exclusive GPU use before running either protocol. The prior DMS supervisor had
exited before these runs; the subsequent guard process waited without allocating
GPU memory. The first standalone census attempt omitted
`HIPENGINE_KERNEL_CENSUS=1`; its empty census is not engagement evidence.
