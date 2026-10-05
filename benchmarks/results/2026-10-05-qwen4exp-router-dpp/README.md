# Ordered router DPP operation screen

On Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`, Qwen3.8-Flash-Next UD-Q4_K_XL
real F32 router weights at layers 0/22/47 and synthetic F32 activations,
the tile4 DPP candidate improves complete logits-plus-top10-selection wall
in all nine cases. Twenty counterbalanced pairs per case:

| Rows | Parent ms | DPP ms | Wall speedup |
| ---: | ---: | ---: | ---: |
| 512 | 1.558–1.561 | 1.335–1.336 | 1.167–1.169x |
| 1024 | 3.062–3.078 | 2.615–2.626 | 1.167–1.172x |
| 4096 | 12.970–12.974 | 11.410–11.423 | 1.135–1.137x |

Both first-arm subsets improve for every case. Logits, selected expert IDs
and routing weights match the parent bit-for-bit. Zero allocations/bytes
remain after close. These are operation results, not model throughput.
Natural-input, full-canonical model state/KV/trajectory and whole-request
A/B gates have not run. No runtime default or profile manifest changes.

## Implementation and correctness

`f32_hidden_token_tile4_dpp_exact` preserves the parent's 256-thread FMA
ownership and reduction association. After one LDS publication barrier,
wave 0 folds strides 128/64/32 in registers and finishes 16/8/4/2/1 through
`permlanex16`/DPP instructions. The strict tile4 parent remains registered.

This is not the removed September 7 shuffle-tail candidate. That candidate
retained two inter-wave reduction barriers and finished 32..1 with a shuffle
sequence; it passed exactness but failed a consistent model-wall gain.
The September 27 warp-per-token candidate changed association and failed
the production envelope. Neither is restored unchanged.

The new RED initially fails because the wrapper/key is absent. Thirty-two
GPU tests then pass, including the existing router regressions. The new
fixtures cover tokens 1/3/5/7/17/513/1024/4096, K 1/255/2047/2048/2560/4097,
non-power-of-two expert counts, duplicated expert ties, cancellation and zero
inputs, repeat raw-bit logits/route weights and exact expert IDs. CPU-reference
`linear` comparison meets KL≤0.05 and top1≥90% on random fixtures.
A 10,000-vector host simulation confirms the register-fold tree's output bits
and demonstrates that an alternate wave-sum association would differ.

Fifty-four focused CPU/registry/profile/reference tests pass. An expanded
bundle has 109 passes and the known unrelated
`test_gfx1151_backend_aliases_gfx1100_kernel_keys` stale `pair32` expectation,
already reproduced at a clean baseline before this unit. No all-green CPU
suite claim; no repair to that unrelated file. Fixture, registry/CPU smoke,
compilation and benchmark-export checks pass.

## Trace and reproduction

Cached-only `rocprofv3 --kernel-trace` shows the DPP producer at
11.170–11.250 ms versus parent 12.781–12.859 ms, with 32 allocated VGPR,
scratch 0 and 4096 B requested dynamic LDS in both arms. The static LDS 0 field
does not include dynamic allocation. Code-object census has 29 VGPR for both;
static instructions 249→251, but parent has one publication plus eight
**dynamic loop** barriers versus the candidate's one publication barrier.
Static instruction count is not dynamic work or an occupancy counter.

```bash
uv run --no-sync python scripts/qwen4exp_router_dpp_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --require-cached-build --output /tmp/qwen4exp-router-dpp-screen.json
uv run --no-sync pytest tests/test_gpu_qwen4exp_router_dpp.py \
  tests/test_gpu_qwen35_router_plan.py -q --suite gpu
```

Use the ROCm environment in
[the worklog](../../../worklog/entries/20261005T232354.215863Z-pi-qwen4exp-router-ordered-dpp-0d1744.md).
The model path, source hashes, exact commands, 180 timed pairs, both order
subsets, output hashes, library hash, cached trace durations/resources and
static census are in [screen.json](screen.json). Raw logs/CSV/disassembly
remain in `/tmp`.
