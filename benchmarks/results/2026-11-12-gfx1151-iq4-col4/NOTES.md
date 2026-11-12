# IQ4_XS four-column verifier tile

The IQ4_XS multirow local32 kernel now owns four output columns per block instead
of eight. Scalar decode keeps eight. Rows 2–4 retain the same K partition,
per-column accumulation order, shuffle tree, cross-wave sum and BF16 rounding.
Other IQ formats and fused pair kernels are unchanged. Existing strict fallback
registrations remain available; no flag or new arithmetic variant is added.

## Evidence

Physical host `gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395, Qwen3.8-27B
`UD-Q4_K_M`, default production profile, BF16 KV, MTP budget 3, prefix cache off,
capacity 2048, one request and 64 greedy outputs with ignored EOS. Compiler,
model fingerprint, baseline commit `793f4b281`, source hashes and profile
manifest are recorded in `summary.json` and `public_ab.json`.

Public `LLM.generate_detailed()` comparison: fresh sessions in true AR,
parent/candidate/candidate/parent order, each with a discarded warmup, over the
canonical ten category prompts plus eight held-outs. The parent arm loads the
unchanged prior IQ library before constructing the model; each session captures
its own graphs. The candidate uses the shipping implementation.

| IQ4 multirow tile | Mean whole-request suite wall |
| --- | ---: |
| Eight output columns | 64.804 s |
| Four output columns | 63.483 s |

Wall is 2.04% lower; all categories improve. All parent/candidate output IDs,
acceptance counts, fallback reasons and repeats match. Every session returns
tracked device allocations to zero on close. These are complete requests,
including prefill, not decode-only rates. The acceptance variation recorded in
the earlier snapshot experiment is not reproduced in this comparison and is
not declared resolved. This is not an AR-exactness or task-quality claim.

Public-request counters motivated the change: four-row IQ4 used 168 VGPRs with
roughly 44–52% measured occupancy. The candidate uses 120 VGPRs with roughly
64–71% occupancy and no scratch; traces identify the four-column template and
doubled grid. Q5 has higher measured occupancy and is unchanged in this unit.
Counter collection perturbs execution and its durations are attribution, not
throughput evidence. Baseline counter requests match unprofiled IDs and controls.

The fixture-derived isolated screen alternates parent/candidate ordering across
eight samples, ten launches per sample, after three warmups. K/N shapes are
5120/17408, 17408/5120, 5120/6144 and 5120/12288, each at rows 2/3/4. Every
candidate output equals the parent byte-for-byte; median HIP-event latency is
6.79–16.59% lower. This is kernel timing, not a model measurement.

## Correctness and reproduction

The new GPU test compares rows 2/3/4 and wave counts 1/2/4 against independent
scalar launches, checks repeat bytes, and checks a NumPy decoded-weight reference
with maximum KL <= 0.05 and top-1 agreement >= 90%. Existing local32 row/decode
coverage passes, as do the focused CPU policy/dispatch bundle and live verifier
1024-boundary/reset regression. No arithmetic
change is intended; the exact parent contract is binding rather than a substitute
for a changed-arithmetic production gate. No W7900 performance claim is made.

Run from the repository root. The wrapper selects `~/miniforge3/envs/therock`.
Archived diagnostic drivers retain their temporary source paths; reconstruct
those sources before rerunning the screen or public parent arm:

```bash
D=benchmarks/results/2026-11-12-gfx1151-iq4-col4
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
git show 793f4b281:hipengine/kernels/hip_gfx1100/quant/gguf_iq_dense.hip > /tmp/qwen38-iq-parent.hip
cp hipengine/kernels/hip_gfx1100/quant/gguf_iq_dense.hip /tmp/qwen38-iq-col4.hip
bash "$P/run.sh" "$D/screen.py"
bash "$P/run.sh" "$D/public_ab.py" --out /tmp/qwen38-iq-public-ab.json
bash "$P/run.sh" benchmarks/results/2026-11-12-gfx1151-qwen38-dense-followup/attribution.py --out /tmp/qwen38-iq-candidate-warm.json
bash "$P/run.sh" -c 'import subprocess; from pathlib import Path; Path("/tmp/qwen38-hipcc-version.txt").write_text(subprocess.check_output(["hipcc", "--version"], text=True))'
bash "$P/run.sh" "$D/counters.py" /tmp/qwen38-iq-candidate-occupancy OccupancyPercent
bash "$P/run.sh" -m pytest tests/test_gpu_iq4_multirow_tile_exact.py tests/test_gpu_gguf_iq_local32_rows.py tests/test_gpu_gguf_iq_local32_decode.py -q
uv run pytest tests/test_unit_gfx1151_iq_dense_policy_parity.py tests/test_unit_gguf_linear_dispatch.py -q
```

Baseline counters used the same driver before the kernel edit, separately with
`FETCH_SIZE`, `OccupancyPercent` and `VALUInsts`. Raw profiler CSV/logs stay outside
Git; compact per-kernel/grid aggregates are archived. In the reused public
A/B driver, `loop` means the eight-column parent, `batch` means four columns,
and `mixed_plan_counts` counts IQ library requests, not mixed-pair dispatches.
