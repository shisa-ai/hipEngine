# halo-box PR #63: shallow-prefill comparison against the current baseline

**Result: the PR head is 13.9% faster on the 4096-token prefill with identical
output.** Measured 2026-09-16 on Framework `gfx1151` (AMD Radeon 8060S), machine
`55ea6c509d0b49eea8de7094a1023668`, both arms on the same host in one session.

## The two arms

| | baseline | candidate |
| --- | --- | --- |
| source | `halo-box/strix-llama.cpp` @ `69946438aa2c432ba365e40bec35c38f6e1de5bb` | PR #63 head `c4aa302294fcd5121af2039fd4d3dee0d472ec03` (branch `codex/halobox-modules-20260914`) |
| worktree | `/home/lhl/comparators-20260915/halobox-strix` | `/home/lhl/comparators-20260915/halobox-pr63` |
| build | `/home/lhl/comparators-20260915/build-halobox` | `/home/lhl/comparators-20260915/build-halobox-pr63` |
| dirty | false | false |

Both were built with the same configuration (`Release`, `AMDGPU_TARGETS=gfx1151`,
`GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_CUDA_FA=ON`) and run with identical
server arguments:

```
-m <first shard of UD-Q4_K_XL> --host 127.0.0.1 --port N --parallel 1 --no-webui
-ngl 999 -fa on -ctk bf16 -ctv bf16 -c 4352 -b 8192 -ub 2048 -t 4 --no-warmup
```

Fixture case `code-p4096` from `qwen4exp_canonical_ar_p512_p1024_p4096.json`
(4096 exact token ids, sha256 `72a815ea…`), one unmeasured warmup request then
one measured request.

`--no-warmup` is new here. It suppresses the server's own empty warmup run, which
is what made the earlier comparator captures span a whole server lifetime. The
baseline re-measured under it reads 5552.5 ms against its historical 5531.0 ms
(0.4% apart), so the flag does not perturb the measured window.

## Rate

The server's own `prompt_ms` is the authoritative rate; it delimits the request
by construction.

| arm | prompt_ms | prompt tokens | tok/s |
| --- | ---: | ---: | ---: |
| baseline `69946438a` | 5552.5 | 4096 | 737.7 |
| PR #63 `c4aa30229` | **4873.5** | 4096 | **840.5** |

**+13.9%** (5552.5 / 4873.5 = 1.139).

## Correctness

Greedy generation (temperature 0, top_k 1) from the fixture's exact token ids,
48 tokens, compared between the two builds:
`scripts/llamacpp_pair_token_compare.py`.

| arm | tokens emitted | identical positions |
| --- | ---: | ---: |
| `69946438a` | 48 | 48/48 |
| `c4aa30229` | 48 | 48/48 |

**Sequences are identical.** The PR is not bit-identical to the baseline — its own
commits record WikiText2 perplexity moving 2.0285 → 2.0259 and then 2.0259 →
2.0254 — but at this prompt the two builds emit the same 48 tokens.

## Where the gain came from

Kernel time per family, both arms, matched protocol. The profiled window covers
the warmup and measured prefill together, so the **deltas** are the meaningful
part; the absolute totals are not a single prefill.

| family | baseline ms | PR63 ms | delta | delta % |
| --- | ---: | ---: | ---: | ---: |
| expert_down | 2301.4 | 1132.8 | −1168.6 | **−50.8%** |
| quantize_pack | 567.9 | 0.0 | −567.9 | **−100.0%** |
| moe_reduce | 405.0 | 208.5 | −196.5 | **−48.5%** |
| elementwise_norm | 1049.8 | 677.9 | −371.9 | −35.4% |
| hyper_connection | 843.0 | 688.8 | −154.2 | −18.3% |
| dense_projection | 3116.4 | 2774.2 | −342.2 | −11.0% |
| gdn | 569.9 | 530.7 | −39.2 | −6.9% |
| expert_gate_up | 1665.5 | 1556.3 | −109.2 | −6.6% |
| qsa_attention | 311.4 | 626.5 | **+315.1** | **+101.2%** |
| indexer | 11.2 | 29.8 | +18.6 | +166.1% |
| other | 14.7 | 12.2 | −2.5 | −17.0% |
| **total** | **10856.3** | **8237.6** | **−2618.7** | **−24.1%** |

The kernel sum falls 24.1% while the wall rate improves 13.9%. The profiled
window contains two prefills and the warmup benefits more than the measured
request, so **the wall number is the one to retain**; the family deltas explain
its composition.

The four largest gains are exactly what the commits describe:

- **`expert_down` −50.8%** and **`quantize_pack` −100%** are one change.
  `08de004` ports the MMB kernels: weights are dequantized to BF16 in LDS and the
  product accumulates on the 16x16x16 BF16 WMMA path, so the Q5_1 MMQ kernel and
  its per-matmul `quantize_mmq_q8_1` activation quantization both disappear. The
  PR63 trace contains pwilkin's `mmb_dense_kernel`, `mmb_routed_kernel`,
  `mmb_routed_glu_kernel` and `mmb_f32split_kernel` verbatim.
- **`moe_reduce` −48.5%** and **`hyper_connection` −18.3%** are `60c26d0`, which
  carries the hyper-connection streams as BF16 and gives the MoE weighted
  reduction BF16 and float4 variants.
- **`elementwise_norm` −35.4%** is largely `0106857`, one wave per row for narrow
  RMS normalization.

### One real regression

`qsa_attention` doubles, and it is the same kernel doing it:

| kernel | baseline | PR63 | dispatches |
| --- | ---: | ---: | ---: |
| `flash_attn_ext_f16<256, 256, 16, 4, false, false, false>` | 270.98 ms | **585.69 ms** | 48 → 48 |

Identical name, identical template arguments, identical dispatch count, 2.16x the
time. This is not reclassification and it is not a shape change. The PR touches
this area directly (`e09da52` adds a masked sparse prefill attention path and
`c4aa302` is the tip commit on selection sentinels), so the likely cause is that
the dense path now does more work per call — but the cause is **not established
here**, only the measurement. It costs ~315 ms against gains of ~2400 ms, so it
does not change the sign of the result.

### One apparent regression that is not one

`indexer` reads +166%, but the baseline's top-k ran through hipCUB and was
classified as `elementwise_norm`:

| | baseline | PR63 |
| --- | ---: | ---: |
| `indexer` family | 11.2 ms | 29.8 ms |
| rocprim trampoline in `elementwise_norm` | 17.4 ms | 0.6 ms |
| **combined** | **28.6 ms** | **30.4 ms** |

`ec05a66` replaces the hipCUB sort with named `top_k_radix_*` kernels. The work
moved between families rather than appearing. Combined, the change is +6%, which
is inside the noise floor.

## Consequence for the comparison target

Both arms pass the gate the review set: **performance improved and output is
identical**. The PR head is therefore the better halo-box comparison target than
`69946438a`, and it is also closer to what pwilkin's arm measures, since after
this merge the two engines share the MMB kernel family.

One caveat to carry: PR #63 was **open, not merged**, when this was measured. If
the head moves, this comparison is against `c4aa30229` specifically, not against
"PR #63".

## Reproducing

```bash
# baseline
python3 scripts/qwen4exp_llamacpp_prefill_comparator.py \
    --server /home/lhl/comparators-20260915/build-halobox/bin/llama-server \
    --label halobox-base-69946438a-bf16 \
    --source-tree /home/lhl/comparators-20260915/halobox-strix \
    --case-id code-p4096 --repetitions 1 --warmups 1 --kv-dtype bf16 \
    --server-arg=--no-warmup --profile --marker-trace \
    --trace-root /tmp/cmp-base-halobox.raw --output /tmp/cmp-base-halobox.json

# PR head: same, with build-halobox-pr63 and halobox-pr63

# correctness
python3 scripts/llamacpp_pair_token_compare.py \
    --server-a .../build-halobox/bin/llama-server    --label-a 69946438a \
    --server-b .../build-halobox-pr63/bin/llama-server --label-b c4aa30229 \
    --model <first shard> --fixture <canonical fixture> --case-id code-p4096 \
    --n-predict 48 --output /tmp/pair-tokens.json

# family attribution
python3 scripts/qwen4exp_comparator_role_map.py \
    --trace <kernel_trace.csv> --label <label> --dialect llamacpp \
    --output benchmarks/results/2026-09-15-flashnext-engine-comparison/roles-<label>.json
```
