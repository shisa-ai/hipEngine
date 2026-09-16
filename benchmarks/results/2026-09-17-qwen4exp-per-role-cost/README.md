# Qwen4Exp per-prefill cost, refreshed, and a shared-taxonomy comparison

Same protocol as
[`../2026-09-16-flashnext-per-role-cost/README.md`](../2026-09-16-flashnext-per-role-cost/README.md),
re-run on 2026-09-17 at `725794c3f` with a clean tracked tree, plus a new
comparison that puts hipEngine and the halo-box comparator on **one** family
taxonomy.

Nothing in the default path changed between the two runs. This record exists to
confirm that, to re-anchor the attribution at a current commit, and to add the
comparison.

## Reproduction

| Measure | 2026-09-16 | 2026-09-17 | Delta |
| --- | ---: | ---: | ---: |
| Kernel count | 9652 | 9652 | 0 |
| Window | 22932.5 ms | 22698.6 ms | −233.9 ms (−1.0%) |
| Attributed | 22368.1 ms | 22083.6 ms | −284.5 ms (−1.3%) |
| Unattributed | 0 ms | 0 ms | — |
| Prefill wall | 22.85 s | 22.70 s | −0.15 s |
| `logits_sha256` | `e717076f…` | `e717076f…` | identical |

The logits digest is the strong check and it is not self-referential: the same
`e717076fe080c887618e65e3810d039bc4fe7356e9b88d9ec6f64d265189a0b7` and the same
`token_id = 248068` appear in
[`../2026-09-16-q8-wmma-layers-recoverable-time/artifact.json`](../2026-09-16-q8-wmma-layers-recoverable-time/artifact.json)
under `fallback`, which is the named production profile on the same case. The
attribution run and the layer-scope work are therefore measuring the same
arithmetic.

Per-operation-class totals moved by less than 0.2%: matmul 14367.8 → 14389.5 ms,
non-matmul 6307.9 → 6308.5 ms, risk-or-repair 1370.3 → 1373.3 ms.

## Protocol

Two environment requirements are load-bearing and neither is recoverable from an
artifact. Both cost a run to rediscover, so they are recorded here.

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151

# Without this the launch census records nothing. --launch-census only names the
# output path. The run still completes and writes a well-formed census with
# total_launches: 0, which nothing in the child report distinguishes from a real
# result. Check total_launches before using the census for shapes.
export HIPENGINE_KERNEL_CENSUS=1

OUT=benchmarks/results/2026-09-17-qwen4exp-per-role-cost
rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format csv \
  -d "$OUT/role-trace" -- \
  $PY scripts/qwen4exp_profile_gap.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --mode prefill --case-id code-p4096 --repetitions 1 \
    --profile --role-markers \
    --launch-census "$OUT/census.json" \
    --compiler-version-file "$OUT/hipcc-version.txt" \
    --prefill-chunk-size 1024 \
    --output "$OUT/child.json"

# rocprofv3 writes into a gfx1151/ subdirectory, not the directory it was given.
TRACE="$OUT/role-trace/gfx1151"
$PY scripts/qwen4exp_role_analyze.py --trace-dir "$TRACE" \
  --measure-prefix qwen4exp_prefill_p4096_ --output "$OUT/role-analysis.json"
$PY scripts/qwen4exp_role_gap_table.py --role-analysis "$OUT/role-analysis.json" \
  --census "$OUT/census.json" --output "$OUT/artifact.json"
$PY scripts/qwen4exp_operation_cost.py --trace-kernels "$OUT/role-analysis.json" \
  --census "$OUT/census.json" --model-shapes "$OUT/model-shapes.json" \
  --output "$OUT/operation-cost.json"

$PY scripts/qwen4exp_shared_family_comparison.py \
  --hipengine-role-analysis "$OUT/role-analysis.json" \
  --hipengine-label hipengine-725794c3f \
  --comparator halobox-base=../2026-09-16-flashnext-delimited-components/attrib-base-69946438a-code4096.json \
  --comparator halobox-pr63=../2026-09-16-flashnext-delimited-components/attrib-pr63-c4aa30229-code4096.json \
  --case-id code-p4096 --strict \
  --output "$OUT/shared-family-comparison.json" \
  --markdown "$OUT/shared-family-comparison.md"
```

`--require-cached-build` is **not** part of this protocol. It fails closed on any
kernel not built under the current environment's cache key and it is absent from
the recorded command of every committed arm.

`model-shapes.json` is copied from the 2026-09-16 record rather than regenerated.
It is a pure function of the GGUF geometry, and the model file and its shard
hash are unchanged, so the copy is the same input.

## Cost by role family

| Role family | ms | % | vs 2026-09-16 |
| --- | ---: | ---: | ---: |
| `linear` | 10565.6 | 47.9 | +13.0 |
| `moe` | 6609.0 | 29.9 | +9.9 |
| `gr_read` | 2546.0 | 11.5 | +6.9 |
| `qsa_prefill` | 1420.9 | 6.4 | −0.2 |
| `gdn` | 820.3 | 3.7 | −4.1 |
| `prefill_boundary` | 94.1 | 0.4 | −0.2 |
| `ple` | 15.4 | 0.1 | — |

Largest individual roles, unchanged in order from the 2026-09-16 record:
`moe:expert_gate` 6619.2 ms (30.0%), `linear:attn_qkv` 2615.8 (11.8%),
`linear:ssm_out` 1960.6 (8.9%), `linear:attn_gate` 1580.9 (7.2%),
`qsa_prefill:attn_q` 1422.4 (6.4%), `gr_read:hc_ffn_down` 1276.2 (5.8%),
`gr_read:hc_attn_down` 1272.8 (5.8%), `linear:attn_q` 1230.4 (5.6%).

## The shared-taxonomy comparison

The 2026-09-16 comparison of our per-role table against the comparator's
per-family table compared two taxonomies. hipEngine carries the tensor role in a
ROCTX range, so one symbol serves several roles; the comparator names the family
in the symbol. `scripts/qwen4exp_comparator_role_map.py` already existed to
express both in one vocabulary and already classified the comparators; it gained
a `hipengine` mapper, and both sides are now classified from kernel symbols by
one module with one family list.

Both columns are the same case (`code-p4096`) on the same host, and they are not
the same arithmetic. Milliseconds are kernel time.

| Family | ours ms | ours % | base ms | ratio | PR #63 ms | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_projection` | 10562.2 | 47.8 | 1536.7 | 6.87x | 1374.3 | **7.69x** |
| `expert_gate_up` | 3414.2 | 15.5 | 819.4 | 4.17x | 761.8 | 4.48x |
| `hyper_connection` | 2523.8 | 11.4 | 428.9 | 5.88x | 335.7 | 7.52x |
| `expert_down` | 2389.0 | 10.8 | 1149.8 | 2.08x | 412.8 | 5.79x |
| `qsa_attention` | 1402.7 | 6.4 | 156.6 | 8.96x | 155.7 | **9.01x** |
| `gdn` | 821.2 | 3.7 | 287.6 | 2.86x | 263.8 | 3.11x |
| `moe_reduce` | 762.5 | 3.5 | 204.0 | 3.74x | 103.9 | 7.34x |
| `elementwise_norm` | 180.2 | 0.8 | 531.4 | **0.34x** | 331.7 | **0.54x** |
| `indexer` | 17.5 | 0.1 | 5.6 | 3.12x | 15.0 | 1.17x |
| `ple` | 9.9 | 0.0 | 0.0 | — | 0.0 | — |
| `other` | 0.2 | 0.0 | 6.2 | 0.03x | 5.7 | 0.04x |
| `quantize_pack` | 0.0 | 0.0 | 284.8 | — | 0.0 | — |
| **total** | **22083.6** | **100.0** | **5411.0** | **4.08x** | **3760.4** | **5.87x** |

### Where the 18.3 s gap to PR #63 actually is

`22083.6 − 3760.4 = 18323.2 ms`. Decomposed by family:

| Family | Gap | Share of gap |
| --- | ---: | ---: |
| `dense_projection` | 9187.9 | **50.1%** |
| `expert_gate_up` | 2652.4 | 14.5% |
| `hyper_connection` | 2188.1 | 11.9% |
| `expert_down` | 1976.2 | 10.8% |
| `qsa_attention` | 1247.0 | 6.8% |
| `moe_reduce` | 658.6 | 3.6% |
| `gdn` | 557.4 | 3.0% |
| `elementwise_norm` | −151.5 | −0.8% |
| remainder | 7.1 | 0.0% |

**Dense projection alone is half the gap, and four families are 87% of it.**
That is a different conclusion from the ratio column alone: `qsa_attention` has
the worst ratio (9.01x) but only 6.8% of the gap, because its absolute size on
both sides is small.

### Two corrections this comparison forces

**Elementwise and norm work is not a gap; we are ahead of the comparator on it.**
Our 180.2 ms against PR #63's 331.7 ms, a 0.54x ratio, with the comparator
spending 8.8% of its kernel time there against our 0.8%. Any "close the
elementwise/norm gap" item is wrong as stated. PR #63's own −37.6% on that
family was a 531.4 → 331.7 ms change, which is still nearly twice our cost.

**`expert_down` is not one of our better families.** Against the base comparator
it reads 2.08x, which looks like our second-best ratio; against PR #63 it is
5.79x, because PR #63 cut that family by 64.1% (1149.8 → 412.8 ms) and we have
no equivalent of that kernel. The base column flatters us on exactly the family
where the comparator moved furthest.

### Caveats on the table

- **Different arithmetic.** Their dense path is BF16 operands; ours is FP32
  scalar FMA on the dense projections, with F16 WMMA on the certified MoE
  scopes and an iu8 risk-and-repair pass. This table says what each operation
  costs each engine, not how efficient either is.
- **The repair fold.** hipEngine's iu8 exact-repair passes (1434.4 ms, kernel
  name matched) are folded into the matmul family they correct, because that is
  the operation whose cost they are. They are also totalled separately as
  `risk_or_repair_ms` so the fold cannot hide the machinery. The comparator has
  no equivalent pass, so that 1.43 s is a real cost of our current default that
  appears inside `expert_gate_up` and `expert_down`. `operation-cost.json`
  reports 1373.3 ms for its `risk_or_repair` class because it classifies by
  role-and-operation label rather than by kernel name.
- **The comparator's own mis-assignment is not inherited.** The withdrawn
  2026-09-15 per-family bucketing had `quantize_mmq_q8_1`, `qsa3_attn_kernel`
  and the rocBLAS `Cijk_*` GEMM in the wrong families. The rules here were not
  reused from that table; they are the ones in
  `qwen4exp_comparator_role_map.py`, and `--strict` fails on any of our kernels
  above 1 ms that the mapper cannot place. It found one on the first run (the
  prompt K/V write, now mapped to `elementwise_norm` to match the comparator's
  treatment of the same bytes as a `set_rows`/`cpy` op).
- **`ple` and `quantize_pack` are not comparable rows.** We spend 9.9 ms on
  `ple` and the comparator's mapper reports 0.0; we spend 0.0 on
  `quantize_pack` and the base comparator spends 284.8. Both are real and both
  are implementation choices, not measurement artifacts.

## Evidence

| File | What it is |
| --- | --- |
| `child.json` | The profiled prefill run: 22.70 s wall, 180.45 tok/s, `token_id` 248068, logits digest, route env, lifecycle |
| `census.json` | 1595 launches over 401 distinct shapes behind the `linear:*` owner calls |
| `role-analysis.json` | 9652 kernels, 100% attributed, 2024 (role, kernel) rows, 52 distinct symbols |
| `artifact.json` | Role-family table, 32 roles, `total_ms` 22083.6 |
| `operation-cost.json` | Per-operation table with real shapes and achieved GFLOP/s |
| `shared-family-comparison.json` | The table above, machine-readable |
| `shared-family-comparison.md` | The same table as markdown |
| `model-shapes.json` | Copied from 2026-09-16; same GGUF, unchanged |
| `hipcc-version.txt` | AMD clang 23.0.0git, HIP 7.15.26333 |
| `role-trace/gfx1151/` | The raw `rocprofv3` CSVs: kernel, marker and HIP API traces |

The raw trace is 13 MB and is **not committed**, matching the 2026-09-16 record.
Every number here is derived from it deterministically, and the digests let a
regeneration be checked:

| File | SHA-256 |
| --- | --- |
| `1902262_kernel_trace.csv` | `b1c8b6418e514026d70e93e3548c5de2cf4f1fd88fa0af0ee9215721c3786dba` |
| `1902262_marker_api_trace.csv` | `c4a10312abac3a4ab283867130e31e3f930a46dc713977c173cba0a251396edd` |
| `1902262_hip_api_trace.csv` | `7c57374491fdb13c65eca5ef969aa59102898a0eef39bae8367e91b2f8f264cd` |
| `1902262_agent_info.csv` | `5239c42eb699d523da3e4a43e3533cdb66e3f5754799e8915830891fd581d5a1` |

Host: Framework `gfx1151`, machine `55ea6c509d0b49eea8de7094a1023668`, rocprofv3
1.3.5, peak allocated 84.4 GB. Source `725794c3f`, `tracked_clean: true`,
`named_profile_intact: true`, `fell_back_to_strict: false`, fixture sha256
`42b562bd8e9644be…`.

## Status

Reproduction confirmed, not a new performance claim. The comparison is the part
with new information, and it repoints the tuning: **half the prefill gap is the
dense projection, and the next three families bring that to 87%**, while the
family with the worst ratio (`qsa_attention`) is 6.8% of the gap and the family
this campaign has been treating as an open gap (`elementwise_norm`) is already
ahead.

Not yet measured, in the order that would settle the most:

1. A memory-counter capture of the `attn_qkv` launch, to close the byte
   accounting the 2026-09-16 record left open.
2. Where the dense projection's stalls are: an occupancy-and-stall breakdown of
   the real kernel, not the probe.
3. The `hyper_connection` 2523.8 ms, which is 11.9% of the gap and has a known
   shape: two passes over the same tensor where the comparator fuses.
4. Whether the 1434.4 ms of risk-and-repair is reducible by tightening the risk
   estimate rather than by changing the arithmetic.
