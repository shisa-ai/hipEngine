# Qwen3.8-27B `Q4_K_M` engine comparison on gfx1151

Date: 2026-09-13. Status: diagnostic comparison, not a retained performance claim.

Five engine builds ran on one physical host against the same GGUF file:
hipEngine production autoregressive (AR) decode, upstream llama.cpp HIP and
Vulkan, and halo-box `strix-llama.cpp` HIP and Vulkan. A sixth arm re-ran the
`strix-llama.cpp` HIP binary with `HIP_LAUNCH_BLOCKING=1`, the workaround that
build documents for `gfx1151`. All arms are single-request AR with no
speculation.

## Result

**Prompt processing:** halo-box `strix-llama.cpp` HIP leads at every shape,
**+9.2% / +12.2% / +16.1%** over hipEngine at 512/128, 1K/128 and 4K/128.
`strix-llama.cpp` Vulkan leads hipEngine by 1-4%. Upstream llama.cpp HIP is
3.7% behind hipEngine at 512/128, level at 1K/128, and 3.0% ahead at 4K/128;
upstream Vulkan is 0.6-5.1% behind at every shape.

**Text generation:** Vulkan leads at every shape, **+5.2% / +6.9% / +4.4%**
over hipEngine. The HIP backends are within 2.2% of hipEngine at 512/128 and
1K/128 and 3.7-4.0% behind at 4K/128.

### Prompt processing (tok/s)

Median of the measured repetitions per shape; bold marks the fastest
engine in each column. The `HIP_LAUNCH_BLOCKING=1` arm is a diagnostic
variant of the `strix-llama.cpp` HIP build and is not marked.

| Engine | Pin | 512/128 | 1K/128 | 4K/128 |
| --- | --- | ---: | ---: | ---: |
| hipEngine production AR | `a2f62c881` | 405.002 | 394.412 | 372.138 |
| llama.cpp HIP | `002a12ad2 (build 10939)` | 390.200 | 389.893 | 383.404 |
| llama.cpp Vulkan | `37b3a9e0c (build 10940)` | 384.437 | 383.276 | 369.867 |
| strix-llama.cpp HIP | `6548035 (build 372)` | **442.293** | **442.332** | **431.938** |
| strix-llama.cpp Vulkan | `6548035 (build 372)` | 409.401 | 402.089 | 388.387 |
| strix-llama.cpp HIP, HIP_LAUNCH_BLOCKING=1 | `6548035 (build 372)` | 442.856 | — | 418.904 |

### Text generation (tok/s)

| Engine | Pin | 512/128 | 1K/128 | 4K/128 |
| --- | --- | ---: | ---: | ---: |
| hipEngine production AR | `a2f62c881` | 12.226 | 11.994 | 12.150 |
| llama.cpp HIP | `002a12ad2 (build 10939)` | 12.346 | 12.258 | 11.702 |
| llama.cpp Vulkan | `37b3a9e0c (build 10940)` | **12.860** | 12.824 | **12.682** |
| strix-llama.cpp HIP | `6548035 (build 372)` | 12.311 | 12.221 | 11.669 |
| strix-llama.cpp Vulkan | `6548035 (build 372)` | 12.858 | **12.825** | 12.682 |
| strix-llama.cpp HIP, HIP_LAUNCH_BLOCKING=1 | `6548035 (build 372)` | 12.299 | — | 11.653 |

### Versus hipEngine

| Engine | Prefill 512/128 | Prefill 1K/128 | Prefill 4K/128 | Decode 512/128 | Decode 1K/128 | Decode 4K/128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp HIP | -3.65% | -1.15% | +3.03% | +0.98% | +2.20% | -3.69% |
| llama.cpp Vulkan | -5.08% | -2.82% | -0.61% | +5.18% | +6.92% | +4.38% |
| strix-llama.cpp HIP | +9.21% | +12.15% | +16.07% | +0.69% | +1.89% | -3.96% |
| strix-llama.cpp Vulkan | +1.09% | +1.95% | +4.37% | +5.17% | +6.93% | +4.38% |
| strix-llama.cpp HIP, HIP_LAUNCH_BLOCKING=1 | +9.35% | — | +12.57% | +0.60% | — | -4.09% |

## Host and model

| Item | Value |
| --- | --- |
| Host | `gfx1151`, Framework Desktop, machine ID `55ea6c509d0b49eea8de7094a1023668` |
| CPU / GPU | AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`, 40 CU), 128 GB unified LPDDR5X |
| Kernel | `7.1.6-1-cachyos` |
| ROCm / compiler | ROCm 10.0.0, HIP 7.15.26333, AMD clang 23.0.0git `8f497e0992fb7513f7f78a6f6b6f1056c375e961` |
| GPU policy | `power_dpm_force_performance_level=high`, CPU governor `performance`, `tuned` profile `accelerator-performance` |
| Model | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (17,106,775,008 bytes, sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`) |
| K/V storage | BF16 on every engine |

## Protocol

The hipEngine row and the llama.cpp-family rows use different timing tiers.
They are reported side by side but are not interchangeable.

**hipEngine** ran `scripts/qwen38_gfx1151_readme_sweep.py` from worktree
`/tmp/hipengine-qwen38-release-20260913` at commit `a2f62c881`: one resident
session per shape, prompt built from repeated token id 9707, one warmup run
followed by three measured runs, 128 decode steps after the first output, graph
capture excluded from the decode wall. This is the protocol behind the
published 404.5 tok/s / 12.2 tok/s row; the 512/128 prefill measured here
reproduces it to +0.13%.

**llama.cpp-family** rows ran `scripts/llamacpp_bench_with_peak.py` over
`llama-bench` with `-ngl 99 -fa 1 -ctk bf16 -ctv bf16 -r 5`: prefill
(`-p N -n 0`) and decode at offset (`-p 0 -n 128 -d N`) are separate processes
with no explicit token array. The tables above report the median of the five
repetitions; `summary.json` also records llama-bench's reported mean
(`avg_ts`). The two HIP prefill cells at 512/128 have 2.3-2.7% CV because the
first timed repetition runs 5.2-5.8% slow, so their medians sit above their
means (llama.cpp HIP 390.200 vs 386.151, `strix-llama.cpp` HIP 442.293 vs
437.323). Every other cell is under 0.7% CV.

Peak GTT for the llama.cpp-family engines is 16.17-16.43 GiB across all shapes,
sampled externally at 10 ms from `/sys/class/drm/card1/device/mem_info_gtt_used`.
hipEngine tracked a 24.15 GiB peak allocation and sampled 24.79-24.85 GiB of
HIP-visible memory. The two memory figures come from different instruments and
are not a like-for-like comparison.

## Exact commands

```bash
# 1. hipEngine production AR
cd /tmp/hipengine-qwen38-release-20260913
env GPU_MAX_HW_QUEUES=2 \
    HIPENGINE_HIP_ARCH=gfx1151 \
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hip1151-t6/hipcc-version-gfx1151.txt \
    HIPENGINE_GGUF_FP16_RECURRENT_STATE=0 \
    HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1 \
    HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE=1 \
    HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256=c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc \
    PYTHONPATH=. /home/lhl/hipEngine/.venv/bin/python scripts/qwen38_gfx1151_readme_sweep.py \
      --prompt-lengths 512 1024 4096 --decode-tokens 128 \
      --warmups 1 --repetitions 3 --output <out>.json

# 2. llama.cpp-family, once per binary
python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench <llama-bench binary> --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --quant gguf_q4_k_m --backend <hip|vulkan> \
  --workloads 512/128 1K/128 4K/128 --repetitions 5 \
  --ngl 99 --flash-attn 1 --cache-type-k bf16 --cache-type-v bf16 \
  --poll 10 --card-name card1 --memory-domain gtt \
  --extra-args "-dev <ROCm0|Vulkan0>" --output <out>.json

# 3. strix-llama.cpp HIP under its documented gfx1151 async workaround
HIP_LAUNCH_BLOCKING=1 python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /tmp/hipengine-halobox-head-20260913/build-hip/bin/llama-bench \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --quant gguf_q4_k_m \
  --backend hip --workloads 512/128 4K/128 --repetitions 5 \
  --ngl 99 --flash-attn 1 --cache-type-k bf16 --cache-type-v bf16 \
  --poll 10 --card-name card1 --memory-domain gtt \
  --extra-args "-dev ROCm0" --output <out>.json
```

`run_comparison.sh` runs the five primary arms in sequence; `BLOCKING_ARM=1`
adds the diagnostic arm. The recorded run used `OUT=/tmp/hip1151-4way` with
`BLOCKING_ARM=1`; the artifacts in `raw/` are those files, copied unmodified.

## Builds

| Engine | Source | Commit | Version | Build |
| --- | --- | --- | --- | --- |
| llama.cpp HIP | `ggerganov/llama.cpp` | `002a12ad25503a93501b2e188c360029830a241a` | build 10939 | `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_NO_VMM=ON`, Release, shared |
| llama.cpp Vulkan | `ggerganov/llama.cpp` | `37b3a9e0ccba261d1cc245a971deae0b18c201ab` | build 10940 | `-DGGML_VULKAN=ON`, Release, shared |
| `strix-llama.cpp` HIP | `halo-box/strix-llama.cpp` | `654803517b06da47f5210553a661bf6c80deb97f` | build 372 | Release, shared, built by the journey campaign |
| `strix-llama.cpp` Vulkan | `halo-box/strix-llama.cpp` | `654803517b06da47f5210553a661bf6c80deb97f` | build 372 | Release, shared, built by the journey campaign |

Upstream master advanced by one CI-only commit (`37b3a9e0c`, "ci : remove
leftover command") between the HIP and Vulkan builds; the HIP binary therefore
sits one commit behind the Vulkan binary. That commit deletes two lines from
`.github/workflows/server-sanitize.yml` and nothing else.

## Limitations

- Output correctness was not verified for any llama.cpp-family row.
  `llama-bench` measures throughput only.
- `strix-llama.cpp` documents that asynchronous HIP batched inference can
  produce wrong outputs on `gfx1151` and sets `HIP_LAUNCH_BLOCKING=1` in its own
  CI. The main `strix-llama.cpp` HIP row ran the asynchronous default; the
  blocking arm is included separately. The blocking arm costs 3.0% at 4K/128
  prefill and nothing measurable at 512/128, so the prefill lead does not depend
  on the workaround, but this comparison does not establish that the
  asynchronous arm computes the right answer.
- The hipEngine rows carry its graph-parity gate (18/18 cases, exact generated
  ids, final logits and recurrent/KV state). No comparable gate ran for the
  llama.cpp-family rows.
- The two timing tiers differ in process model, prompt construction and
  repetition handling, as described under [Protocol](#protocol). Small deltas
  between hipEngine and the llama.cpp-family engines are inside that tier
  difference; the `strix-llama.cpp` HIP prefill lead is much larger than it.
- All arms ran sequentially in one session, not counterbalanced pairs, so
  thermal drift is not controlled.

## Files

| File | Contents |
| --- | --- |
| `run_comparison.sh` | Runs the five primary arms sequentially; `BLOCKING_ARM=1` adds the diagnostic arm |
| `assemble.py` | Builds `summary.json` and prints the tables above |
| `summary.json` | Machine-readable summary with artifact hashes and both medians and means |
| `raw/hipengine-production-ar.json` | hipEngine resident sweep, three shapes |
| `raw/upstream-hip.json`, `raw/upstream-vulkan.json` | Upstream llama.cpp HIP and Vulkan |
| `raw/halobox-hip.json`, `raw/halobox-vulkan.json` | `strix-llama.cpp` HIP and Vulkan |
| `raw/halobox-hip-launch-blocking.json` | `strix-llama.cpp` HIP with `HIP_LAUNCH_BLOCKING=1` |
| `raw/run_comparison.log` | Full console log of the recorded run |
