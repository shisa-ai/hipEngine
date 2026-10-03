# gufo vs hipEngine — same-file, same-host comparison (gfx1151)

Date: 2026-10-04
Host: AMD Radeon 8060S Graphics (gfx1151, 20 CUs), AMD RYZEN AI MAX+ 395,
125 GiB unified LPDDR5X, 32 threads, kernel `7.1.6-1-cachyos`,
machine-id `55ea6c509d0b49eea8de7094a1023668`.
Host power state: `tuned-adm active` = `accelerator-performance`,
`scaling_governor` = `performance`, verified at the start and end of every
measurement block. No other GPU process ran during any block.

This directory compares the prebuilt gufo edge binary against hipEngine on the
**same model files** and the **same host**. It is not an old→new comparison and
no rate here is a delta against a published hipEngine row measured elsewhere.

## Engine identities

| | gufo | hipEngine |
| --- | --- | --- |
| Binary | `/home/lhl/gate-runs/gufo-edge-20260925/build/gufo` | in-tree JIT (no AOT binary) |
| Binary sha256 | `8ca63994d7325dee3d635c9dd38d0278fae5aaafd888228966724dc5c88cdf17` | — |
| Reported version | `98641a6503da2ec5d6dbb1888ddc95f8a3e13b28` | repo HEAD `f31997b737be3c88eed6e46ea7436281210b25e0` |
| Source tree | `/home/lhl/gufo-src` HEAD `9cad13974cf6da0cd3674b4e0a88b14b7e4a2908` (shallow, 1 commit) | branch `qwen3.8-flash-next`, clean under `hipengine/` and `kernels/` |
| Build source | `/home/lhl/gate-runs/gufo-edge-20260925/source` HEAD `98641a6` + `source.patch` | — |
| Toolchain | GNU 16.2.1 C/C++, HIP Clang 23.0.0, hipcc from TheRock `_rocm_sdk_devel` | `/tmp/reprof/compiler.txt`: HIP 7.15.26333, AMD clang 23.0.0git |
| `gufo diagnose` | `[PASS]` / `[supported]`, fingerprint `d8fd8211…`, reports ROCm 7.2.3 | — |

**The binary was not built from the current `gufo-src` HEAD.** It reports
version `98641a6` and `CMakeCache.txt` records `GUFO_VERSION:STRING=98641a6…`;
the build source tree is a separate clone at `98641a6`, one commit *ahead* of
the shallow clone's `9cad139`, with `source.patch` applied (it changes
`IsQwen38Artifact` in `src/models/qwen/chat_template.cpp` to stop selecting the
Qwen3.8 chat template from tensor shape alone). Full record:
`raw/identity.json`, `raw/gufo-diagnose.txt`.

All hipEngine measurements ran while HEAD moved between `5b753adc84` and
`f31997b73`; `git diff 5b753adc84 f31997b73 -- hipengine kernels scripts` is
empty, so the code state is identical. Only `docs/` and `worklog/` commits
landed in that window.

## gufo cannot load our Qwen3.6-35B-A3B

```
$ ./gufo bench -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 32 -n 4 -d 0
HIP: found 1 ROCm devices (Total VRAM: 122880 MiB):
  Device 0: AMD Radeon 8060S Graphics, gfx1151, Wave Size: 32, VRAM: 122880 MiB
Error creating Qwen GPU executor: Missing required Qwen tensor: blk.0.ffn_gate.weight
[Model Load]: 0.0188839 s (failed)
exit=1
```

The 35B-A3B file is `qwen35moe` with `ffn_gate_exps.weight` /
`ffn_gate_shexp.weight`; the gufo Qwen weight loader requires dense
`ffn_gate`/`ffn_up`/`ffn_down` and has no MoE module. The published
`gufo Qwen3.6-35B-A3B-Q6dense` scoreboard (3095 pp / 68.43 tg) is therefore
**not reproducible from this gufo tree on our file**, and the 2.13x prefill /
1.22x decode framing cannot be checked same-file. No same-file 35B comparison
exists in this artifact.

## B. Performance

### B1. Qwen3.8-27B `Q4_K_M` (sha256 `7e78da5d…`, both engines)

gufo (our file, prebuilt binary):
```
./gufo bench -m /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  -p 512,1024,2048,4096,32768,65536 -n 128 -d 0 -r 2 -v
./gufo bench -m /models/gguf/Qwen3.8-27B-Q4_K_M.gguf -p 2048 -n 128 -d 0,4096,16384 -r 2 -v
```
hipEngine all-in-one (one resident process, session sized 65666):
```
HIPENGINE_COMPILER_VERSION_FILE=/tmp/reprof/compiler.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 \
$PY scripts/qwen35_readme_sweep.py --engine gguf \
  --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --quant gguf_q4_k_m --backend hip_gfx1151 \
  --workloads 512/128 1K/128 4K/128 32K/128 64K/128 \
  --warmup-runs 1 --measured-runs 2 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk --use-wmma-prefill --use-gemv-decode \
  --graph-replay-decode --compiler-version-file /tmp/reprof/compiler.txt --require-cached-build
```
hipEngine per-shape: identical flags with one workload per process
(`raw/hipengine-27b-q4km-pershape-*.json`). gufo `-r 2` = 2 measured runs;
hipEngine = 1 discarded warmup + 2 measured.

| shape | gufo pp | hipEngine pp (per-shape) | hipEngine pp (1 session) | hipEngine pp (committed 09-12) | gufo tg | hipEngine tg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 383.53 ± 0.22 | 357.10 (357.03–357.17) | 407.99 (407.70–408.28) | 380.37 | — | 12.18 |
| 1K/128 | 392.36 ± 0.10 | 406.18 (406.10–406.27) | 405.90 (405.82–405.99) | 377.61 | — | 11.95 |
| 4K/128 | 382.74 ± 0.01 | 398.82 (398.69–398.95) | 398.56 (398.53–398.59) | 361.50 | — | 12.11 |
| 32K/128 | 360.22 ± 0.03 | — | 293.95 (293.86–294.04) | — | — | 10.72 |
| 64K/128 | 336.92 ± 0.09 | — | 256.99 (256.93–257.05) | — | — | 9.64 |
| tg128 @ d0 | 12.11 | — | 12.18 | — | — | — |

gufo depth ladder (`-p 2048`): pp2048 393.30 @d0, 386.21 @d4096, 366.24 @d16384;
tg128 12.12 @d0, 11.98 @d4096, 11.52 @d16384. Committed hipEngine 09-12
tg128 was 12.21/11.98/12.13 at 512/1K/4K. gufo `pp` is the marginal rate of
one batch at the given depth; hipEngine `tg` at 4K/32K/64K is decode *after* a
prompt of that length, not at a cached-prefix depth.

**Protocol caveat.** gufo sizes its context per point (4096 for pp512/1K/2K,
32769 for 32K, 65537 for 64K) while the hipEngine all-in-one run uses a 65666
session for every point; the per-shape run is right-sized but uses a different
prefill chunk profile for the 512 session. The prompt token sequences also
differ (gufo `((i%1000)+100)` cycle vs hipEngine repeated id 9707). These are
the reasons the 512 row is not a clean single number. Ratios:

- 1K/4K prefill: hipEngine per-shape **+3.5% / +4.2%** over gufo; the
  all-in-one gives the same +3.4% / +4.1%.
- 512 prefill: gufo 383.5 vs hipEngine 357.1 (per-shape, gufo +7.4%) and 408.0
  (one session, hipEngine +6.4%). **Inconclusive** — protocol-sensitive.
- 32K/64K prefill: gufo **+22.6% / +31.1%**.
- decode: essentially tied at d0/4K (hipEngine +0.6% / +1.1%).

### B2. Flash-Next `UD-Q4_K_XL` (four shards, gufo-published sha256 match)

gufo AR:
```
./gufo bench -m <shard1> -p 512,1024,2048,4096 -n 128 -d 0 -r 2 -v
```
gufo MTP with **gufo's published shared sidecar** (downloaded from
`unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee`, sha256
`5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`):
```
./gufo bench -m <shard1> -p 512,1024,2048,4096 -n 128 -d 0 -r 2 -v \
  --speculative mtp --mtp-model <shared-Q8_0.gguf> --draft-tokens 7
```
hipEngine AR (`qwen4exp_canonical_ar_bench.py hipengine`, 4 categories ×
p512/1K/4K, warm PLE cache, 1 warmup + 2 measured, `--require-cached-build`);
hipEngine MTP (`qwen4exp_journey_mtp_bench.py --engine hipengine --max-tokens 16
--budget 2 --repetitions 3`, 10 category prompts, request wall incl. prefill).

| shape | gufo AR pp | gufo MTP pp | hipEngine AR pp (mean, range) | hipEngine AR tg | gufo AR tg | gufo MTP tg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1168.78 ± 2.61 | 1162.83 ± 3.56 | 254.31 (252.60–257.03) | 19.87 | — | — |
| 1K | 1510.14 ± 0.01 | 1494.80 ± 0.81 | 251.65 (248.54–255.29) | 17.77 | — | — |
| 2K | 1717.60 ± 1.22 | 1699.47 ± 1.66 | — | — | — | — |
| 4K | 1695.37 ± 0.61 | 1674.76 ± 0.63 | 241.53 (238.78–244.35) | 15.07 | — | — |
| tg128 @ d0 | — | — | — | — | 26.71 ± 0.24 | 39.24 ± 0.97 |

- **gufo prefill is 4.6x / 6.0x / 7.0x hipEngine's** at 512 / 1K / 4K.
- gufo decode is **1.34x** hipEngine's at 512 context (26.71 vs 19.87) and
  flatter with depth (gufo d4096 published 25.98).
- gufo MTP is **1.47x its own AR** on the bench prompt (`drafted=98 accepted=68`),
  and the bench output sha256 is byte-identical to AR
  (`98c708d7827afad9912654dfce06b561fadec8de7ea62a3e7c33902f9b5d1402`).
- hipEngine MTP is **0.801x its own AR** on the 10-prompt request wall
  (15.84 → 12.69 tok/s; every category 0.77–0.82x). Protocols differ
  (bench pp/tg vs request wall), so the two MTP ratios are **not comparable**.

**gufo cannot load our sidecar.** With
`--mtp-model /models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
(sha256 `9db03a68…`) gufo fails:
`Error creating Qwen3.8-Flash-Next model: missing tensor blk.48.nextn.hc_head_norm.weight`.
The head therefore could not be held constant across engines; gufo's own
published shared sidecar was fetched instead (sha256 verified exact) and used
for all gufo MTP rows. Cross-engine MTP is a different head and is not claimed.

## C. Accuracy

gufo's HTTP/bench/prompt surface exposes **no logits**; it emits only a
`[TokenTrace]`/`[QwenBenchTrace]` sha256 over the generated token IDs. A
cross-engine KL/top-1 envelope is therefore **impossible** against gufo. The
comparison is exact greedy token-ID agreement plus the engines' own task gates.

Protocol: greedy (`temperature 0`), 16 generated tokens, raw joined message
text (no chat template on either side), committed suite
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`
(sha256 `fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a`,
10 prompts / 4 categories). gufo AR and MTP both use gufo's shared Q8_0 head;
hipEngine AR and MTP both use our Q8_0 sidecar.

| check | result |
| --- | --- |
| gufo AR vs gufo MTP, per prompt | **10/10 identical** |
| hipEngine AR vs hipEngine MTP, per prompt | **10/10 identical** |
| hipEngine AR vs hipEngine MTP, per category | code 4/4, general_en 2/2, general_ja 2/2, mixed_ja_en 2/2 |
| gufo AR vs hipEngine AR, per prompt | **9/10 identical** |

The single cross-engine divergence is `general_ja_explain` (general_ja,
heldout): tokens 0–14 are identical and the sequences split at generated-token
index **15** — hipEngine emits `や` (id 67813), gufo emits `）`. gufo's CLI
exposes only the hash, so the position was recovered by byte-comparing the
printed text (`raw/gufo-prompt-suite/general_ja_explain.ar.rep0.log`) against
the hipEngine token decode. `general_en_explain` and all other prompts agree
token-for-token.

### Correctness verdicts (stated separately)

- **hipEngine (Flash-Next, production profile):** within this suite MTP is
  token-exact vs its own AR on all 10 prompts and deterministic across
  repetitions; the registered production plan for
  `qwen4_exp_gguf`/`hip_gfx1151`/`gguf_ud_q4_k_xl` passes the calibrated
  numerical envelope per `docs/EXECUTION-PROFILES.md` §2.9 (mean/p95/p99/max KL
  `3.065e-4/1.785e-3/4.271e-3/7.771e-3`, top-1 `0.99556`), with the documented
  task-axis caveat (`requires_task_review`: 15/18 strict-exact task outputs).
  That gate was **not re-run here**; it is cited, not re-measured.
- **gufo (Flash-Next):** internally consistent — AR and MTP produce identical
  greedy token IDs on all 10 prompts, and the bench synthetic prompt output
  sha256 is identical between AR and MTP. This confirms gufo's own
  `QUALITY.md` claim on our suite. gufo exposes no logits, so no KL/top-1
  envelope can be applied and no independent numerical verdict is possible
  here.

The two engines are **not** averaged into one conclusion.

## D. The published 27B row is not reproducible from this binary

gufo's published single-user 27B Q4 row is **656.33 pp / 12.37 tg at d0**
(`docs/models/qwen3.8-27b/BENCHMARKS.md`) for the **UD-Q4_K_XL** file
(sha256 `3f227079…`) with gufo revision `628ed18e` (binary `8734dac7…`) on a
different physical host (fingerprint `46337fb5…`, kernel `7.1.8`, total memory
`134309883904`; ours `d8fd8211…`, kernel `7.1.6`, `134304342016`). The published
setup pairs the target with the Q4_K_M DFlash2 draft, so only the **pp** axis is
compared here.

Measured with the prebuilt `98641a6` binary on this host:

- **Path hypothesis — rejected.** `gufo serve llm --context 133760` on our
  Q4_K_M file reports `prompt_per_second = 380.37` at `prompt_n = 2481`, i.e.
  the same rate as `gufo bench` (393.31 at pp2048). The HTTP path is not the
  gap (`raw/gufo-serve-q4km.probe.json`).
- **File hypothesis — supported.** The standard Q4_K_M tensor mix is
  F32/Q4_K/Q6_K/Q5_K; `Qwen3.8-27B-UD-Q4_K_M.gguf` is an IQ4_XS-mixed dynamic
  quant (117 `IQ4_XS` tensors, plus Q5_K/Q8_0/Q4_K) and gufo labels it
  "IQ4_XS mixed". The same binary measures **pp2048 761.90** on it
  (pp512 484.99, pp4096 754.73, tg128 12.79) — **1.94x** the standard Q4_K_M
  rate. That single axis brackets the published 656.33.
  `Qwen3.8-27B-UD-Q4_K_S.gguf` fails to load in gufo entirely
  (`Invalid GGUF tensor shape or unsupported storage type: blk.0.ffn_gate.weight`).
- **Mechanism — the axis is quant-layout dispatch, not bytes.** On the same
  binary, same host and same `-p` ladder, the IQ4_XS file ramps **1.57x** from
  pp512 (484.99) to pp2048 (761.90) while standard `Q4_K_M` is **flat**
  (383.53 -> 393.31, **+2.5%**). A 4% file-size difference cannot produce that
  contrast, so the two files select different prefill kernels and gufo's
  published 656.33 is a property of the path its file selects as much as of the
  file. The ramp contrast is measured; the kernel-path reading is an inference.
- **Build hypothesis — untestable.** The published revision `628ed18e` is not in
  the local 409-commit history or the shallow clone, so its binary cannot be
  rebuilt or measured. Its contribution is not separable from the file axis.

No further inference is made: the exact UD-Q4_K_XL file is not local, and the
two axes cannot be fully separated without it.

## Where hipEngine wins, loses, or is not comparable

- **27B Q4_K_M, 1K/4K prefill:** hipEngine wins by ~3.5–4.2% at matched shapes.
- **27B Q4_K_M, 512 prefill:** inconclusive (protocol/session-size sensitive,
  measured range 357–408 pp).
- **27B Q4_K_M, 32K/64K prefill:** gufo wins by 22.6% / 31.1%.
- **27B Q4_K_M decode:** tied (hipEngine +0.6–1.1%).
- **Flash-Next prefill:** gufo wins by 4.6–7.0x.
- **Flash-Next decode (d0):** gufo wins by 1.34x.
- **Flash-Next MTP:** not comparable — gufo cannot load our sidecar, and the
  available protocols (bench pp/tg vs 10-prompt request wall) differ. gufo's
  own MTP speeds up (1.47x AR); hipEngine's slows down (0.80x AR) on its own
  protocol.
- **Accuracy:** both engines are internally AR==MTP exact; cross-engine greedy
  AR agrees on 9/10 prompts and diverges by one near-tie token on
  `general_ja_explain`. A cross-engine KL envelope is not reachable because
  gufo exposes no logits.

## Artifacts

- `raw/identity.json`, `raw/gufo-diagnose.txt`, `raw/host-fingerprint.txt`
- `raw/gufo-27b-pp-ladder.log`, `raw/gufo-27b-depth-ladder.log`,
  `raw/gufo-27b-udq4km-bench.log`, `raw/gufo-27b-udq4ks-bench.log`,
  `raw/gufo-35b-load.log`, `raw/gufo-serve-q4km.probe.json`
- `raw/hipengine-27b-q4km-sweep.json`, `raw/hipengine-27b-q4km-pershape-*.json`
- `raw/gufo-flashnext-ar-bench.log`, `raw/gufo-flashnext-mtp-shared-bench.log`
- `raw/hipengine-flashnext-ar-canonical.json`,
  `raw/hipengine-flashnext-journey-mtp.json`, `raw/hipengine-flashnext-traces.json`
- `raw/gufo-prompt-suite-ar.jsonl`, `raw/gufo-prompt-suite-mtp.jsonl`,
  `raw/gufo-prompt-suite/*.log`
- `performance-27b-q4km.json`, `performance-flashnext.json`,
  `accuracy-token-identity.json`, `gufo-published-row-gap.json`

## Proposed `benchmarks/CHANGELOG.md` one-liner

> **2026-10-04 UTC**: **Same-file gfx1151 gufo-vs-hipEngine comparison on Radeon 8060S.** Prebuilt gufo edge `98641a6` (binary `8ca63994…`) vs hipEngine `f31997b7`. Qwen3.8-27B `Q4_K_M`: hipEngine prefill +3.5%/+4.2% at 1K/4K (per-shape), gufo +22.6%/+31.1% at 32K/64K, decode tied (+0.6–1.1%); Flash-Next `UD-Q4_K_XL`: gufo prefill 4.6–7.0x and decode 1.34x hipEngine. gufo **cannot load** the 35B-A3B (`Missing required Qwen tensor: blk.0.ffn_gate.weight`) nor our MTP sidecar (`missing tensor blk.48.nextn.hc_head_norm.weight`). gufo's published 27B 656.33 pp is not reproducible from this binary; `gufo serve` and `gufo bench` agree (380 vs 393) and an IQ4_XS-mixed proxy file measures 761.90 pp, so the gap is file/quant composition, not the HTTP path (build revision untestable). Accuracy: both engines AR==MTP token-exact on the 10-prompt suite; cross-engine AR 9/10 (near-tie at token 15 of `general_ja_explain`); no cross-engine KL (gufo exposes no logits). No hipEngine default or registry entry changed. [Evidence](results/2026-10-04-gfx1151-gufo-vs-hipengine-same-file/README.md).
