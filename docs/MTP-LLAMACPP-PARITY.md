# GGUF MTP llama.cpp Parity Trace and Roadmap

## 2026-06-30 / 2026-07-03 - DUAL-ENGINE PER-STAGE ATTRIBUTION (speed goal closed): where the MTP gap actually is

Original goal: stand up clean per-stage profiling for BOTH engines on the same model
(Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151) and attribute the MTP tok/s gap to specific
stages/kernels. Full write access to `~/llama.cpp` granted; HIP and Vulkan both in
scope. Builds: llama.cpp HIP+Vulkan at `6e9007ae6` (master, clean). Model 21.1 GiB.

### SUMMARY — full tok/s ladder + gap decomposition (read this first)

| config (same model, gfx1151) | AR tok/s | MTP tok/s | cycle wall / output | uplift | role |
| --- | ---: | ---: | ---: | ---: | --- |
| hipEngine default exact B5 | 54.79 parallel-attn full | 61.98 parallel-attn full | 16.162 ms | 1.1312× | Correctness-preserving control lane after the shared attention fix. |
| hipEngine `llama-compat` B2, no-copy verifier capture + llama-style direct partial commit, natural24 cyclecap24 | 54.79 natural24 cyclecap24 full | **71.52 natural24 cyclecap24 full** | **14.005 ms** | **1.3055×** | Active apples-to-apples llama.cpp replication lane with `--max-output-tokens 24` and enough verify cycles for every prompt to reach the cap. Rejected/partial bulk blocks commit the captured verifier row, matching llama.cpp's normal MTP accept path; not serial-prefix-equivalent. The `f32head` artifact name is misleading: this retained row did **not** enable `--verify-lm-head-q6-top1-dp4a`. The actual current-shape verifier-head route regressed to **66.45 tok/s / 15.072 ms/output** with unchanged acceptance/economy, so it is rejected for now. The fixed-cycle provenance row remains **72.23 tok/s / 13.865 ms/output**. |
| hipEngine `llama-compat` B2, copied verifier capture + llama-style direct partial commit | 54.78 directcommit full | 60.56 directcommit full | 16.534 ms | 1.1055× | Superseded by the no-copy GDN state-row capture. It paid a full recurrent-state D2D copy before every captured GDN prefill layer. |
| hipEngine `llama-compat` B2, semantic-safe direct state + serial state-only partial replay | 54.74 serial-state full | 51.85 serial-state full | 19.308 ms | 0.9472× | Exact semantic control. Rejected/partial bulk blocks restore and serial-replay state, but skip replay LM-head sampling. |
| hipEngine `llama-compat` B2, unsafe direct-state diagnostic | 54.79 draft-dense-Q8 draft-only full | 75.15 draft-dense-Q8 draft-only full | 13.325 ms | 1.3716× | Superseded: this row direct-committed rejected/partial bulk-block state that the lifecycle comparator proved is not prefix-equivalent. |
| llama.cpp HIP/ROCm B2 (dp4a) | 51.38 suite / 52.13 traced / 51.98 rerun | 67.3 suite / 72.12 traced / 71.91 rerun / 75.4 cli prompt | **14.269 ms rerun** | ~1.31× suite / 1.383× traced/rerun / 1.47× cli | Timing target and source reference. |
| llama.cpp **Vulkan** (dp4a) | **62.65** | **84.6 cli prompt** | n/a | ~1.35× | Backend ceiling reference, not the HIP parity target. |

The working HIP target is now the no-copy llama-style direct-commit
`llama-compat` B2 natural24 row versus the rerun llama.cpp HIP B2 natural24 row.
The implementation intentionally
splits contracts: default/exact and the `serialstate` control preserve
serial-prefix state equivalence, while the active llama replication lane commits
the captured verifier row on rejected/partial blocks just like llama.cpp's
`common_speculative_accept()` path updates `pending_h` from `verify_h`. This
moves the active compat row to **14.005 ms/output vs llama.cpp's rerun
14.269 ms/output** using the same natural24 output-token cap. The active HIP
stage-wall bucket is slightly faster, but the request-level throughput headline
still trails llama.cpp by about **0.39 tok/s** (**71.52 vs 71.91 tok/s**). The
fixed-cycle hipEngine provenance row remains **72.23 tok/s /
13.865 ms/output**. The key speed fixes were removing the per-layer
recurrent-state D2D copy in the prefill-GDN verifier capture path and adding
llama.cpp's direct partial-commit lifecycle. The follow-up harness fix was adding
llama.cpp's tail
rule (`draft_n_max = min(B, n_remaining - 1)`) so the comparison no longer mixes
hipEngine fixed-cycle overshoot with llama.cpp's server `max_tokens=24`
behavior. Replay is still not the gap:
`target_block_replay_or_commit` is **0.044 ms/output**,
`target_verify_replay_rows=0`, and the remaining exposed cleanup is row economy
(**1.171 vs 1.148 target rows/output**) plus target semantic parity after
full-accept bonus sampling.

Verifier-head attribution correction: artifact
`benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json`
is a retained active-route rerun, but its route did not include
`--verify-lm-head-q6-top1-dp4a`. The measured current-shape route that does
include the flag is
`benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-vlmheadtop1-full.json`;
it regresses to **66.45 tok/s**, **15.072 ms/output**, and
`target_block_lm_head_sample` **2.118 ms/output** versus the active route's
**71.52 tok/s**, **14.005 ms/output**, and **1.068 ms/output**, with unchanged
acc/output **0.596**, draft acceptance **0.777**, and target rows/output
**1.171**. Do not attribute the retained 71.52 tok/s row to the verifier-head
path.

### NATURAL24 c=1/c=4/c=8 serving diagnostic - current packed-prefill server rows

This table is separate from the stage-wall parity tracker above. The hipEngine
direct-suite rows are kept for c=1 context; the c>N hipEngine rows are OpenAI
server full-request throughput from `scripts/mtp-bench.py`. The llama.cpp rows
use the comparable client/full-request aggregate fields from
`scripts/llamacpp_mtp_bench.py` with `max_tokens=24`, reasoning off, greedy
sampling, f16 KV, flash-attn on, and B2 `--spec-type draft-mtp
--spec-draft-n-max 2`. The llama.cpp HIP server binary was
`/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server` at commit
`1ebf790cda38d827559548f67b0469189690cc8c` with local dirty state recorded in
the artifacts; use these as diagnostics, not as a replacement for the clean
instrumented HIP stage target. Llama.cpp decode-only aggregate numbers remain in
the artifacts and are called out in the reading column where useful.

| engine / path | c | AR serving/direct tok/s | MTP serving/direct tok/s | ratio | budget / route | reading |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hipEngine exact direct suite | 1 | 54.80 | **52.13** | 0.951x | B1 fastest; B2 52.04, B5 50.65 | Under the natural24 token cap, B2 is faster than B5, but B1 is fastest and no exact budget beats AR. This does not supersede the retained 10-cycle exact B5 row **61.98 tok/s / 1.131x AR**. |
| hipEngine `llama-compat` direct suite | 1 | 54.79 | **71.52** | 1.3055x | B2 directcommit/no-copy | Direct suite only; not a server concurrency row. |
| hipEngine `llama-compat` server MTP | 1 | 41.24 | **34.44** | 0.835x | B2 resident slots, zero batch window, pooled target/draft state | Diagnostic blocked: zero batch window and persistent target-session / MTP draft-runner pools remove the deliberate 100 ms queue delay plus draft-open tax, but warmed c=1 MTP still loses to AR. Timing buckets show AR server-vs-direct is mostly prompt prefill, while MTP is dominated by target verify. |
| hipEngine `llama-compat` server MTP | 2 | 41.27 | **34.22** | 0.829x | B2 resident slots, zero batch window, pooled target/draft state | Zero-window c=2 did not coalesce in the rerun: no `slots_*` phase buckets, effectively independent c=1 requests. |
| hipEngine `llama-compat` server MTP | 2 | 66.15 | **59.94** | 0.906x | B2, 5 ms batch window, packed AR prefill/decode + default-on packed MTP prefill for eligible batches + MTP stream-draft + packed target verify | Packed MTP prefill removes most c=2 prompt-open overhead after warmup. |
| hipEngine `llama-compat` server MTP | 4 | 67.68 | **66.60** | 0.984x | same | c=4 is now near same-server AR and faster than llama.cpp HIP/Vulkan full-request MTP c=4 (**49.69/48.10 tok/s**). |
| hipEngine `llama-compat` server MTP | 8 | 61.72 | **54.88** | 0.889x | same; c=8 first wave still uses serial prompt open, trailing c=2 wave uses packed MTP prefill, target verify streams chunk-4 groups | c=8 now beats llama.cpp HIP/Vulkan full-request MTP c=8 (**50.56/54.25 tok/s**) but remains below current hipEngine AR; verifier still dominates. |
| llama.cpp HIP server B2 | 1 | 38.10 | **48.22** | 1.266x | B2 | Full-request/client aggregate. Decode-only aggregate is **52.19/75.56 tok/s**. |
| llama.cpp HIP server B2 | 4 | 68.59 | **49.69** | 0.724x | B2 | Full-request/client aggregate. Decode-only aggregate is **108.33/78.21 tok/s**. |
| llama.cpp HIP server B2 | 8 | 76.76 | **50.56** | 0.659x | B2 | Full-request/client aggregate. Decode-only aggregate is **124.71/78.56 tok/s**. |
| llama.cpp Vulkan server B2 | 1 | 40.65 | **48.96** | 1.204x | B2 | Full-request/client aggregate. Decode-only aggregate is **64.15/91.48 tok/s**. |
| llama.cpp Vulkan server B2 | 4 | 68.32 | **48.10** | 0.704x | B2 | Full-request/client aggregate. Decode-only aggregate is **124.68/92.19 tok/s**. |
| llama.cpp Vulkan server B2 | 8 | 78.50 | **54.25** | 0.691x | B2 | Full-request/client aggregate. Decode-only aggregate is **139.71/103.57 tok/s**. |

Artifacts:

- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c1-bw0-timing-pooled-warm.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw0-timing-pooled.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c4-bw0-timing-pooled.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw0-timing-pooled.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw0-phase-serial.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw5-phase-serial.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw5-packed-smoke.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c4-bw0-phase-serial.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c4-bw5-packed-smoke.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw0-phase-serial.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw5-packed-smoke.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw5-packed-chunk4-warm.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c2-bw5-packed-streamchunks-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c4-bw5-packed-streamchunks-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c8-bw5-packed-streamchunks-rerun2.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c2-bw5-packed-prefill-finalrows-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c4-bw5-packed-prefill-finalrows-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c8-bw5-packed-prefill-finalrows-rerun2.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c2-bw5-streamdraft-server-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c4-bw5-streamdraft-server-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c8-bw5-streamdraft-server-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw5-streamdraft-warm.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c4-bw5-streamdraft-rerun2.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw5-streamdraft-rerun2.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw5-streamverify-rerun2.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c2-bw5-packedstage-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c4-bw5-packedstage-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-c8-bw5-packedstage-rerun.json`
- `benchmarks/results/2026-07-05-hipengine-server-mtp-natural24-sweep.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c2-bw5-default-after-batcher.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c4-bw5-default-after-batcher.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c8-bw5-default-after-batcher.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c2-bw5-packed-ar.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c4-bw5-packed-ar.json`
- `benchmarks/results/2026-07-05-hipengine-server-ar-natural24-c8-bw5-packed-ar.json`
- `benchmarks/results/2026-07-03-ar-mtp-default-natural24-budget-sweep-c1.json`
- `benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json`
- `benchmarks/results/2026-07-03-llamacpp-hip-mtp-natural24-c1.json`
- `benchmarks/results/2026-07-03-llamacpp-hip-mtp-natural24-c4.json`
- `benchmarks/results/2026-07-03-llamacpp-hip-mtp-natural24-c8.json`
- `benchmarks/results/2026-07-03-llamacpp-vulkan-mtp-natural24-c1.json`
- `benchmarks/results/2026-07-03-llamacpp-vulkan-mtp-natural24-c4.json`
- `benchmarks/results/2026-07-03-llamacpp-vulkan-mtp-natural24-c8.json`

Serving-route flag status: the default policy remains MTP serving off, but the
capability flag truthfully reports `speculative_mtp.serving_route=true` when the
server is started with `--speculative-mtp-serving opt_in` or `auto` and the
loaded GGUF engine exposes the NextN tensors. Explicit opt-in requests use
`"speculative_mtp": true`; `auto` routes only compatible greedy-fast requests.
The route is guarded by the existing sampling-incompatibility checks and rejects
streaming, non-greedy sampling, and unsupported engines.

What landed is the c=N llama-compat resident-slot server milestone plus the
first production packed target verifier. The OpenAI
path calls `LLM.generate_speculative_mtp_detailed()`, which enters a GGUF
llama-compat MTP hook with one shared target-weight runner, pooled target
sessions, pooled MTP draft runners, per-request draft K/V state, verifier
hidden-state capture/commit, and MTP K/V lifecycle ownership. Slots are advanced
in one process; this is true resident state isolation, not parallel child
processes and not the old single-session reset loop. The current c>N loop is now
phase-serial (`draft -> target verify -> commit`) to mirror llama.cpp's lifecycle
and expose `slots_draft_phase_ms`, `slots_verify_phase_ms`, and
`slots_commit_phase_ms`; the target verify phase now calls
`Qwen35GGUFResidentSession.verify_target_blocks_batch()` on packed multi-slot
blocks when the shape is supported.
Short prompts that cannot safely build the shifted context-replay rows still
fall back to ordinary GGUF greedy AR under the same request. The exact default
MTP route and streaming MTP remain unclaimed. The sequence of
fixes is now measured: shared prepared target weights removed the catastrophic
per-request GGUF rematerialization, zero batch window removed the intentional
100 ms/request coalescing delay, and pooled target/draft state moved warmed c=1
MTP **30.09 -> 34.44 usage tok/s** by eliminating draft-open cost. Server AR
moved only **40.56 -> 41.24 tok/s** because the remaining direct-suite gap is
prompt prefill, not session construction.

Implementation status after packed AR prefill/decode, c=8 AR chunk-stream
decode, stream-slot MTP draft, c=8 stream-verify chunks, and default-on packed
MTP prompt prefill for eligible four-slot batches: c=2 is **66.15 AR / 59.94
MTP tok/s**, c=4 is **67.68 / 66.60**, and c=8 is **61.72 / 54.88**. This keeps
AR c>N fixed and materially improves MTP c>N scaling. c=4 is now near
same-server AR, and c=8 now beats llama.cpp HIP/Vulkan full-request MTP
(**50.56/54.25 tok/s**), while still trailing hipEngine AR and llama.cpp
decode-only aggregate scaling. The real MTP concurrency target is therefore
narrower but still open: reduce verifier wall and cold-shape latency, and then
revisit c=8 full packed prompt prefill beyond the four-slot cap.

Follow-up packed-verifier instrumentation now splits the target verifier chunk
inside the server artifact. On the warm c=8 diagnostic rerun
(`2026-07-05-hipengine-server-mtp-natural24-c8-bw5-packedstage-rerun.json`),
the headline is within the same retained regime (**52.68 tok/s**, diagnostic
only), with `slots_verify_phase_ms=12248.655` and
`target_packed_verify_total_ms=11653.156`. The biggest exposed sub-buckets are:

| packed verifier c=8 bucket | total ms |
| --- | ---: |
| linear-attention layers | 6080.186 |
| LM-head/sample drain | 2183.854 |
| full-attention layers | 1958.680 |
| final stream sync | 846.360 |
| setup + initial state import + token upload | 456.700 |
| scatter outputs | 97.646 |
| hidden readback | 3.268 |

An opt-in HIP-event split
(`HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS=1`) confirms the same target
without per-leaf synchronizes. The c=8 event run is slower due instrumentation
(**47.17 tok/s**) but exposes queued GPU work: compact-WMMA selected gate/up
**1860.286 ms**, verifier LM-head/sample **2049.108 ms**, full-attention layers
**1984.223 ms**, QKV/gate projection **1295.096 ms**, selected down
**805.975 ms**, and `wmma_total` read **163.563 ms**. Use it for ranking
kernel bodies, not for headline speed.

Packed MTP prompt-prefill update: 2026-07-06 promotes
`HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL=1` to the default for eligible c=2/c=4
serving batches. It reuses packed prompt rows and returns FP32 prompt hidden rows
for MTP catch-up. Warm c=2/c=4/c=8 move **46.75/49.65/52.18 ->
59.94/66.60/54.88 tok/s**. The four-slot guard remains: uncapped c=8 packed
prefill was previously rejected, and the current c=8 win comes from the trailing
c=2 wave plus normal chunk-4 verifier streaming. Cold first packed-prefill
shapes can be slow, so startup warmup/cached-build policy remains a follow-up.

This rejects the cheap next knobs: B1 server budget regressed warm c=8 to
**50.94 tok/s** and verifier chunk-size 3 regressed to **51.21 tok/s**; chunk
size 2 failed under four concurrent stream chunks. Follow-up diagnostics also
reject skipping the compact-WMMA `wmma_total` host read and selected-WMMA
launch-bounds tuning as defaults: the no-read probe measured **52.05/51.96
tok/s** at c=8, while launch-bounds=4 helped c=8 in isolation
(**53.22/53.44 tok/s**) but regressed c=4 (**49.20/49.04 tok/s** vs retained
**49.65**). The next useful fix should start in packed verifier
linear-attention rows and sampling/drain with a real kernel/body change, not
budget shape, smaller chunking, scalar-read removal, or launch-bounds tuning.

Follow-up AR c>N audit: the default OpenAI batcher now coalesces compatible
non-MTP requests even when each request has its own cancellation token; grouped
requests receive a composite cancellation token just like the MTP route. This
fixes the scheduler precondition but did not by itself create AR backend
scaling. Default-on packed AR decode first moved AR to
**50.89/56.79/59.17 tok/s** by using `step_batch_native(...,
scatter_state=False)`, decode-shaped GEMV projection, deferred packed-state
scatter across cycles, and parallel chunk-stream execution for c>4. Default-on
packed final-row prompt prefill then moves the current retained AR rows to
**66.15/67.68/61.72 tok/s** by packing slot-major prompt rows, scattering
final KV/recurrent state back to each resident session, and sampling only each
slot's final prompt row instead of sampling/copying every prompt row. The older
attempt to reuse the MTP
`verify_target_blocks_batch()` primitive for one-token AR decode was rejected:
c=2/c=4/c=8 measured **32.12/41.32/41.22 tok/s**, with
`target_verify_batch_ms` dominating (**1192/1382/1391 ms mean per request**).
That rejected verifier-shaped path is superseded by the retained packed decode
and packed prefill path.

### CLOSURE AUDIT - speed target, exact-path portability, and remaining risk

Decision: close the speed-gap sprint. The current `llama-compat` replication
lane is **14.005 ms/output** at the stage-wall level versus llama.cpp HIP
**14.269 ms/output**, so the measured stage wall is already slightly faster
than the reference. The request-throughput headline still differs by only
**0.39 tok/s** (**71.52 vs 71.91 tok/s**, about **0.5%**) and is now explained
by semantic/proposal economy rather than an exposed verifier or draft wall-time
bucket.

Do not keep the single-token target-logit chase as a P0 speed task. The live
`mixed_ja_en_translate` mismatch at task 9 / cycle 3 / row 2 has the same top-8
token set in both engines, but hipEngine samples `8940` while llama.cpp samples
`668`. The focused margin is hipEngine `8940 - 668 = +0.51934`, llama.cpp
`8940 - 668 = -0.00961`, gap **+0.52895 logits**. The broad F32 verifier stack
only moves the live margin to **+0.48450** and does not flip the token. That
remains useful parity evidence, but it is no longer a speed-gap blocker.

Portability audit: every known llama-compat optimization that is exact-safe and
non-regressive has either been promoted to the exact/default lane or is tracked
with a concrete reason it is not promotable.

| compat finding | exact/default handling | reason |
| --- | --- | --- |
| Shared parallel `mtp_dense_attn_f32` draft attention | **Promoted** to the exact default route. Exact B5 moved **60.8 -> 61.98 tok/s**, cycle **16.496 -> 16.162 ms/output**, draft drain **1.921 -> 1.899 ms/output** with unchanged acceptance. | Bit-exact and full-suite non-regressive. |
| Q8 shared-dual draft shared gate/up | **Default-on** for the exact resident draft path as well as `llama-compat`; opt-out cleanup remains in `docs/REFACTOR.md`. | Bit-exact versus two single GEMVs and validated by focused tests. |
| Earlier exact-route wins: Q6_K rowtile verifier lm-head, minrows2, `p_min=0.5`, cap32k/B1-probe | **Already default** in the exact suite route. | These are exact-mode economics wins and are represented by the current **61.98 tok/s** default row. |
| Llama-style direct partial commit plus no-copy prefill-GDN captured-row commit | **Compat-only**; exact analysis uses the default exact route or the `serialstate` control. | This intentionally follows llama.cpp's normal accept lifecycle and is not serial-prefix-equivalent on rejected/partial bulk blocks. It is the right replication lane, not an exact-default promotion. |
| dp4a verifier, dense raw-Q8 dp4a sidecars, draft dense-Q8 dp4a, q8_1 Q6 top-1, selected-down X8, F32 `ssm_out` route | **Compat-only / default-off globally.** | Accuracy-traded or route-contract-specific mechanisms; no exact/non-regressive replacement has been proven. |
| Resident device-chain/no-probe draft shape | **Not promoted** to exact default. | Bit-exact pieces exist, but full-suite exact evidence did not produce a retained speed win and the no-probe llama.cpp economy does not transfer to the exact default lane. |
| Verifier-head top-1 dp4a, shared-Q8 verifier, selected gate/up X8/raw, Q5/T32, fused selected down/SiLU diagnostics | **Rejected or retained only as diagnostics.** | Same-suite results regressed speed, acceptance, or both. |

Current conclusion: there is no known exact-safe llama-compat speed win left
unpromoted. Future exact-mode work should start from a new exact kernel or a
fresh full-suite default rerun, not from the closed single-logit parity chase.

### RETAINED TRACKING - default vs llama-compat vs llama.cpp HIP

This is the canonical dashboard for the closed speed-parity sprint. Keep this
section as a three-lane comparison if MTP work is reopened: hipEngine default
exact, hipEngine `llama-compat`, and llama.cpp HIP. Update it after each
retained or diagnostic MTP run that changes any lane. The headline speed row
uses retained full-suite numbers where available; the stage rows use
instrumented/deep traces on the same model and gfx1151 host. `llama-compat` is
the route that mirrors llama.cpp's no-probe B2 structure, so its delta column is
the audit ledger for any future work.

Tracking invariant when reopened: the first question is always "where is
`llama-compat` still slower than llama.cpp HIP, and by how much?" Keep the
default exact lane beside it as a regression guard, but do not let exact-mode
concerns obscure the replication lane. The standing table must answer that
question at a glance: current speed, per-stage cost, compat gap size, and the
next llama.cpp source path or kernel family to compare. In the current closed
state, the stage-wall delta is not positive; only the request-level semantic /
proposal-economy delta remains.

Update rule: every future `llama-compat` run that changes the route shape or moves
throughput must update the standing snapshot, source-artifact row, headline gap,
stage ledger, full-suite bucket inventory, all-sync leaf attribution, active gap
budget, and target map below in the same commit. Keep the three-lane tables
visible and in the same order: hipEngine default exact, hipEngine
`llama-compat`, llama.cpp HIP, then the compat delta. Do not replace them with
prose-only conclusions or a single tok/s headline. Old "FINAL" sections further
down are historical once they disagree with this active tracker.

Required refresh shape for each retained or diagnostic parity run:

| required row/table | what must change | why |
| --- | --- | --- |
| Standing three-lane snapshot | Current headline speed, cycle wall, draft drain, verifier drain, row economy, acceptance, and compat delta. | Gives the sprint a single "where are we behind right now" table. |
| Source artifacts | Exact route name, flags/env, artifact path, and whether the row is retained, all-sync attribution-only, or rejected. | Prevents mixing headline speed rows with diagnostic sync-split rows. |
| Three-lane speed gap | hipEngine default exact, hipEngine `llama-compat`, llama.cpp HIP, and the compat delta. | Keeps the top-line "how many tok/s or ms are left" visible. |
| Three-lane stage ledger | The large buckets that have real cross-engine meaning: draft drain, verifier drain, row economy, and non-gaps. | Shows where the remaining couple of ms actually lives. |
| Full-suite bucket inventory | Every high-level bucket emitted by the current hipEngine full-suite artifacts plus the closest llama.cpp analog when one exists. | Makes the next target mechanical: pick the largest positive compat delta with a valid analog. |
| All-sync leaf attribution | Fine-grained `llama-compat` split rows that explain the large full-suite buckets, with the exact attribution-only artifact named. | Prevents mixing headline speed rows with extra-sync diagnostic rows while still showing which kernel body to attack next. |
| Active gap budget / target map | Remaining ms or rows to close and the llama.cpp source area to inspect next. | Keeps the implementation work tied to a measured llama-compat gap, not intuition. |
| Llama.cpp source anchors | Exact llama.cpp file/function or kernel family for each live gap row. | Makes the next copy-or-retune target explicit once `llama-compat` structurally mirrors llama.cpp. |

Current parity state: the active
`denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit` compat
lane now uses llama-style direct-state transactions: prefill-shaped GDN capture
for all block commits, including rejected or partial bulk blocks. That
full-suite row is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-full.json`.
The active apples-to-apples natural24 row is
`benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json`.
Despite the filename, that artifact uses the normal active direct-commit route
and does not enable `--verify-lm-head-q6-top1-dp4a`.
It is **0.264 ms/output faster** than the rerun llama.cpp HIP B2 measured stage
row (**14.005 vs 14.269 ms/output**) but slightly behind llama.cpp's request
throughput headline (**71.52 vs 71.91 tok/s**). It beats true AR
(**71.52 tok/s, 1.3055x AR**). The fixed-cycle provenance row remains **72.23 tok/s /
13.865 ms/output**. The active llama.cpp timing target remains
`benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json`;
it was collected with local llama.cpp instrumentation patches, so treat it as a
stage target rather than a clean upstream performance claim. The semantic-safe
`serialstate` control remains
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`
at **51.85 tok/s / 19.308 ms/output** and is the row to use when the question is
serial-prefix equivalence rather than llama.cpp replication. Direct commit plus
no-copy GDN state-row capture moves the active natural24 replication lane
**51.85 -> 71.52 tok/s**, cycle **19.308 -> 14.005 ms/output**, verifier drain
**16.891 -> 11.436 ms/output**, replay/commit **2.489 -> 0.044 ms/output**,
and replay rows **38 -> 0**. The fixed-cycle provenance row for the same route
is **72.23 tok/s / 13.865 ms/output / 11.405 ms verifier drain**. The speed
target is closed: partial replay and verifier drain no longer explain the
request-level llama.cpp deficit. The remaining gap is semantic/economic:
the corrected full-suite run reaches all **240/240** requested output tokens, but
the focused `mixed_ja_en_translate` trace first diverges after both engines fully
accept draft `[11, 567]`; hipEngine samples bonus token `8940` while llama.cpp
samples `668`. Serial-exact hipEngine reproduces the same hipEngine bonus token,
so this is target-state/logit parity, not the bulk verifier scheduler or
direct-commit shortcut. Evidence:
`benchmarks/results/2026-07-02-mtp-proposal-trace-compare-natural24-mixed-ja-en-translate.json`
and
`benchmarks/results/2026-07-02-hipengine-mtp-serialexact-natural24-mixed-ja-en-translate.json`.
Economy denominator correction:
`benchmarks/results/2026-07-03-mtp-economy-denominator-reconcile.json`
shows the apparent accepted/output deficit is a denominator mismatch. The
request-level llama.cpp summary is **136/240 = 0.5667** accepted/output, while
the stage-timing "measured excluding first task" bucket is **136/223 = 0.6099**.
hipEngine is **143/240 = 0.5958**, so there is no full-request acceptance
deficit. The remaining request-level gap is **71.52 vs 71.91 tok/s** with lower
hipEngine draft acceptance (**0.777 vs 0.805**) and prompt-level variance; the
largest per-prompt speed delta is still `mixed_ja_en_translate` (**-10.21
tok/s**) where the known bonus-token semantic mismatch occurs.
The latest prefix-state fingerprint diagnostic makes the status sharper: the
stage-wall improvement work is no longer the active blocker, and the current
progress is semantic/economy attribution. The prefill-GDN captured-row lifecycle
changes the committed prefix state immediately after the first full-accept
cycle; by forced pair 12, default prefix state rejects draft token `539` for
`26126` by **0.003027**, while the prefill-GDN prefix accepts `539` by
**0.295256**. All full-attention KV layers differ and 29/30 linear-state layers
differ. That means layer 35 is an amplification point, not the root cause. The
next required llama.cpp comparison is raw `verify_h`/`pending_h`/draft-seed
values around the matching task, so we can decide which hipEngine prefix
lifecycle actually mirrors llama.cpp before optimizing more row kernels.
Evidence:
`benchmarks/results/2026-07-03-mtp-prefix-state-fingerprint-default-vs-prefillgdn.json`.
That comparison is now available in
`benchmarks/results/2026-07-03-mtp-hidden-lifecycle-default-vs-prefillgdn-vs-llamacpp.json`.
llama.cpp's own lifecycle handoff is exact: `draft_seed_input` equals
`process_h_input[0]`, `verify_h[0]` equals `process_h_input[1]`, and
`verify_h[1]` equals `process_h_input[2]` with zero full-vector delta. Against
llama.cpp task 9 / cycle 18, prefill-GDN is slightly closer for the initial
prefix seed (**0.0630 MAE** vs default **0.0668**), but default is closer at the
decisive verifier row 1 (**0.0690** vs **0.0806**) and is the only hipEngine
lane that matches llama.cpp's reject decision (`539 - 26126`: llama
**-0.00896**, default **-0.00303**, prefill-GDN **+0.29526**). The active fix is
therefore not "make the seed more prefill-GDN-like"; it is to preserve
llama.cpp's exact hidden handoff while preventing the row-1 drift/score flip in
the fast captured-state lifecycle.

Prefix-state numeric summary follow-up:
`benchmarks/results/2026-07-03-mtp-prefix-state-numeric-summary-default-vs-prefillgdn.json`.
This adds `--prefix-state-numeric-summary` to the forced-target probe and a
compact reducer over the default-prefix and prefill-GDN-prefix cycle-12
artifacts. It confirms the drift is broad before the row-1 score flip:
**58/60** linear Conv/GDN state components and **20/20** full-attention KV
components hash differently. The row-1 margin is unchanged from the raw probe
(default rejects `539` by **-0.003027**, prefill-GDN accepts it by
**+0.295256**). The largest compact summary deltas are Conv-state samples in
layers **33**, **26**, **18**, **32**, and **30**, while the largest KV summary
deltas are key-cache tails in full-attention layers **15**, **11**, **27**,
**31**, and **35**. This is a triage map, not final attribution: the probe stores
summary statistics, not full pairwise raw state arrays, so the next actionable
instrumentation is selected raw state dumps around those ranked layers plus the
known layer-23/layer-35 verifier-output split.

Selected raw-state follow-up:
`benchmarks/results/2026-07-03-mtp-prefix-state-rawselected-default-vs-prefillgdn.json`.
The forced-target probe now supports selected raw prefix-state dumps via
`--raw-prefix-linear-state-layer` and `--raw-prefix-kv-state-layer`, and the
state-summary reducer computes true pairwise MAE/RMSE/max/cosine when both
inputs contain raw buffers. For the ranked layers above, the largest selected
linear-state deltas are all Conv state, not recurrent state: layer **26** Conv
MAE **0.02723**, layer **30** Conv **0.02611**, layer **33** Conv **0.02408**,
layer **32** Conv **0.02260**, and layer **18** Conv **0.01956**. The matching
recurrent-state MAEs are only **4.8e-05..7.6e-05**. Full-attention KV key deltas
are smaller but real: layer **27** key MAE **0.01591**, layer **31** key
**0.01555**, layer **35** key **0.01449**, layer **15** key **0.01238**, and
layer **11** key **0.01018**. A diagnostic hybrid
`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV=1` then tested whether
using the F32 chain Conv state-row kernel with the fast prefill-GDN recurrent
capture would fix the row-1 accept flip. It did not:
`benchmarks/results/2026-07-03-mtp-prefix-state-summary-prefillgdn-vs-chainconv.json`
shows **0/60** linear-state and **0/20** KV hash changes versus prefill-GDN, and
the row-1 margin remains **+0.295256**. Therefore the selected Conv-state drift
is a downstream symptom of earlier hidden/input history divergence, not the
causal copy target. The next target remains the hidden history before state-row
capture, using llama.cpp `verify_h`/target-layer outputs rather than more Conv
kernel variants.

Hidden-lifecycle ladder follow-up:
`benchmarks/results/2026-07-03-mtp-hidden-lifecycle-ladder-default-vs-prefillgdn-vs-llamacpp.json`.
This aggregates raw llama.cpp hidden lifecycle traces against hipEngine default
and prefill-GDN forced probes for hip cycles **1/3/7/11/12** (llama.cpp task 9
cycles **7/9/13/17/18**). The result changes the diagnosis again: prefill-GDN is
closer to llama.cpp on the prefix hidden seed in **5/5** sampled cycles and
closer on the decisive verifier row in **3/5**, but default is closer on the
decisive token margin in **4/5** and is the only lane that matches the cycle-12
reject. Examples: cycle 7 has prefill-GDN much closer on prefix hidden
(**0.08376 MAE** vs default **0.19385**) and row-1 hidden (**0.07594** vs
**0.10654**), yet default has the closer margin error (**0.16846** vs
**0.24053**). Cycle 12 is the critical near tie: prefill-GDN is closer on
prefix hidden (**0.06304** vs **0.06679**), but default is closer on row-1
hidden (**0.06902** vs **0.08056**) and on token margin (**0.00594** error vs
**0.30422**). Therefore the live accept/economy gap is not solved by making the
resident hidden seed simply "more llama-like"; it is an output-norm / LM-head /
target-score accumulation sensitivity problem around near ties. The next useful
copy target is llama.cpp's final hidden-to-logit path for the decisive rows,
including output norm dtype, LM-head quant/dequant, and any row-specific score
buffering, while retaining the prefill-GDN state lifecycle for speed unless a
full-suite row proves otherwise.

Follow-up scored-boundary handoff diagnostic:
`benchmarks/results/2026-07-03-mtp-process-h-input-vs-layer0-hiddenin-noncapture.json`
and
`benchmarks/results/2026-07-03-mtp-process-h-input-vs-layer0-hiddenin-prefillgdn.json`.
The reducer now supports boundary-only comparisons and merges llama.cpp
`top_k` plus `candidate_scores` sample rows. Result: both hipEngine lanes have
the **same** scored layer-0 `hidden_in` vector for row 1, while comparing it to
llama.cpp `process_h_input` gives the same invalid large delta in both lanes
(**1.80063 MAE**, cosine **0.0593**). This is a label/contract lesson, not a
copy target: `process_h_input` is the MTP process hidden input, not the target
model's layer-0 token embedding. Target-layer input comparisons must use
llama.cpp `model.input_embed` for layer 0 or `verify_layer_output_{N-1}` for
later layers. The active-vs-default flip is therefore not at the row-1 layer-0
input handoff. The same current-env path comparison still says the first
active-vs-noncapture layer-output split over **1e-3 MAE** is layer **23**
(**0.001022**), with the material jump at full-attention layer **35**
(layer-output **0.006418** vs noncapture; final layer-39 **0.007415**). Next
implementation work should inspect the captured-state linear-attention scoring
contract that feeds later full-attention layers, especially the prefill-GDN path
that scores from `recurrent_bf16` while the noncapture side-match scores from
the F32 recurrent output.

Current bonus-row target-logit split, focused on `mixed_ja_en_translate`, cycle
3, verifier row 2 after both engines accept draft `[11, 567]`:

| engine / verifier path | sampled row-2 token | logit 8940 | logit 668 | `8940 - 668` | artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| llama.cpp HIP B2 token trace | **668** | 25.536228 | **25.545841** | **-0.009613** | `benchmarks/results/2026-07-02-llamacpp-mtp-token-trace-b2-natural24-mixed-ja-en-translate.jsonl` |
| hipEngine bulk active verifier | **8940** | **25.841198** | 25.321857 | **+0.519341** | `benchmarks/results/2026-07-02-mtp-target-bonus-row-hipengine-bulk-cycle3.json` |
| hipEngine bulk + `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` | **8940** | **25.795452** | 25.382889 | **+0.412563** | `benchmarks/results/2026-07-02-mtp-target-bonus-row-hipengine-bulk-f32res-cycle3.json` |
| hipEngine bulk + wide F32 verifier-boundary flags | **8940** | **25.798450** | 25.361250 | **+0.437201** | `benchmarks/results/2026-07-02-mtp-target-bonus-row-hipengine-bulk-f32wide-cycle3.json` |
| hipEngine serial-exact verifier | **8940** | **25.754116** | 25.289141 | **+0.464975** | `benchmarks/results/2026-07-02-mtp-target-bonus-row-hipengine-serialexact-cycle3.json` |

Layer-split tensor comparison for the same row is now compacted in
`benchmarks/results/2026-07-02-mtp-bonus-row-verifier-tensor-compare-layer-split.json`
from a llama.cpp `LLAMA_MTP_TENSOR_TRACE` midpoint capture and matching
hipEngine forced-target raw layer-output capture:

| target boundary, row 2 | MAE | RMSE | max abs | cosine | readout |
| --- | ---: | ---: | ---: | ---: | --- |
| layer 0 output | 0.0000547 | 0.0000813 | 0.00189 | 0.999990 | Close; early embedding/layer-0 path is not the divergence. |
| layer 1 output | 0.0000980 | 0.000143 | 0.00310 | 0.999980 | Still close. |
| layer 5 output | 0.000294 | 0.000394 | 0.00497 | 0.999943 | Small drift only. |
| layer 10 output | 0.000693 | 0.000898 | 0.00594 | 0.999844 | Still below the 1e-3 MAE split threshold. |
| layer 20 output | **0.00375** | **0.00474** | 0.0194 | 0.997873 | Coarse split crossed 1e-3 here; refined split below moves the first crossing to layer 13. |
| layer 30 output | 0.00598 | 0.00751 | 0.0255 | 0.998522 | Drift continues to accumulate. |
| layer 39 / pre-output_norm | **0.01227** | **0.01561** | 0.0792 | 0.998601 | Final residual drift before output norm. |
| `verify_h` vs hipEngine target hidden seed | **0.10931** | **0.13915** | 0.534 | 0.998551 | Output norm amplifies the late residual drift into the row used for sampling/commit. |

Refined split artifacts:
`benchmarks/results/2026-07-02-mtp-bonus-row-verifier-tensor-compare-layer-10-20.json`
and
`benchmarks/results/2026-07-02-mtp-bonus-row-verifier-tensor-compare-layer-12-14.json`.

| refined target boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| layer 12 output | 0.000975 | 0.00124 | Still effectively below the split threshold. |
| layer 13 output | **0.00110** | **0.00142** | First measured layer above 1e-3 MAE. |
| layer 14 output | **0.00253** | **0.00319** | First larger jump. |

Layer-boundary follow-up:
`benchmarks/results/2026-07-02-mtp-bonus-row-layer13-linear-attn-compare.json`,
`benchmarks/results/2026-07-02-mtp-bonus-row-layer13-moe-taps-compare.json`,
`benchmarks/results/2026-07-02-mtp-bonus-row-layer14-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-02-mtp-bonus-row-layer14-moe-taps-compare.json`.
Layer 13 is a small router/ranking drift: all 8 selected experts are common,
but ranks 2-4 are permuted; router MAE is **0.01357** and post-MoE MAE is
**0.000943**. Layer 14 is the first material expert-set difference: router top-k
rank 7 differs (**hipEngine expert 175 vs llama.cpp expert 32**), router MAE is
**0.02493**, and the MoE path jumps to **0.00227 ffn_out MAE / 0.00245
post-MoE MAE**. Attention output itself is smaller
(layer 14 linear-attn output **0.000612 MAE**, attention residual
**0.000962 MAE**). The scored live-path split below supersedes the earlier
"copy MoE/router first" target: the first complete layer-14 sub-boundary above
threshold is already the layer-14 attention RMSNorm input, before layer-14
projection, conv/GDN, or MoE router code runs.

New live-path instrumentation (2026-07-03) closes one diagnostic ambiguity:
`scripts/gguf_mtp_forced_target_probe.py` can now emit
`--scored-layer-boundary-row` / `--raw-scored-layer-boundary-row` captures from
the actual bulk/native verifier pass that scored the target rows, rather than
only from isolated single-row layer replay. The artifact
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer14-cycle3.json`
was collected on the same bonus row with direct-state linear-row capture and
`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`; it records the active diagnostic
env separately from route-derived flags. The compact scored-vs-llama comparison
`benchmarks/results/2026-07-03-mtp-bonus-row-layer14-scored-moe-taps-compare.json`
confirms the live scored path has the same material top-k mismatch
(hipEngine `[61, 68, 7, 60, 249, 37, 178, 175]` vs llama.cpp
`[61, 68, 7, 60, 249, 37, 178, 32]`). Scored-path router MAE is **0.02175**
(isolated replay was **0.02493**), `ffn_out` MAE is **0.00228**, and
post-MoE/layer-output MAE is **0.00253**. The small scored-vs-isolated numeric
shift proves the new capture is observing the live verifier path, but it does
not change the token-level failure: layer-14 router-input/logit precision near
the cutoff still produces the expert `175` vs `32` split. Future live-path
bisection should prefer the scored capture block, and use isolated captures only
for narrower single-layer experiments.

The matching scored linear-attention/input split is compacted in
`benchmarks/results/2026-07-03-mtp-bonus-row-layer14-scored-linear-attn-compare.json`.
It uses the same row, a patched temp llama.cpp trace with row-2 values, and an
explicit `task_id=9` filter so the reducer selects the `mixed_ja_en_translate`
cycle-3 record rather than the same cycle from the warmup task.

| layer-14 scored boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `attn_norm_14` vs hipEngine `attn_norm` | **0.02051** | **0.02654** | First complete sub-boundary above threshold; CPU RMSNorm now proves this is incoming hidden drift, not layer-14 RMSNorm arithmetic. |
| `z_14` vs hipEngine `linear_z` | **0.01564** | **0.02035** | Projection output follows the already-drifted input. |
| `beta_14` vs hipEngine `ssm_beta` | 0.00914 | 0.01342 | Near the 1e-2 split threshold. |
| `conv_output_silu_14` vs hipEngine `conv_out` | 0.00120 | 0.00272 | Drift is present but smaller after conv/SILU. |
| `linear_attn_out_14` vs hipEngine `attn_out` | 0.000653 | 0.000844 | Smaller than the input/RMSNorm split. |
| `attn_residual_14` vs hipEngine `attn_residual` | 0.00109 | 0.00140 | Residual carries the input-side drift forward. |
| `attn_post_norm_14` vs hipEngine `attn_post_norm` | **0.02736** | **0.03498** | Post-attention RMSNorm amplifies the residual drift before the MoE router. |
| `ffn_out_14` vs hipEngine reconstructed `ffn_out` | 0.00228 | 0.00290 | Includes the expert-set split. |
| `post_moe_14` vs hipEngine layer output | 0.00253 | 0.00319 | Matches the layer-14 output checkpoint. |

Formula follow-up: the row-2 raw llama.cpp rerun exposes
`verify_layer_output_13`, `attn_norm_14`, and `process_h_input` values. The
valid target-layer input comparison is hipEngine `hidden_in` vs llama.cpp
`verify_layer_output_13`: **0.00109696 MAE / 0.00142193 RMSE**. CPU RMSNorm
with `blk.14.attn_norm.weight` exactly reproduces llama.cpp `attn_norm_14`
from `verify_layer_output_13` and exactly reproduces hipEngine `attn_norm` from
hipEngine `hidden_in` (best candidate max-abs **0** for both). The larger
normalized-space delta (**0.02051 MAE / 0.02654 RMSE**) is therefore expected
amplification of the incoming layer-13 output drift. `process_h_input` is kept
only as context: it is the MTP draft-context hidden input, not the target layer
hidden state, so do not compare it to hipEngine verifier `hidden_in`.

Layer-13 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer13-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer13-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer13-scored-moe-taps-compare.json`.

| layer-13 scored boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_12` vs hipEngine `hidden_in` | **0.000975** | **0.001244** | Incoming layer-12 output is just below the 1e-3 MAE split threshold. |
| `attn_norm_13` vs hipEngine `attn_norm` | **0.01954** | **0.02534** | CPU RMSNorm reproduces both engines within trace rounding; this is incoming hidden drift amplified by normalization. |
| `z_13` vs hipEngine `linear_z` | **0.01662** | **0.02135** | Projection follows the normalized input drift. |
| `conv_output_silu_13` vs hipEngine `conv_out` | 0.00119 | 0.00268 | Small downstream drift; qkv/beta raw taps are trace-label/layout caveated. |
| `linear_attn_out_13` vs hipEngine `attn_out` | 0.000526 | 0.000695 | Not a linear-attention output cliff. |
| `attn_residual_13` vs hipEngine `attn_residual` | 0.00101 | 0.00128 | Carries the incoming drift forward. |
| `ffn_out_13` vs hipEngine reconstructed `ffn_out` | 0.000539 | 0.000714 | MoE projection/combination remains small. |
| `post_moe_13` vs hipEngine layer output | **0.001097** | **0.001422** | Exactly the incoming layer-14 hidden delta measured above. |

Layer-13 MoE is not the next copy target. It only permutes ranks 2/3 among the
same eight selected experts, with no hip-only or llama-only expert; common
expert rows remain close and aggregate `ffn_out` MAE is **0.000539**. The
temporary llama.cpp `linear_attn_qkv_mixed_13`, `alpha_13`, and `beta_13` raw
taps are not reliable semantic oracles in this trace: `alpha_13` aliases
`gate_13`, and qkv/beta differ far more than downstream conv/linear-attention
outputs.

Layer-13 interpretation: that split moved the semantic target to the scored
layer-12 boundary/internal split, not layer-14 MoE, layer-14 RMSNorm, or
layer-13 MoE. Do not copy a different MoE selection rule: the target model is
qwen35moe softmax gating, and the expert-set mismatch is downstream of
accumulated hidden precision reaching a router cutoff.

Layer-12 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer12-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer12-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer12-scored-moe-taps-compare.json`.

| layer-12 scored boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_11` vs hipEngine `hidden_in` | **0.000996** | **0.001264** | Incoming layer-11 output is right at the 1e-3 split threshold. |
| `attn_norm_12` vs hipEngine `attn_norm` | **0.01826** | **0.02342** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_12` vs hipEngine `linear_z` | **0.01290** | **0.01663** | Projection follows the normalized input drift. |
| `conv_output_silu_12` vs hipEngine `conv_out` | 0.000803 | 0.00175 | No conv cliff; qkv/beta raw taps remain trace-label/layout caveated. |
| `linear_attn_out_12` vs hipEngine `attn_out` | 0.000607 | 0.000778 | Not a linear-attention output cliff. |
| `attn_residual_12` vs hipEngine `attn_residual` | 0.000956 | 0.00120 | Carries the incoming drift forward. |
| `ffn_out_12` vs hipEngine reconstructed `ffn_out` | 0.000445 | 0.000592 | MoE projection/combination remains small. |
| `post_moe_12` vs hipEngine layer output | 0.000975 | 0.001244 | Exactly the incoming layer-13 hidden delta measured above. |

Layer-12 MoE is also not the next copy target. After adding
`--llamacpp-task-id` to the MoE reducer, the correct task-9 record shows only a
rank 5/6 swap among the same experts (`71` and `194`), aggregate `ffn_out` MAE
**0.000445**, and post-MoE MAE **0.000975**. At that point the semantic target
moved to the scored layer-11 boundary/internal split.

Layer-11 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer11-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer11-scored-full-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer11-scored-moe-taps-compare.json`.

| layer-11 scored full-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_10` vs hipEngine `hidden_in` | **0.000693** | **0.000898** | Incoming layer-10 output is below the 1e-3 split threshold. |
| `attn_norm_11` vs hipEngine `attn_norm` | **0.01169** | **0.01554** | CPU RMSNorm mostly explains the normalized-space delta from the input drift. |
| `attn_output_11` vs hipEngine `attn_out` | 0.000485 | 0.000630 | No full-attention output cliff. |
| `attn_residual_11` vs hipEngine `attn_residual` | 0.000811 | 0.00103 | Carries the incoming drift forward. |
| `attn_post_norm_11` vs hipEngine `attn_post_norm` | **0.02146** | **0.02755** | Second RMSNorm amplifies the residual-space drift before MoE. |
| `ffn_out_11` vs hipEngine reconstructed `ffn_out` | 0.000595 | 0.000746 | MoE projection/combination remains small. |
| `post_moe_11` / `verify_layer_output_11` vs hipEngine layer output | **0.000996** | **0.001245** | Exactly the incoming layer-12 hidden delta measured above. |

Layer-11 MoE top-k selection matches exactly:
`[210, 147, 7, 154, 107, 27, 106, 251]`. Router logits differ by
**0.0142 MAE / 0.0179 RMSE**, but the cutoff is not crossed; selected weighted
rows are **0.000137 MAE**, aggregate `ffn_out` is **0.000595 MAE**, and
post-MoE is **0.000996 MAE**. Layer 11 is the first full-attention layer in this
backtrace, and it is not where hipEngine diverges from llama.cpp. The next
semantic target moves upstream again to the scored layer-10 boundary/internal
split.

Trace gotcha for future raw layer-output captures: llama.cpp exports target
layer outputs as `verify_layer_output_N`, but the raw tensor value allowlist
must request the pre-translation graph label `l_out_N` in
`LLAMA_MTP_TENSOR_TRACE_VALUE_LABELS`; requesting `verify_layer_output_N` only
captures summaries without `values`.

Layer-10 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer10-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer10-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer10-scored-moe-taps-compare.json`.

| layer-10 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_9` vs hipEngine `hidden_in` | **0.000648** | **0.000832** | Incoming layer-9 output is already the live boundary drift. |
| `attn_norm_10` vs hipEngine `attn_norm` | **0.01561** | **0.02018** | CPU RMSNorm mostly explains the normalized-space delta from the input drift. |
| `z_10` vs hipEngine `linear_z` | **0.01463** | **0.01892** | Projection follows normalized input drift. |
| `conv_output_silu_10` vs hipEngine `conv_out` | 0.000714 | 0.00173 | No conv cliff; qkv/beta raw taps remain trace-label/layout caveated. |
| `linear_attn_out_10` vs hipEngine `attn_out` | 0.000359 | 0.000466 | Not a linear-attention output cliff. |
| `attn_residual_10` vs hipEngine `attn_residual` | 0.000661 | 0.000857 | Carries the incoming drift forward. |
| `attn_post_norm_10` vs hipEngine `attn_post_norm` | **0.01845** | **0.02388** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_10` vs hipEngine reconstructed `ffn_out` | 0.000334 | 0.000426 | MoE projection/combination remains small. |
| `post_moe_10` / `verify_layer_output_10` vs hipEngine layer output | **0.000693** | **0.000898** | Exactly the incoming layer-11 hidden delta measured above. |

Layer-10 MoE top-k selection also matches exactly:
`[91, 24, 252, 73, 92, 165, 105, 72]`. Router logits differ by
**0.0126 MAE / 0.0165 RMSE**, selected weighted rows are **0.0000857 MAE**,
aggregate `ffn_out` is **0.000334 MAE**, and post-MoE is **0.000693 MAE**.
Layer 10 is therefore not the copy target either; the next semantic split is
the scored layer-9 boundary/internal split.

Layer-9 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer9-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer9-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer9-scored-moe-taps-compare.json`.

| layer-9 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_8` vs hipEngine `hidden_in` | **0.000542** | **0.000712** | Incoming layer-8 output remains the live boundary drift. |
| `attn_norm_9` vs hipEngine `attn_norm` | **0.01446** | **0.01873** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_9` vs hipEngine `linear_z` | **0.01229** | **0.01586** | Projection follows normalized input drift. |
| `conv_output_silu_9` vs hipEngine `conv_out` | 0.000748 | 0.00186 | No conv cliff; qkv/beta raw taps remain trace-label/layout caveated. |
| `linear_attn_out_9` vs hipEngine `attn_out` | 0.000386 | 0.000502 | Not a linear-attention output cliff. |
| `attn_residual_9` vs hipEngine `attn_residual` | 0.000570 | 0.000730 | Carries the incoming drift forward. |
| `attn_post_norm_9` vs hipEngine `attn_post_norm` | **0.01774** | **0.02282** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_9` vs hipEngine reconstructed `ffn_out` | 0.000357 | 0.000450 | MoE projection/combination remains small. |
| `post_moe_9` / `verify_layer_output_9` vs hipEngine layer output | **0.000648** | **0.000832** | Exactly the incoming layer-10 hidden delta measured above. |

Layer-9 MoE is the first upstream split in this backtrace where router rank
order differs, but it is only a rank 2/3 permutation among the same eight
experts: hipEngine `[148, 217, 78, 61, 123, 227, 183, 115]` vs llama.cpp
`[148, 217, 61, 78, 123, 227, 183, 115]`. There are still no hip-only or
llama-only experts. Common-expert rows remain close, selected weighted rows are
**0.000684 MAE**, aggregate `ffn_out` is **0.000357 MAE**, and post-MoE is
**0.000648 MAE**. This is a useful cutoff-pressure signal, but not yet a layer
to copy; the next semantic split moves upstream to scored layer 8.

Layer-8 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer8-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer8-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer8-scored-moe-taps-compare.json`.

| layer-8 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_7` vs hipEngine `hidden_in` | **0.000459** | **0.000593** | Incoming layer-7 output remains the live boundary drift. |
| `attn_norm_8` vs hipEngine `attn_norm` | **0.01301** | **0.01682** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_8` vs hipEngine `linear_z` | **0.01194** | **0.01527** | Projection follows normalized input drift. |
| `conv_output_silu_8` vs hipEngine `conv_out` | 0.000902 | 0.00214 | No conv cliff; qkv/beta raw taps remain trace-label/layout caveated. |
| `linear_attn_out_8` vs hipEngine `attn_out` | 0.000342 | 0.000449 | Not a linear-attention output cliff. |
| `attn_residual_8` vs hipEngine `attn_residual` | 0.000505 | 0.000689 | Carries the incoming drift forward. |
| `attn_post_norm_8` vs hipEngine `attn_post_norm` | **0.01620** | **0.02089** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_8` vs hipEngine reconstructed `ffn_out` | 0.000301 | 0.000384 | MoE projection/combination remains small. |
| `post_moe_8` / `verify_layer_output_8` vs hipEngine layer output | **0.000542** | **0.000712** | Exactly the incoming layer-9 hidden delta measured above. |

Layer-8 MoE has more rank movement, but still no expert-set mismatch:
hipEngine `[168, 74, 207, 98, 29, 137, 200, 3]` vs llama.cpp
`[168, 74, 98, 207, 29, 3, 200, 137]`. Common-expert rows remain close,
selected weighted rows are **0.00135 MAE**, aggregate `ffn_out` is
**0.000301 MAE**, and post-MoE is **0.000542 MAE**. This again argues against
copying MoE selection or layer-8 linear attention; the next semantic split moves
upstream to scored layer 7.

Layer-7 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer7-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer7-scored-full-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer7-scored-moe-taps-compare.json`.

| layer-7 scored full-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_6` vs hipEngine `hidden_in` | **0.000310** | **0.000438** | Incoming layer-6 output remains the live boundary drift. |
| `attn_norm_7` vs hipEngine `attn_norm` | **0.00733** | **0.01101** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `attn_output_7` vs hipEngine `attn_out` | 0.000337 | 0.000434 | No full-attention output cliff. |
| `attn_residual_7` vs hipEngine `attn_residual` | 0.000440 | 0.000582 | Carries the incoming drift forward. |
| `attn_post_norm_7` vs hipEngine `attn_post_norm` | **0.01339** | **0.01728** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_7` vs hipEngine reconstructed `ffn_out` | 0.000240 | 0.000304 | MoE projection/combination remains small. |
| `post_moe_7` / `verify_layer_output_7` vs hipEngine layer output | **0.000459** | **0.000593** | Exactly the incoming layer-8 hidden delta measured above. |

Layer-7 MoE top-k selection matches exactly:
`[40, 56, 37, 74, 192, 120, 10, 158]`. Selected weighted rows are
**0.0000644 MAE**, aggregate `ffn_out` is **0.000240 MAE**, and post-MoE is
**0.000459 MAE**. Layer 7 is a full-attention layer and still not the copy
target; the next semantic split moves upstream to scored layer 6.

Layer-6 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer6-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer6-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer6-scored-moe-taps-compare.json`.

| layer-6 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_5` vs hipEngine `hidden_in` | **0.000294** | **0.000394** | Incoming layer-5 output is still the live boundary drift. |
| `attn_norm_6` vs hipEngine `attn_norm` | **0.00811** | **0.01078** | CPU RMSNorm mostly explains the normalized-space delta from the input drift. |
| `z_6` vs hipEngine `linear_z` | 0.00914 | 0.01175 | Below the projection close threshold. |
| `beta_6` vs hipEngine `ssm_beta` | 0.00837 | 0.01045 | Below the projection close threshold. |
| `conv_output_silu_6` vs hipEngine `conv_out` | 0.000332 | 0.000755 | No conv cliff. |
| `linear_attn_out_6` vs hipEngine `attn_out` | 0.000197 | 0.000267 | Not a linear-attention output cliff. |
| `attn_residual_6` vs hipEngine `attn_residual` | 0.000300 | 0.000424 | Carries the incoming drift forward. |
| `attn_post_norm_6` vs hipEngine `attn_post_norm` | **0.00872** | **0.01124** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_6` vs hipEngine reconstructed `ffn_out` | 0.000119 | 0.000151 | MoE projection/combination remains small. |
| `post_moe_6` / `verify_layer_output_6` vs hipEngine layer output | **0.000310** | **0.000438** | Exactly the incoming layer-7 hidden delta measured above. |

Layer-6 is the first reduced linear-attention layer where stable pre-SSM labels
show no projection/conv cliff: `z`, `beta`, convolved q/k/v, and
`linear_attn_out` are all close. MoE only swaps ranks 1/2 among the same expert
set, hipEngine `[42, 166, 162, 193, 126, 177, 73, 55]` vs llama.cpp
`[42, 162, 166, 193, 126, 177, 73, 55]`, with selected weighted rows
**0.000465 MAE**, aggregate `ffn_out` **0.000119 MAE**, and post-MoE
**0.000310 MAE**. This points upstream again, to scored layer 5.

Layer-5 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer5-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer5-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer5-scored-moe-taps-compare.json`.

| layer-5 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_4` vs hipEngine `hidden_in` | **0.000243** | **0.000324** | Incoming layer-4 output remains the live boundary drift. |
| `attn_norm_5` vs hipEngine `attn_norm` | **0.00751** | **0.00998** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_5` vs hipEngine `linear_z` | 0.00805 | 0.01031 | Below the projection close threshold. |
| `beta_5` vs hipEngine `ssm_beta` | 0.00531 | 0.00623 | Below the projection close threshold. |
| `conv_output_silu_5` vs hipEngine `conv_out` | 0.000296 | 0.000753 | No conv cliff. |
| `linear_attn_out_5` vs hipEngine `attn_out` | 0.000164 | 0.000214 | Not a linear-attention output cliff. |
| `attn_residual_5` vs hipEngine `attn_residual` | 0.000262 | 0.000382 | Carries the incoming drift forward. |
| `attn_post_norm_5` vs hipEngine `attn_post_norm` | **0.00871** | **0.01119** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_5` vs hipEngine reconstructed `ffn_out` | 0.000165 | 0.000209 | MoE projection/combination remains small. |
| `post_moe_5` / `verify_layer_output_5` vs hipEngine layer output | **0.000294** | **0.000394** | Exactly the incoming layer-6 hidden delta measured above. |

Layer-5 MoE top-k selection matches exactly:
`[144, 24, 169, 11, 249, 14, 212, 158]`. Selected weighted rows are
**0.0000400 MAE**, aggregate `ffn_out` is **0.000165 MAE**, and post-MoE is
**0.000294 MAE**. Layer 5 again has no projection/conv/MoE copy target; the
next semantic split moves upstream to scored layer 4.

Layer-4 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer4-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer4-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer4-scored-moe-taps-compare.json`.

| layer-4 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_3` vs hipEngine `hidden_in` | **0.000204** | **0.000297** | Incoming layer-3 output remains the live boundary drift. |
| `attn_norm_4` vs hipEngine `attn_norm` | **0.00716** | **0.00952** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_4` vs hipEngine `linear_z` | 0.00798 | 0.01029 | Below the projection close threshold. |
| `beta_4` vs hipEngine `ssm_beta` | 1.23614 | 1.65309 | Trace-label/layout caveated; downstream linear-attention output remains close. |
| `conv_output_silu_4` vs hipEngine `conv_out` | 0.000328 | 0.000907 | No conv cliff. |
| `linear_attn_out_4` vs hipEngine `attn_out` | 0.000157 | 0.000207 | Not a linear-attention output cliff. |
| `attn_residual_4` vs hipEngine `attn_residual` | 0.000223 | 0.000293 | Carries the incoming drift forward. |
| `attn_post_norm_4` vs hipEngine `attn_post_norm` | **0.00793** | **0.01004** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_4` vs hipEngine reconstructed `ffn_out` | 0.000112 | 0.000142 | MoE projection/combination remains small. |
| `post_moe_4` / `verify_layer_output_4` vs hipEngine layer output | **0.000243** | **0.000324** | Exactly the incoming layer-5 hidden delta measured above. |

Layer-4 MoE top-k selection matches exactly:
`[17, 74, 254, 25, 160, 122, 190, 104]`. Router logits differ by
**0.00851 MAE / 0.01103 RMSE**, selected weighted rows are **0.0000318 MAE**,
aggregate `ffn_out` is **0.000112 MAE**, and post-MoE is **0.000243 MAE**.
Layer 4 again has no linear-attention or MoE copy target; the next semantic
split moves upstream to scored layer 3.

Layer-3 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer3-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer3-scored-full-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer3-scored-moe-taps-compare.json`.

| layer-3 scored full-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_2` vs hipEngine `hidden_in` | **0.000152** | **0.000224** | Incoming layer-2 output is already the remaining boundary drift. |
| `attn_norm_3` vs hipEngine `attn_norm` | **0.00528** | **0.00804** | CPU RMSNorm reproduces both engines; normalized drift is input-driven. |
| `attn_output_3` vs hipEngine `attn_out` | 0.000104 | 0.000135 | No full-attention output cliff. |
| `attn_residual_3` vs hipEngine `attn_residual` | 0.000181 | 0.000265 | Carries the incoming drift forward. |
| `attn_post_norm_3` vs hipEngine `attn_post_norm` | **0.00693** | **0.00879** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_3` vs hipEngine reconstructed `ffn_out` | 0.000101 | 0.000127 | MoE projection/combination remains small. |
| `post_moe_3` / `verify_layer_output_3` vs hipEngine layer output | **0.000204** | **0.000297** | Exactly the incoming layer-4 hidden delta measured above. |

Layer-3 MoE top-k selection matches exactly:
`[70, 202, 171, 90, 19, 220, 206, 6]`. Router logits differ by
**0.00689 MAE / 0.00857 RMSE**, selected weighted rows are **0.0000265 MAE**,
aggregate `ffn_out` is **0.000101 MAE**, and post-MoE is **0.000204 MAE**.
Layer 3 is not the copy target; the next semantic split moves upstream to
scored layer 2.

Layer-2 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer2-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer2-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer2-scored-moe-taps-compare.json`.

| layer-2 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_1` vs hipEngine `hidden_in` | **0.0000980** | **0.000143** | Incoming layer-1 output is now the remaining boundary drift. |
| `attn_norm_2` vs hipEngine `attn_norm` | **0.00465** | **0.00640** | CPU RMSNorm exactly reproduces both engines; normalized drift is input-driven. |
| `z_2` vs hipEngine `linear_z` | 0.00612 | 0.00788 | Projection follows normalized input drift. |
| `beta_2` vs hipEngine `ssm_beta` | 0.00529 | 0.00678 | Below the projection close threshold. |
| `conv_output_silu_2` vs hipEngine `conv_out` | 0.000150 | 0.000425 | No conv cliff. |
| `linear_attn_out_2` vs hipEngine `attn_out` | 0.000103 | 0.000135 | Not a linear-attention output cliff. |
| `attn_residual_2` vs hipEngine `attn_residual` | 0.000136 | 0.000200 | Carries the incoming drift forward. |
| `attn_post_norm_2` vs hipEngine `attn_post_norm` | **0.00525** | **0.00664** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_2` vs hipEngine reconstructed `ffn_out` | 0.0000819 | 0.000103 | MoE projection/combination remains small. |
| `post_moe_2` / `verify_layer_output_2` vs hipEngine layer output | **0.000152** | **0.000224** | Exactly the incoming layer-3 hidden delta measured above. |

Layer-2 MoE top-k selection matches exactly:
`[33, 64, 127, 239, 198, 238, 87, 134]`. Router logits differ by
**0.00944 MAE / 0.01225 RMSE**, selected weighted rows are **0.0000187 MAE**,
aggregate `ffn_out` is **0.0000819 MAE**, and post-MoE is **0.000152 MAE**.
Layer 2 has no projection/conv/MoE copy target; the next semantic split moves
upstream to scored layer 1.

Layer-1 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer1-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer1-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer1-scored-moe-taps-compare.json`.

| layer-1 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| `verify_layer_output_0` vs hipEngine `hidden_in` | **0.0000547** | **0.0000813** | Incoming layer-0 output is now the remaining boundary drift. |
| `attn_norm_1` vs hipEngine `attn_norm` | **0.00358** | **0.00518** | CPU RMSNorm mostly explains the normalized-space delta from the input drift. |
| `z_1` vs hipEngine `linear_z` | 0.00639 | 0.00820 | Projection follows normalized input drift. |
| `beta_1` vs hipEngine `ssm_beta` | 0.00460 | 0.00591 | Below the projection close threshold. |
| `conv_output_silu_1` vs hipEngine `conv_out` | 0.000171 | 0.000602 | No conv cliff. |
| `linear_attn_out_1` vs hipEngine `attn_out` | 0.0000678 | 0.000122 | Not a linear-attention output cliff. |
| `attn_residual_1` vs hipEngine `attn_residual` | 0.0000855 | 0.000140 | Carries the incoming drift forward. |
| `attn_post_norm_1` vs hipEngine `attn_post_norm` | **0.00388** | **0.00504** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_1` vs hipEngine reconstructed `ffn_out` | 0.0000583 | 0.0000780 | MoE projection/combination remains small. |
| `post_moe_1` / `verify_layer_output_1` vs hipEngine layer output | **0.0000980** | **0.000143** | Exactly the incoming layer-2 hidden delta measured above. |

Layer-1 MoE top-k selection matches exactly:
`[238, 158, 112, 250, 199, 107, 128, 242]`. Router logits differ by
**0.01023 MAE / 0.01162 RMSE**, selected weighted rows are **0.0000152 MAE**,
aggregate `ffn_out` is **0.0000583 MAE**, and post-MoE is **0.0000980 MAE**.
Layer 1 still has no projection/conv/MoE copy target; the next semantic split
moves upstream to scored layer 0.

Layer-0 follow-up:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-scored-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-scored-moe-taps-compare.json`.
The llama.cpp trace patch now also exposes `model.input_embed`, so the layer-0
target input is no longer inferred from the context-only `process_h_input`
label. Patch/artifacts:
`benchmarks/results/2026-07-03-llamacpp-model-input-embed-trace-patch.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-inputembed-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-inputembed-moe-taps-compare.json`.

| layer-0 scored linear-attention boundary, row 2 | MAE | RMSE | readout |
| --- | ---: | ---: | --- |
| pre-layer-0 target hidden vs llama.cpp `model.input_embed` | **0.00000643** | **0.0000103** | Target input construction is essentially aligned; `process_h_input` remains context-only draft input and must not be compared to target `hidden_in`. |
| `attn_norm_0` vs hipEngine `attn_norm` | **0.00115** | **0.00207** | Small normalized input delta is already present at the first target layer. |
| `z_0` vs hipEngine `linear_z` | 0.00485 | 0.00625 | Projection follows the normalized input delta; no layer-0 projection cliff. |
| `beta_0` vs hipEngine `ssm_beta` | 0.00177 | 0.00273 | Below the projection close threshold. |
| `conv_output_silu_0` vs hipEngine `conv_out` | 0.000118 | 0.000865 | No conv cliff. |
| `linear_attn_out_0` vs hipEngine `attn_out` | 0.0000353 | 0.0000619 | Linear-attention output is already very close. |
| `attn_residual_0` vs hipEngine `attn_residual` | 0.0000374 | 0.0000504 | Carries the tiny first-layer output drift forward. |
| `attn_post_norm_0` vs hipEngine `attn_post_norm` | **0.00193** | **0.00265** | Second RMSNorm amplifies residual-space drift before MoE. |
| `ffn_out_0` vs hipEngine reconstructed `ffn_out` | 0.0000386 | 0.0000494 | MoE projection/combination remains small. |
| `post_moe_0` / `verify_layer_output_0` vs hipEngine layer output | **0.0000547** | **0.0000813** | Exactly the incoming layer-1 hidden delta measured above. |

F32-residual + F32-attention-norm rerun:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32res-attnnorm-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32res-attnnorm-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32res-attnnorm-moe-taps-compare.json`.
This is diagnostic-only (`performance_claim=false`). Important implementation
detail: `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` is inert unless
`HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` also allocates/passes the F32 hidden
buffers. The reducer now emits a separate `attn_norm_f32_scratch` bucket so the
BF16 mirror and the actual F32 scratch are not conflated.

| layer-0 F32-residual+attn-norm diagnostic, row 2 | baseline MAE | F32 diagnostic MAE | readout |
| --- | ---: | ---: | --- |
| `attn_norm_0` vs BF16 mirror `attn_norm` | 0.001155 | 0.001155 | BF16 mirror is intentionally unchanged. |
| `attn_norm_0` vs `attn_norm_f32_scratch` | n/a | **0.000729** | F32 scratch closes about one third of the norm-space delta; remaining delta comes from the tiny input embedding difference amplified by RMSNorm. |
| `z_0` vs `linear_z` | 0.004850 | **0.003687** | Projection input precision helps but does not remove the gap. |
| `conv_output_silu_0` vs `conv_out` | 0.000118 | **0.0000919** | Small improvement, no conv cliff. |
| `linear_attn_out_0` vs `attn_out` | 0.0000353 | **0.0000314** | Small improvement. |
| `post_moe_0` / layer output | 0.0000547 | **0.0000495** | Small improvement, but the sampled bonus still stays `8940`, not llama.cpp's `668`. |

F32-token-embedding + F32-residual + F32-attention-norm rerun:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32embed-f32res-attnnorm-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32embed-f32res-attnnorm-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32embed-f32res-attnnorm-moe-taps-compare.json`.
This is diagnostic-only (`performance_claim=false`) and uses
`HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING=1` to seed the verifier F32 residual
buffer directly from dequantized `token_embd.weight` rows while leaving the BF16
mirror embedding path intact.

| layer-0 F32 token-embedding diagnostic, row 2 | prior F32-res+attn-norm MAE | F32 token-embed MAE | readout |
| --- | ---: | ---: | --- |
| pre-layer-0 target hidden vs `model.input_embed` | 0.00000643 | **0.0** | Input construction is now exact for the captured row. |
| `attn_norm_0` vs `attn_norm_f32_scratch` | 0.000729 | **0.0** | F32 RMSNorm scratch now exactly matches llama.cpp. |
| `attn_norm_0` vs BF16 mirror `attn_norm` | 0.001155 | **0.000906** | Mirror remains BF16 and is not the verifier F32 residual source. |
| `z_0` vs `linear_z` | 0.003687 | **0.002233** | Remaining first split is now projection/dequant from exact F32 normalized input. |
| `beta_0` vs `ssm_beta` | 0.001775 | **0.001754** | Essentially unchanged. |
| `conv_output_silu_0` vs `conv_out` | 0.0000919 | **0.0000667** | Smaller, still no conv cliff. |
| `linear_attn_out_0` vs `attn_out` | 0.0000314 | **0.0000296** | Slightly closer. |
| `attn_post_norm_0` vs `attn_post_norm` | 0.001766 | **0.001667** | Residual-space norm drift remains. |
| `post_moe_0` / layer output | 0.0000495 | **0.0000493** | Essentially unchanged. |

F32 projection-output rerun:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32proj-f32embed-f32res-attnnorm-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-f32embed-f32res-attnnorm-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-f32embed-f32res-attnnorm-moe-taps-compare.json`.
This is diagnostic-only (`performance_claim=false`) and uses
`HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS=1` to route compatible
linear-attention `attn_qkv`/`attn_gate` projections through the raw-Q8 dp4a
F32-output dual wrapper, while preserving BF16 mirror buffers for downstream
consumers and captures.

| layer-0 F32 projection-output diagnostic, row 2 | F32 token-embed MAE | F32 projection-output MAE | readout |
| --- | ---: | ---: | --- |
| pre-layer-0 target hidden vs `model.input_embed` | **0.0** | **0.0** | Input construction remains exact. |
| `attn_norm_0` vs `attn_norm_f32_scratch` | **0.0** | **0.0** | F32 norm scratch remains exact. |
| `z_0` vs `linear_z` | 0.002233 | **0.0000000841** | Q8 `attn_gate` projection output is now effectively closed. |
| `beta_0` vs `ssm_beta` | 0.001754 | **0.001754** | Unchanged: `ssm_beta.weight` is dense F32, not Q8_0, so the raw-Q8 output wrapper cannot cover it. |
| `conv_output_silu_0` vs `conv_out` | 0.0000667 | **0.00000000283** | Conv/GDN input from qkv is now effectively closed. |
| `linear_attn_out_0` vs `attn_out` | 0.0000296 | **0.0000284** | Slightly closer, but still not a cliff. |
| `attn_post_norm_0` vs `attn_post_norm` | 0.001667 | **0.001607** | Residual-space norm drift remains. |
| `post_moe_0` / layer output | 0.0000493 | **0.0000493** | Essentially unchanged. |

Dense-F32 alpha/beta F32-output rerun:
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-cycle3.json`,
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-linear-attn-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-moe-taps-compare.json`.
The post-projection/pre-`ssm_out` reducer refresh is
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-pre-ssm-linear-attn-compare.json`.
The F32 post-attention-norm diagnostic rerun is
`benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32proj-densef32ab-f32postnorm-keepf32-f32embed-f32res-attnnorm-cycle3.json`
with reducer
`benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-f32postnorm-keepf32-f32embed-f32res-attnnorm-pre-ssm-linear-attn-compare.json`.
This is diagnostic-only (`performance_claim=false`). It adds the missing
registry-dispatched dense-F32 F32-input/F32-output GEMV route for
`ssm_alpha`/`ssm_beta` and fixes the capture guard that was overwriting the new
F32 scratch with BF16-widened mirrors.

| layer-0 dense-F32 alpha/beta F32-output diagnostic, row 2 | prior F32 projection-output MAE | dense-F32 alpha/beta MAE | readout |
| --- | ---: | ---: | --- |
| pre-layer-0 target hidden vs `model.input_embed` | **0.0** | **0.0** | Input construction remains exact. |
| `attn_norm_0` vs `attn_norm_f32_scratch` | **0.0** | **0.0** | F32 norm scratch remains exact. |
| `z_0` vs `linear_z` | 0.0000000841 | **0.0000000841** | Q8 `attn_gate` projection remains closed. |
| `beta_0` vs `ssm_beta` | 0.001754 | **0.0000000740** | Dense-F32 `ssm_beta` projection is now effectively closed. |
| `conv_output_silu_0` vs `conv_out` | 0.00000000283 | **0.00000000283** | Conv/GDN input remains effectively closed. |
| `linear_attn_out_0` vs `attn_out` | 0.0000284 | **0.0000285** | Unchanged: remaining drift is after the projection/conv inputs. |
| `attn_post_norm_0` vs `attn_post_norm` | 0.001607 | **0.001603** | Residual-space norm drift remains. |
| `post_moe_0` / layer output | 0.0000493 | **0.0000488** | Essentially unchanged. |

Post-projection split result:

| layer-0 post-projection / post-norm diagnostic, row 2 | dense-F32 alpha/beta MAE | F32 post-norm MAE | readout |
| --- | ---: | ---: | --- |
| `recurrent_bf16` vs llama.cpp `final_output_0` | **0.00000623** | **0.00000623** | Pre-`ssm_out` is closed for the live scored row. The JSONL contains `final_output_cont_0` summary fields but not full raw `values`, so the reducer uses the full-valued `final_output_0` fallback. |
| `linear_attn_out_0` vs `attn_out` | **0.0000285** | **0.0000285** | `ssm_out` remains closed; no recurrent-GDN/`ssm_out` copy target. |
| `attn_post_norm_0` vs BF16 mirror | **0.001603** | **0.001603** | BF16 mirror unchanged. |
| `attn_post_norm_0` vs F32 scratch / router input | n/a | **0.001528** | F32 post-norm helps the local post-norm bucket slightly, but not enough to affect the token decision. |
| `post_moe_0` / layer output | **0.0000488** | **0.0000460** | Small local improvement only. |

The live branch still samples hipEngine bonus `8940`, not llama.cpp bonus `668`.
The scored block remains `[11, 567, 8940]`; row 2 scores `8940=25.67706`
rank 1 and `668=25.31575` rank 3, so `668` is still **0.36131** behind. This
now closes the target-input/RMSNorm-scratch, Q8 `attn_qkv`/`attn_gate`,
dense-F32 `ssm_beta`, conv-input, recurrent-GDN, and `ssm_out` hypotheses for
layer 0. Enabling F32 post-norm still samples `8940`; row 2 scores
`8940=25.71944` rank 1 and `668=25.26696` rank 4, so the margin worsens to
**+0.45249**. The next semantic target is accumulated residual/RMSNorm drift
across layers and final LM-head amplification, not layer-0 recurrent-GDN,
`ssm_out`, post-norm buffer selection, attention, or MoE implementation.

Layer-0 MoE top-k selection matches exactly:
`[57, 6, 56, 66, 127, 110, 106, 157]`. Router logits differ by
**0.00389 MAE / 0.00464 RMSE**, aggregate `ffn_out` by **0.0000394 MAE**, and
post-MoE by **0.0000488 MAE**. Layer 0 now has no projection/conv/MoE copy
target.

The live token mismatch remains a sensitive LM-head near-tie after accumulated
hidden drift, not a layer body cliff. On the same task-9/cycle-3/row-2 branch,
llama.cpp samples bonus `668` with logits `668=25.54584`, `8940=25.53623`
(only **0.00961** apart), while hipEngine samples `8940` with logits
`8940=25.67706`, `668=25.31575` (**0.36131** apart). The next instrumentation
target is therefore accumulated residual/RMSNorm drift and final hidden/LM-head
amplification, not copying a different layer-0 attention, recurrent-GDN,
`ssm_out`, post-norm-buffer, or MoE implementation.

Important correction for the next implementation pass: this GGUF advertises
`general.architecture = qwen35moe`, and the current llama.cpp qwen35moe MTP path
calls `build_moe_ffn(..., exp_probs_b=nullptr, gating=SOFTMAX)`. The Step35
`sigmoid(logit) + ffn_exp_probs_b.bias` selection path does **not** apply to this
model. The observed layer-14 expert swap is therefore not a missing router-bias
mechanism; softmax-over-all-experts followed by selected-weight renormalization
is equivalent to hipEngine's softmax over the selected raw logits. The live gap
is accumulated hidden/RMSNorm precision that reaches the router logits near the
top-k cutoff. New boundary captures keep the legacy `attn_post_norm` field for
compatibility but also emit `attn_post_norm_bf16` so future comparisons do not
confuse the BF16 scratch capture with the verifier's optional F32 residual
diagnostics.

A tempting exact-mode
shortcut remains rejected:
`--target-block-direct-partial-replay-mode bulk-state-only` still emitted the
same visible cycle-3 token `[65342]`, but the lifecycle comparator found
`first_mismatch` at cycle 3 with hidden seed plus Conv/GDN state mismatches
across 61 fingerprints. So the next fix must be a real prefix-equivalent partial
commit/capture path, not just `verify_target_block(..., advance_state_only=True)`
after snapshot restore. A second scheduler-only shortcut,
`native-state-only`, also failed in the active bulk-scoring shape: visible tokens
still matched, but cycle 3 diverged across 59 hidden/linear-state fingerprints.
The explicit llama-style direct-commit lifecycle diagnostic
`benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`
also diverges from the serial replay baseline at cycle 3 with matching visible
tokens; that is expected for the replication lane and is why the serial-state
control is kept separate.

Historical semantic target before the lifecycle fix was the long proposal
trace's pair-12 target accept/reject mismatch:
both engines draft `[15495, 539]`, but hipEngine accepts `539` while llama.cpp
rejects it and samples `26126`. The forced-prefix target score diagnostic
narrows this to a row-1 logits tie-break after input token `15495`: hipEngine
serial-exact keeps `539` ahead of `26126` by **0.118 logits**, while llama.cpp
puts `26126` ahead of `539` by **0.009 logits**. The follow-up hidden-row and
per-layer checkpoint diagnostics correct two tensor-alignment traps:
llama.cpp's `process_h_input` is shifted, so `process_h_input` row `i+1`
corresponds to verifier `verify_h` row `i`, and the target graph has two
`h_nextn`-like drains unless the `verify_h`/`verify_pre_output_norm` labels are
used. With the corrected labels, row-1 target residual drift is gradual, not a
single bad layer: hipEngine serial-exact vs llama.cpp rises from about
**5.3e-4 MAE after layer 1** to **1.02e-2 MAE after layer 39/pre-output_norm**,
then final output_norm scales that to **7.79e-2 MAE** on `verify_h`. The
layer-31 sub-boundary diagnostic shows no hidden single-stage cliff inside that
late layer: hipEngine layer output vs llama.cpp `l_out_31`/`post_moe_31` is
**0.00528 MAE / 0.00671 RMSE / 0.99871 cosine**, and direct reconstructed
hipEngine MoE `ffn_out` vs llama.cpp `ffn_out_31` is **0.00446 MAE /
0.00562 RMSE / 0.99161 cosine**. CPU recomputation of
`output_norm` from each engine's raw pre-output row exactly reproduces that
engine's `verify_h`, and applying the same CPU norm to both pre-output rows
reproduces the final **0.07789 MAE** hidden delta. A raw row-1 hidden dump plus
CPU-dequantized `output.weight` rows for tokens `539` and `26126` reproduces
each engine's ranking, so the remaining mismatch is accumulated target hidden
production precision drift before final output_norm/lm-head, not a different
lm-head ordering or output_norm implementation. The live implementation
hypothesis is that hipEngine's BF16 verifier layer boundaries do not exactly
match llama.cpp's F32 target `l_out` graph tensors. The first opt-in
`HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` slice confirms residual-boundary
precision is active, but the residual-only version flips an earlier hipEngine
cycle-2 near-tie (`25` vs `1590`) instead of proving the llama.cpp cycle-12
decision. Extending the same diagnostic to feed layer-entry attention RMSNorm
from FP32 residual rows reaches the old cycle-12 branch, but still samples
`[15495, 539, 1151]` and increases the wrong row-1 `539` over `26126` margin to
**+0.143 logits** versus llama.cpp's **-0.009**. That rules out attention-norm
input precision alone. It is still not full llama.cpp F32 graph parity because
attention norm outputs plus selected/shared MoE projection inputs continue
through BF16 scratch boundaries. The cross-engine fine MoE tap comparison for
layer 31 completes the next split: router logits are already extremely close
(**0.0160 MAE / 0.0205 RMSE / 0.999995 cosine**) but the top-k cutoff is close
enough to change selection. hipEngine picks
`[221, 95, 240, 60, 88, 19, 212, 59]`; llama.cpp picks
`[221, 95, 240, 60, 19, 88, 212, 75]`. For the seven common experts, selected
SwigLU/down/weighted rows are close once aligned by expert id, the shared-gate
logit differs by only **0.00356**, and aggregate `ffn_out` / post-MoE deltas
remain **0.00446 / 0.00528 MAE**. That puts the live semantic port back on
accumulated F32/BF16 target hidden production feeding router inputs, not on a
different selected-MoE combine rule or a layer-31 projection cliff. The
follow-up all-layer router trace makes that concrete: layer 0 top-k matches
llama.cpp, then the first router top-k divergence is already layer 1. Layer 1
differs only at rank 8 (hipEngine expert `126`, llama.cpp expert `63`) with
router logits still **0.00562 MAE / 0.00694 RMSE / 0.999999 cosine** and
routing weights **0.00054 MAE**. Across 40 layers, **29 layers match top-k** and
11 have near-tie rank/cutoff differences. The next semantic split should
therefore compare raw layer-0 output / layer-1 router input and audit the
earliest residual boundary, not later MoE projection math.

That layer-0/1 boundary split now confirms the handoff direction:
hipEngine's own layer-0 output to layer-1 input is exact (**0 MAE**), but that
same row is already **0.000203 MAE / 0.000263 RMSE / 0.999950 cosine** away
from llama.cpp `post_moe_0` before layer 1 begins. The first layer-1 router
cutoff split is still the same rank-7 near tie (hipEngine expert `126` vs
llama.cpp expert `63`); the cutoff logits differ by about **0.0100** for
expert `126` and **0.00333** for expert `63`, while full layer-1 router logits
are **0.00562 MAE / 0.00694 RMSE / 0.999999 cosine**. Layer-1 output is still
close (**0.000535 MAE / 0.000664 RMSE / 0.999834 cosine**). One dirty
llama.cpp label caveat remains: `attn_norm_0` does not align with hipEngine's
layer-0 `attn_norm` in this trace even though downstream layer-0 residual,
FFN output, and post-MoE do align closely, so do not use `attn_norm_0` as the
split until the llama.cpp trace label is revalidated. The next implementation
target is therefore true GGML-like F32 projection/output contracts through the
early target layer, not a broken hipEngine layer handoff.

One implementation bug in that diagnostic lane is now fixed: the
`HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8` sub-path only affected the
shared dp4a path; BF16 pair/concat fallbacks still consumed `scratch.post_norm`.
The fallback now bypasses pair fusion and uses supported F32-input singleton
shared gate/up projections when the layout has a registered F32 dispatch. That
removes the earlier artificial cycle-7 failure in the full F32 post-norm slice,
but it does **not** solve the llama.cpp semantic mismatch: the retained smoke
still reaches pair 12 and ranks `539` over `26126` by **+0.123926 logits**.
That made the selected/shared internals testable rather than suspect by
omission.

The next F32 projection-input split is now measured. A new default-off
`HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` diagnostic materializes layer-entry
attention RMSNorm into FP32 scratch, keeps the BF16 mirror for unsupported
callers, and routes dense-Q8 dp4a QKV / QKV+gate consumers from the FP32 tensor.
On the active pair-12 bulk verifier probe this moves the row-1
`539 - 26126` margin in the right direction, from **+0.31369** under the
FP32-residual bulk control to **+0.18198**, but it still samples
`[15495, 539, 1151]` and accepts 2 versus llama.cpp's `[15495, 26126]`.
So attention-norm output + dense-Q8 projection input precision is part of the
hidden drift budget, but it is not the missing parity fix. The remaining live
suspects are projection/output contracts that still round through BF16 after
the projection: selected/shared gate/up/down outputs, SwigLU intermediates,
shared output, and the combine path versus llama.cpp's GGML F32 graph tensors.
The follow-up `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` split keeps the
linear-attention `ssm_out` projection output in FP32 through the post-attention
residual/RMSNorm add, then preserves the BF16 mirror for existing consumers. It
only nudges the same pair-12 bulk margin from **+0.18198** to **+0.17663** and
still accepts `539`, so the linear-attention output-to-residual BF16 round is
not the missing semantic lever either. The next llama.cpp source-shaped split
also landed negative: llama.cpp builds `beta` and `alpha` from the same
normalized `cur` used by QKV/Z in
`src/models/qwen35moe.cpp::build_layer_attn_linear`, while hipEngine had only
upgraded QKV/Z consumers. The new
`HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1` diagnostic routes row-bulk
`ssm_alpha`/`ssm_beta` from the FP32 attention-norm tensor, but the pair-12 row
is byte-identical to the prior attention-output slice: still
`[15495, 539, 1151]`, accepted 2, with row-1 `539 - 26126` **+0.17663**. So
alpha/beta projection input precision is ruled out for this branch as well. The
full-attention output boundary is now measured too: the same
`HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` diagnostic routes row-bulk full-attention
`attn_output` through a BF16-input/FP32-output path when the raw Q8 sidecar is
available, casts the BF16 mirror, and feeds the FP32 output into the
post-attention residual/RMSNorm helper. Pair 12 still samples
`[15495, 539, 1151]`, accepts 2, and the row-1 `539 - 26126` margin worsens to
**+0.27480** (`26.22991 - 25.95511`). So the full-attention `attn_output` BF16
output round is not the parity lever either.

The follow-up layer-0/1 fine-MoE comparison now rules out a hidden early
selected/shared combine cliff. Layer 0 router top-k matches llama.cpp; selected
weighted rows are at most **0.0000764 MAE** by common expert, shared-gate logit
differs by **0.00170**, `ffn_out` is **0.000126 MAE**, and post-MoE/layer output
is exactly the already-known **0.000203 MAE** boundary drift. Layer 1's first
local semantic split is again router cutoff, not projection math: router logits
are **0.00562 MAE / 0.00694 RMSE / 0.999999 cosine**, the rank-7 cutoff is
hipEngine expert `126` vs llama.cpp expert `63`, common-expert weighted rows are
only **<=0.0001005 MAE**, shared-gate logit differs by **-0.00474**, and
`ffn_out`/post-MoE stay close (**0.000489 / 0.000535 MAE**). The active target
therefore returns to the accumulated layer-output/router-input precision
boundary before layer 1, not selected/shared intermediate or down-output math.

The layer-0 linear-attention split now rules out a large projection/conv cliff.
A generated patch applied to a temporary llama.cpp tree adds raw target labels
for `linear_attn_qkv_mixed_0`, `z_0`, `beta_0`, `conv_output_silu_0`,
`q_conv_0`, `k_conv_0`, `v_conv_0`, `linear_attn_out_0`, `attn_residual_0`,
`attn_post_norm_0`, `ffn_out_0`, and `post_moe_0`. Comparing those labels
against hipEngine's forced row-1 layer-0 boundary capture shows stable
pre-`ssm_out` labels are already close: `z` projection **0.004111 MAE /
0.999996 cosine**, `beta` projection **0.001866 MAE / 0.999999 cosine**,
`conv_output_silu` **0.0001452 MAE / 0.9999995 cosine**, and q/k/v conv-view
slices **<=0.0001865 MAE**. Downstream, `linear_attn_out` remains
**0.0001595 MAE / 0.0002100 RMSE / 0.999952 cosine**, `attn_residual`
**0.0001630 MAE / 0.0002197 RMSE / 0.999962 cosine**, `attn_post_norm` scales
that to **0.005668 MAE** at post-norm RMS ~0.64, then `ffn_out` adds only
**0.0001256 MAE** and final post-MoE/layer output is the known
**0.0002027 MAE**. The active target is accumulated small target-hidden drift,
not a bad layer-0 projection, conv layout, selected-MoE projection, or combine
path.

Two llama.cpp labels remain trace caveats rather than semantic evidence:
`linear_attn_qkv_mixed_0` does not align directly with hipEngine `linear_qkv`
even though downstream `conv_output_silu_0` matches closely, and `alpha_0` is
byte-identical to `gate_0` in the trace. Do not use those labels as raw
projection oracles until the debug extractor is revalidated.

The `final_output_0` caveat is now resolved: it was a bad trace oracle, not a
hipEngine GDN magnitude issue. The generated
`scripts/llamacpp_mtp_final_output_cont_trace_patch.py` diagnostic patch adds
`final_output_cont_` to both llama.cpp trace allowlists, emits
`final_output_cont`, and feeds `ssm_out` from that contiguous tensor in the
temporary trace tree so the tensor is materialized. With that projectable tap,
llama.cpp `final_output_cont_0 -> ssm_out` reconstructs llama.cpp
`linear_attn_out_0` at **0.000157 MAE / 0.000207 RMSE**, while hipEngine
`recurrent_out -> ssm_out` still reconstructs hipEngine `attn_out` at **0 MAE**.
The direct pre-`ssm_out` boundary is also tight: hipEngine `recurrent_out` vs
llama.cpp `final_output_cont_0` is **0.0000176 MAE / 0.0000391 RMSE /
0.999999 cosine**, and post-`ssm_out` remains **0.0001595 MAE**. Layer-0
GDN/recurrent layout and `ssm_out` projection are therefore ruled out as the
semantic mismatch source; keep chasing the accumulated layer-output/router-input
precision drift after the already-known **0.0002027 MAE** layer-0 output delta.

The selected-FFN precision ladder now has the first pair-12 side-matching split.
On top of the residual + attention-norm-output + attention-output + alpha/beta
+ F32 MoE combine + FP32 selected-down stack, the default-off
`HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1` diagnostic computes
`silu(gate) * up` into FP32 scratch, preserves the BF16 mirror, and feeds the
FP32 selected intermediate into selected-down. That changes the active pair-12
row-1 decision from accepting draft token `539` to sampling `26126`: sampled
tokens become `[15495, 26126, 1151]`, accepted drafts fall from 2 to 1, and
row-1 `539 - 26126` moves from **+0.00536** to **-0.00303** (`26.04795 -
26.05098`). llama.cpp is still a little farther negative at about **-0.00896**,
but this is the first hipEngine verifier slice on the same side of the
near-tie. The selected SwigLU/intermediate BF16 round is therefore a real
llama.cpp parity contract, not just another noisy F32 probe. Next semantic work
is to validate the same contract across longer proposal traces/full-suite
acceptance and decide whether the llama-compat verifier should adopt a cohesive
F32 selected-FFN/MoE graph mode rather than a pile of independent probes.

The first live validation of that side-matching slice is negative. A corrected
`--target-block-replay-state-commit` diagnostic now scores the block with the
non-capturing bulk verifier, then restores and replays only the accepted prefix
through the serial-exact path for resident state. This proves the state lifecycle
can be made transactional (`38` replay rows, `0` direct commits on the 13-cycle
smoke), but it is not a llama.cpp replication fix: the run diverges earlier at
cycle 2, where non-capturing bulk scores `[40798, 1590, 1103]` and emits
`[40798, 1590]` instead of the active direct-state trace's
`[40798, 25, 1103]`. Therefore the pair-12 forced-prefix win is prefix-local;
the live path still needs capture-path / F32 verifier graph parity, not a
two-pass score-bulk/serial-replay mode.

The direct capture-path split is now narrower and more useful. Artifact
`benchmarks/results/2026-07-02-mtp-capture-path-diagnostics.json` compares the
same forced pair-12 row with and without `capture_linear_state_rows` and adds
default-off capture diagnostics for BF16 GDN output, prefill-shaped Conv/GDN
row snapshots, and "score with prefill math, commit chain rows". Results:

| forced pair-12 verifier path | row-state capture | sampled tokens | accepted | row-1 `539 - 26126` | reading |
| --- | --- | --- | ---: | ---: | --- |
| non-capturing bulk current cycle | no | `[15495, 26126, 1151]` | 1 | **-0.00303** | Prefix-local side match. |
| default direct-state capture | yes | `[15495, 539, 1151]` | 2 | **+0.08004** | Active live mismatch. |
| capture + BF16 GDN output | yes | `[15495, 539, 1151]` | 2 | **+0.29526** | BF16-ing the chain output is worse. |
| capture + prefill Conv/GDN state rows | yes | `[15495, 539, 1151]` | 2 | **+0.29526** | Replacing row-state kernels is not enough. |
| capture score-prefill + chain commit | yes | `[15495, 539, 1151]` | 2 | **+0.29526** | Scoring current rows differently is not enough once prior verifier hidden/KV history changes. |

The key correction is that the active split is not a bad BF16 layer boundary in
isolation. Default capture versus non-capturing bulk has identical raw layer-0
boundary tensors in the isolated tap, including `layer_out`, but the scored
FP32 residual/hidden mirror already differs after layer 0 (**0.0000615 MAE**),
then grows through layer 1 (**0.000502 MAE**) and layer 39/pre-output
(**0.007225 MAE**). That points the next fix at the verifier FP32 hidden/KV
history contract used by direct-state block verification versus llama.cpp, not
at simply copying a different Conv/GDN row-state kernel.

This split now has a repeatable reducer:
`scripts/gguf_mtp_compare_forced_target_paths.py`. It compares two hipEngine
forced-target artifacts by sampled token, accepted count, candidate-token margin,
and per-layer hidden-vector drift. The first retained outputs are
`benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-default-compare.json`
and
`benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-prefillconvgdn-compare.json`;
both are diagnostic-only (`performance_claim=false`).

| forced pair-12 path comparison | row-1 `539 - 26126` movement vs noncapture | sampled-token change | accepted delta | layer-0 MAE | first layer >= 1e-3 MAE | layer-39 MAE | reading |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| noncapture -> default direct-state capture | **+0.083063** (`-0.00303 -> +0.08004`) | `26126 -> 539` | +1 | 0.0000615 | layer 11, 0.001095 | 0.007225 | Active direct-state path perturbs the FP32 hidden mirror enough to cross the near-tie. |
| noncapture -> capture + prefill Conv/GDN rows | **+0.298283** (`-0.00303 -> +0.29526`) | `26126 -> 539` | +1 | 0.0000000 | layer 23, 0.001022 | 0.007415 | Layer-0 state rows can be made identical and the final target decision still diverges. |

The reducer output is the active acceptance/economy watchpoint for the next
direct-state verifier fix: a useful patch should move the default-capture margin
toward the noncapture/llama side, preserve or reduce the hidden drift ladder, and
eventually change row 1 back to `26126` without transactional serial replay.

The same reducer now has an optional scored-boundary mode for raw
`scored_layer_boundary_captures` arrays. Fresh current-env probes captured layers
0 and 1 for the F32 selected-intermediate pair-12 branch:
`benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-default-boundary-l0-l1-compare.json`
and
`benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-prefillgdn-boundary-l0-l1-compare.json`.
The raw scored-boundary dumps are intentionally not the retained compact
evidence; the compare artifacts ignore stale `recurrent_out` from the fused
noncapture path.

| current-env scored-boundary comparison | row-1 margin movement | layer-0 split | layer-1 split | reading |
| --- | ---: | --- | --- | --- |
| noncapture -> default chain capture | **+0.083063** | `hidden_in`/F32 attn-norm exact; `attn_out` **3.06e-05 MAE**, post-attn norm **0.001200**, router logits **0.003050**, layer out **6.15e-05** | layer input **6.15e-05**, F32 attn-norm **0.002402**, router logits **0.004681**, layer out **0.000502** | Default chain scoring perturbs the FP32 residual mirror immediately. Keep it as a diagnostic, not the active no-copy compat path. |
| noncapture -> active prefill-GDN no-copy capture | **+0.298283** | `hidden_in`, F32 attn-norm, `attn_out`, post-attn norm, router logits, and layer out all **0 MAE** | layer input/F32 attn-norm/projections exact; `attn_out` **3.01e-05**, post-attn norm **0.000874**, router logits **0.001316**, layer out **6.73e-05** | Active no-copy row-state capture is layer-0 exact. The remaining accept/economy bug is later accumulated verifier hidden/MoE history, not the layer-0 Conv/GDN capture kernel. |

This changes the next copy target: do not spend more time on layer-0
Conv/GDN/no-copy row materialization unless a later reducer regresses it. The
next split should capture active prefill-GDN scored boundaries deeper in the
stack around the first hidden-drift threshold and then compare the corresponding
llama.cpp GGML tensor contract for post-attn norm, router input, selected
SwigLU/down, and final residual accumulation.

That deeper split is now captured for the active F32 selected-intermediate
environment at layers 22-24. The retained compact artifacts are
`benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-prefillgdn-boundary-l22-l24-compare.json`,
`benchmarks/results/2026-07-03-mtp-noncapture-vs-llamacpp-layer22-24-compare.json`,
and
`benchmarks/results/2026-07-03-mtp-capture-prefillgdn-vs-llamacpp-layer22-24-compare.json`.
They are diagnostic-only (`performance_claim=false`), but they make the current
status precise: active capture is no longer losing at layer 0, and the first
large live drift is the layer-23-era verifier hidden/MoE contract.

| current-env forced pair-12 comparison | row-1 `539 - 26126` | sampled token | layer-22 out MAE | layer-23 out MAE | layer-24 out MAE | reading |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| noncapture side-match vs llama.cpp | hip **-0.00303**, llama **-0.00896** | both `26126` | 0.001864 | 0.001975 | 0.001909 | Noncapture is still the semantic side-match for the near-tie, but it shares about 0.0019 layer-output drift vs llama.cpp by layers 22-24. |
| active prefill-GDN capture vs noncapture | **+0.298283** movement (`-0.00303 -> +0.29526`) | `26126 -> 539` | 0.000906 | **0.001022** | 0.001047 | Active capture crosses the 1e-3 hidden-drift threshold at layer 23 and flips the verifier decision. |
| active prefill-GDN capture vs llama.cpp | hip **+0.29526**, llama **-0.00896** | hip `539`, llama `26126` | 0.001878 | 0.001994 | 0.001904 | The cross-engine gap is not explained by layer-0 row materialization; compare the layer-23 sub-boundaries directly against llama.cpp. |

The largest active capture-vs-noncapture sub-boundary deltas in this window are
post-attn norm / router input, router logits, and selected MoE SwigLU/down:
layer 22 post-attn norm **0.012459 MAE**, layer 23 post-attn norm
**0.012981**, router logits **0.008348**, selected SwigLU **0.007505**, and
layer 24 router logits **0.017107**.

The corresponding llama.cpp sub-boundary trace now exists. The retained compact
cross-engine reducers are
`benchmarks/results/2026-07-03-mtp-capture-prefillgdn-vs-llamacpp-layer22-24-subboundary-compare.json`
and
`benchmarks/results/2026-07-03-mtp-noncapture-vs-llamacpp-layer22-24-subboundary-compare.json`.
They compare the same row/layers for attn norm, derived post-attention residual,
post-attn norm, router logits, selected MoE, shared expert, reconstructed FFN
output, and final `post_moe`/layer output. The important result is negative:
capture and noncapture have nearly the same cross-engine sub-boundary profile.
For layer 23, capture vs llama.cpp has derived residual **0.154562 MAE**,
FFN out **0.154545**, post-attn norm **0.026442**, but final `post_moe` only
**0.001994**; noncapture is essentially the same (**0.155768** residual,
**0.155727** FFN out, **0.026246** post-attn norm, **0.001975** final
`post_moe`). This rules out a capture-only expert-selection or selected-FFN
sub-boundary bug in layers 22-24. The active next target is the small
capture-vs-noncapture hidden drift that accumulates after layer 1 and flips the
near-tie at the final target score, not the large raw MoE/FFN labels that cancel
inside both hip paths.

The prefix-state fingerprint split supersedes the earlier read that this was
mostly a current-cycle scorer problem. `_replay_prior_cycles()` uses captured
rows for prior blocks, so `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` changes
the committed verifier prefix before cycle 12 is scored. The compact artifact is
`benchmarks/results/2026-07-03-mtp-prefix-state-fingerprint-default-vs-prefillgdn.json`
(`performance_claim=false`).

| prefix diagnostic | position / prev | sampled tokens | row-1 `539 - 26126` | hidden seed | state mismatch | reading |
| --- | --- | --- | ---: | --- | --- | --- |
| cycle 1 default vs prefill-GDN | pos 43 / prev 727 | both `[10562, 87682, 1494]` | n/a | differs | 29/30 linear layers, 10/10 full-attn KV layers | Prefix state diverges immediately after cycle 0, before a visible token mismatch. |
| cycle 12 default prefix | pos 72 / prev 653 | `[15495, 26126, 1151]` | **-0.003027** | `0c31da6a556e81ff1b93e855df7e7049` | baseline | This side matches llama.cpp's reject decision for the near-tie. |
| cycle 12 prefill-GDN prefix | pos 72 / prev 653 | `[15495, 539, 1151]` | **+0.295256** | `47f3a377a5c77b2a53b7ccec9d0794fe` | 29/30 linear layers, 10/10 full-attn KV layers | Active no-copy lifecycle flips the decision because prior prefix state changed, not because layer-35 alone is wrong. |

The raw hidden lifecycle comparison is now retained as
`benchmarks/results/2026-07-03-mtp-hidden-lifecycle-default-vs-prefillgdn-vs-llamacpp.json`.
It uses llama.cpp task 9 / cycle 18 and hipEngine forced pair 12 with raw prefix
and verifier-row hidden values. llama.cpp's handoff rows are bit-identical:
`draft_seed_input == process_h_input[0]`,
`verify_h[0] == process_h_input[1]`, and
`verify_h[1] == process_h_input[2]` all have **0.0 MAE**. HipEngine prefill-GDN
is closer to llama.cpp at the initial prefix seed (**0.0630 MAE** vs default
**0.0668**) and row 0 (**0.1310** vs **0.1370**), but default is closer at the
decisive row 1 (**0.0690** vs **0.0806**) and keeps the llama-side reject
decision. The next implementation target is therefore the captured-state row-1
handoff/scoring drift after the seed, not another attempt to make only the
cycle-start seed closer.

Use the tables in this order when choosing the next fix: first the canonical
three-lane speed-gap board, then the standing snapshot/source artifacts, then
the full-suite bucket inventory, then the attribution-only all-sync and
rocprof leaf tables, and finally the source-anchor table for the matching
llama.cpp implementation point. A new fine-grained bucket only becomes an
active parity target when it rolls up into `draft_initial`,
`target_block_verify_total`, target rows/output, or total cycle wall.

Dashboard contract:

| lane | role in this sprint | how to read it |
| --- | --- | --- |
| hipEngine default exact | Correctness-preserving control lane. | Keeps us honest about exact-mode regressions, but it is not the llama replication target. |
| hipEngine `llama-compat` | Active replication lane. | Should structurally match llama.cpp's B2/no-probe MTP path; every positive delta vs llama.cpp is a concrete optimization target. |
| llama.cpp HIP | Timing target and implementation reference. | When `llama-compat` mirrors the shape but stays slower, inspect the corresponding llama.cpp stage/kernel and copy or retune the mechanism. |

Active comparison rules:

| rule | requirement |
| --- | --- |
| Keep the three lanes together. | Every active speed or stage table must keep columns for hipEngine default exact, hipEngine `llama-compat`, llama.cpp HIP, and the compat delta. A compat-only diagnostic is allowed only in the leaf attribution section and must name the parent row it is expected to move. |
| Track the gap in ms/output first. | Tok/s stays visible, but the working budget is the compat ms/output delta vs llama.cpp. A change is not parity progress until it moves `cycle_wall_ms_per_output`, `draft_initial`, `target_block_verify_total`, or target rows/output in the retained full-suite lane. |
| Separate rollups from leaves. | Rollup rows decide priority; all-sync and rocprof rows identify the kernel or source path. Do not replace the rollup gap with an attribution-only number. |
| Compare source only after structure matches. | Once `llama-compat` has the same B2/no-probe shape for a row, any remaining positive delta becomes a source-code comparison task against the named llama.cpp path or kernel family. |

#### Canonical live three-lane speed/stage gap tracker (update every parity run)

Last refreshed from the full-suite artifacts
`benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json`
and
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-full.json`,
plus the rerun llama.cpp HIP B2 row in
`benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json`.

Preserve this as the top-of-file board for the parity sprint. It is the active
speed-gap comparison table requested for this work: hipEngine default exact,
hipEngine `llama-compat`, llama.cpp HIP, then the `llama-compat` gap. The gap
column is the live optimization budget; the final column names the next source
path or kernel family to compare. If a new fine-grained bucket has no direct
llama.cpp analog, leave the direct comparison to the detailed inventory and move
the gap only through its nearest parent row here.

The table schema is fixed for the parity sprint: every live row must keep the
same three lanes, the `llama-compat` delta, and the next llama.cpp source or
kernel family to inspect. New fine-grained buckets belong in the full-suite
inventory or attribution tables first; promote them to this board only when they
either map cleanly to a llama.cpp stage or move the retained full-suite parent
bucket. This keeps the remaining couple of milliseconds visible as an explicit
stage budget instead of burying it in prose.

| stage / bucket | hipEngine default exact B5 | hipEngine `llama-compat` B2 | llama.cpp HIP B2 | compat gap | target / next comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| Total MTP wall | 16.162 ms/output | **14.005 ms/output** | 14.269 ms/output | **-0.264 ms/output** | Stage wall is still slightly faster than the rerun llama.cpp HIP stage row; request-level tok/s remains **71.52 vs 71.91**. |
| Draft drain | 1.899 ms/output | **2.101 ms/output** | 2.141 ms/output | **-0.040 ms/output** | Draft parent is at parity. |
| Draft visible sampler/GPU drain | 1.129 ms/output | **1.933 ms/output** | 1.888 ms/output | **+0.045 ms/output** | Small residual; compare through draft drain because bucket names differ across engines. |
| Draft transformer body | 0.141 ms/output | **0.124 ms/output** | 0.250 ms/output | compat faster | Not an active target. |
| Serial verifier probe / tail cleanup | 6.508 ms/output | **0.151 ms/output** | 0.000 ms/output | +0.151 ms/output | Natural24 tail cleanup only; fixed-cycle compat still has the serial probe removed. |
| Target verifier drain | 7.728 ms/output | **11.436 ms/output** | 12.120 ms/output | **-0.684 ms/output** | No longer a llama.cpp speed gap after no-copy GDN state-row capture. The current-shape verifier-head route is rejected because it raises verifier drain to **12.501 ms/output** without improving acceptance. |
| Target rows / output | 1.163 | **1.171** | 1.148 | **+0.023 rows/output** | Row economy remains visible: 0 replay rows and 41 discarded rows over 240 outputs. |
| Replay / commit | 0.019 ms/output | **0.044 ms/output** | 0.004 ms/output | **+0.040 ms/output** | Small residual; serial-state remains the exact control. |
| Setup/snapshot/commit/accounting | 0.125 ms/output | **0.045 ms/output** | 0.188 ms/output | compat faster | No longer an active target after direct partial commit removes snapshots/replay. |

This board is intentionally redundant with the detailed ledgers below. Keep it
short, current, and three-lane so the next optimization target is visible without
reading the historical sections. If a new instrumented hipEngine bucket has no
llama.cpp analog, keep it in the full-suite inventory or leaf attribution table
and compare it only through the nearest rollup row here.

#### Standing three-lane parity snapshot (update first)

This is the first table to refresh after any retained or diagnostic parity run.
It keeps the active speed gap and the high-level stage split in one place. Use
retained full-suite rows for headline tok/s and async cycle wall; use all-sync
rows only in the leaf attribution table below.

Gap sign convention: `compat gap / reading` is always measured from hipEngine
`llama-compat` to llama.cpp HIP. Positive ms/rows mean compat is slower or doing
more verifier work; negative tok/s means a throughput deficit. Keep hipEngine
default exact in the same table so every fix shows whether it is a
llama-replication-only change, an exact-mode regression risk, or a real
cross-lane improvement. The goal for this sprint is to spend down the
`llama-compat` deltas here until the B2 structure and timing match llama.cpp.

| metric | hipEngine default exact B5 | hipEngine `llama-compat` B2 | llama.cpp HIP B2 | compat gap / reading |
| --- | ---: | ---: | ---: | --- |
| MTP tok/s | 61.98 parallel-attn full | **71.52 natural24 cyclecap24 full** | 67.3 suite / 72.12 traced / 71.91 rerun | Request-level compat is **-0.39 tok/s** vs llama.cpp; fixed-cycle compat provenance remains **72.23 tok/s**. The artifact filename contains `f32head`, but the retained row did not enable the verifier-head flag. |
| Cycle wall / output | 16.162 ms | **14.005 ms** | 14.269 ms | **-0.264 ms/output**; compat is slightly faster than the measured-excluding-first llama.cpp stage row. |
| Draft drain, `draft_initial` | 1.899 ms | **2.101 ms** | 2.141 ms | **-0.040 ms/output**; draft parent remains effectively at parity. |
| Visible draft sampler/GPU drain | 1.129 ms | **1.933 ms** | 1.888 ms | **+0.045 ms/output**; small compared with total wall. |
| Serial verify probe / tail cleanup | 6.508 ms | **0.151 ms** | 0.000 ms | Natural24 tail cleanup only; fixed-cycle compat stays at zero serial verify. |
| Target verifier drain | 7.728 ms | **11.436 ms** | 12.120 ms | **-0.684 ms/output**; no longer a llama.cpp speed gap. |
| Replay / commit | 0.019 ms | **0.044 ms** | 0.004 ms | **+0.040 ms/output**; small residual, not P0. |
| Target rows / output | 1.163 | **1.171** | 1.148 | **+0.023 rows/output**; 0 replay rows and 41 discarded rows over 240 outputs. |
| Accepted / output | 0.535 | **0.596** | 0.567 request / 0.610 stage-measured | Full-request hipEngine is **+0.029** accepted/output; the **-0.014** reading uses llama.cpp's 223-token stage denominator, not its 240-token request denominator. |

The current retained HIP stage-wall target is therefore closed for the
llama-replication lane, but the request-level headline is not: corrected
cyclecap24 is **71.52 tok/s** vs llama.cpp **71.91 tok/s**. Draft wall is still
at parity, verifier wall is faster than the traced llama.cpp HIP target, and
direct partial commit has removed serial accepted-prefix replay from the
replication lane. The measured verifier-head top-1 route is rejected for speed
(**66.45 tok/s / 15.072 ms/output**) because it raises
`target_block_lm_head_sample` from **1.068** to **2.118 ms/output** without
changing acceptance or row economy. Further work is now target
semantic/economy cleanup with the corrected denominator: preserve the
request-level acceptance advantage, improve draft acceptance/row-discard mix
where it is genuinely worse, explain the full-accept bonus-token divergence,
keep the no-copy capture path under all-sync/rocprof watch, and decide whether
the exact semantic lane can share any of this machinery without direct-commit
state divergence.
The explicit bulk state-only replay shortcut is not valid: artifact
`benchmarks/results/2026-07-02-mtp-state-lifecycle-bulk-state-only-partial-replay-compare.json`
reports `first_mismatch` at cycle 3, replay source
`serial_exact_accepted_prefix`, direct source `bulk_state_only_replay`, matching
visible token `[65342]`, and 61 hidden/linear-state fingerprint mismatches.
`native-state-only` replay keeps bulk scoring and uses the native verifier only
for the accepted-prefix replay, but it is also invalid:
`benchmarks/results/2026-07-02-mtp-state-lifecycle-native-state-only-partial-replay-active-compare.json`
has the same cycle-3 visible token and 59 hidden/linear-state mismatches.

Current source artifacts:

| lane | route / artifact | why it is in the table |
| --- | --- | --- |
| hipEngine default exact | `benchmarks/results/2026-07-02-ar-mtp-default-parallelattn-full.json` plus prior retained exact suite rows | Shipped correctness-preserving MTP lane; useful as a control, not the llama replication target. The shared `mtp_dense_attn_f32` parallel-attention fix moves exact B5 **60.8 -> 61.98 tok/s**, cycle **16.496 -> 16.162 ms/output**, and draft drain **1.921 -> 1.899 ms/output** with unchanged acc/output **0.535**, draft acceptance **0.723**, and target rows/output **1.163**. |
| hipEngine llama-compat no-copy direct-commit | route `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit`, active natural24 cyclecap24 artifact `benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json`, prior cyclecap24 artifact `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-full.json`, rejected current-shape verifier-head artifact `benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-vlmheadtop1-full.json`, fixed-cycle artifact `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-full.json`; all-sync attribution `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-allsync-smoke.json`; lifecycle diagnostic `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`; proposal comparison `benchmarks/results/2026-07-02-mtp-proposal-trace-compare-natural24-mixed-ja-en-translate.json`; serial-exact one-prompt A/B `benchmarks/results/2026-07-02-hipengine-mtp-serialexact-natural24-mixed-ja-en-translate.json` | Active no-probe B2 llama.cpp replication lane; the suite route now records and applies `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`. Full-accept and rejected/partial blocks commit captured verifier row state instead of serial-replaying accepted prefixes. The prefill-GDN state-row kernel reads live recurrent state without the old full-state D2D copy. Natural24 cyclecap24 full suite: **71.52 tok/s**, **14.005 ms/output**, **1.3055x AR**, acc/output **0.596**, draft acceptance **0.777**, target rows/output **1.171**, verifier drain **11.436 ms/output**, replay/commit **0.044 ms/output**, replay rows **0**, discarded rows **41**. The `f32head` filename is misleading; this active artifact did not enable `--verify-lm-head-q6-top1-dp4a`. The measured current-shape verifier-head route regresses to **66.45 tok/s**, **15.072 ms/output**, verifier drain **12.501 ms/output**, and `target_block_lm_head_sample` **2.118 ms/output** with unchanged acceptance/economy. Fixed-cycle provenance remains **72.23 tok/s**, **13.865 ms/output**, **1.319x AR**, acc/output **0.609**, draft acceptance **0.780**, target rows/output **1.172**. The lifecycle diagnostic intentionally diverges from serial replay at cycle 3; this is llama replication, not exact-mode safety. |
| hipEngine llama-compat prefix-state fingerprint | `benchmarks/results/2026-07-03-mtp-prefix-state-fingerprint-default-vs-prefillgdn.json` | Diagnostic-only forced pair-12 prefix comparison for the active F32 selected-intermediate environment; `performance_claim=false`. It proves prefill-GDN captured-row prior replay changes the committed prefix immediately after the first full-accept cycle: cycle 1 already has different hidden seed, 29/30 linear-state layer fingerprints, and 10/10 full-attention KV fingerprints. By cycle 12, default prefix state samples `[15495, 26126, 1151]` with row-1 `539 - 26126 = -0.003027`, while the prefill-GDN prefix samples `[15495, 539, 1151]` with `+0.295256`. The raw source comparison below answers which parts of that split match llama.cpp. |
| hipEngine llama-compat prefix-state numeric summary | `benchmarks/results/2026-07-03-mtp-prefix-state-numeric-summary-default-vs-prefillgdn.json` | Diagnostic-only compact summary reducer; `performance_claim=false`. It adds `--prefix-state-numeric-summary` to `scripts/gguf_mtp_forced_target_probe.py` and `scripts/gguf_mtp_compare_prefix_state_summaries.py`. At forced pair 12, default vs prefill-GDN differ in **58/60** linear state components and **20/20** full-attention KV components. Largest summary deltas rank Conv-state layers **33/26/18/32/30** and full-attention key-cache layers **15/11/27/31/35**. This ranks the next raw-dump targets but is not final pairwise MAE. |
| hipEngine llama-compat selected raw prefix-state comparison | `benchmarks/results/2026-07-03-mtp-prefix-state-rawselected-default-vs-prefillgdn.json` | Diagnostic-only selected raw-buffer comparison; `performance_claim=false`. It adds selected raw prefix-state dumps to `scripts/gguf_mtp_forced_target_probe.py` and pairwise raw decoding to `scripts/gguf_mtp_compare_prefix_state_summaries.py`. Selected Conv-state MAE is much larger than recurrent-state MAE: Conv layers **26/30/33/32/18** are **0.02723/0.02611/0.02408/0.02260/0.01956**, while the corresponding recurrent states are only **4.8e-05..7.6e-05**. Selected KV key MAEs are **0.01018..0.01591**. This rules in hidden/history drift before state capture, not a recurrent-state copy bottleneck. |
| hipEngine llama-compat prefill-GDN chain-Conv hybrid | `benchmarks/results/2026-07-03-mtp-prefix-state-summary-prefillgdn-vs-chainconv.json` | Negative diagnostic-only artifact; `performance_claim=false`. Adds default-off `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV=1` to test whether chain Conv state rows plus fast prefill-GDN recurrent rows fix the row-1 accept flip. The hybrid is byte-identical to prefill-GDN at forced pair 12 (**0/60** linear-state and **0/20** KV changes) and keeps row-1 `539 - 26126 = +0.295256`, so Conv-state raw drift is downstream, not the causal implementation fix. |
| hipEngine vs llama.cpp hidden lifecycle comparison | `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-default-vs-prefillgdn-vs-llamacpp.json` | Diagnostic-only raw hidden-vector reducer over llama.cpp task 9 / cycle 18 and hipEngine forced pair 12; `performance_claim=false`. It adds `scripts/llamacpp_mtp_compare_hidden_lifecycle.py` and raw prefix hidden output in `scripts/gguf_mtp_forced_target_probe.py`. llama.cpp's internal handoff is exact (`draft_seed_input == process_h_input[0]`, `verify_h[0] == process_h_input[1]`, `verify_h[1] == process_h_input[2]`, all **0.0 MAE**). Prefill-GDN is slightly closer at the cycle-start seed (**0.0630 MAE** vs default **0.0668**) but default is closer at decisive verifier row 1 (**0.0690** vs **0.0806**) and matches llama.cpp's row-1 reject margin. This moves the next target to captured-state row-1 drift/score parity after the seed. |
| hipEngine vs llama.cpp hidden lifecycle ladder | `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-cycle1-default-vs-prefillgdn-vs-llamacpp.json`, `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-cycle3-default-vs-prefillgdn-vs-llamacpp.json`, `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-cycle7-default-vs-prefillgdn-vs-llamacpp.json`, `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-cycle11-default-vs-prefillgdn-vs-llamacpp.json`, and aggregate `benchmarks/results/2026-07-03-mtp-hidden-lifecycle-ladder-default-vs-prefillgdn-vs-llamacpp.json` | Diagnostic-only hidden lifecycle ladder; `performance_claim=false`. Adds `scripts/llamacpp_mtp_hidden_lifecycle_ladder.py`. Across hip cycles **1/3/7/11/12**, prefill-GDN is closer on prefix hidden in **5/5** cycles and closer on the decisive verifier row in **3/5**, but default is closer on token margin in **4/5** and is the only lane that matches the cycle-12 reject. This moves the next copy target from state-row capture to final hidden-to-logit scoring: output norm, LM-head quant/dequant, and near-tie score accumulation. |
| hipEngine scored layer-0 handoff caveat | `benchmarks/results/2026-07-03-mtp-process-h-input-vs-layer0-hiddenin-noncapture.json`, `benchmarks/results/2026-07-03-mtp-process-h-input-vs-layer0-hiddenin-prefillgdn.json` | Diagnostic-only boundary comparison using `scripts/llamacpp_mtp_compare_verifier_tensors.py`; `performance_claim=false`. The reducer now permits boundary-only comparisons and merges llama.cpp `top_k` plus `candidate_scores`. Both lanes have identical row-1 layer-0 `hidden_in`, and both show the same invalid large cross-engine delta to llama.cpp `process_h_input` (**1.80063 MAE**), proving that label is not the target-layer input oracle. Use llama.cpp `model.input_embed` or prior `verify_layer_output` for future target-layer input splits. The active-vs-default flip starts after layer-0 input, first crosses **1e-3 MAE** at layer **23**, and jumps materially by full-attention layer **35**. |
| hipEngine llama-compat copied-state direct-commit | route `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit`, artifact `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`; smoke `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-smoke.json` | Superseded active lane before the no-copy GDN capture kernel. It paid a full recurrent-state D2D copy before each captured prefill-GDN layer: **60.56 tok/s**, **16.534 ms/output**, verifier drain **14.071 ms/output**. The no-copy replacement keeps acceptance/economy identical and moves full-suite **60.56 -> 72.23 tok/s**. |
| hipEngine llama-compat semantic-safe | route `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-serialstate`, artifact `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json` | Semantic-safe control. Full-accept blocks still direct-commit captured state, while rejected/partial bulk blocks restore and serial-replay the accepted prefix without replay LM-head sampling. Full suite: **51.85 tok/s**, **19.308 ms/output**, **0.9472x AR**, acc/output **0.606**, draft acceptance **0.770**, target rows/output **1.331**, replay/commit **2.489 ms/output**, replay rows **38**, discarded rows **46**. |
| hipEngine llama-compat prior serial-full replay | route `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly`, artifact `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directstate-prefillgdn-partialfix-full.json` | Superseded semantic-safe control. It used full serial LM-head sampling during accepted-prefix replay: **50.96 tok/s**, **19.645 ms/output**, **0.9312x AR**, verifier drain **17.222 ms/output**, replay/commit **2.775 ms/output**. The serial-state-only row keeps the same acceptance/row economy and removes only replay sampling. |
| hipEngine llama-compat bulk state-only partial replay | `benchmarks/results/2026-07-02-mtp-state-lifecycle-bulk-state-only-partial-replay-compare.json` | Rejected diagnostic, not a speed row. It tests replacing serial accepted-prefix replay after direct-state partial/reject blocks with bulk `advance_state_only` replay. The first partial/reject at cycle 3 has matching visible token `[65342]`, but hidden seed plus Conv/GDN fingerprints diverge across 61 entries, so this shortcut cannot replace the semantic-safe serial replay bucket. |
| hipEngine llama-compat native state-only partial replay | `benchmarks/results/2026-07-02-mtp-state-lifecycle-native-state-only-partial-replay-active-compare.json` | Rejected diagnostic, not a speed row. It keeps the original bulk block for scoring, then uses native row-serial-attention `advance_state_only` only for direct-state partial/reject replay. Cycle 3 still keeps visible token `[65342]`, but hidden seed plus Conv/GDN fingerprints diverge across 59 entries, so changing replay scheduler alone is not enough. |
| hipEngine llama-compat unsafe direct-state | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-draftonly-full.json` | Superseded performance diagnostic. It moved full-suite **74.39 -> 75.15 tok/s** and cycle **13.463 -> 13.325 ms/output**, but direct-committed rejected/partial bulk-block state that the lifecycle comparator later proved is not prefix-equivalent to serial accepted-prefix replay. Do not use as the active compat row. |
| hipEngine llama-compat prior retained parallel-attn | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-parallelattn-clean-rerun-full.json` | Superseded active lane before limiting dense-Q8 dp4a to draft forward leaves. The parallel-attention clean rerun moved full-suite **71.84 -> 74.39 tok/s**, cycle **13.940 -> 13.463 ms/output**, and draft drain **2.684 -> 2.204 ms/output** with unchanged acc/output **0.621**, draft acceptance **0.820**, and target rows/output **1.136**. |
| hipEngine llama-compat prior retained shared-gate | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-sharedgate-routerrow-full.json` | Superseded active lane before parallelizing `hipengine_mtp_dense_attn_f32`. The shared-gate scalar-dot fix moved full-suite **71.34 -> 71.84 tok/s**, cycle **14.037 -> 13.940 ms/output**, and draft drain **2.747 -> 2.684 ms/output** with unchanged acc/output **0.621**, draft acceptance **0.820**, and target rows/output **1.136**. |
| hipEngine llama-compat prior retained resident-init | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-residentinit-routerrow-full.json` | Superseded active lane before routing the draft shared-gate scalar dot through the row-parallel F32 router kernel. Resident initial KV moved full-suite **64.41 -> 71.34 tok/s**, cycle **15.547 -> 14.037 ms/output**, acc/output **0.578 -> 0.621**, draft acceptance **0.685 -> 0.820**, target rows/output **1.266 -> 1.136**, and verifier drain **12.166 -> 10.966 ms/output**. |
| hipEngine llama-compat prior retained routerrow | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-routerrow-full.json` | Superseded active lane before the resident initial prompt KV writer fix. It was structurally close but still seeded prompt MTP KV through the legacy writer, producing the old **64.41 tok/s / 15.547 ms/output** full-suite row and the now-closed **+1.316 ms/output** parent gap. |
| hipEngine llama-compat prior retained f32ssm | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-full.json` | Superseded active lane before row-parallel draft router logits. Router-row moves full-suite **63.63 -> 64.41 tok/s**, cycle **15.735 -> 15.547 ms/output**, and draft drain **3.252 -> 3.055 ms/output** with unchanged acc/output **0.578**, draft acceptance **0.685**, and target rows/output **1.266**. Draft-chain rocprof shows `draft_run_ffn_router_linear` **0.508 -> 0.048 ms/cycle** in the sync-stage leaf table. |
| hipEngine llama-compat prior retained x8top1 | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-full.json` | Superseded active lane before direct-state F32 `ssm_out` q8_1/raw-Q8 dp4a. F32 `ssm_out` moves full-suite **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**, target verifier drain **12.662 -> 12.158 ms/output**, and target rows/output **1.299 -> 1.266**. |
| hipEngine llama-compat prior retained denseq8all + Q8 shared dual | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json` | Superseded active lane before X8-packed Q6_K draft lm-head top-1. Same acceptance/economy; X8 top-1 moves full-suite **61.19 -> 61.31 tok/s**, cycle **16.364 -> 16.331 ms/output**, and draft drain **3.378 -> 3.352 ms/output**. |
| hipEngine llama-compat prior retained x8q6 | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rowhist-full.json` | Superseded active lane. Better acceptance/row economy than `denseq8all`, but slower wall and farther from llama.cpp's dense MMVQ mechanism. |
| hipEngine llama-compat row-hist smoke | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rowhist-smoke.json` | Historical instrumentation-only smoke that proved `cycle_histograms` flow through suite output before the full retained rerun above. Do not use for headline tok/s. |
| hipEngine llama-compat no-copy verifier all-sync split | `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-allsync-smoke.json`; prior copied-state control `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-allsync-smoke.json` | Attribution-only smoke with extra sync points inside verifier layer families and selected-MoE gate/up/down. The copied-state GDN leaf `target_block_linear_attn_prefill_gdn_state_rows` drops **2.913 -> 0.785 ms/output** after removing the full recurrent-state D2D copy. Do not use for headline tok/s. |
| hipEngine llama-compat prior verifier all-sync split | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync-smoke.json` | Historical attribution-only smoke before directcommit/no-copy capture. Do not use for headline tok/s. |
| hipEngine llama-compat verifier block rocprof split | `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2.json` | Diagnostic-only B2-shaped `verify_target_block` kernel trace for the retained compat route (`--mode block-verify --verify-dp4a --selected-down-x8-repack q6 --record-stage-timings`). Use it to rank verifier kernel families; do not use it for headline tok/s. |
| llama.cpp HIP verifier-shape pp4 rocprof proxy | `benchmarks/results/2026-07-01-llamacpp-hip-pp4-kernel-summary.json` | Diagnostic-only `llama-bench -p 4 -b 4 -ub 4 -n 0` kernel-family summary. This remains a verifier-shaped source/kernel proxy, not a headline MTP timing row. |
| llama.cpp HIP MTP whole-run rocprof proxy | `benchmarks/results/2026-07-02-llamacpp-mtp-rocprof-token32-gen8-whole-run.json` | Diagnostic-only `llama-server` MTP request under `rocprofv3 --kernel-trace`, with `LLAMA_MTP_STAGE_TIMINGS` enabled. It produces the first llama.cpp MTP kernel-family bucket split in this tracker, but it is whole-process and required `SIGKILL` after profiler finalize timeout, so use it only as a source/kernel proxy. |
| llama.cpp HIP MTP ROCTX range proxy | `benchmarks/results/2026-07-02-llamacpp-mtp-rocprof-token32-gen8-roctx-ranges.json` | Diagnostic-only rerun after llama.cpp commit `dd7ec418c` added `LLAMA_MTP_ROCTX=1` ranges around the existing MTP stage timers. The artifact includes `range_name_summaries` for stage-window kernel buckets, but it is still whole-process and includes warmup/prompt/server ranges, so do not use it as a headline timing row. |
| hipEngine vs llama.cpp target layer checkpoint diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer-checkpoints-diagnostic.json` | Diagnostic-only forced pair-12 row-1 per-layer split; `performance_claim=false`. hipEngine now emits post-layer BF16 residual rows through `--layer-output-row LAYER:ROW` / `--raw-layer-output-row LAYER:ROW`; local llama.cpp dirty instrumentation traces target graph `l_out` tensors as `verify_layer_output_N`. Corrected row alignment shows no single large layer cliff: full-vector MAE is **0.000535** after layer 1, **0.00269** after layer 28, **0.00539** after layer 32, and **0.01015** after layer 39/pre-output_norm, with cosine still **0.99931** pre-output_norm. Final `verify_h` remains **0.07789 MAE / 0.09815 RMSE / 0.99908 cosine**, enough to flip the `539` vs `26126` near-tie. The active hypothesis is accumulated BF16 verifier-boundary drift in hipEngine versus llama.cpp's F32 `l_out` graph tensors. |
| hipEngine vs llama.cpp target layer-31 sub-boundary diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer31-subboundary-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. hipEngine now emits `--layer-boundary-row LAYER:ROW` / `--raw-layer-boundary-row LAYER:ROW` snapshots from an isolated replay session, covering layer input, `attn_norm`, attention output/residual, post-attn norm, selected MoE down rows, shared MoE output, reconstructed combined MoE `ffn_out`, rounded post-MoE, and layer output. Local llama.cpp dirty instrumentation traces target `attn_norm_31`, `attn_residual_31`, `attn_post_norm_31`, `ffn_out_31`, `post_moe_31`, and `l_out_31`. Result: no layer-31 cliff. hipEngine `attn_norm` vs llama `attn_norm_31` is **0.01623 MAE / 0.02082 RMSE / 0.99969 cosine**; residual vs llama `attn_residual_31` is **0.00303 MAE / 0.00405 RMSE / 0.99943 cosine**; post-attn norm is **0.03616 MAE / 0.04541 RMSE / 0.99940 cosine** at RMS ~1.30; reconstructed MoE `ffn_out` vs llama `ffn_out_31` is **0.00446 MAE / 0.00562 RMSE / 0.99161 cosine**; reconstructed rounded post-MoE exactly equals hip layer output (**0 MAE**) and hip layer output vs llama `post_moe_31`/`l_out_31` is **0.00528 MAE / 0.00671 RMSE / 0.99871 cosine**. This keeps the semantic blocker on accumulated residual-boundary/final-output-norm precision drift, not one bad late-layer substage. |
| hipEngine target layer-31 fine MoE tap diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer31-fine-moe-taps-diagnostic.json` | Diagnostic-only forced pair-12 row-1 split; `performance_claim=false`. The hipEngine boundary capture now emits llama.cpp-shaped MoE internals that are durable after the layer run: router logits, selected SwigLU/intermediate, selected down rows, per-expert weighted down rows, selected weighted sum before/after BF16 rounding, shared intermediate/out, sigmoid-gated shared contribution, reconstructed `ffn_out`, and rounded post-MoE. The scored row is unchanged: sampled `[15495, 539, 1151]`, row-1 `539 - 26126` **+0.118217**. Layer 31 selects experts `[221, 95, 240, 60, 88, 19, 212, 59]`; selected weighted sum RMS is **0.039687**, shared-gated RMS is **0.015372** from shared-gate logit **-2.042253** / sigmoid **0.114838**, reconstructed `ffn_out` RMS is **0.042586**, and `post_moe_rounded_from_components` hashes exactly equal `layer_out`. This confirms hipEngine's local combine reconstruction is self-consistent; the cross-engine row below completes the matching llama.cpp tensor comparison. |
| hipEngine vs llama.cpp target layer-31 fine MoE cross-engine diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer31-fine-moe-cross-engine-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. The new `scripts/llamacpp_mtp_compare_target_moe_taps.py` helper compares raw hipEngine taps against local llama.cpp dirty tensor traces for `ffn_moe_logits_31`, `ffn_moe_weights_norm_31`, selected `ffn_moe_swiglu/down/weighted/out_31`, shared `ffn_shexp/gate/gated_31`, `ffn_out_31`, `post_moe_31`, and `verify_layer_output_31`; duplicate llama labels for `ffn_moe_out_31` and `ffn_out_31` are identical (`max_abs_vs_first=0`). Result: the first layer-31 local semantic split is router top-k selection, not selected/shared projection math. Router logits differ only **0.0160 MAE / 0.0205 RMSE / 0.999995 cosine**, but hipEngine selects `[221, 95, 240, 60, 88, 19, 212, 59]` while llama.cpp selects `[221, 95, 240, 60, 19, 88, 212, 75]`; ranks 4/5 swap and the cutoff is hip-only expert `59` vs llama-only expert `75`. For common experts aligned by expert id, selected weighted rows stay at or below **0.00179 MAE**, shared-gate logit differs by **0.00356**, aggregate `ffn_out` is **0.00446 MAE / 0.00562 RMSE / 0.99161 cosine**, and post-MoE/layer output remains **0.00528 MAE / 0.00671 RMSE / 0.99871 cosine**. This strengthens the accumulated router-input/residual precision hypothesis; it does not identify a separate layer-31 combine rule to copy from llama.cpp. |
| hipEngine vs llama.cpp target all-layer router trace diagnostic | `benchmarks/results/2026-07-02-mtp-target-router-trace-cross-engine-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. The new `--router-trace-row 1` forced-probe path replays one isolated row through all 40 target layers and records per-layer MoE router logits, selected experts, routing weights, shared gate, and hidden/layer summaries. A focused local llama.cpp dirty rerun captured raw values for `ffn_moe_logits_N`, `ffn_moe_weights_norm_N`, and `shared_expert_gate_N` for all layers. Result: layer 0 top-k matches llama.cpp; the first top-k divergence is layer 1, rank 8 only (hipEngine expert `126`, llama.cpp expert `63`) with router logits still **0.00562 MAE / 0.00694 RMSE / 0.999999 cosine** and routing weights **0.00054 MAE**. Overall **29/40 layers match top-k**; the 11 mismatches are near-tie swaps or cutoff differences, including the known layer-31 split. Next target: raw layer-0 output / layer-1 router-input precision. |
| hipEngine vs llama.cpp target layer0/1 boundary cross-engine diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer0-1-boundary-cross-engine-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. The new `scripts/llamacpp_mtp_compare_early_boundary.py` helper compares the raw hipEngine layer-0 and layer-1 boundary capture against local llama.cpp dirty tensor traces for `post_moe_0`, layer-1 router logits/weights, and layer-1 output. Result: hipEngine's local layer-0 output to layer-1 input is exact (**0 MAE**), but that same boundary is already **0.000203 MAE / 0.000263 RMSE / 0.999950 cosine** away from llama.cpp `post_moe_0` before layer 1 begins. Layer 0 router top-k still matches llama.cpp; layer 1 has the same rank-7 cutoff split (hipEngine expert `126`, llama.cpp expert `63`) with layer-1 router logits **0.00562 MAE / 0.00694 RMSE / 0.999999 cosine**, routing weights **0.000541 MAE**, shared-gate logit delta **-0.00474**, and layer-1 post-MoE/layer output **0.000535 MAE / 0.000664 RMSE / 0.999834 cosine**. The artifact marks llama.cpp `attn_norm_0` as label-alignment suspect because it disagrees while downstream residual/post-MoE tensors align; do not chase that label without revalidating the llama.cpp trace instrumentation. |
| hipEngine vs llama.cpp target layer0/1 fine MoE cross-engine diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer0-fine-moe-cross-engine-diagnostic.json`, `benchmarks/results/2026-07-02-mtp-target-layer1-fine-moe-cross-engine-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. Local llama.cpp dirty instrumentation was extended to convert BF16/F16 debug tensors to F32 before exporting raw `LLAMA_MTP_TENSOR_TRACE_VALUES`, because early selected/shared MoE internals were otherwise summary-only. Result: layer 0 top-k matches llama.cpp and selected/shared internals are very close: router logits **0.0107 MAE / 0.0138 RMSE / 0.999998 cosine**, routing weights **0.00169 MAE**, common-expert selected weighted rows **<=0.0000764 MAE**, shared-gate logit delta **+0.00170**, `ffn_out` **0.000126 MAE**, post-MoE/layer output **0.000203 MAE**. Layer 1 repeats the first router cutoff split (hipEngine `126`, llama.cpp `63`) while common-expert selected weighted rows are still **<=0.0001005 MAE**, shared-gate logit delta **-0.00474**, `ffn_out` **0.000489 MAE**, and post-MoE/layer output **0.000535 MAE**. This rules out selected/shared projection or combine math as the early semantic cliff; the remaining target is layer-output/router-input precision before the layer-1 cutoff. |
| hipEngine vs llama.cpp target layer-0 linear-attention cross-engine diagnostic | `benchmarks/results/2026-07-02-mtp-target-layer0-linear-attn-cross-engine-diagnostic.json` | Diagnostic-only forced pair-12 / llama cycle-18 row-1 split; `performance_claim=false`. `scripts/llamacpp_mtp_linear_attn_trace_patch.py` generates the temp-tree llama.cpp patch that exposes early linear-attention labels; `scripts/llamacpp_mtp_compare_layer0_linear_attn.py` compares those labels to hipEngine raw layer-0 boundary taps. Result: no large layer-0 projection/conv cliff. Stable pre-`ssm_out` labels are close: `z` projection **0.004111 MAE / 0.999996 cosine**, `beta` projection **0.001866 MAE / 0.999999 cosine**, `conv_output_silu` **0.0001452 MAE / 0.9999995 cosine**, and q/k/v conv-view slices **<=0.0001865 MAE**. Downstream `linear_attn_out` remains **0.0001595 MAE / 0.0002100 RMSE / 0.999952 cosine**, attention residual **0.0001630 MAE / 0.0002197 RMSE / 0.999962 cosine**, attention post-norm **0.005668 MAE**, `ffn_out` **0.0001256 MAE**, and post-MoE/layer output **0.0002027 MAE**. The artifact marks `linear_attn_qkv_mixed_0` and `alpha_0` as trace-label caveats, and its direct `final_output_0` caveat is superseded by the projectable contiguous tap row below. |
| hipEngine scored layer-14 input/linear-attention cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-bonus-row-layer14-scored-linear-attn-compare.json` | Diagnostic-only `mixed_ja_en_translate` task 9 / cycle 3 / row 2 split; `performance_claim=false`. `scripts/llamacpp_mtp_compare_layer0_linear_attn.py` now supports arbitrary layers, scored hipEngine captures, `--llamacpp-task-id`, raw input-boundary values, and a CPU RMSNorm formula audit. Result: the first complete layer-14 sub-boundary above threshold is before layer-14 projection/MoE: `attn_norm_14` vs hipEngine `attn_norm` is **0.02051 MAE / 0.02654 RMSE**. The valid input comparison is hipEngine `hidden_in` vs llama.cpp `verify_layer_output_13` at only **0.00109696 MAE / 0.00142193 RMSE**; CPU RMSNorm exactly reproduces both engines' `attn_norm` rows from their own inputs, so layer-14 RMSNorm arithmetic is not the bug. `process_h_input` is context-only draft hidden input and must not be compared to target `hidden_in`. Downstream `z` is **0.01564 MAE**, `beta` **0.00914**, `conv_output_silu` **0.00120**, `linear_attn_out` **0.000653**, post-attention norm **0.02736**, `ffn_out` **0.00228**, and post-MoE/layer output **0.00253**. The next instrumentation target is the scored layer-13 boundary/internal split. |
| hipEngine scored layer-13 input/linear-attention cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer13-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer13-scored-linear-attn-compare.json` | Diagnostic-only `mixed_ja_en_translate` task 9 / cycle 3 / row 2 split; `performance_claim=false`. Same scored bulk/native verifier path as the layer-14 row, but capturing layer 13. Result: incoming hidden vs llama.cpp `verify_layer_output_12` is **0.000975 MAE / 0.001244 RMSE**, `attn_norm_13` is **0.01954 / 0.02534**, `z_13` is **0.01662 / 0.02135**, `conv_output_silu_13` is **0.00119 / 0.00268**, `linear_attn_out_13` is **0.000526 / 0.000695**, attention residual is **0.00101 / 0.00128**, `ffn_out_13` is **0.000539 / 0.000714**, and post-MoE/layer output is **0.001097 / 0.001422**. CPU RMSNorm reproduces both engines within trace rounding, so the split moves upstream to layer 12. The temp llama.cpp `linear_attn_qkv_mixed_13`, `alpha_13`, and `beta_13` raw taps are trace-label/layout caveated. |
| hipEngine scored layer-13 MoE cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-bonus-row-layer13-scored-moe-taps-compare.json` | Diagnostic-only companion to the scored layer-13 split. Layer-13 MoE is not the next copy target: router top-k only permutes ranks 2/3 among the same experts (`172` and `239`), with no hip-only or llama-only expert. Common-expert selected rows stay close, aggregate `ffn_out` is **0.000539 MAE**, and post-MoE is **0.001097 MAE**, matching the layer-13 output/incoming layer-14 hidden delta. |
| hipEngine scored layer-12 input/linear-attention cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer12-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer12-scored-linear-attn-compare.json` | Diagnostic-only `mixed_ja_en_translate` task 9 / cycle 3 / row 2 split; `performance_claim=false`. Same scored bulk/native verifier path, now capturing layer 12. Result: incoming hidden vs llama.cpp `verify_layer_output_11` is **0.000996 MAE / 0.001264 RMSE**, `attn_norm_12` is **0.01826 / 0.02342**, `z_12` is **0.01290 / 0.01663**, `conv_output_silu_12` is **0.000803 / 0.00175**, `linear_attn_out_12` is **0.000607 / 0.000778**, attention residual is **0.000956 / 0.00120**, `ffn_out_12` is **0.000445 / 0.000592**, and post-MoE/layer output is **0.000975 / 0.001244**. CPU RMSNorm exactly reproduces both engines, so the split moves upstream to layer 11. qkv/beta raw taps remain trace-label/layout caveated. |
| hipEngine scored layer-12 MoE cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-bonus-row-layer12-scored-moe-taps-compare.json` | Diagnostic-only companion to the scored layer-12 split. Added `--llamacpp-task-id` to `scripts/llamacpp_mtp_compare_target_moe_taps.py` because the multi-task llama.cpp JSONL otherwise selected task 0 cycle 3. Correct task-9 result: only a rank 5/6 swap among the same experts (`71` and `194`), no hip-only or llama-only experts, aggregate `ffn_out` **0.000445 MAE**, and post-MoE **0.000975 MAE**. |
| hipEngine scored layer-0 input/linear-attention/MoE cross-engine diagnostic | `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-scored-linear-attn-compare.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-scored-moe-taps-compare.json` | Superseded diagnostic-only `mixed_ja_en_translate` task 9 / cycle 3 / row 2 split; `performance_claim=false`. This first scored layer-body walk showed there was no large layer-0 attention/MoE copy target: `attn_norm_0` **0.00115 MAE**, `conv_output_silu_0` **0.000118**, `linear_attn_out_0` **0.0000353**, `ffn_out_0` **0.0000386**, and post-MoE/layer output **0.0000547**; layer-0 MoE top-k matched exactly. Later F32 token/projection diagnostics below supersede the stale input-hidden next-target note from this row. |
| hipEngine scored layer-0 dense-F32 alpha/beta + pre-`ssm_out` / post-norm diagnostic | `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-pre-ssm-linear-attn-compare.json`, `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32proj-densef32ab-f32postnorm-keepf32-f32embed-f32res-attnnorm-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-f32postnorm-keepf32-f32embed-f32res-attnnorm-pre-ssm-linear-attn-compare.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32proj-densef32ab-keepf32-f32embed-f32res-attnnorm-moe-taps-compare.json` | Diagnostic-only follow-up; `performance_claim=false`. Adds the dense-F32 F32-input/F32-output `ssm_alpha`/`ssm_beta` projection route, fixes the F32 scratch overwrite guard, and updates the linear-attention reducer to compare pre-`ssm_out` plus post-attention-norm variants. Layer-0 target input and `attn_norm_f32_scratch` are exact, `z_0` is **8.41e-08 MAE**, `beta_0` closes **0.001754 -> 7.40e-08 MAE**, and `conv_output_silu_0` is **2.83e-09 MAE**. The new pre-`ssm_out` split closes recurrent-GDN/`ssm_out`: `recurrent_bf16` vs llama.cpp `final_output_0` is **6.23e-06 MAE** and `linear_attn_out_0` is **2.85e-05 MAE**. F32 post-norm improves the local norm bucket **0.001603 -> 0.001528 MAE** and post-MoE **4.88e-05 -> 4.60e-05 MAE**, but still samples `8940`; row 2 worsens to `8940=25.71944` rank 1 vs `668=25.26696` rank 4 (**+0.45249**). Next target: accumulated residual/RMSNorm drift and final LM-head amplification, not layer-0 projection/conv/GDN/`ssm_out`/post-norm-buffer/MoE. |
| llama.cpp `model.input_embed` layer-0 target-input trace | `benchmarks/results/2026-07-03-llamacpp-model-input-embed-trace-patch.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-inputembed-linear-attn-compare.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-inputembed-moe-taps-compare.json` | Diagnostic-only refinement of the scored layer-0 split; `performance_claim=false`. The llama.cpp patch adds `model.input_embed` to both target tensor-trace allowlists. Result: hipEngine `hidden_in` vs llama.cpp `model.input_embed` is essentially aligned (**0.00000643 MAE / 0.0000103 RMSE / 0.999999 cosine**). The first complete semantic split is attention RMSNorm output: llama.cpp `attn_norm_0` exactly matches F32 RMSNorm over `model.input_embed`, while hipEngine's captured `attn_norm` exactly matches the BF16 mirror. |
| hipEngine layer-0 F32 residual + F32 attention-norm scored diagnostic | `benchmarks/results/2026-07-03-mtp-target-bonus-row-hipengine-scored-layer0-f32res-attnnorm-cycle3.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32res-attnnorm-linear-attn-compare.json`, `benchmarks/results/2026-07-03-mtp-bonus-row-layer0-f32res-attnnorm-moe-taps-compare.json` | Diagnostic-only follow-up; `performance_claim=false`. `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` alone is inert without `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1`; the capture now emits `attn_norm_f32_scratch` to separate the actual F32 scratch from the BF16 mirror. With both flags, `attn_norm_f32_scratch` vs llama.cpp `attn_norm_0` is **0.000729 MAE / 0.00116 RMSE**, `z_0` improves **0.004850 -> 0.003687 MAE**, `linear_attn_out_0` improves **0.0000353 -> 0.0000314 MAE**, and post-MoE improves **0.0000547 -> 0.0000495 MAE**. The branch still samples bonus `8940`, so this is a small measured semantic improvement, not the remaining parity fix. |
| hipEngine validation of llama.cpp `final_output_0` tap | `benchmarks/results/2026-07-02-mtp-target-layer0-final-output-reprojection-diagnostic.json` | Superseded diagnostic-only model-backed projection check; `performance_claim=false`. The helper loads only hipEngine's layer-0 `ssm_out` weight with decode repack enabled, reprojects hipEngine `recurrent_out`, and reprojects traced llama.cpp `final_output_0`. Result: hipEngine `recurrent_out -> ssm_out` exactly reconstructs the captured hipEngine `attn_out` (**0 MAE**), but traced llama.cpp `final_output_0 -> ssm_out` is **0.2977 MAE / 0.3772 RMSE / -0.0302 cosine** from llama.cpp `linear_attn_out_0`. Keep this only as proof that the old `final_output_0` label was not a projectable semantic pre-`ssm_out` oracle; the corrected contiguous tap below supersedes it. |
| hipEngine validation of llama.cpp `final_output_cont_0` tap | `benchmarks/results/2026-07-02-mtp-target-layer0-final-output-cont-reprojection-diagnostic.json`; patch artifact `benchmarks/results/2026-07-02-llamacpp-final-output-cont-trace.patch` | Diagnostic-only materialized contiguous post-GDN/pre-`ssm_out` tap; `performance_claim=false`. `scripts/llamacpp_mtp_final_output_cont_trace_patch.py` generates a local llama.cpp patch that adds `final_output_cont_` to both tensor-trace allowlists, emits `final_output_cont`, and feeds `ssm_out` from that contiguous tensor in the temporary trace tree. Cycle 18 remains the active llama.cpp branch (`[15495, 539]` drafted, one accepted, bonus `26126`), and the tap is projectable: llama.cpp `final_output_cont_0 -> ssm_out` reconstructs `linear_attn_out_0` at **0.000157 MAE / 0.000207 RMSE / 0.999953 cosine**. Direct hipEngine `recurrent_out` vs llama.cpp `final_output_cont_0` is only **0.0000176 MAE / 0.0000391 RMSE / 0.999999 cosine**, and hipEngine `attn_out` vs llama.cpp `linear_attn_out_0` remains **0.0001595 MAE / 0.0002100 RMSE / 0.999952 cosine**. Layer-0 GDN/recurrent layout and `ssm_out` projection are ruled out; keep the active semantic search on accumulated layer-output/router-input precision drift. |
| hipEngine vs llama.cpp target output_norm recompute diagnostic | `benchmarks/results/2026-07-02-mtp-target-output-norm-recompute-diagnostic.json` | Diagnostic-only CPU recompute from the raw row-1 pre-output residuals; `performance_claim=false`. Using `output_norm.weight` and `eps=1e-6`, CPU `x * weight / sqrt(mean(x^2)+eps)` exactly reproduces hipEngine `verify_h` from hipEngine `pre_output_norm` and exactly reproduces llama.cpp `verify_h` from llama.cpp `verify_pre_output_norm` (**0 MAE** for both). The pre-output residual delta is **0.01015 MAE / 0.01273 RMSE / 0.99931 cosine**; applying the same CPU output_norm to both rows deterministically produces **0.07789 MAE / 0.09815 RMSE / 0.99908 cosine**, exactly matching the observed final hidden delta. Rounding llama.cpp pre-output to BF16 barely changes it (**0.07787 MAE**), so final output_norm and final-boundary BF16 rounding are not separate implementation suspects. |
| hipEngine FP32 residual-boundary verifier slice | `benchmarks/results/2026-07-02-mtp-target-f32-residual-diagnostic.json` | Diagnostic-only opt-in `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1`; `performance_claim=false`. The slice keeps verifier target layer residual outputs in FP32 while preserving BF16 mirrors for existing projection inputs. It proves residual precision is semantically active: replaying the old cycle-12 target trace fails earlier at cycle 2, where exact hipEngine samples `[40798, 25, 1103]` and accepts 2, while the FP32-residual slice samples `[40798, 1590, 1103]` and accepts 1. Row-1 logits flip from exact `25` rank 1 / `1590` rank 2 to FP32-residual `1590` rank 1 / `25` rank 2. Exact-vs-slice row-1 pre-output hidden moves **0.00793 MAE / 0.01048 RMSE / 0.99944 cosine** and post-output hidden moves **0.06757 MAE / 0.08528 RMSE / 0.99943 cosine**. This confirms the precision hypothesis is live, but the residual-only slice changes the cycle path before the old pair-12 accept/reject. |
| hipEngine FP32 residual + attention-norm-input verifier slice | `benchmarks/results/2026-07-02-mtp-target-f32-residual-attnnorm-diagnostic.json` | Diagnostic-only extension of `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1`; `performance_claim=false`. The verifier now feeds layer-entry attention RMSNorm from FP32 residual rows when available. This reaches the old cycle-12 branch, but it still samples `[15495, 539, 1151]` and accepts 2. Row 1 ranks token `539` over `26126` with logits **26.05737** vs **25.91428**, margin **+0.14309**. That is farther from llama.cpp than the prior hipEngine serial-exact margin (**+0.11822**) and still opposite llama.cpp (**-0.00896**, `26126` over `539`). Attention-norm input precision alone is therefore not the missing parity fix; remaining suspects are BF16 projection-input/output boundaries inside attention and selected/shared FFN/MoE. |
| hipEngine FP32 attention-norm output + dense-Q8 projection-input slice | `benchmarks/results/2026-07-02-mtp-target-f32-residual-bulk-control-diagnostic.json`, `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-output-denseq8-diagnostic.json` | Diagnostic-only bulk verifier A/B; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` materializes attention RMSNorm into FP32 scratch, casts a BF16 mirror for existing kernels, and routes dense-Q8 dp4a QKV / QKV+gate consumers from the FP32 tensor when `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32=1` is already active. Pair-12 still samples `[15495, 539, 1151]` and accepts 2, but the row-1 `539 - 26126` margin moves from the FP32-residual bulk control **+0.31369** (`26.15284 - 25.83915`) to **+0.18198** (`26.19658 - 26.01460`). This is the first projection-input F32 slice that moves toward llama.cpp, but it remains opposite llama.cpp's **-0.00896** and does not close semantic parity. |
| hipEngine FP32 linear-attention output-to-residual slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` routes row-bulk linear-attention `ssm_out` through the Q8 dp4a `f32_f32_out` kernel, casts the BF16 mirror for captures/downstream consumers, and feeds the FP32 attention output directly into the FP32 residual + post-attention RMSNorm helper. Pair-12 still samples `[15495, 539, 1151]` and accepts 2. The row-1 `539 - 26126` margin only moves **+0.18198 -> +0.17663** (`26.17307 - 25.99644`), still opposite llama.cpp's **-0.00896**. This rules out the linear-attention `ssm_out` BF16 output round as the main remaining semantic gap. |
| hipEngine FP32 linear-attention alpha/beta projection-input slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1` routes row-bulk linear-attention `ssm_alpha` and `ssm_beta` from the FP32 attention-norm tensor, matching llama.cpp's source shape where `beta`/`alpha` are `build_lora_mm(..., cur)` in `src/models/qwen35moe.cpp::build_layer_attn_linear`. Pair-12 is unchanged from the prior attention-output slice: still samples `[15495, 539, 1151]`, accepts 2, and row-1 `539 - 26126` remains **+0.17663** (`26.17307 - 25.99644`). This rules out alpha/beta projection input precision for the active branch. |
| hipEngine FP32 full-attention output-to-residual slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-fullattnout-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` now also routes row-bulk full-attention `attn_output` through a BF16-input/FP32-output path when the raw Q8 sidecar is present, casts the BF16 mirror for captures/downstream consumers, and feeds the FP32 attention output directly into the FP32 residual + post-attention RMSNorm helper. Pair-12 still samples `[15495, 539, 1151]` and accepts 2. The row-1 `539 - 26126` margin worsens from the alpha/beta slice **+0.17663** to **+0.27480** (`26.22991 - 25.95511`), still opposite llama.cpp's **-0.00896**. This rules out the full-attention `attn_output` BF16 output round as the missing semantic gap. |
| hipEngine FP32 MoE selected-sum accumulator slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1` keeps the selected-expert weighted sum in FP32 in the F32-residual MoE combine, instead of BF16-rounding that selected sum before adding residual and sigmoid-gated shared output. Pair-12 still samples `[15495, 539, 1151]` and accepts 2, but row-1 `539 - 26126` narrows from the full-attention-output slice **+0.27480** to **+0.03385** (`26.04901 - 26.01516`). This proves the selected-sum BF16 combine boundary is semantically active, but it is still not sufficient: llama.cpp remains opposite at about **-0.00896**. Combining this with `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` did not produce a pair-12 result because prior-cycle replay diverged at cycle 2 (`[40798, 1590, 1103]` vs trace `[40798, 25, 1103]`). |
| hipEngine FP32 selected-down output slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1` requires the F32 MoE combine diagnostic and routes X8 Q5/Q6 selected-down GEMV into FP32 selected rows before the FP32 selected/shared combine. Pair-12 still samples `[15495, 539, 1151]` and accepts 2, but row-1 `539 - 26126` narrows again from **+0.03385** to **+0.00536** (`26.06115 - 26.05580`). This nearly reaches llama.cpp's tie but still stays on the wrong side; llama.cpp is about **-0.00896**, so the remaining gap is about **0.0143 logits**. |
| hipEngine FP32 selected-SiLU intermediate slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-selectedintermediate-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1` requires the F32 MoE combine + selected-down stack, computes selected `silu(gate) * up` into FP32 scratch, preserves the BF16 mirror, and passes the FP32 activation into selected-down. This is the first slice that flips the active pair-12 decision onto the llama.cpp side: sampled tokens become `[15495, 26126, 1151]`, accepted drafts fall to 1, and row-1 `539 - 26126` moves from **+0.00536** to **-0.00303** (`26.04795 - 26.05098`). llama.cpp is about **-0.00896**, so the residual semantic gap is now about **0.00593 logits** and the selected SwigLU/intermediate BF16 boundary is confirmed as a parity contract. |
| hipEngine transactional bulk-score / serial-state replay diagnostic | `/tmp/hipengine-mtp-proposal-trace/hipengine-active-draftdenseq8-draftonly-f32selectedintermediate-replaystate-fixed2-c13.json` | Diagnostic-only harness mode; not retained and no performance claim. New `--target-block-replay-state-commit` behavior scores with the selected block verifier without linear-state capture, then restores and replays accepted rows with `verify_target_block_serial_exact()` for resident state. The artifact proves the lifecycle wiring (`target_verify_replay_rows=38`, `target_verify_serial_rows=38`, `target_verify_direct_commit_rows=0`) and records corrected effective cycle fields (`target_block_direct_state_commit=false`). It is negative semantically and economically: cycle 2 now scores `[40798, 1590, 1103]`, emits `[40798, 1590]`, and leaves the old pair-12 prefix; 13-cycle speed falls to **31.14 tok/s** because every accepted prefix is replayed. Do not treat score-bulk/serial-replay as the llama.cpp replication path. |
| hipEngine FP32 shared-down output slice | `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-shareddown-denseq8-diagnostic.json` | Diagnostic-only extension; `performance_claim=false`. New flag `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN=1` requires the F32 MoE combine and selected-down stack, routes `ffn_down_shexp` through a BF16-input/FP32-output linear path into `scratch.moe_shared_out_f32`, preserves the BF16 mirror, and combines FP32 selected rows with FP32 shared rows. The split is semantically active but regresses the active near-tie: pair-12 still samples `[15495, 539, 1151]` and accepts 2, and row-1 `539 - 26126` widens from **+0.00536** to **+0.03043** (`26.12703 - 26.09660`). Shared-down output precision alone is ruled out as the missing parity fix. |
| hipEngine FP32 post-attention-norm consumer split | `benchmarks/results/2026-07-02-mtp-target-f32-postnorm-split-diagnostic.json` | Diagnostic-only extension adding `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` plus sub-flags for router, selected q8_1, and shared q8_1 consumers; `performance_claim=false`. Combined router+selected-q8 fails the old trace at cycle 7: row 1 flips from trace token `413` to draft token `4071`. Split margins for `413 - 4071`: control **+0.13053**, router-only **+0.08784**, selected-q8-only **-0.14458**, combined **-0.03290**. Router-only reaches pair 12 but worsens the original mismatch: `539 - 26126` becomes **+0.33520** versus control **+0.14309** and llama.cpp **-0.00896**. This rules out post-attn norm/router/input-q8 precision as the missing fix and pushes the suspect to true GGML-like F32 projection/output contracts. |
| hipEngine FP32 post-norm shared fallback fix smoke | `benchmarks/results/2026-07-02-mtp-target-f32-postnorm-shared-fallback-smoke.json` | Diagnostic-only forced pair-12 smoke after fixing the `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8` fallback: when shared gate/up weights support F32 activation, the fallback now bypasses BF16 pair fusion and launches F32-input singleton shared projections. The full `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1 HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` slice now reaches the pair-12 branch instead of the earlier cycle-7 failure, but still samples `[15495, 539, 1151]` and accepts 2. Row 1 ranks `539` over `26126`: logits **26.064096** vs **25.940170**, margin **+0.123926**. This fixes the diagnostic coverage bug but does not close semantic parity; the later selected-intermediate slice identifies the first side-matching FFN contract. |
| hipEngine vs llama.cpp target pre-output-norm diagnostic | `benchmarks/results/2026-07-02-mtp-target-pre-output-norm-diagnostic.json` | Superseded diagnostic-only forced pair-12 row-1 split. Its hipEngine capture is still valid, but the llama-side raw `h_nextn_pre_output_norm` trace was later shown to be label-alignment ambiguous: the corrected `verify_pre_output_norm`/`verify_layer_output_39` capture above gives **0.01015 MAE**, not the old **0.2481 MAE** outlier. Keep this artifact only as provenance for why per-layer target labels were added. |
| hipEngine vs llama.cpp raw target hidden + lm-head diagnostic | `benchmarks/results/2026-07-02-mtp-target-hidden-raw-lmhead-diagnostic.json` | Diagnostic-only raw row-1 split at the active pair-12 mismatch; `performance_claim=false`. The hipEngine probe emits full FP32 hidden values for row 1 via `--raw-hidden-row 1`; local llama.cpp emits raw `verify_h` row-1 values with `LLAMA_MTP_HIDDEN_TRACE_VALUES=1`, `LLAMA_MTP_HIDDEN_TRACE_VALUE_LABELS=verify_h`, and `LLAMA_MTP_HIDDEN_TRACE_VALUE_ROWS=1`. Dequantizing only `output.weight` rows `539` and `26126` and dotting them with the raw hidden vectors reproduces the observed ranking: hipEngine serial CPU margin `539-26126` **+0.1235** vs observed **+0.1182**, llama CPU margin **-0.0019** vs observed **-0.0090**. The mismatch is therefore target hidden production drift, not lm-head implementation ordering. |
| hipEngine vs llama.cpp forced target hidden diagnostic | `benchmarks/results/2026-07-02-mtp-target-hidden-compare-diagnostic.json` | Diagnostic-only direct hidden-row comparison at the active pair-12 mismatch; `performance_claim=false`. The hipEngine probe now records the cycle-start pending seed and target verifier hidden rows; local llama.cpp instrumentation now emits `verify_h` rows from `common/speculative.cpp`. This corrects the earlier row alignment: llama `process_h_input` is shifted by one row, while `verify_h` is the direct target. Pending seed matches structurally but not bit-for-bit (first8 MAE **0.0909**, last8 MAE **0.0953**). Direct row-1 `verify_h` hidden deltas are small but nonzero: hipEngine bulk vs llama first8 MAE **0.0773**, last8 MAE **0.0609**; hipEngine serial-exact vs llama first8 MAE **0.0785**, last8 MAE **0.0391**. The row-1 logits still flip: hipEngine serial-exact ranks `539` over `26126` by **0.1182**, llama.cpp ranks `26126` over `539` by **0.0090**. |
| hipEngine vs llama.cpp forced target score diagnostic | `benchmarks/results/2026-07-02-mtp-target-score-compare-diagnostic.json` | Diagnostic-only forced-prefix score comparison at the active pair-12 mismatch; `performance_claim=false`. hipEngine reconstructs the active prefix with block-cycle replay and then probes row 1 after input token `15495`. Bulk target verification samples `[15495, 539, 1151]` and ranks `539` over `26126` by **0.336 logits**; final serial-exact probing on the same bulk-replayed prefix still samples `[15495, 539, 1151]` and ranks `539` over `26126` by **0.118 logits**. llama.cpp HIP samples `[15495, 26126]` and ranks `26126` over `539` by **0.009 logits**. This rules out draft generation and a bulk-only verifier bug; the live semantic target is sub-0.12-logit target-row numerical drift/tie-break parity. |
| hipEngine vs llama.cpp active long proposal trace diagnostic | `benchmarks/results/2026-07-02-mtp-proposal-trace-compare-active-draftdenseq8-draftonly-long-diagnostic.json` | Diagnostic-only same-prompt token trace using the then-active `draftdenseq8-draftonly` `llama-compat` route for 32 measured hipEngine cycles and a fresh llama.cpp HIP B2 120-token trace; `performance_claim=false`. It supersedes the short trace: proposals stay aligned at the first real mismatch, but target acceptance diverges at pair 12. Both engines draft `[15495, 539]`; hipEngine accepts both and emits `[15495, 539, 1151]`, while llama.cpp rejects `539` and emits `[15495, 26126]`. Serial-exact hipEngine A/B reproduces the same row-12 decision. This remains semantic provenance; the current live perf target is safe rejected/partial state commit. |
| hipEngine vs llama.cpp short proposal trace diagnostic | `benchmarks/results/2026-07-02-mtp-proposal-trace-compare-active-draftdenseq8-draftonly-diagnostic.json` | Superseded diagnostic-only same-prompt token trace using the then-active `draftdenseq8-draftonly` `llama-compat` route and the earlier llama.cpp HIP B2 token trace; `performance_claim=false`. It remains useful because it showed the old pair-3 proposal mismatch was closed, but its final-row boundary interpretation is superseded by the longer trace above. |
| hipEngine vs llama.cpp prior proposal trace diagnostic | `benchmarks/results/2026-07-02-mtp-proposal-trace-compare-diagnostic.json` | Historical diagnostic-only same-prompt token trace before the resident initial MTP KV writer fix was retained in the active route; `performance_claim=false`. The v3 comparison adds per-row stream offsets and token-divergence row locations. It proves the old prompt/target state aligned for the first three measured cycles, then draft proposals diverged before any rejection; keep as provenance for the resident-initial-KV semantic fix, not as the current active proposal gap. |
| hipEngine draft-context/logit A/B diagnostic | `benchmarks/results/2026-07-02-mtp-draft-context-logit-ab-diagnostic.json` | Diagnostic-only follow-up on the same prompt; `performance_claim=false`. It rules out hipEngine accepted-row KV commit and llama.cpp's public backend-sampling toggle as the primary cause, shows resident full-logit draft is not closer, and records cycle-3 top-k score margins with/without hipEngine MTP device KV. |
| llama.cpp draft score trace diagnostic | `benchmarks/results/2026-07-02-mtp-llamacpp-draft-score-trace-diagnostic.json` | Historical diagnostic-only same-prompt score trace after llama.cpp commit `0f7d32267` added `draft_sample_trace` to `LLAMA_MTP_TOKEN_TRACE` rows. At the pre-fix first divergence llama.cpp ranks token `8` first and `65342` second by only **0.100 logits**, while hipEngine with device KV ranks `65342` first and token `8` third, **1.260 logits** behind. This showed the then-active blocker was MTP seed/context/logit parity at seq position 49, not merely accepted-row KV commit or sampler toggles. |
| MTP hidden-state parity diagnostic | `benchmarks/results/2026-07-02-mtp-hidden-state-parity-diagnostic.json` | Diagnostic-only same-prompt hidden summary trace after hipEngine commit `6190fd08` added `--record-draft-hidden-stats` and llama.cpp commit `c0f750604` added `draft_hidden_state_trace`. The `draft_seed_input` row is close (`rms_delta` **0.00456**, first8 mean abs **0.0798**), but depth-0 `draft_next_seed` has a much larger first8 mean abs delta (**0.731**) before token selection. The next target is depth-0 draft attention/KV/rope/context state, not the `pending_h` seed handoff. |
| Resident MTP RoPE-dimension fix diagnostic | `benchmarks/results/2026-07-02-mtp-resident-rope-dim-fix-diagnostic.json` | Diagnostic-only semantic fix; `performance_claim=false`. The GGUF metadata has `qwen35moe.rope.dimension_count=64` while resident MTP used `qk_head_dim=256` as `rotary_dim` for draft Q/K and accepted-row MTP K/V commit. Fixing that cuts the seq-position-49 depth-0 `draft_next_seed` first8 mean abs delta **0.731 -> 0.329**, but hipEngine still drafts `[65342, 18078]` while llama.cpp drafts `[8, 1411]`; token `8` is now rank 2 but still **1.391 logits** behind `65342`. |
| llama.cpp tensor-stage parity diagnostic | `benchmarks/results/2026-07-02-mtp-llamacpp-tensor-stage-trace-diagnostic.json` | Diagnostic-only same-prompt tensor summary trace after local llama.cpp commit `687c17d26` added `LLAMA_MTP_TENSOR_TRACE=1` for selected `graph_mtp` labels. At seq position 49 / depth 0, token embed, e/h norm, projected state, post-RoPE Q/K, and V are close; the first large jump is `draft_stage_attn_pregate` (**rms 1.056 hipEngine vs 1.410 llama.cpp**, first8 MAE **0.147**). The next semantic target is draft attention history/KV rows, context length, or mask/softmax behavior. |
| MTP attention-history row trace diagnostic | `benchmarks/results/2026-07-02-mtp-attention-history-row-trace-diagnostic.json` | Diagnostic-only same-prompt row-aware trace after hipEngine added `--record-draft-cache-rows` and llama.cpp commit `1ebf790cd` added process-row tensor tracing. At seq position 49 / depth 0, hipEngine dense K/V rows match llama.cpp at sampled positions 40, 48, and 49 (`first8_mae` **0.0118-0.0468** on reliable rows), while `draft_stage_attn_pregate` still differs (**rms 1.056 vs 1.410**). The next semantic target is effective attention execution: visible count/mask, GQA head mapping, score scale/softmax, or FA-on math. |
| MTP attention-debug host recompute diagnostic | `benchmarks/results/2026-07-02-mtp-attention-debug-diagnostic.json` | Diagnostic-only same-prompt hipEngine host recomputation of resident dense attention after adding `--record-draft-attention-debug`; `performance_claim=false`. At seq position 49 / depth 0, hipEngine GPU `draft_stage_attn_pregate` matches a host recompute over its own dense K/V cache (`cpu_device_mae_mean` **4.1e-7**, max abs **9.3e-6**). The top attention row is 48 for 7 heads and 49 for 9 heads, so the previously sampled late rows cover the dominant hipEngine attention weights. Next split is llama.cpp FA-on effective attention/mask/weight distribution, not hipEngine dense-attention kernel math. |
| Resident initial MTP KV writer diagnostic | `benchmarks/results/2026-07-02-mtp-resident-initial-kv-diagnostic.json` | Diagnostic-only semantic fix; `performance_claim=false`. The active llama-compat path now seeds initial prompt MTP KV with `resident_write_kv_rows` instead of legacy `run_draft(..., kv_write_only=True)`. At seq position 49 / depth 0, hipEngine now drafts **`[8, 1411]`**, matching llama.cpp, while the prior path drafted `[65342, 18078]`. Row 2 now matches closely on high-impact heads (`qh7` weight **0.227 vs 0.218**, `qh12` **0.408 vs 0.391**; K/V first4 max deltas `<=0.0378` on row 2). Row 0 remains a boundary-row residual, but it no longer changes this draft decision. |
| hipEngine llama-compat draft lm-head all-sync split | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-top1split128-allsync-smoke.json` | Attribution-only smoke with extra sync points inside the Q6 top-1 draft lm-head path, including stage1 vs stage2/gather. Do not use for headline tok/s. |
| hipEngine llama-compat draft-chain rocprof split | `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-dual.json` | Diagnostic-only ROCTX/kernel trace for the retained B2 resident draft chain (`--q6-top1-dp4a --selected-down-x8-repack q6 --record-stage-timings`) with default-on Q8 shared dual enabled. Use it to rank draft kernel families; do not use it for headline tok/s. |
| hipEngine llama-compat draft-chain fine sync split | `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-routerrow-sharedgate-fine-sync.json`; prior resident-init artifact `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-residentinit-fine-sync.json`; prior router-row A/B controls `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-control-fine-sync.json`, `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-fine-sync.json` | Attribution-only ROCTX/kernel trace plus `--sync-stage-timings` for the active X8 Q6 top-1 route after the profiler was aligned with resident initial prompt KV seeding and the shared-gate scalar-dot fix. The current profile is **6.743 ms/cycle host**, **5.767 ms/cycle kernel**, **90.25 calls/cycle**; `draft_run_ffn_shared_gate_linear` moves **0.222 -> 0.027 ms/cycle**. The router-row A/B remains the provenance for the prior non-Q6 leaf fix: `draft_run_ffn_router_linear` **0.508 -> 0.048 ms/cycle**, host **7.569 -> 6.971 ms/cycle**, and kernel **6.461 -> 5.983 ms/cycle**. Use it to target draft leaves; do not use it for headline tok/s because it adds sync points. |
| hipEngine llama-compat draft-chain parallel-attn GPU-event split | `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-routerrow-sharedgate-parallelattn-gpuevents.json` | Diagnostic-only non-sync HIP event split after replacing the thread-0 dense-attention body with a parallel reduction kernel. Result: host **6.529 ms/cycle**, kernel **5.498 ms/cycle**, **90.5 calls/cycle**, `draft_device_chain_drain` **5.258 ms/cycle**, `draft_gpu_run_attention` **0.243**, and `hipengine_mtp_dense_attn_f32` **0.033 ms/cycle**. Versus the prior GPU-event split, attention falls **0.558 -> 0.243 ms/cycle** and the kernel-family attention core falls **0.345 -> 0.033 ms/cycle**. |
| hipEngine vs llama.cpp draft-kernel compare | `benchmarks/results/2026-07-02-mtp-draft-kernel-compare-draftdenseq8-draftonly.json` | Diagnostic-only offline join of the now-superseded unsafe hipEngine draft-chain GPU-event profile, the unsafe `75.15 tok/s` full-suite row, llama.cpp MTP ROCTX range profile, and retained llama.cpp HIP stage rerun. It still records useful Q6 top-1 per-call parity: hipEngine `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1` **1.786 ms/call** vs llama.cpp Q6_K `mul_mat_vec_q` **1.781 ms/call** (**+0.005 ms/call**). Treat the parent-wall conclusion in this artifact as superseded by the semantic-safe partialfix full-suite row; the active gap is now verifier replay, not Q6 top-1. |
| hipEngine llama-compat draft-chain GPU-event split | `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-routerrow-sharedgate-gpuevents.json` | Diagnostic-only non-sync HIP event split for the same active B2 draft route (`--record-stage-timings --gpu-event-stage-timings`, no per-leaf sync). It keeps the ROCTX/kernel trace while adding `draft_gpu_*` queued-work buckets. Result: host **6.896 ms/cycle**, kernel **5.793 ms/cycle**, **90.0 calls/cycle**, `draft_device_chain_drain` **5.505 ms/cycle**, `draft_gpu_run_lm_head` **3.707**, `draft_gpu_decode_initial` **3.034**, and `draft_gpu_decode_next` **3.005**. This proves the async `draft_topk_readback` bucket is real queued draft GPU work, not D2H; use it before adding more sync-only splits. |
| hipEngine llama-compat rejected fused-head-q8 check | `benchmarks/results/2026-07-02-gguf-mtp-draft-fusedheadq8-rejected.json` | Diagnostic only: a temporary exact fused draft lm-head prep kernel combined final RMSNorm, BF16 rounding, and q8_1 activation quantization for the Q6_K top-1 path. The byte-level check passed, but the GPU-event A/B rejected it: host wall was noise (**6.650 -> 6.649 ms/cycle**), while kernel work regressed (**5.514 -> 5.572 ms/cycle**), `draft_device_chain_drain` rose **5.266 -> 5.370 ms/cycle**, `draft_gpu_run_lm_head` rose **3.723 -> 3.781 ms/cycle**, and the norm/cast/quant bucket rose **0.076 -> 0.145 ms/cycle**. Code and route were reverted; do not add this flag. |
| hipEngine llama-compat rejected draft dense-Q8 dp4a check | `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-draftdenseq8-fine-sync.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-full.json` | Diagnostic only: routes resident draft dense Q8_0 F32 projections through F32->q8_1 plus raw-Q8 dp4a float-output wrappers (`--resident-mtp-draft-dense-q8-dp4a`). Draft-chain profile moved the intended dense bucket (**5.983 -> 5.570 ms/cycle kernel**, `draft_dense_shared_gemv` **0.784 -> 0.374 ms/cycle**), but full-suite B2 regressed versus the active router-row lane (**64.41 -> 64.14 tok/s**, cycle **15.547 -> 15.612 ms/output**) because acceptance/row economy worsened (`acc/output` **0.578 -> 0.573**, target rows/output **1.266 -> 1.282**) and verifier drain rose (**12.166 -> 12.324 ms/output**). Keep as evidence only. |
| hipEngine llama-compat rejected selected SiLU/down fusion check | `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-control-fine-sync.json`, `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-fine-sync.json` | Diagnostic only: adds exact BF16-equivalent `--resident-mtp-draft-selected-silu-down-fused`, fusing selected MoE `silu(gate)*up` into the Q5_K selected-down GEMV. It removes one launch but the fused Q5 body is slower: kernel time **5.973 -> 6.054 ms/cycle**, draft host wall **7.044 -> 7.206 ms/cycle**, `draft_run_moe_down_combine` **0.487 -> 0.531 ms/cycle**, and `gguf_k_selected_prefill_out` **0.325 ms/cycle** becomes `gguf_k_selected_silu_prefill_out` **0.391 ms/cycle**. No full-suite run; the parent draft profile regressed. |
| hipEngine llama-compat Q8 shared-dual A/B | `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-control.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-dual.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shared-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-smoke.json` | Retained exact draft-path improvement: Q8 shared gate/up launches collapse from two single raw-Q8 GEMVs to one dual F32/F32 launch per draft layer. The isolated kernel delta is small, but same-session async smoke moved **69.44 -> 70.20 tok/s** with identical acceptance; full-suite compat moved **60.96 -> 61.19 tok/s**. |
| hipEngine llama-compat rejected Q6 top-1 t64 check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-top1split-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-smoke.json` | Diagnostic only: llama.cpp's RDNA3 Q6_K MMVQ uses a two-warp single-column shape, but hipEngine's pack8 top-1 stage1 remains faster at the existing 128-thread launch on the real route. |
| hipEngine llama-compat rejected Q6 top-1 row-shape check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row-smoke.json` | Diagnostic only: this copies llama.cpp's one-output-row-per-block Q6_K MMVQ shape and signed `__vsubss4`/dot4 body more closely than the t64 check, but the larger final reduce erases the tiny stage1 gain and async smoke regresses. |
| hipEngine llama-compat rejected Q6 top-1 scalehoist check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rerun-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist-smoke.json` | Diagnostic only: keeps pack8's `vocab/8` final reduce but hoists Q6_K `d*scale` values into shared memory. Same-session smoke rejected it, so no full-suite run and no headline update. |
| hipEngine llama-compat rejected Q6 top-1 pack8-llama-body check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama-b2-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama-control-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama-allsync-smoke.json` | Diagnostic only: keeps pack8's `vocab/8` final reduce but swaps in llama.cpp's Q6_K vecdot body. All-sync stage1 improves slightly, but the async B2 smoke regresses, so no full-suite run and no headline update. |
| hipEngine llama-compat rejected Q6 top-1 pack16 check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-pack16-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-pack16-smoke.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-pack16.json` | Diagnostic only: doubles the retained pack8 stage1 output group to 16 vocab rows per block to reduce q8 activation reloads and final reduce entries. Same-session denseq8all smoke is neutral/slightly worse (**71.74 -> 71.72 tok/s**, `draft_initial` **2.479 -> 2.487 ms/output**), and draft rocprof shows the stage1 family itself slows **3.603 -> 3.684 ms/cycle**, so no full-suite run and no headline update. |
| hipEngine llama-compat retained Q6 top-1 X8 lm-head sidecar | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-smoke.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-full.json` | Superseded retained lane: materializes `output.weight[:vocab]` as X8 tiles for the accuracy-traded draft Q6_K top-1 path. Same-session smoke moved **71.53 -> 71.76 tok/s** with identical acceptance; draft rocprof shows `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1` at **3.558 ms/cycle** vs the prior pack8 **3.603 ms/cycle**; full-suite compat moves **61.19 -> 61.31 tok/s** and draft drain **3.378 -> 3.352 ms/output**. The F32 `ssm_out` row supersedes it. |
| hipEngine llama-compat rejected Q6 top-1 X8 dscale sidecar | `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8dscale.json` | Diagnostic only: adds an X8-aligned FP32 `d*scale` sidecar to test whether repeated Q6_K scale unpack/multiply is the retained X8 bottleneck. It regresses the draft-chain profile: host wall **6.805 -> 8.023 ms/cycle**, kernel time **6.427 -> 7.615 ms/cycle**, and `draft_lm_head_q6_top1` **3.648 -> 4.859 ms/cycle** versus the retained X8 artifact. Extra sidecar memory traffic/register pressure is worse than recomputing scales, so no full-suite run and no headline update. |
| hipEngine llama-compat retained verifier F32 `ssm_out` raw-Q8 dp4a | `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-f32ssm.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-full.json` | Superseded retained lane before router-row: direct-state `ssm_out` has F32 activations, so it needs a separate F32 q8_1 quantizer before the raw-Q8 dp4a singleton body. Isolated block profile moved kernel **23.893 -> 22.881 ms/block**; same-session smoke moved **70.74 -> 71.43 tok/s** with identical acceptance; full-suite compat moved **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**, verifier drain **12.662 -> 12.158 ms/output**, acc/output **0.567 -> 0.578**, and target rows/output **1.299 -> 1.266**. |
| hipEngine llama-compat rejected verifier shared-Q8 dp4a check | `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-refresh.json`, `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-sharedq8.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-full.json` | Diagnostic only: routes verifier shared-expert `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` through raw-Q8 q8_1/dp4a helpers. The isolated block profile was slightly positive (kernel **23.893 -> 23.648 ms/block**) and smoke improved **70.64 -> 71.66 tok/s**, but full-suite B2 regressed **61.31 -> 59.63 tok/s**, cycle **16.331 -> 16.793 ms/output**, verifier drain **12.662 -> 13.038 ms/output**, and acceptance **0.567 -> 0.556**. Do not promote. |
| hipEngine llama-compat rejected verifier Q6 top-1 X8 lm-head check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-vlmheadtop1-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-vlmheadtop1-smoke.json` | Diagnostic only: materializes an X8 `root.lm_head` sidecar and routes verifier row sampling through q8_1/dp4a Q6_K top-1, skipping full logits plus argmax. Same-session smoke rejected it: control **71.12 tok/s**, cycle **14.086 ms/output**, `target_block_lm_head_sample` **1.058 ms/output**, verifier **11.277 ms/output**; verifier-top1 **65.18 tok/s**, cycle **15.363 ms/output**, `target_block_lm_head_sample` **1.874 ms/output**, verifier **12.473 ms/output**, with identical acceptance. No full-suite run and no headline update. |
| hipEngine llama-compat rejected q5/both X8 selected-down check | `benchmarks/results/2026-07-01-llama-compat-b2-x8-selected-down-dp4a-current-micro.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8both-smoke.json` | Diagnostic only: q6-only X8 selected-down is retained for the accuracy-traded compat lane; q5/both smoke regressed vs q6-only, so q5 stays on T16. |
| hipEngine llama-compat rejected Q4 X8 selected gate/up check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-control-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-allsync-smoke.json` | Diagnostic only: materializing Q4_K selected gate/up experts as X8 q8_1/dp4a replacement layout regressed same-session smoke **67.62 -> 59.08 tok/s** and target verifier drain **12.005 -> 14.117 ms/output** with identical acceptance. All-sync attributes the loss to selected gate/up GEMV, not q8_1 quantize. |
| hipEngine llama-compat rejected Q4 raw selected gate/up check | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup-control-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup-control-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup-allsync-smoke.json` | Diagnostic only: materializing Q4_K selected gate/up experts as raw GGUF copies llama.cpp's `mul_mat_vec_q_moe` body/layout more directly, but same-session smoke regressed **68.55 -> 62.04 tok/s** and target verifier drain **11.792 -> 13.328 ms/output** with identical acceptance. All-sync attributes the loss to the raw selected gate/up GEMV body. |
| hipEngine llama-compat rejected Q5 T16 selected-down one-wave check | `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-t64-rerun-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-q5t32-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q4-t16-selected-dual-dp4a-q5t32-control-micro.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q5t32-smoke.json` | Diagnostic only: llama.cpp MoE MMVQ uses one wave/token, and Q5 selected-down improved in isolation at 32 threads, but the real compat route regressed vs the retained 64-thread/Q6-X8 lane. |
| hipEngine llama-compat rejected fused-SiLU check | `benchmarks/results/2026-07-01-llama-compat-b2-q4-t16-selected-dual-dp4a-micro.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-fusedsilu-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-fusedsilu-smoke.json` | Diagnostic only: micro/all-sync suggested launch removal could help, but async smoke regressed the retained compat row. |
| hipEngine llama-compat rejected Q8T16 pair t64 check | `benchmarks/results/2026-07-01-q8-t16-pair-threads-micro.json` | Diagnostic only: the actual verifier `attn_qkv+attn_gate` Q8T16 pair shape is faster at the existing 128-thread launch than at 64 threads. |
| hipEngine llama-compat rejected Q8T16 q8_1/dp4a pair check | `benchmarks/results/2026-07-01-q8-t16-pair-q8-1-dp4a-micro.json` | Diagnostic only: applying llama.cpp-style q8_1/dp4a to the existing T16 tile layout is much slower than the exact pair because T16 stores four-K dot4 bytes strided by output column. |
| hipEngine llama-compat rejected Q8T16 pair rowtile check | `benchmarks/results/2026-07-01-q8-t16-pair-rowtile-micro.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-rowtilepair-full.json` | Diagnostic only: exact row-amortized rowtile4 wins the isolated pair and smoke, but full-suite compat regresses, so runtime default remains the existing pair kernel. |
| hipEngine llama-compat rejected Q8T16 rowtile-all check | `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-q8rowtileall.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall-control-smoke.json` | Diagnostic only: broad exact rowtile coverage lowers the isolated block verifier dense-Q8 kernel bucket, but same-session async smoke loses to the retained `x8q6` route, so no full-suite run and no headline update. |
| hipEngine llama-compat rejected raw-Q8 dp4a rowtile-pair sidecar | `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-full.json` | Diagnostic only: the fused raw-Q8 sidecar pair launch improves smoke verifier sub-buckets but full-suite B2 regresses **60.36 -> 59.42 tok/s** with lower acceptance and more target rows/output. This pair-only variant is rejected; the later `denseq8all` route supersedes it as the retained dense-Q8 llama-replication lane. |
| hipEngine llama-compat retained raw-Q8 dp4a all-sidecar + Q8 shared dual | `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-dual.json` | Superseded active lane: raw Q8_0 x q8_1 rowtile coverage for pair, singleton, and Q/K/V triple verifier projections cuts the isolated dense-Q8 bucket **11.420 -> 8.902 ms/block**. Default-on resident Q8 shared gate/up dual GEMV cuts the draft shared projection call count and moved the retained full-suite lane to **61.19 tok/s**, total wall **16.364 ms/output**, verifier drain **12.666 ms/output**, with unchanged acceptance **0.567** and target rows/output **1.299**. The X8 top-1 and F32 `ssm_out` rows supersede it. |
| llama.cpp HIP current rerun | `benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.json`, `benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.jsonl` | Diagnostic-retained same-protocol B2 rerun with local llama.cpp instrumentation patches (`llama_cpp_commit=1ebf790c`, dirty). Use this as the active stage target: base **51.98 tok/s**, MTP **71.91 tok/s**, cycle **14.269 ms/output**, draft drain **2.141 ms/output**, visible sampler **1.888 ms/output**, verifier **12.120 ms/output**, accepted/output **0.610** in measured stage rows, draft acceptance **0.805**, and target rows/output **1.148**. |
| llama.cpp HIP prior deep trace | `benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.json` | Superseded instrumented llama.cpp HIP B2 trace. Keep as the prior comparison row: **72.12 tok/s**, cycle **14.231 ms/output**, draft drain **2.140 ms/output**, visible sampler **1.886 ms/output**, verifier **12.083 ms/output**. |

#### Latest proposal trace finding

Active long-trace artifact:
`benchmarks/results/2026-07-02-mtp-proposal-trace-compare-active-draftdenseq8-draftonly-long-diagnostic.json`.
This is a one-prompt diagnostic, not a retained performance claim. It reruns
the active `draftdenseq8-draftonly` `llama-compat` route for 32 measured cycles
and reruns llama.cpp HIP B2 with a 120-token budget so the comparison is no
longer clipped at the old 10-row trace boundary.

| comparison | hipEngine active `llama-compat` | llama.cpp HIP B2 | reading |
| --- | ---: | ---: | --- |
| cycle rows compared | 32 | 32 | Same measured prompt after llama.cpp warmup exclusion. |
| exact draft rows | 30 / 32 | 30 / 32 | Draft proposal shape is mostly aligned; the first real mismatch is not a draft-token mismatch. |
| exact output rows | 29 / 32 | 29 / 32 | The old final-row boundary explanation is superseded by a real target accept/reject mismatch. |
| accepted-count match rows | 30 / 32 | 30 / 32 | Same reading: target acceptance diverges before proposal chunking drifts. |
| visible output tokens in compared rows | 88 | 89 | One-prompt diagnostic only; do not generalize to full-suite row economy. |
| accepted / output | 0.636 | 0.640 | Same caution: one prompt only. |
| draft acceptance | 0.875 | 0.891 | Same caution: one prompt only. |
| flattened output stream prefix | 33 tokens | 33 tokens | Streams match beyond the old 26-token short-trace boundary, then diverge at pair 12. |
| first stream token divergence | token index 33 = `539` | token index 33 = `26126` | This is a target verifier accept/reject difference, not a proposal-generation difference. |

First real cycle-level divergence:

| field | hipEngine | llama.cpp |
| --- | --- | --- |
| pair / cycle | pair 12 / cycle 12 | pair 12 / traced cycle 18 |
| draft tokens | `[15495, 539]` | `[15495, 539]` |
| accepted draft tokens | 2 | 1 |
| output tokens | `[15495, 539, 1151]` | `[15495, 26126]` |
| rejected draft token | none | `539` |

Interpretation: the old pair-3 proposal mismatch is closed, but the short
10-row active trace was insufficient to prove sustained semantic parity. The
longer same-protocol trace shows the current live mismatch is target verifier
semantic parity at pair 12: both engines propose the same two draft tokens
`[15495, 539]`, then hipEngine accepts both while llama.cpp rejects `539` and
samples `26126`.

Follow-up A/B checks:

| A/B | result | reading |
| --- | --- | --- |
| hipEngine active block verifier vs `--target-block-verify-mode serial-exact` | Same row-12 decision: `[15495, 539, 1151]`. | Rules out row-bulk verifier scheduling and direct-state commit as the local cause. |
| Disable selected-down X8 in the serial-exact verifier run | Rows 0-15 remain identical to active serial-exact, including row 12. | The selected-down X8 sidecar is not the isolated cause. |
| Disable `--verify-dp4a` in the serial-exact verifier run | Rows 0-15 remain identical to active serial-exact, including row 12. | The mismatch is not isolated to the selected-expert dp4a opt-in. |
| Disable dense verifier dp4a | Stream diverges earlier by accepting `[262, 4071]` where active hipEngine and llama.cpp reject `4071`. | Dense verifier dp4a is required for earlier trace alignment on this prompt; simply removing it is not a fix. |

Forced target score diagnostic:
`benchmarks/results/2026-07-02-mtp-target-score-compare-diagnostic.json`.
This adds `scripts/gguf_mtp_forced_target_probe.py` on the hipEngine side and a
local llama.cpp `server-context.cpp` target-score trace under
`LLAMA_MTP_TOKEN_TRACE`. The hipEngine probe reconstructs the active prefix by
replaying prior trace cycles through the same block verifier/commit path; naive
serial visible-token replay diverges early and is not a valid forced-prefix
state.

| row-1 target score after input `15495` | sampled tokens | accepted drafts | top-1 vs top-2 |
| --- | --- | ---: | --- |
| hipEngine bulk verifier | `[15495, 539, 1151]` | 2 | `539` logit **26.3945**, `26126` **26.0583**; `539` ahead by **0.3362**. |
| hipEngine serial-exact final probe with bulk prefix replay | `[15495, 539, 1151]` | 2 | `539` logit **26.2025**, `26126` **26.0843**; `539` ahead by **0.1182**. |
| llama.cpp HIP target trace | `[15495, 26126]` | 1 | `26126` logit **26.1047**, `539` **26.0957**; `26126` ahead by **0.0090**. |

Reading: the mismatch is now a target row logits parity problem on a near-tie,
not proposal generation, selected-down X8, selected-expert dp4a, dense verifier
dp4a removal, row-bulk scheduling, or direct-state commit. The next source
comparison should split target row-1 hidden/logit numerics at this forced prefix:
target hidden row after consuming input `15495`, output norm, lm-head input, and
lm-head GEMV/output ordering. This remains semantic provenance from before the
direct-state lifecycle correction; the active performance gap is now the
semantic-safe rejected/partial replay cost tracked at the top of this document.

Forced target hidden diagnostic:
`benchmarks/results/2026-07-02-mtp-target-hidden-compare-diagnostic.json`.
This adds a cycle-start pending seed summary to
`scripts/gguf_mtp_forced_target_probe.py` and a local llama.cpp `verify_h` trace
point in `common/speculative.cpp` (dirty external diagnostic instrumentation,
not a hipEngine dependency). It also fixes the earlier hidden-row interpretation:
llama.cpp's `process_h_input` is the shifted MTP draft-context input, so
`process_h_input` row `i+1` corresponds to target verifier `verify_h` row `i`.
Do not compare hipEngine verifier row `i` to llama `process_h_input` row `i`.

| hidden/logit check | hipEngine bulk | hipEngine serial-exact | llama.cpp HIP | reading |
| --- | ---: | ---: | ---: | --- |
| cycle pending seed vs llama draft seed | first8 MAE **0.0909**, last8 **0.0953** | same replay state | reference | Handoff is structurally aligned, not a missing `pending_h`/shift bug. |
| row 0 `verify_h` after input `653` | first8 MAE **0.0976**, last8 **0.1922** | first8 **0.1464**, last8 **0.1936** | sampled `15495` | Hidden differs but both engines choose the same high-margin token. |
| row 1 `verify_h` after input `15495` | first8 MAE **0.0773**, last8 **0.0609** | first8 **0.0785**, last8 **0.0391** | sampled `26126` | The hidden rows are close but not bit-identical exactly where the logits are a tie. |
| row 1 score for `539` | **26.3945** rank 1 | **26.2025** rank 1 | **26.0957** rank 2 | hipEngine keeps the draft token ahead. |
| row 1 score for `26126` | **26.0583** rank 2 | **26.0843** rank 2 | **26.1047** rank 1 | llama.cpp wins the tie by **0.0090** logits. |
| row 2 `verify_h` after input `539` | first8 MAE **0.1733**, last8 **0.1536** | first8 **0.1792**, last8 **0.1286** | sampled `1151` | Downstream row also chooses the same top token, so row 1 remains the branch point. |

Updated reading: target hidden-row lifecycle is aligned, but target forward
numerics are not bit-identical. The next split should dump raw row-1
post-output-norm hidden values and score `539`/`26126` through a common lm-head
path, or instrument the target forward earlier (pre-output-norm residual and
per-layer hidden checkpoints) to find the first source of the small row-1 drift.
This remains semantic parity work; it is not a retained full-suite performance
gap.

Raw row-1 hidden + lm-head diagnostic:
`benchmarks/results/2026-07-02-mtp-target-hidden-raw-lmhead-diagnostic.json`.
This reruns the forced probes with `--raw-hidden-row 1` and reruns llama.cpp
with raw values filtered to `verify_h` row 1. Then it dequantizes only
`output.weight` rows `539` and `26126` using `dequantize_gguf_data()` and dots
those rows against each raw hidden vector.

| row-1 raw/cross-score check | hipEngine bulk | hipEngine serial-exact | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| hidden MAE vs llama row 1 | **0.08067** | **0.07789** | 0 |
| hidden RMSE vs llama row 1 | **0.10226** | **0.09815** | 0 |
| hidden max abs delta vs llama row 1 | **0.38638** | **0.38062** | 0 |
| hidden cosine vs llama row 1 | **0.999005** | **0.999078** | 1.0 |
| CPU-dequant lm-head margin `539 - 26126` | **+0.32916** | **+0.12350** | **-0.00192** |
| observed margin `539 - 26126` | **+0.33617** | **+0.11822** | **-0.00896** |
| CPU-vs-observed margin error | **0.00701** | **0.00528** | **0.00704** |

Reading: the same dequantized lm-head rows reproduce each engine's ranking from
the raw hidden vector. The row-1 accept/reject mismatch is therefore not an
lm-head ordering or candidate-sort issue. It is explained by the target hidden
row that reaches the lm-head.

Pre-output-norm target residual diagnostic:
`benchmarks/results/2026-07-02-mtp-target-pre-output-norm-diagnostic.json`.
This reruns the same forced pair-12 prefix with hipEngine
`--raw-pre-output-norm-row 1` and local llama.cpp target graph tensor tracing for
`h_nextn_pre_output_norm` row 1. The local llama.cpp checkout needed diagnostic
plumbing in `qwen35moe.cpp`, `llama-graph.cpp`, `llama-context.cpp`, and
`common/speculative.cpp`; those external changes are not part of hipEngine.

| row-1 hidden checkpoint | hipEngine bulk vs llama | hipEngine serial-exact vs llama | reading |
| --- | ---: | ---: | --- |
| pre-output_norm residual MAE / RMSE | **0.24833 / 0.31823** | **0.24815 / 0.31797** | The residual stream entering final output_norm is already substantially different. |
| pre-output_norm residual cosine | **0.65539** | **0.65759** | Bulk and serial-exact are close to each other but both are far from llama.cpp at this checkpoint. |
| post-output_norm `verify_h` MAE / RMSE | **0.08067 / 0.10226** | **0.07789 / 0.09815** | Output norm compresses the residual-space drift into a much smaller post-norm hidden delta. |
| post-output_norm `verify_h` cosine | **0.999005** | **0.999078** | The rows are visually close after norm, but the candidate logits are a near-tie. |
| observed margin `539 - 26126` | **+0.33617** | **+0.11822** | llama.cpp is **-0.00896**, so the remaining post-norm drift still flips acceptance. |

Superseded reading: this table was useful because it moved the search before
the lm-head, but its llama-side `h_nextn_pre_output_norm` label was later shown
to be ambiguous. The corrected per-layer target labels put row-1
pre-output_norm/layer-39 drift at **0.01015 MAE**, not **0.2481 MAE**. The
subsequent layer-31 sub-boundary split also shows no single late-layer cliff:
layer output vs llama.cpp `l_out_31` is **0.00528 MAE / 0.00671 RMSE /
0.99871 cosine**. A CPU output_norm recompute then shows the output_norm
implementation itself is identical on both raw rows: the row-1 pre-output delta
is simply amplified by the shared norm/weights into the final `verify_h` delta.
The active target is now accumulated BF16-vs-F32 residual-boundary drift, not a
specific bad attention/MoE substage or final output_norm bug.

Layer-31 sub-boundary diagnostic:
`benchmarks/results/2026-07-02-mtp-target-layer31-subboundary-diagnostic.json`.
hipEngine emits this through `--layer-boundary-row LAYER:ROW` or
`--raw-layer-boundary-row LAYER:ROW`; the capture runs in an isolated replay
session so the sub-layer tap cannot perturb the scored verifier path. Local
llama.cpp traces `attn_norm_31`, `attn_residual_31`, `attn_post_norm_31`,
`ffn_out_31`, `post_moe_31`, and `l_out_31` for row 1.

| layer-31 row-1 checkpoint | hipEngine serial-exact vs llama.cpp | reading |
| --- | ---: | --- |
| attn_norm | **0.01623 MAE / 0.02082 RMSE / 0.99969 cosine** | Direct initial norm comparison; close, but normalized space magnifies existing residual drift. |
| attention residual | **0.00303 MAE / 0.00405 RMSE / 0.99943 cosine** | Still in the gradual per-layer drift band. |
| post-attn norm | **0.03616 MAE / 0.04541 RMSE / 0.99940 cosine** | Normalized-space values amplify the residual delta, but direction remains very close. |
| reconstructed hip MoE `ffn_out` vs llama `ffn_out_31` | **0.00446 MAE / 0.00562 RMSE / 0.99161 cosine** | Direct component reconstruction, no longer just `layer_out - residual`; no standalone MoE-combine cliff. |
| layer output vs llama `post_moe_31` / `l_out_31` | **0.00528 MAE / 0.00671 RMSE / 0.99871 cosine** | Confirms layer 31 is not the source of a large jump. |
| reconstructed rounded post-MoE vs hip layer output | **0 MAE / 0 RMSE** | The host reconstruction matches the fused hip combine kernel's BF16 selected-branch and final rounding contract. |

Fine MoE tap follow-up:
`benchmarks/results/2026-07-02-mtp-target-layer31-fine-moe-taps-diagnostic.json`.
This reruns the same forced pair-12 row-1 hipEngine capture and keeps the scored
decision unchanged (`539 - 26126` **+0.118217**). The capture now exposes the
durable llama.cpp-shaped target MoE leaves available after a layer run:
router logits, selected SwigLU/intermediate, selected down rows, per-expert
weighted down rows, selected weighted sum before/after the BF16 combine
boundary, shared intermediate/out, sigmoid-gated shared contribution, combined
`ffn_out`, and rounded post-MoE.

| layer-31 hipEngine fine MoE tap | value | reading |
| --- | ---: | --- |
| selected experts | `[221, 95, 240, 60, 88, 19, 212, 59]` | Top-k selection is now explicitly captured for the target row. |
| shared-gate logit / sigmoid | **-2.042253 / 0.114838** | Shared contribution is small but nonzero. |
| selected weighted sum RMS | **0.039687** | Direct analog for the aggregate after llama.cpp `ffn_moe_weighted` add reduction. |
| selected weighted BF16 RMS | **0.039687** | The selected branch's BF16 combine boundary barely moves the RMS at this layer. |
| shared-gated RMS | **0.015372** | Direct analog for llama.cpp `ffn_shexp_gated`. |
| reconstructed `ffn_out` RMS | **0.042586** | Selected plus gated-shared contribution before residual add. |
| rounded post-MoE hash vs layer output hash | **same (`73c355f4e86f4bf8`)** | HipEngine's host reconstruction exactly matches the captured layer output. |

This is not yet a cross-engine comparison because current llama.cpp target tensor
tracing still exposes only the coarser `ffn_out_31` / `post_moe_31` /
`l_out_31` labels. The next source comparison is to expose and drain the
matching llama.cpp target graph labels from `build_moe_ffn()` and
`build_layer_ffn()`: `ffn_moe_logits`, `ffn_moe_weights_norm`,
`ffn_moe_swiglu`, `ffn_moe_down`, `ffn_moe_weighted`, `ffn_moe_out`,
`ffn_shexp`, `shared_expert_gate`, and `ffn_shexp_gated` for the same row.

Output_norm recompute diagnostic:
`benchmarks/results/2026-07-02-mtp-target-output-norm-recompute-diagnostic.json`.
This is a CPU-only recompute using raw `pre_output_norm` rows and GGUF
`output_norm.weight`.

| row-1 output_norm check | result | reading |
| --- | ---: | --- |
| CPU norm(hip pre-output) vs hipEngine `verify_h` | **0 MAE / 0 RMSE** | hipEngine output_norm capture is exactly explained by the raw pre-output row. |
| CPU norm(llama pre-output) vs llama.cpp `verify_h` | **0 MAE / 0 RMSE** | llama.cpp output_norm capture is exactly explained by the same formula. |
| hip pre-output vs llama pre-output | **0.01015 MAE / 0.01273 RMSE / 0.99931 cosine** | This is the real residual-stream delta before final norm. |
| CPU norm(hip pre-output) vs CPU norm(llama pre-output) | **0.07789 MAE / 0.09815 RMSE / 0.99908 cosine** | Final output_norm deterministically amplifies the residual delta; it is not an independent mismatch. |
| CPU norm(hip pre-output) vs CPU norm(BF16-rounded llama pre-output) | **0.07787 MAE / 0.09823 RMSE / 0.99908 cosine** | A final BF16 boundary cast alone does not explain the difference. |

F32 residual-boundary verifier slice:
`benchmarks/results/2026-07-02-mtp-target-f32-residual-diagnostic.json` and
`benchmarks/results/2026-07-02-mtp-target-f32-residual-attnnorm-diagnostic.json`;
the attention-norm-output dense-Q8 split is
`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-output-denseq8-diagnostic.json`
with bulk control
`benchmarks/results/2026-07-02-mtp-target-f32-residual-bulk-control-diagnostic.json`;
the linear-attention output-to-residual split is
`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-denseq8-diagnostic.json`;
the post-attention norm split is
`benchmarks/results/2026-07-02-mtp-target-f32-postnorm-split-diagnostic.json`.
The MoE selected-sum and selected-down output splits are
`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-denseq8-diagnostic.json`
and
`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-denseq8-diagnostic.json`.
This is an opt-in verifier-only experiment:
`HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1`. It converts the verifier block's token
embeddings from BF16 to FP32 once, accumulates target layer residual outputs in
FP32, optionally feeds layer-entry attention RMSNorm from the FP32 residual
stream, can materialize attention RMSNorm output into FP32 scratch for dense-Q8
projection-input consumers, keeps BF16 mirrors for the existing projection
kernels, and runs final output_norm from the FP32 residual stream. It is
default-off and not a speed claim.

| FP32 residual slice check | exact current | `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` | reading |
| --- | ---: | ---: | --- |
| Old cycle-12 replay | reaches pair-12 trace | fails earlier at cycle 2 | The FP32 residual slice changes target samples before the old pair-12 branch point, so the old trace cannot be replayed unchanged to compare `539` vs `26126`. |
| Cycle-2 sampled target tokens | `[40798, 25, 1103]` | `[40798, 1590, 1103]` | The precision slice is semantically active. |
| Cycle-2 accepted draft tokens | **2** | **1** | The changed row-1 choice changes acceptance economy on this trace. |
| Row-1 rank 1 | token `25`, logit **29.51889** | token `1590`, logit **29.54754** | The near-tie flips. |
| Row-1 rank 2 | token `1590`, logit **29.42545** | token `25`, logit **29.50755** | This is the same class of sensitivity as the pair-12 `539`/`26126` mismatch, but at a different cycle. |
| exact vs FP32-residual pre-output hidden | n/a | **0.00793 MAE / 0.01048 RMSE / 0.99944 cosine** | Residual-boundary precision alone moves the pre-output row by the same order as the hip-vs-llama layer-39 delta. |
| exact vs FP32-residual post-output hidden | n/a | **0.06757 MAE / 0.08528 RMSE / 0.99943 cosine** | Final output_norm again amplifies a small residual-stream movement into logit-visible hidden drift. |

The stronger FP32-residual + attention-norm-input slice reaches the old pair-12
branch, but does not move toward llama.cpp:

| pair-12 row-1 check | prior hipEngine serial-exact | FP32 residual + attn-norm-input | llama.cpp HIP | reading |
| --- | ---: | ---: | ---: | --- |
| sampled target tokens | `[15495, 539, 1151]` | `[15495, 539, 1151]` | `[15495, 26126]` | The extended slice still accepts `539`. |
| accepted draft tokens | **2** | **2** | **1** | Acceptance mismatch remains. |
| `539 - 26126` logit margin | **+0.11822** | **+0.14309** | **-0.00896** | Attention-norm input precision moves the wrong near-tie farther from llama.cpp. |
| row-1 top 2 | `539` then `26126` | `539` logit **26.05737**, `26126` logit **25.91428** | `26126` then `539` | The missing parity lever is downstream or inside remaining BF16 projection boundaries, not the layer-entry norm input alone. |

The attention-norm-output + dense-Q8 projection-input split is more informative:
it moves the bulk pair-12 margin toward llama.cpp, but still not far enough:

| pair-12 bulk row-1 check | FP32 residual bulk control | + FP32 attention-norm output / dense-Q8 inputs | llama.cpp HIP | reading |
| --- | ---: | ---: | ---: | --- |
| sampled target tokens | `[15495, 539, 1151]` | `[15495, 539, 1151]` | `[15495, 26126]` | The new slice still accepts `539`. |
| accepted draft tokens | **2** | **2** | **1** | Acceptance mismatch remains. |
| `539 - 26126` logit margin | **+0.31369** | **+0.18198** | **-0.00896** | Attention norm output + dense-Q8 input precision explains part of the hidden drift but not the remaining tie-break. |
| row-1 top 2 | `539` **26.15284**, `26126` **25.83915** | `539` **26.19658**, `26126` **26.01460** | `26126` then `539` | The next split must move beyond projection inputs and audit projection outputs / intermediates. |

The first projection-output boundary split keeps linear-attention `ssm_out` in
FP32 through the residual add/post-attention RMSNorm. It is measurable but too
small to explain parity:

| pair-12 bulk row-1 check | + FP32 attention-norm output / dense-Q8 inputs | + FP32 linear-attention output-to-residual | llama.cpp HIP | reading |
| --- | ---: | ---: | ---: | --- |
| sampled target tokens | `[15495, 539, 1151]` | `[15495, 539, 1151]` | `[15495, 26126]` | The new slice still accepts `539`. |
| accepted draft tokens | **2** | **2** | **1** | Acceptance mismatch remains. |
| `539 - 26126` logit margin | **+0.18198** | **+0.17663** | **-0.00896** | Avoiding the `ssm_out` BF16 round helps by only **0.00535 logits**. |
| row-1 top 2 | `539` **26.19658**, `26126` **26.01460** | `539` **26.17307**, `26126` **25.99644** | `26126` then `539` | The remaining split must target later selected/shared/FFN output contracts or accumulated multi-layer router-input drift. |

The next split carried post-attention RMSNorm itself into an FP32 scratch buffer
under `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1`, with independent consumers for
router, selected q8_1, and shared q8_1 inputs. The combined mode is **not** a
fix: it fails the old trace before pair 12. At cycle 7 the row-1 target decision
is a near tie between trace token `413` and draft token `4071`:

| cycle-7 row-1 slice | sampled row-1 token | accepted draft tokens | `413 - 4071` logit margin | reading |
| --- | ---: | ---: | ---: | --- |
| FP32 residual + attention-norm-input control | `413` | **1** | **+0.13053** | Current trace path is preserved. |
| FP32 post-norm, router only | `413` | **1** | **+0.08784** | Router F32 input moves the tie but does not flip it. |
| FP32 post-norm, selected q8_1 only | `4071` | **2** | **-0.14458** | Selected projection input quantized from FP32 is the early trace-breaking slice. |
| FP32 post-norm, router + selected q8_1 | `4071` | **2** | **-0.03290** | Router partially compensates, but combined mode still accepts the wrong row for this trace. |

Router-only reaches the old pair-12 branch, but also moves away from llama.cpp:
row 1 still samples `539`, and the `539 - 26126` margin becomes **+0.33520**
versus the control **+0.14309** and llama.cpp **-0.00896**. Therefore the
post-attention norm/router/input-q8 precision boundary is useful instrumentation,
but not the missing parity fix. The remaining likely differences are inside the
projection/output contracts themselves: llama.cpp's GGML HIP MoE path consumes
F32 graph tensors and produces F32 outputs, while hipEngine's selected/shared
paths still quantize to q8_1 and store BF16 gate/up/down/intermediate outputs.

The selected-MoE output precision split is the largest positive semantic move so
far. `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1` narrows the pair-12 row-1
`539 - 26126` margin from the full-attention-output slice **+0.27480** to
**+0.03385** by avoiding the BF16 selected-sum combine boundary. Adding
`HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1` exposes X8 Q5/Q6 selected-down
GEMV with FP32 output and combines those FP32 selected rows directly; the same
row moves again to **+0.00536** (`26.06115 - 26.05580`). That leaves only
about **0.0143 logits** to llama.cpp's **-0.00896** tie-break, but it still
samples `[15495, 539, 1151]` and accepts 2 instead of llama.cpp's
`[15495, 26126]` / accepted 1.

The follow-up `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN=1` split carries the
shared-expert down projection output in FP32 as well, preserves the BF16 mirror,
and combines FP32 selected rows with FP32 shared rows. It is semantically active
but moves the tie the wrong way: row-1 `539 - 26126` widens to **+0.03043**
(`26.12703 - 26.09660`). Shared-down output precision alone is therefore ruled
out as the missing parity fix. The next semantic split should target selected
gate/up/intermediate precision or a fuller llama.cpp-style F32 verifier FFN/MoE
graph, not this isolated shared-down boundary.

Reading: residual-boundary precision is confirmed as a real semantic lever, not
just a bookkeeping theory. However, this slice is **not** full llama.cpp F32 graph
parity. The extended version rules out one specific suspect: layer-entry
attention RMSNorm input precision. Attention norm outputs and selected/shared
expert projection inputs still pass through existing BF16 scratch buffers, and
the selected-down output split only covers the X8 selected-down leg. Shared
expert down output, shared-gated contribution, selected gate/up/intermediate, and
non-X8 selected-down paths still use BF16 contracts. The shared-down split above
rules out isolated shared-down output precision, so the next port should carry
the verifier FFN/MoE graph farther through selected gate/up/intermediate FP32
contracts or a cohesive F32 verifier FFN/MoE graph. Do not use this mode as a
retained performance row.

Superseded short-trace artifact:
`benchmarks/results/2026-07-02-mtp-proposal-trace-compare-active-draftdenseq8-draftonly-diagnostic.json`.
That 10-row diagnostic remains useful provenance because it proved the old
pair-3 proposal mismatch was closed, but its "only a trace-boundary mismatch"
reading is now superseded by the longer same-protocol trace above.

The historical diagnostics below explain how the old pair-3 mismatch was found
and closed by resident initial MTP KV seeding. They remain useful provenance,
but they are not the current active blocker on this prompt.

Follow-up A/B artifact:
`benchmarks/results/2026-07-02-mtp-draft-context-logit-ab-diagnostic.json`.
This is historical one-prompt diagnostic evidence from before the resident
initial MTP KV writer fix was retained, not a performance claim.

| A/B | result | reading |
| --- | --- | --- |
| hipEngine resident host replay with `--no-mtp-device-kv-cache` vs llama.cpp | Same first divergence as the prior proposal trace: pair 3, hipEngine drafts `[65342, 18078]`, llama.cpp drafts `[8, 1411]`. | The accepted-row KV commit writer was not the primary cause on this prompt. |
| llama.cpp with `--no-spec-draft-backend-sampling` | Same prior first divergence and same stream prefix as the regular llama.cpp trace. | The public llama.cpp backend-sampling toggle was not the explanation. |
| hipEngine resident full-logit draft path | Diverges earlier: cycle 2 drafts `[40798, 25]` while target rows are `[40798, 1590]`; 14/20 accepted in the 10-cycle diagnostic. | Removing the retained Q6 top-1 route does not move hipEngine toward llama.cpp. |
| hipEngine resident Q6 scores, no device KV | At cycle 3 depth 0, token `8` is rank 3 and **1.876 logits** behind token `65342`. | llama.cpp's divergent token is a plausible candidate but not a tie in hipEngine. |
| hipEngine resident Q6 scores, with device KV | Device KV changes logits and moves token `8` closer, but it is still rank 3 and **1.260 logits** behind token `65342`. | Historical evidence that KV history influenced the scorer before the resident initial-KV fix. |

Llama-side score trace artifact:
`benchmarks/results/2026-07-02-mtp-llamacpp-draft-score-trace-diagnostic.json`.
This adds a diagnostic patch to the local llama.cpp tree at commit `0f7d32267`
(`tools: add MTP draft score trace`) and records candidate IDs, logits,
probabilities, and logit margins in `LLAMA_MTP_TOKEN_TRACE` stage rows.

At the historical first same-prompt divergence:

| token at seq position 49 / depth 0 | llama.cpp rank / margin | hipEngine device-KV rank / margin | hipEngine no-device-KV rank / margin |
| ---: | ---: | ---: | ---: |
| `8` | **1 / 0.000** | 3 / 1.260 | 3 / 1.876 |
| `65342` | 2 / **0.100** | **1 / 0.000** | **1 / 0.000** |
| `13787` | 3 / 0.743 | 2 / 0.904 | 2 / 1.320 |
| `18078` | 4 / 1.072 | 4 / 2.255 | 4 / 2.828 |

Historical reading: llama.cpp's divergent choice was a borderline draft row, but
it was not just a tie-break difference in hipEngine. With device KV enabled,
hipEngine still had to move token `8` by about **1.26 logits** to match
llama.cpp's `ctx_dft` ordering, and no-device replay was farther away. That
ruled out accepted-row KV commit and public backend-sampling as first-order
causes and led to the resident initial-KV investigation.

Historical target that led to the resident initial-KV fix: compare hipEngine's
resident MTP seed, `pending_h`, and `ctx_dft`
row contents against llama.cpp `common_speculative_process()` plus
`common_speculative_impl_draft_mtp::draft()` at seq position 49. The next useful
instrumentation is not another Q6_K body variant; it is a hidden/KV row dump or
row-level checksum around the llama.cpp `verify_h -> pending_h -> ctx_dft`
handoff and the hipEngine `pending_hidden_row_index` / MTP device-KV rows.
hipEngine now records `cycle_start_seq_position`, `cycle_end_seq_position`,
`draft_topk_scores`, and `draft_topk_margins` behind
`--record-draft-topk-scores` for this exact comparison. Device-chain top-1 paths
still leave score arrays empty because they intentionally avoid materializing
full vocab logits.

Hidden-state parity artifact:
`benchmarks/results/2026-07-02-mtp-hidden-state-parity-diagnostic.json`.
hipEngine commit `6190fd08` adds `--record-draft-hidden-stats`; llama.cpp commit
`c0f750604` adds `draft_hidden_state_trace` to `LLAMA_MTP_TOKEN_TRACE` rows.

At the same historical seq-position-49 divergence:

| hidden row | reading | rms delta | first8 mean abs delta | interpretation |
| --- | --- | ---: | ---: | --- |
| `draft_seed_input` / `pending_h` | token `1103`, position 49, depth -1 | **0.00456** | **0.0798** | The pending hidden seed handed into the draft is close across engines. This is not the primary mismatch. |
| depth-0 `draft_next_seed` | token `1103`, position 49, before token selection | **0.01458** | **0.7310** | The first MTP/`ctx_dft` decode output hidden row has already drifted substantially before sampling flips token `8` vs `65342`. |

Historical target after hidden tracing: the first-order semantic mismatch was
inside the depth-0 draft decode, not the verifier pending-h handoff. That led to
the later draft attention/KV row and resident initial MTP KV writer diagnostics.
Keep this as provenance for the fix rather than as the active target.

RoPE-dimension follow-up artifact:
`benchmarks/results/2026-07-02-mtp-resident-rope-dim-fix-diagnostic.json`.
This is a semantic parity fix and diagnostic trace, not a retained speed row.
The GGUF model reports `qwen35moe.rope.dimension_count=64`,
`qwen35moe.rope.dimension_sections=[11, 11, 10, 0]`, and
`blk.40.attn_q_norm.weight.shape=(256,)`. hipEngine resident MTP was using
`qk_head_dim=256` as the RoPE `rotary_dim` in both `_run_one()` and
accepted-row `_write_one_kv()`, while llama.cpp `graph_mtp` passes
`n_rot=64` to `ggml_rope_multi()`. Resident MTP now derives `rotary_dim` from
the RoPE table width.

Same-prompt result after the fix:

| row | before fix | after fix | llama.cpp | reading |
| --- | ---: | ---: | ---: | --- |
| depth-0 `draft_next_seed` first8 mean abs delta vs llama | **0.731** | **0.329** | n/a | RoPE was a real semantic mismatch. |
| depth-1 `draft_next_seed` first8 mean abs delta vs llama | **1.689** | **1.316** | n/a | Later chain remains far apart because depth-0 token still differs. |
| hipEngine seq-pos-49 depth-0 drafts | `[65342, 18078]` | `[65342, 18078]` | `[8, 1411]` | Proposal parity is not closed. |
| token `8` depth-0 margin | 1.260 logits behind rank-1 `65342` | 1.391 logits behind rank-1 `65342` | rank 1; `65342` is 0.100 behind | Hidden row got closer but the lm-head ordering still differs. |
| 10-cycle same-prompt smoke | 70.51 tok/s / 14.201 ms-output | 69.95 tok/s / 14.317 ms-output | 72.12 traced tok/s | Diagnostic only; do not replace the full-suite tracker from this row. |

Historical target after the RoPE fix: compare the remaining post-RoPE MTP graph
tensors against llama.cpp labels (`mtp_Qcur_normed`, `mtp_Kcur_normed`,
`mtp_Vcur`, `mtp_attn_pregate`, `mtp_attn_residual`, `h_nextn`) or dump
llama.cpp `ctx_dft` K/V row checksums. This led to the attention/KV row and
resident initial MTP KV writer diagnostics; it is no longer the current active
proposal gap on this prompt.

Tensor-stage follow-up artifact:
`benchmarks/results/2026-07-02-mtp-llamacpp-tensor-stage-trace-diagnostic.json`.
llama.cpp commit `687c17d26` adds `LLAMA_MTP_TENSOR_TRACE=1`, shape-aware
summaries for selected Qwen3.5 MoE `graph_mtp` tensors, and post-RoPE Q/K
labels. The comparison below uses the same prompt, hipEngine post-RoPE cycle 3,
and llama.cpp stage row 9: token `1103`, seq position 49, depth 0.

| depth-0 stage | hipEngine rms | llama.cpp rms | first8 mean abs delta | reading |
| --- | ---: | ---: | ---: | --- |
| `draft_stage_token_embed` | 0.010809 | 0.010809 | 0.000000 | Exact match. |
| `draft_stage_e_norm` | 0.295241 | 0.295241 | 0.000000 | Exact match. |
| `draft_stage_h_norm` | 0.515393 | 0.516368 | 0.013300 | Close; residual pending-h difference only. |
| `draft_stage_projected` | 0.234909 | 0.233961 | 0.006362 | Close. |
| `draft_stage_attn_normed` | 0.986629 | 0.986522 | 0.019746 | Close. |
| `draft_stage_query_rope` | 1.504878 | 1.505277 | 0.007978 | RoPE/Q path is no longer the large mismatch. |
| `draft_stage_key_cur_rope` | 1.880685 | 1.879979 | 0.013736 | Current K path is close. |
| `draft_stage_value_cur` | 1.146771 | 1.144939 | 0.018544 | Current V path is close. |
| `draft_stage_attn_pregate` | **1.056160** | **1.409786** | **0.147055** | First large jump; compare draft attention history/KV rows, context length, and mask/softmax behavior. |
| `draft_stage_attn_out` | 0.108522 | 0.093688 | 0.011352 | Output projection partly compresses the attention delta. |
| `draft_stage_attn_residual` | 0.262644 | 0.245346 | 0.013410 | Residual is still close in first coordinates. |
| `draft_stage_attn_post_norm` | 1.554944 | 1.618222 | 0.111419 | Attention-history difference survives normalization. |
| `draft_stage_ffn_out` | 0.249094 | 0.148834 | 0.197250 | FFN amplifies the existing attention-state mismatch. |
| `draft_next_seed` | 2.803440 | 2.827997 | 0.328866 | Final hidden row is still too different for proposal parity. |

Interpretation: the remaining semantic miss is not current-token embedding,
normalization, projection, or RoPE Q/K/V. Those now match closely enough to move
the investigation downstream. The next required split is the attention input
history: hipEngine dense MTP device-KV rows versus llama.cpp `ctx_dft` K/V rows,
plus the effective attention length/mask at seq position 49. If those rows
match and `draft_stage_attn_pregate` still differs, then inspect the attention
kernel math/softmax scaling; otherwise fix the KV/history mirror first.

Attention-history row follow-up artifact:
`benchmarks/results/2026-07-02-mtp-attention-history-row-trace-diagnostic.json`.
hipEngine now has `--record-draft-cache-rows`, which extends
`--record-draft-stage-stats` with selected dense MTP K/V history rows.
llama.cpp commit `1ebf790cd` adds row-indexed process/catch-up tensor tracing
to the local HIP checkout. This is still one-prompt diagnostic evidence and
`performance_claim=false`.

At the same seq-position-49 / depth-0 divergence:

| K/V row comparison | hipEngine rms | llama.cpp rms | first8 mean abs delta | reading |
| --- | ---: | ---: | ---: | --- |
| row 40 key | 1.870004 | 1.869944 | 0.011814 | Close; process row and single-row draft agree in llama.cpp. |
| row 40 value | 1.124036 | 1.117155 | 0.046811 | Close using the reliable single-row draft value. Multi-row process `Vcur` remains low-confidence for non-last rows. |
| row 48 key | 1.851114 | 1.850622 | 0.019440 | Close; previous visible row matches. |
| row 48 value | 1.345635 | 1.346136 | 0.021270 | Close; previous visible row matches. |
| row 49 key | 1.880685 | 1.879979 | 0.013736 | Close; current row matches. |
| row 49 value | 1.146771 | 1.144939 | 0.018544 | Close using the reliable single-row draft value. |

This pushes the active semantic target past sampled K/V generation and sampled
cache contents. hipEngine and llama.cpp still differ at
`draft_stage_attn_pregate` (**1.056160 vs 1.409786 rms**, first8 MAE
**0.147055**) even though current and nearby history K/V rows are close.
The next split must instrument effective attention execution: visible count and
mask, GQA head mapping, score scale/softmax, and FA-on math at position 49.

Two negative A/B checks are also retained in the artifact:

| A/B | result | reading |
| --- | --- | --- |
| llama.cpp FA-on default cache vs `--cache-type-k f32 --cache-type-v f32` | No proposal or attention-row movement at the seq-position-49 divergence. | F32 KV cache storage is not the missing parity lever; FA-on casts K/V as needed inside `build_attn_mha()`. |
| llama.cpp FA-on vs `--flash-attn off` | No-FA changes the proposal/acceptance path before this FA-on divergence and drafts `[198, 262]` at the comparable later cycle instead of `[8, 1411]`. | Flash attention changes semantics enough that the active parity target remains llama.cpp FA-on unless we explicitly change the route decision. |

Attention-debug host-recompute follow-up artifact:
`benchmarks/results/2026-07-02-mtp-attention-debug-diagnostic.json`.
hipEngine now has `--record-draft-attention-debug`, which requires
`--record-draft-stage-stats` and `--mtp-device-kv-cache`. It reads back the
resident draft Q/K/V cache plus GPU attention output, recomputes dense causal
GQA attention on the host, and records per-head top score/weight rows. This is
still one-prompt diagnostic evidence and `performance_claim=false`.

At the same seq-position-49 / depth-0 divergence, hipEngine GPU attention is
internally consistent with the dense-cache formula:

| attention debug row | token / position | cache tokens | CPU-vs-GPU MAE mean | CPU-vs-GPU max abs | top-row histogram | reading |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| depth 0 | `1103` / 49 | 50 | **4.1e-7** | **9.3e-6** | row 48: 7 heads; row 49: 9 heads | hipEngine dense attention kernel and host formula match for the exact row that diverges from llama.cpp. |
| depth 1 | `65342` / 50 | 51 | **4.0e-7** | **1.025e-5** | row 49: 2 heads; row 50: 14 heads | The second draft step also matches host recompute; late rows dominate as expected. |

Example depth-0 per-head weights:

| query head | kv head | top rows | top weights | CPU-vs-GPU max abs |
| ---: | ---: | --- | --- | ---: |
| 0 | 0 | 48, 49, 46, 42, 47 | 0.5268, 0.1461, 0.0784, 0.0722, 0.0492 | 1.19e-6 |
| 1 | 0 | 49, 38, 42, 20, 48 | 0.9910, 0.0021, 0.0021, 0.0011, 0.0007 | 4.8e-7 |
| 10 | 1 | 49, 10, 47, 48, 2 | 0.9892, 0.0033, 0.0014, 0.0011, 0.0011 | 2.4e-7 |

Interpretation: the active mismatch is no longer hipEngine's resident dense
attention math, nor obviously missing early prompt rows. At depth 0, every head
has its largest attention weight on row 48 or 49, and those sampled K/V rows are
already close to llama.cpp. The next required split is llama.cpp-side FA-on
effective attention visibility: mask keep count, logical row order/layout, GQA
head mapping, scale, and per-head top score/weight distribution for the same
seq-position-49 row.

#### Latest route decision

Keep this table as the top-level route decision log for the active
`llama-compat` sprint. This lane is explicitly accuracy-traded to mirror
llama.cpp, so a full-suite row may replace the retained lane when it is
speed-positive and structurally closer to llama.cpp even if acceptance or row
economy regresses. That regression must stay visible here and in the rowhist
tracker. Smoke and all-sync rows can only name the next kernel target.

| route | status | MTP tok/s | cycle wall | acc/output | draft acceptance | target rows/output | target verifier drain | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit` + no-copy prefill-GDN capture + llama-style direct partial commit + natural24 cyclecap24 tail clamp | **active llama-replication lane** | **71.52 natural24 cyclecap24 full** | **14.005 ms/output** | **0.596** | **0.777** | **1.171** | **11.436 ms/output** | Current apples-to-apples comparison lane vs llama.cpp HIP B2. Rejected/partial bulk blocks commit the captured verifier row, matching llama.cpp's normal MTP accept lifecycle rather than serial-prefix replay. The no-copy GDN state-row kernel removes the old per-layer recurrent-state D2D copy, and `--max-output-tokens 24` clamps the last draft window like llama.cpp server. The corrected suite uses `--cycles 24` so all prompts reach 24 output tokens; replay/commit is **0.044 ms/output**, with **0** replay rows, **95** direct-commit rows, and **41** discarded rows. The retained artifact filename includes `f32head`, but this route did not enable the verifier-head flag. Fixed-cycle provenance remains **72.23 tok/s / 13.865 ms/output**. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit-vlmheadtop1` | rejected full-suite diagnostic | 66.45 natural24 cyclecap24 full | 15.072 ms/output | 0.596 | 0.777 | 1.171 | 12.501 ms/output | This is the actual current-shape route that enables `--verify-lm-head-q6-top1-dp4a` plus no-copy prefill-GDN capture. It keeps acceptance/economy identical to the active route but regresses throughput **71.52 -> 66.45 tok/s** and cycle wall **14.005 -> 15.072 ms/output**. The regression is concentrated in `target_block_lm_head_sample` (**1.068 -> 2.118 ms/output**), so do not retain it as a speed path. Artifact: `benchmarks/results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-vlmheadtop1-full.json`. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit` + copied prefill-GDN capture + llama-style direct partial commit | superseded copied-state lane | 60.56 full | 16.534 ms/output | 0.609 | 0.780 | 1.172 | 14.071 ms/output | Prior active lane before no-copy GDN capture. The all-sync attribution showed `target_block_linear_attn_prefill_gdn_state_rows` at **2.913 ms/output**; no-copy drops that leaf to **0.785 ms/output**, the corrected apples-to-apples natural24 cyclecap24 row is **71.52 tok/s**, and fixed-cycle provenance remains **72.23 tok/s**. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-serialstate` + prefill-GDN capture + reject-safe serial state-only replay | semantic-safe control | 51.85 full | 19.308 ms/output | 0.606 | 0.770 | 1.331 | 16.891 ms/output | Exact-state control. Rejected/partial bulk blocks restore and serial-replay the accepted prefix, but replay now advances exact state only and skips LM-head sampling. Lifecycle comparator stays clean; replay/commit is **2.489 ms/output**, with **38** replay rows and **46** discarded rows. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly` + prefill-GDN capture + full serial partial replay | superseded semantic-safe control | 50.96 full | 19.645 ms/output | 0.606 | 0.770 | 1.331 | 17.222 ms/output | Prior safe lane before replay LM-head removal. It remains useful as an A/B control: serial state-only replay moves **50.96 -> 51.85 tok/s**, cycle **19.645 -> 19.308 ms/output**, and replay/commit **2.775 -> 2.489 ms/output** with unchanged acceptance/economy. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly` | superseded unsafe direct-state diagnostic | 75.15 full | 13.325 ms/output | 0.621 | 0.820 | 1.136 | 10.933 ms/output | Not retained as a valid semantic lane. The lifecycle comparator proved rejected/partial direct commit diverges from serial accepted-prefix replay; this row remains only as the cost of the now-unsafe shortcut. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow` + Q8 shared dual + resident initial KV + shared-gate scalar row-dot + parallel MTP attention | superseded retained diagnostic | 74.39 clean rerun | 13.463 ms/output | 0.621 | 0.820 | 1.136 | 10.929 ms/output | Prior active lane before draft-only dense-Q8 dp4a. Parallelizing `hipengine_mtp_dense_attn_f32` removed the thread-0 dense-attention bottleneck and moved draft drain **2.684 -> 2.204 ms/output**. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow` + Q8 shared dual + resident initial KV + shared-gate scalar row-dot | superseded retained diagnostic | 71.84 shared-gate full | 13.940 ms/output | 0.621 | 0.820 | 1.136 | 10.929 ms/output | Prior active lane before parallelizing `hipengine_mtp_dense_attn_f32`. The new row kept identical acceptance and row economy while moving draft drain **2.684 -> 2.204 ms/output**. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow` + Q8 shared dual + resident initial KV | superseded retained diagnostic | 71.34 resident-init full | 14.037 ms/output | 0.621 | 0.820 | 1.136 | 10.966 ms/output | Prior active lane before reusing `qwen35_router_logits_f32_f32w(..., num_rows=1)` for the draft shared-gate scalar dot. The shared-gate row keeps identical acceptance and row economy while moving draft drain **2.747 -> 2.684 ms/output**. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-siludown` | rejected draft-profile diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | Exact chain-equivalent selected SiLU/down fusion regressed the draft parent profile before full-suite: kernel **5.973 -> 6.054 ms/cycle**, host **7.044 -> 7.206 ms/cycle**, and selected-down family **0.325 -> 0.391 ms/cycle** despite one fewer launch. Do not promote or run full-suite unless a different fused Q5 body beats the active parent profile. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8` | rejected full-suite diagnostic | 64.14 full | 15.612 ms/output | 0.573 | 0.670 | 1.282 | 12.324 ms/output | Legacy all-stage draft dense-Q8 dp4a route, including initial KV seeding stages. Draft profile was positive, but the full suite lost acceptance/row economy and verifier drain. The retained `-draftonly` route proves the useful subset is draft forward leaves only; keep this all-stage row as rejection evidence. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm` + Q8 shared dual | superseded retained diagnostic | 63.63 f32ssm full | 15.735 ms/output | 0.578 | 0.685 | 1.266 | 12.158 ms/output | Prior active lane before row-parallel draft router logits. Keep as the direct control row for the retained router-row A/B. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1` + Q8 shared dual | superseded retained diagnostic | 61.31 x8top1 full | 16.331 ms/output | 0.567 | 0.655 | 1.299 | 12.662 ms/output | Prior active lane before F32 `ssm_out`; keep as the direct control row for the retained F32 verifier A/B. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-sharedq8` | rejected full-suite diagnostic | 59.63 full | 16.793 ms/output | 0.556 | 0.625 | 1.333 | 13.038 ms/output | The isolated block profile and smoke looked positive, but the full suite regressed vs its x8top1 control (**61.31 tok/s**, **16.331 ms/output**, **0.567 acc/output**, **12.662 ms/output verifier**) and is farther from the current F32 lane. Extra shared-expert q8_1/dp4a launches and changed verifier numerics do not transfer across categories; keep as evidence only. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-vlmheadtop1` | rejected smoke diagnostic | 65.18 smoke | 15.363 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 12.473 ms/output smoke | Same-session active-route control was **71.12 tok/s**, cycle **14.086 ms/output**, verifier **11.277 ms/output**, and `target_block_lm_head_sample` **1.058 ms/output** with identical acceptance. Direct q8_1/dp4a verifier Q6 top-1 worsens sampler cost to **1.874 ms/output** and verifier drain to **12.473 ms/output**; do not run full-suite or update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all` + Q8 shared dual | superseded retained diagnostic | 61.19 shared-dual full | 16.364 ms/output | 0.567 | 0.655 | 1.299 | 12.666 ms/output | Same route before X8-packed Q6_K draft lm-head top-1. Keep as the direct control row for the X8 top-1 A/B. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all` | superseded retained diagnostic | 60.96 rowhist / 60.89 prior | 16.427 ms/output | 0.567 | 0.655 | 1.299 | 12.727 ms/output | Same route before resident Q8 shared gate/up dual GEMV default-on. Keep as the control row for the shared-dual A/B. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` | superseded retained diagnostic | 60.28 rowhist / 60.36 prior | 16.610 ms/output | 0.583 | 0.700 | 1.250 | 13.038 ms/output | Better row economy than `denseq8all`, but slower wall and less faithful to llama.cpp's dense `mul_mat_vec_q` mechanism. Keep as the control lane for future precision/economy A/Bs. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup` | rejected smoke diagnostic | 59.08 smoke | 16.948 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 14.117 ms/output smoke | Same-session retained control was **67.62 tok/s / 14.810 ms/output / 12.005 ms verifier** with identical smoke acceptance. Q4 X8 selected gate/up is slower than retained T16 dp4a for this route. Do not run full-suite or update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup` | rejected smoke diagnostic | 62.04 smoke | 16.142 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 13.328 ms/output smoke | Same-session retained control was **68.55 tok/s / 14.612 ms/output / 11.792 ms verifier** with identical smoke acceptance. Raw GGUF selected gate/up copies llama.cpp's MoE MMVQ body more directly, but is slower than retained T16 dp4a. Do not run full-suite or update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8` | rejected diagnostic | 59.42 | 16.852 ms/output | 0.559 | 0.635 | 1.322 | 13.093 ms/output | Rowtile-pair raw-Q8 sidecar improved smoke/all-sync verifier timing, but full-suite economics regressed. Do not update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall` | rejected smoke diagnostic | 68.54 smoke | 14.614 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 11.790 ms/output smoke | Same-session control smoke was faster at **68.78 tok/s / 14.561 ms/output** with the same acceptance. Do not run full-suite or update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama` | rejected smoke diagnostic | 67.92 smoke | 14.747 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 11.920 ms/output smoke | Same-session retained control was **68.88 tok/s / 14.541 ms/output / 11.722 ms verifier** with identical smoke acceptance. The all-sync leaf showed the intended Q6 stage1 movement (**1.220 -> 1.205 ms/output**), but the async parent row regressed, so do not run full-suite or update the headline gap. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-pack16` | rejected smoke diagnostic | 71.72 smoke | 13.963 ms/output smoke | 0.667 smoke | 1.000 smoke | 1.000 smoke | 11.133 ms/output smoke | Same-session denseq8all control was **71.74 tok/s / 13.961 ms/output / 11.147 ms verifier** with identical acceptance. Pack16 slightly worsens `draft_initial` (**2.479 -> 2.487 ms/output**) and draft rocprof shows `gguf_q6_k_pack16_gemv_q8_1_dp4a_top1_stage1` slower than pack8 (**3.684 vs 3.603 ms/cycle**), so do not run full-suite or update the headline gap. |

#### Three-lane speed gap

| metric | hipEngine default exact B5 | hipEngine `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit` B2 | llama.cpp HIP B2 | compat gap vs llama.cpp | active reading |
| --- | ---: | ---: | ---: | ---: | --- |
| AR tok/s | 54.79 parallel-attn full | 54.82 natural24 full | 51.38 suite / 52.13 traced / 51.98 rerun | hipEngine faster | AR is not the blocker. |
| MTP tok/s | 61.98 parallel-attn full | **71.52 natural24 cyclecap24 full** | 67.3 suite / 72.12 traced / 71.91 rerun | **-0.39 tok/s** vs rerun | The remaining request-level gap is small but real. |
| uplift over own AR | 1.1312x parallel-attn full | **1.3055x natural24 cyclecap24 full** | ~1.31x suite / 1.383x traced/rerun | slightly below llama rerun | The replication lane has the same shape but not the same target semantics/economy. |
| cycle wall / output | 16.162 ms parallel-attn full | **14.005 ms natural24 cyclecap24 full** | 14.269 ms rerun | **-0.264 ms/output** | Stage wall is slightly faster than the rerun timing target. |
| accepted / output | 0.535 | **0.596** | 0.610 rerun | -0.014 | Remaining compatibility delta is acceptance/economy. |
| draft acceptance | 0.723 | **0.777** | 0.805 | -0.028 | Slightly lower than llama, but not a wall gap. |
| target passes / output | 0.567 | **0.403** | 0.390 | +0.013 | Pass economy is close; natural24 includes one AR tail cycle. |
| target rows / output | 1.163 | **1.171** | 1.148 | +0.023 | Row economy is close; replay rows are now zero. |

#### Three-lane stage gap ledger

| bucket | hipEngine default exact B5 | hipEngine llama-compat B2 | llama.cpp HIP B2 | compat gap vs llama.cpp | current interpretation / next target |
| --- | ---: | ---: | ---: | ---: | --- |
| `cycle_wall_ms_per_output` | 16.162 | **14.005** | 14.269 | **-0.264** | Stage wall remains faster after no-copy GDN capture and cyclecap24 tail-clamp instrumentation. The actual verifier-head route regresses to **15.072 ms/output**. |
| `draft_initial` | 1.899 | **2.101** | 2.141 | **-0.040** | Draft parent is effectively at parity. |
| `draft_mtp_layer_forward` | 0.141 | **0.124** | 0.250 decode subtotal | compat faster | Draft transformer work is not the problem. |
| `draft_topk_readback` / llama `llama_draft_sample_topk` | 1.129 | **1.940** | 1.888 | **+0.052** | Small residual; not a wall gap. |
| `target_serial_verify_step` | 6.508 | **0.151** | 0.000 | +0.151 | Natural24 tail cleanup only: one cycle reaches the cap with zero drafts. The fixed-cycle compat row stays at zero serial verify. |
| `target_block_verify_total` | 7.728 | **11.436** | 12.120 | **-0.684** | No longer a llama.cpp speed gap. |
| `target_block_layer_total` | 6.874 | **10.065** | n/a | n/a | HipEngine verifier cost center; compare through verifier total. |
| `target_block_linear_attn_layers` | 5.055 | **7.482** | n/a | n/a | Largest hipEngine verifier layer family. |
| `target_block_full_attn_layers` | 1.819 | **2.584** | n/a | n/a | Secondary verifier layer family. |
| `target_block_lm_head_sample` | 0.579 | **1.068** | n/a | n/a | Visible verifier-side target after layer GEMVs. |
| `target_block_replay_or_commit` | 0.019 | **0.044** | 0.004 | **+0.040** | Small residual for the replication lane; not P0. |
| `mtp_device_kv_commit` | n/a | **0.295** | n/a | n/a | HipEngine directcommit/KV bookkeeping bucket; compare only through total wall. |
| `target_block_setup` + commit/accounting | 0.125 | **0.045** | 0.188 comparable visible overhead | compat faster | No longer a gap after direct partial commit removes serial replay. |
| llama `mtp_context_replay_append` | n/a | 0.008 | **11.348** | n/a | In llama, verifier GPU drain lands here; do not compare raw `target_block_forward`. |

Rule for this table: if a bucket has no true llama.cpp analog, leave the llama
cell as `n/a` and compare it only through `target_block_verify_total` or
`cycle_wall_ms_per_output`. Raw `target_block_forward` is async-misaligned across
engines and is not a valid parity target by itself.

#### Three-lane full-suite bucket inventory

This is the current full-suite cycle-stage inventory for the two hipEngine lanes,
plus the llama.cpp HIP B2 deep-trace analog where the bucket semantics line up.
It lists every high-level bucket emitted by the current hipEngine full-suite
artifacts and the llama.cpp native buckets that explain the comparable totals.
Rows with `n/a` in the llama column are still real hipEngine work, but they must
roll up through `draft_initial`, `target_block_verify_total`, or total cycle wall
before we claim parity movement. Fine-grained kernel splits are attribution-only
and live in the next section.

| bucket | hipEngine default exact B5 | hipEngine llama-compat B2 | llama.cpp HIP B2 analog | compat gap vs llama.cpp | action |
| --- | ---: | ---: | ---: | ---: | --- |
| `cycle_wall_ms_per_output` | 16.162 | **14.005** | 14.269 | **-0.264** | Stage wall is closed after no-copy GDN capture and llama-style cyclecap24 tail clamp; request-level tok/s remains **71.52 vs 71.91**. The measured verifier-head route regresses to **15.072 ms/output**. |
| `accept_policy_and_seed` | 0.002 | 0.002 | 0.002 | -0.001 | Already noise-level. |
| `draft_initial` | 1.899 | **2.101** | 2.141 | **-0.040** | Draft parent is still effectively at parity. |
| `draft_prepare_inputs` | 0.086 | 0.025 | n/a | n/a | hipEngine-only prep, already small. |
| `draft_seed_upload` | 0.102 | 0.047 | n/a | n/a | Not a current target. |
| `draft_mtp_layer_forward` | 0.141 | 0.123 | 0.250 llama draft decode subtotal | compat faster | Draft transformer body is not the gap. |
| `draft_diagnostic_topk` | 0.000 | n/a | n/a | n/a | Default diagnostic-only row. |
| `draft_device_chain_ensure_embed_table` | n/a | 0.000 | n/a | n/a | No target. |
| `draft_device_topk_gather` | n/a | 0.000 | n/a | n/a | No target. |
| `draft_device_chain_drain` | n/a | **1.933** | n/a | n/a | hipEngine compat draft drain bucket; compare through `draft_initial`. |
| `draft_topk_d2h` | n/a | 0.006 | n/a | n/a | D2H is too small to explain the gap. |
| `draft_topk_readback` / llama `llama_draft_sample_topk` | 1.129 | **1.940** | 1.888 | **+0.052** | Small residual; names are not perfectly isomorphic. |
| llama `llama_draft_decode_initial` | n/a | n/a | 0.118 | n/a | Llama native row; included in its 0.250 ms draft decode subtotal. |
| llama `llama_draft_decode_next` | n/a | n/a | 0.132 | n/a | Llama native row; included in its 0.250 ms draft decode subtotal. |
| llama `llama_draft_prepare_initial_batch` | n/a | n/a | 0.001 | n/a | Llama-only setup, not a gap. |
| llama `llama_draft_prepare_next_batch` | n/a | n/a | 0.000 | n/a | Llama-only setup, not a gap. |
| llama `llama_draft_finalize` | n/a | n/a | 0.000 | n/a | Llama-only setup, not a gap. |
| `target_serial_verify_step` | 6.508 | **0.151** | 0.000 | +0.151 | Natural24 tail cleanup only; fixed-cycle compat stays at zero serial verify. |
| `target_block_verify_total` | 7.728 | **11.436** | 12.120 | **-0.684** | No longer a llama.cpp speed gap. |
| `target_block_setup` | 0.101 | 0.045 | n/a | n/a | Not a gap. |
| `target_block_embedding` | 0.013 | 0.025 | n/a | n/a | Not a current target. |
| `target_block_forward` | 7.706 | 11.387 | n/a | n/a | Async-misaligned; compare through verifier total. |
| `target_block_layer_total` | 6.874 | **10.065** | n/a | n/a | hipEngine verifier cost center. |
| `target_block_linear_attn_layers` | 5.055 | **7.482** | n/a | n/a | Biggest hipEngine verifier family. |
| `target_block_full_attn_layers` | 1.819 | **2.584** | n/a | n/a | Secondary verifier family. |
| `target_block_output_norm_hidden` | 0.123 | 0.152 | n/a | n/a | Below top targets. |
| `target_block_lm_head_sample` | 0.579 | **1.068** | n/a | n/a | Verifier-side lm-head/sample target after layer GEMVs. |
| `target_block_hidden_readback` | 0.005 | 0.008 | n/a | n/a | Not a target. |
| `target_block_acceptance_accounting` | 0.001 | 0.002 | 0.188 | -0.186 | Not a gap; llama charges more visible accounting here. |
| `target_block_replay_or_commit` | 0.019 | **0.044** | 0.004 | **+0.040** | Small residual for the replication lane. |
| `target_block_cursor_update` | 0.001 | 0.002 | n/a | n/a | Not a target. |
| `target_block_snapshot` | n/a | n/a | 0.001 | n/a | Directcommit no longer emits a nonzero snapshot bucket. |
| `mtp_device_kv_commit` | n/a | **0.294** | n/a | n/a | HipEngine directcommit/KV bookkeeping bucket; compare only through total wall. |
| `mtp_context_replay_append` / llama verifier drain | n/a | 0.008 | **11.369** | n/a | Same label is not semantically aligned: in llama this is verifier GPU drain; in hipEngine directcommit it is only small append bookkeeping. |
| llama `llama_process_build_draft_batch` | n/a | n/a | 11.252 | n/a | Dominant llama verifier/process sub-row inside `mtp_context_replay_append`. |
| llama `llama_process_decode_ctx_dft` | n/a | n/a | 0.115 | n/a | Llama draft-context process sub-row. |
| llama `llama_process_copy_verify_h` | n/a | n/a | 0.001 | n/a | Llama-only hidden copy, not a gap. |
| llama `llama_process_scan_batch` | n/a | n/a | 0.000 | n/a | Llama-only scan, not a gap. |
| llama `llama_accept_update_pending_h` | n/a | n/a | 0.000 | n/a | Llama-only accept update, not a gap. |
| setup/commit/accounting rollup | 0.125 | **0.045** | 0.188 | compat faster | Replay/control overhead is no longer a gap. |

#### Active gap budget

This is the sprint target list. Keep the `llama-compat` row structurally matched
to llama.cpp B2, then spend the gap down against the rerun llama.cpp HIP row.
All-sync sub-buckets are attribution aids only; validate wins with the async/full
suite row before moving the headline numbers.

| target area | current hipEngine llama-compat B2 | llama.cpp HIP B2 target | budget to close | current named work |
| --- | ---: | ---: | ---: | --- |
| Total cycle wall | **14.005 ms/output** | 14.269 ms/output | **-0.264 ms/output** | Stage wall is closed after no-copy GDN capture and llama-style natural24 cyclecap24 tail clamp. |
| Draft drain | **2.101 ms/output** | 2.141 ms/output | **-0.040 ms/output** | Draft parent remains at parity; D2H remains tiny. |
| Target verifier drain | **11.436 ms/output** | 12.120 ms/output | **-0.684 ms/output** | No longer a llama.cpp speed gap. |
| Replay / commit | **0.044 ms/output** | 0.004 ms/output | **+0.040 ms/output** | Small residual; keep as a regression guard. |
| Target rows / output | **1.171** | 1.148 | +0.023 rows/output | Compat pays 41 discarded rows over 240 outputs and no replay rows. |
| Non-gaps | AR faster; fixed-cycle serial verify removed; natural24 cyclecap24 has only two zero-draft cap-tail cycles; draft parent near parity; verifier drain faster than llama.cpp | n/a | n/a | Spend time on the remaining request-level tok/s, target semantic parity, row economy, and exact semantics. |

Post-directcommit correction: the old `75.15 tok/s / 13.325 ms/output` row was an
unsafe direct-state shortcut, but the active replication lane now intentionally
matches llama.cpp's captured-row partial commit rather than serial-prefix replay.
Using the current directcommit full-suite `llama-compat` row and the traced
llama.cpp HIP B2 row:

| decomposition metric | hipEngine `llama-compat` B2 | llama.cpp HIP B2 | reading |
| --- | ---: | ---: | --- |
| visible outputs / cycle | **2.474** | **2.563** | HipEngine gets **0.089 fewer** visible outputs/cycle after natural24 cyclecap24 tail clamping. |
| cycle wall / output | **14.005 ms** | **14.269 ms** | HipEngine is **0.264 ms/output faster** on the retained rerun stage target. |
| inferred wall / cycle | **34.654 ms** | **36.575 ms** | HipEngine spends **1.921 ms/cycle less**, which offsets most of the weaker output/cycle amortization. |
| amortization share | n/a | n/a | If hipEngine matched llama's output/cycle at the current hipEngine cycle cost, it would save about **0.49 ms/output**. |
| residual cycle-cost share | n/a | n/a | The previous cycle-cost residual was the copied recurrent-state GDN capture; no-copy capture removes it and leaves row economy as the exposed delta. |
| Q6_K lm-head dispatch | **1.786 ms/call** (`gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1`) | **1.781 ms/call** (`mul_mat_vec_q<GGML_TYPE_Q6_K,ncols=1>`) | Per-call Q6 body is effectively at parity (**+0.005 ms/call** in `benchmarks/results/2026-07-02-mtp-draft-kernel-compare-draftdenseq8-draftonly.json`). |

The active interpretation changes accordingly: the semantic-safe control still
uses serial state-only replay when exact serial-prefix state is required, but the
llama-replication lane now deliberately follows llama.cpp's captured-row direct
commit behavior. For parity performance work, stop treating accepted-prefix
replay as the P0 gap. The next useful implementation work is target verifier
forward time: compare the row-bulk verifier layer kernels and verifier
lm-head/sample path against llama.cpp's HIP graph and MMVQ/MMVQ-MoE kernels.

Proposal trace instrumentation is now available for that next comparison.
llama.cpp commit `ef8050cec` adds `LLAMA_MTP_TOKEN_TRACE=1` to the server MTP
stage JSONL rows, recording `draft_token_ids`, `sampled_token_ids`,
`accepted_token_ids`, `output_token_ids`, `bonus_token_id`, and
`rejected_draft_token_id`. hipEngine wrappers expose it as
`scripts/llamacpp_mtp_bench.py --stage-token-trace` and
`scripts/llamacpp_mtp_rocprof.py --token-trace`; `_summarize_stage_timings`
preserves the first rows under `proposal_trace_sample`. A short non-retained
token-repeat smoke confirmed the emitted fields, so the next retained
diagnostic can compare llama.cpp proposal/acceptance rows against hipEngine's
existing per-cycle `draft_tokens`, `comparison_target_tokens`, and
`output_tokens`.
Use `scripts/mtp_proposal_trace_compare.py --hipengine <gguf_mtp_bench.json>
--llamacpp <stage.jsonl-or-wrapper.json>` for that comparison. It normalizes
both engines to draft IDs, accepted prefix, emitted output IDs, bonus token, and
first rejected draft token, then reports exact-row match rates and the first
proposal/acceptance divergence.

#### Row-economy histogram tracker

`cycle_histograms` are now a required output bucket for hipEngine MTP metrics,
category aggregation, suite rollup, and the llama.cpp stage-timing summary. The
hipEngine columns below are full-suite timing reruns of the current default
exact B5 and `llama-compat` B2 lanes. The llama.cpp values are measured by summarizing
`benchmarks/results/2026-07-02-llamacpp-mtp-stage-timing-b2-natural24-rerun.jsonl`
with the current harness, excluding warmup task `0`.

| row-economy bucket | hipEngine default exact B5 | hipEngine `llama-compat` B2 | llama.cpp HIP B2 | compat reading |
| --- | --- | --- | --- | --- |
| histogram source | `...default-parallelattn-full.json` | `...directcommit-nocopy-natural24-cyclecap24-full.json` | `...natural24-rerun.jsonl`, measured rows | Active compat uses llama-style captured-row direct commit for partial/reject blocks and the same natural24 tail clamp as llama.cpp server. |
| cycles / visible outputs | 100 / 215 | 97 / 240 | 87 / 223 | Compat is faster per cycle but emits **0.089 fewer** visible outputs/cycle than llama's rerun row. |
| `generated_draft_tokens` | `{0: 25, 1: 39, 2: 10, 3: 11, 4: 8, 5: 7}` | `{0: 2, 1: 6, 2: 89}` | `{1: 5, 2: 82}` | Natural24 cyclecap24 exposes two hipEngine cap-tail cycles with zero drafts and six one-draft tail cycles. |
| `accepted_draft_tokens` | `{0: 41, 1: 36, 2: 7, 3: 5, 4: 5, 5: 6}` | `{0: 19, 1: 13, 2: 65}` | `{0: 11, 1: 16, 2: 60}` | HipEngine has fewer full accepts and more zero-accept cycles than llama.cpp. |
| `visible_output_tokens` | `{1: 41, 2: 36, 3: 7, 4: 5, 5: 5, 6: 6}` | `{1: 19, 2: 13, 3: 65}` | `{1: 11, 2: 16, 3: 60}` | Output distribution is close, but llama keeps a better full-accept/zero-accept mix. |
| `target_verify_rows_evaluated` | `{1: 34, 2: 30, 3: 10, 4: 11, 5: 8, 6: 7}` | `{1: 2, 2: 6, 3: 89}` | `{2: 5, 3: 82}` | Compat evaluates one B2 block per cycle except the natural24 tail cycles. |
| `target_verify_replay_rows` | n/a | `{0: 97}` | n/a | Replay rows are gone in the active replication lane. |
| `target_verify_direct_commit_rows` | n/a | `{0: 2, 1: 95}` | n/a | Every block cycle direct-commits one captured verifier row; zero-draft tail cycles do not. |
| `target_verify_discarded_rows` | `{0: 83, 1: 5, 2: 7, 3: 4, 4: 1}` | `{0: 72, 1: 9, 2: 16}` | `{0: 63, 1: 15, 2: 9}` | Compat discards 41 rows/240 outputs; llama discards 33 rows/223 outputs. |
| `target_verify_rows_minus_visible_output` | `{0: 83, 1: 5, 2: 7, 3: 4, 4: 1}` | `{0: 72, 1: 9, 2: 16}` | `{0: 63, 1: 15, 2: 9}` | The residual row-economy gap is discarded block rows plus the natural24 tail, not serial replay. |

Request-level economy reconciliation:
`benchmarks/results/2026-07-03-mtp-economy-denominator-reconcile.json` compares
the retained hipEngine full-suite row, the llama.cpp request rows, and the
llama.cpp stage rows in one artifact. It confirms that the stage histogram above
is useful for row-cost attribution, but its accepted/output denominator is not
the llama.cpp request denominator: request-level hipEngine is **143/240 =
0.5958**, request-level llama.cpp is **136/240 = 0.5667**, and the stage-measured
llama.cpp row is **136/223 = 0.6099**. The remaining request tok/s gap
(**-0.3909 tok/s**) is therefore not explained by a full-request
accepted/output deficit. The prompt row with a real local acceptance and speed
loss is `mixed_ja_en_translate`: hipEngine accepts **13** draft tokens vs
llama.cpp **14** and is **10.21 tok/s** slower on that prompt.

Live target-score instrumentation is now available for the next semantic split:
`scripts/gguf_mtp_bench.py --record-target-topk-scores
--target-score-candidate-tokens ...` asks the block verifier to copy the
already-materialized full target lm-head logits back to host and emits compact
per-row `target_lm_head_score_rows` in the cycle JSON. The row records include
the verifier input token, sampled target token, target top-k scores, and scores
for draft/target plus explicit extra candidates such as the known near-tie
tokens `668`, `8940`, `26126`, and `539`. This is diagnostic-only and not a
timing route because it adds a full-logit D2H copy per target block. Smoke
artifact:
`benchmarks/results/2026-07-03-mtp-target-score-capture-smoke.json` on the
active `llama-compat` direct-commit shape produced three target verifier rows
with candidate scores populated, so the next `mixed_ja_en_translate` run can
compare hipEngine's live margins directly against llama.cpp's token trace
without forced-target replay ambiguity.

Focused live comparison:
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-scores-live.json`
and reducer output
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-target-score-compare-live-vs-llamacpp.json`
select `mixed_ja_en_translate` task 9 / cycle 3 / row 2, after both engines
accept draft `[11, 567]`. The live hipEngine verifier samples `8940`; llama.cpp
samples `668`. The target top-8 token set is identical across engines
(`668`, `8940`, `3019`, `1318`, `1144`, `1220`, `28663`, `60445`), but the
near-tie order is wrong: hipEngine has `8940 - 668 = +0.51934` while llama.cpp
has `8940 - 668 = -0.00961`, for a **+0.52895 logit** hip-minus-llama margin
gap. This rules out a target argmax/vocab-label bug for this row; the mismatch
is target hidden/score drift amplified at the final lm-head.

Live hidden-seed follow-up:
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-hidden-scores-live.json`
adds `target_hidden_seed_rows` beside the live target-score rows, and
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-target-hidden-score-compare-live-vs-llamacpp.json`
preserves the selected hipEngine row in the reducer output. This is still
diagnostic-only, not a timing claim. For task 9 / cycle 3 / row 2, hipEngine's
scored hidden seed for input token `567` has `sha256_16 =
b5ad63cdd8c205e6`, mean **-0.02255636**, RMS **2.58293229**, position **75**,
and first8 `[0.47027054, 2.82138753, -0.15778151, 0.49233231, 5.39549446,
-3.17500472, 0.38539246, 3.24579096]`. The matching llama.cpp tensor trace's
`verify_h` row has mean **-0.02630296**, RMS **2.58591292**, position **75**,
and first8 `[0.45559129, 2.80407262, 0.01650394, 0.59033644, 5.51924992,
-3.06079221, 0.56728262, 3.08224630]`. The sampled-token and margin result is
unchanged: hipEngine still samples `8940`, llama.cpp samples `668`, and the
`8940 - 668` gap is **+0.52895 logits** hip-minus-llama. The next split should
be a raw hidden-vector comparator for this exact live row, then a final
hidden-to-logit contract check against llama.cpp's output norm / lm-head path.

The same live diagnostic with the current F32 selected-FFN stack
(`HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1`,
`HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1`,
`HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1`,
`HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1`,
`HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1`,
`HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1`, and
`HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1`) is
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-scores-f32selectedintermediate-live.json`
with reducer output
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-target-score-compare-f32selectedintermediate-vs-llamacpp.json`.
It is directionally closer but still wrong: hipEngine still samples `8940`, and
`8940 - 668` only moves **+0.51934 -> +0.48450**. Acceptance/economy on the
focused prompt remains unchanged (**13/20** accepted drafts, **24** output
tokens). Therefore F32 selected-intermediate is a confirmed parity lever but not
the live direct-commit fix; keep the active target on reducing the final target
hidden/score drift rather than copying a different argmax or MoE top-k rule.

#### Llama-compat target map

Use this table as the short work queue when comparing a new `llama-compat` run
against llama.cpp. The left side is the hipEngine bucket that must move; the
right side is the llama.cpp bucket or source area to inspect when the structures
match but the timings do not.

| priority | gap area | hipEngine buckets to update | llama.cpp comparison point | current delta | next fix class |
| ---: | --- | --- | --- | ---: | --- |
| S | Target verifier semantic parity | proposal trace `target_tokens`, `accepted_draft_tokens`, forced-prefix target score/top-k rows, forced-prefix pending seed and `verify_h` rows, raw row-1 hidden/lm-head cross-score, pre-output/per-layer hidden checkpoints, F32 verifier-boundary probes, and `scripts/gguf_mtp_compare_forced_target_paths.py` capture-path vs non-capturing block-verifier reducer outputs | llama.cpp `sampled_token_ids`/accept accounting in `tools/server/server-context.cpp`, local `target_sample_trace`, local `verify_h`/raw-value trace in `common/speculative.cpp`, target hidden source around `llama_decode()`, and GGML target graph tensor dtype boundaries in `src/models/qwen35moe.cpp` | Diagnostic pair 12: both draft `[15495, 539]`; hipEngine accepts 2 and emits `[15495, 539, 1151]`, llama.cpp accepts 1 and emits `[15495, 26126]`. The F32 selected-SiLU intermediate slice is the first forced-prefix side match: row-1 `539 - 26126` moves to **-0.00303** vs llama.cpp about **-0.00896**. Live active prefill-GDN/no-copy capture still accepts `539`; it is layer-0 exact vs noncapture, first crosses the 1e-3 hidden-output threshold at layer 23 (**0.001022 MAE**), and moves the margin **+0.298283** to **+0.29526**. Cross-engine layer outputs at 22-24 are about **0.0019 MAE** for both hip paths vs llama.cpp, with layer 23 largest. Layer22-24 llama.cpp sub-boundary tracing is now done and shows capture/noncapture have nearly the same raw residual/FFN/post-MoE profile vs llama.cpp. | The selected SwigLU/intermediate BF16 boundary is a confirmed llama.cpp parity contract, but it is not sufficient in the live direct-state path. Active no-copy row-state capture is layer-0 exact, and layer22-24 sub-boundaries do not expose a capture-only expert-selection or selected-FFN bug. A useful patch should now move the active prefill-GDN reducer margin toward negative and reduce the capture-vs-noncapture hidden-drift ladder without transactional serial replay; the next source comparison is hidden/KV history ordering and final target-score accumulation, not raw selected-MoE labels that cancel inside both hip paths. |
| 1 | Total MTP wall | `cycle_wall_ms_per_output`, retained MTP tok/s | rerun B2 cycle wall plus suite tok/s | **-0.264 ms/output stage**, **-0.39 tok/s request** | Stage wall is closed; request-level throughput still trails. |
| 2 | Target verifier drain | `target_block_verify_total`, `target_block_linear_attn_layers`, `target_block_full_attn_layers`, `target_block_lm_head_sample`, verifier rocprof kernel-family rows | llama verifier drain inside `mtp_context_replay_append` / `mul_mat_vec_q` / `mul_mat_vec_q_moe` | **-0.684 ms/output** | No longer a speed gap after no-copy GDN capture; keep as a regression guard. The actual verifier-head route worsens this bucket to **12.501 ms/output**. |
| 3 | Proposal / row economy | `target_rows_per_output`, `target_passes_per_output`, accepted/output, draft acceptance, visible outputs/cycle, proposal trace stream/chunking, draft top-k scores/margins, draft hidden summaries | llama B2 no-probe draft proposal, `common_speculative_process()`, and accept accounting | Request tok/s **-0.3909**, full-request accepted/output **+0.029**, draft acceptance **-0.0276**, stage target rows/output **+0.0229** | Do not chase a broad accepted/output deficit; it is a denominator artifact. Focus on lower draft acceptance / discarded-row mix and the `mixed_ja_en_translate` semantic mismatch where hipEngine loses one accepted draft and **10.21 tok/s**. |
| 4 | Draft operation drain | `draft_initial`, `draft_device_chain_drain`, `draft_topk_readback`, GPU-event `draft_gpu_run_lm_head`, `draft_gpu_decode_initial`, `draft_gpu_decode_next`, all-sync `draft_run_lm_head_q6_top1_dp4a_x8_stage1`, draft rocprof `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1`, fine-sync draft body leaves | `llama_draft_sample_topk` plus llama draft decode/lm-head path | **-0.040 ms/output** | Parent draft drain is closed. |
| 5 | Replay / commit | `target_block_replay_or_commit`, `target_verify_replay_rows`, `target_verify_direct_commit_rows`, lifecycle comparator state hashes | llama partial accept/checkpoint/update path in `common_speculative_process()` and `common_speculative_accept()` | **+0.040 ms/output replay/commit** | Small residual for the llama-replication lane. The semantic-safe serial-state control remains separate. |
| 6 | Non-targets | AR tok/s, `target_serial_verify_step`, draft parent drain | n/a | n/a | Keep as regression guards, not active gap work. |

#### 2026-07-03 full-attention capture split

Retained compact artifacts:

- `benchmarks/results/2026-07-03-mtp-capture-vs-noncapture-f32selectedintermediate-fullattn31-35-39-boundary-compare.json`
- `benchmarks/results/2026-07-03-mtp-noncapture-vs-llamacpp-fullattn31-35-39-subboundary-compare.json`
- `benchmarks/results/2026-07-03-mtp-capture-prefillgdn-vs-llamacpp-fullattn31-35-39-subboundary-compare.json`

Forced pair: hipEngine cycle 12 row 1 maps to llama.cpp task 9 / cycle 18,
draft `[15495, 539]`.  The measured throughput gap is no longer the live
bottleneck for this lane: noncapture and llama.cpp agree on the near-tie target
decision, while active prefill-GDN/no-copy capture flips it.

| path | sampled token | `539 - 26126` margin | interpretation |
| --- | ---: | ---: | --- |
| hipEngine noncapture F32 selected-intermediate | `26126` | **-0.003027** | Same side as llama.cpp; rejects draft token `539`. |
| hipEngine active prefill-GDN/no-copy capture | `539` | **+0.295256** | Wrong side; accepts draft token `539`. |
| llama.cpp HIP | `26126` | **-0.008963** | Reference side for this pair. |

Full-attention scored-boundary ladder, mean absolute delta:

| layer | cap vs noncap hidden in | cap vs noncap attn out | cap vs noncap post-attn norm | cap vs noncap attn residual | cap vs noncap FFN out | cap vs noncap layer out | noncap vs llama layer out | cap vs llama layer out |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 0.001516 | 0.000421 | 0.018186 | 0.002690 | 0.002008 | 0.001676 | 0.005588 | 0.005604 |
| 35 | 0.002249 | 0.000891 | 0.020058 | 0.006337 | 0.000806 | 0.006418 | 0.006520 | 0.008735 |
| 39 | 0.007230 | 0.001413 | 0.030264 | 0.007032 | 0.002441 | 0.007415 | 0.009060 | 0.010481 |

Layer 35 is the first large amplification point after the earlier layer-23
1e-3 threshold crossing: the active capture path enters layer 35 at
`0.002249` hidden MAE vs noncapture and exits at `0.006418`.  By layer 39 the
input drift is already `0.007230` and the output drift is `0.007415`.
Selected-MoE internals can look large, especially layer-39 selected SwigLU
(`0.129213` MAE), but the weighted selected sum is only `0.000869` and the
final layer output remains in the `0.0074` band.  This matches the layer22-24
negative result: raw residual/FFN labels can be large or noisy, but the
remaining user-visible mismatch is the active captured-row hidden/state ladder
and final target-score tie, not a standalone selected-expert label.

Current status: yes, the project is still improving, but the improvement target
has moved.  The `llama-compat` stage wall is effectively closed; the active
work is semantic parity for captured verifier row state.  A useful patch should
move the active capture margin back toward the noncapture/llama side without
returning to transactional serial replay.  The next splits should target
full-attention captured-row materialization and hidden/KV history ordering
around layers 35/39, plus final LM-head score accumulation for the near-tie
tokens `539` and `26126`.

#### Llama.cpp source anchors for the live gap

This table is not a new measurement. It pins each live `llama-compat` budget row
to the llama.cpp HIP implementation area that should be inspected when the
structure matches but the timing does not. Update it when the llama.cpp commit,
route shape, or stage labels change.

| live budget row | llama.cpp HIP source anchor | hipEngine rows that must move |
| --- | --- | --- |
| Total MTP wall | `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`: `common_speculative_impl_draft_mtp` (`:841`), stage accounting (`common_speculative_mtp_stage_add`, `:56`), and the B2 process/draft/accept loop (`:1009`-`:1230`). | `cycle_wall_ms_per_output`, retained MTP tok/s, and the standing three-lane snapshot. |
| Draft drain | `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`: `llama_draft_sample_topk`, `llama_draft_decode_initial`, `llama_draft_decode_next`, plus `common_speculative_impl_draft_mtp::process()` for shifted target-batch mirroring; `/home/lhl/llama.cpp/llama.cpp-hip/tools/server/server-context.cpp`: `server_slot::update_batch()`, `common_speculative_process()`, `common_context_seq_rm()` after accept; `/home/lhl/llama.cpp/llama.cpp-hip/common/sampling.cpp`; `/home/lhl/llama.cpp/llama.cpp-hip/src/llama-sampler.cpp`; `/home/lhl/llama.cpp/llama.cpp-hip/ggml/src/ggml-cuda/mmvq.cu`: `mul_mat_vec_q` and Q6_K dispatch. | `draft_initial`, `draft_device_chain_drain`, `draft_topk_readback`, GPU-event `draft_gpu_run_lm_head` / `draft_gpu_decode_initial` / `draft_gpu_decode_next`, proposal trace stream/chunking, draft all-sync Q6 top-1 stages, and draft-chain rocprof rows. |
| Target verifier drain | `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`: `llama_process_build_draft_batch` (`:1047`) and `llama_process_decode_ctx_dft` (`:1051`); `/home/lhl/llama.cpp/llama.cpp-hip/ggml/src/ggml-cuda/mmvq.cu`: `mul_mat_vec_q` (`:477`), `mul_mat_vec_q_moe` (`:683`), and dispatch switch (`:976`-`:1216`). | `target_block_verify_total`, `target_block_layer_total`, `target_block_linear_attn_layers`, `target_block_full_attn_layers`, `target_block_lm_head_sample`, and verifier rocprof kernel-family rows. |
| Verify row economy | `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`: batch scan/build, `verify_h` capture, `pending_h` update, and accept accounting (`llama_process_scan_batch`, `llama_process_build_draft_batch`, `llama_accept_update_pending_h`); `/home/lhl/llama.cpp/llama.cpp-hip/tools/server/server-context.cpp`: accepted-token insertion and post-accept context trim. | `target_rows_per_output`, `target_passes_per_output`, accepted/output, draft acceptance, proposal trace chunking, and the row-economy histograms. |

Latest verifier/draft split attribution after q6top1dp4a plus q6-only X8,
raw-Q8 dp4a all-sidecar, X8 draft lm-head top-1, and F32 `ssm_out` raw-Q8 dp4a uses
two attribution-only smokes plus ROCTX/kernel traces. Verifier leaf rows come from
`benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync-smoke.json`;
the active dense-Q8 block rocprof refresh comes from
`benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-f32ssm.json`;
draft lm-head leaf rows come from
`benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-top1split128-allsync-smoke.json`.
Draft kernel-family rows come from
`benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`
for the active X8 top-1 route; the prior pack8/shared-dual control is
`benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-dual.json`.
The current fine-grained sync-stage draft split comes from
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-routerrow-sharedgate-fine-sync.json`,
which uses bulk target prefill hidden-row capture plus
`resident_write_kv_rows` for initial prompt MTP KV seeding and the row-parallel
F32 router kernel for the draft shared-gate scalar dot. The prior resident-init
fine-sync artifact is
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-residentinit-fine-sync.json`.
The prior router-row A/B artifacts are
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-fine-sync.json`
and control artifact
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-control-fine-sync.json`.
The rejected draft dense-Q8 dp4a profile is
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-draftdenseq8-fine-sync.json`.
The rejected selected SiLU/down fused Q5 profile is
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-fine-sync.json`
with same-session control
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-control-fine-sync.json`.
It supersedes the earlier `...fine-sync-ffn.json` attribution artifact for the
active route because router-row is now retained in the llama-compat lane.
Do not use the all-sync or rocprof rows for headline tok/s; use them only to
choose which leaf bucket should move the full-suite `draft_initial` or
`target_block_verify_total` row next.

#### Current draft-chain rocprof attribution

This table profiles the retained `llama-compat` draft shape directly:
`scripts/gguf_mtp_draft_rocprof.py --steps 4 --warmup 2 --q6-top1-dp4a
--q6-top1-stage1-shape x8 --selected-down-x8-repack q6
--router-row-parallel --record-stage-timings --sync-stage-timings
--dense-q8-dp4a --dense-q8-dp4a-stages draft
--require-cached --skip-warmbuild`.
The child runs under `rocprofv3 --kernel-trace --marker-trace` with
`--require-cached`, and each measured ROCTX range covers only the B2 resident
draft chain. The target step used to refresh the next seed is outside the
marker window.

Summary for the B2-shaped draft profile with default-on Q8 shared dual, X8
Q6_K top-1, row-parallel F32 router logits, resident initial prompt KV seeding,
the shared-gate scalar-dot fix, the parallel MTP dense-attention body, and
dense-Q8 dp4a limited to draft forward leaves:
**6.145 ms/cycle host wall**, **5.055 ms/cycle kernel time**, **82.3% kernel
share**, **96.5 kernel calls/cycle**, and **1.090 ms/cycle host residual** in
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-draftdenseq8-draftonly-gpuevents.json`.
Versus the prior active GPU-event artifact, the draft-only selector moves host
**6.529 -> 6.145 ms/cycle**, kernel **5.498 -> 5.055 ms/cycle**,
`draft_dense_shared_gemv` **0.800 -> 0.379 ms/cycle**, and
`draft_device_chain_drain` **5.258 -> 4.864 ms/cycle**, while Q6 top-1 stays at
parity. The call count rises **90.5 -> 96.5 calls/cycle** because of added
q8_1 quantize/dp4a launches, but the dense GEMV body savings win at the parent
row.

The old all-stage draft dense-Q8 dp4a diagnostic added the same wrappers to
initial KV seeding as well as draft forward leaves. Its isolated draft profile
improved, but the full-suite route regressed the then-active lane
**64.41 -> 64.14 tok/s** with worse row economy. The retained `-draftonly` row
is the important split: do not apply draft dense-Q8 dp4a to initial prompt KV
seeding; keep it limited to draft forward leaves unless a future full-suite row
proves otherwise.
The selected SiLU/down fusion diagnostic added an exact BF16-equivalent Q5_K
selected-down body that computes `silu(gate) * up` inside the GEMV. It is rejected
at the draft-profile level: launch count falls **90.75 -> 88.5 calls/cycle**, but
kernel time rises **5.973 -> 6.054 ms/cycle**, host wall rises **7.044 -> 7.206
ms/cycle**, and `draft_run_moe_down_combine` rises **0.487 -> 0.531 ms/cycle**.
This says the active separate `silu_mul_separate_out_bf16` plus
`gguf_k_selected_prefill_out` chain is still better than the first fused Q5 body;
do not spend a full-suite run on it.
The current GPU-event diagnostic artifact
`benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-draftdenseq8-draftonly-gpuevents.json`
adds `--gpu-event-stage-timings` without `--sync-stage-timings`, so it attributes
the queued async draft drain with HIP event elapsed time instead of pushing a
device sync after every leaf. It is still diagnostic-only (`performance_claim=false`)
because event recording perturbs host enqueue time, but it answers the immediate
accounting question: the large async `draft_topk_readback` / `draft_device_chain_drain`
bucket is queued GPU draft work, not D2H copy or Python readback. The largest
event bucket is still the Q6 lm-head/top-1 path, followed by selected gate/up,
selected down/combine, QKV/dense-Q8 projection work, and the now-small attention
bucket.
The cross-engine draft-kernel compare artifact
`benchmarks/results/2026-07-02-mtp-draft-kernel-compare-draftdenseq8-draftonly.json` joins this
hipEngine profile with the llama.cpp ROCTX range profile and the retained
llama.cpp stage rerun. It shows the dominant Q6 lm-head dispatch is not the
remaining source-level gap in that profile: hipEngine is **1.786 ms/call**,
llama.cpp is **1.781 ms/call**, and the delta is only **+0.005 ms/call**. The
parent-wall conclusion in that artifact is now superseded because its
full-suite row used unsafe rejected/partial direct-state commit. Treat Q6 top-1
as a guardrail and keep the active work on semantic-safe verifier replay.
Normalizing the short profiler windows per MTP cycle gives hipEngine
**5.055 ms/cycle** kernel total vs llama.cpp `llama_draft_sample_topk`
**4.340 ms/cycle**, with Q6 equal (**3.571 vs 3.561 ms/cycle**) and the
residual in non-Q6 secondary draft kernels (**1.483 vs 0.778 ms/cycle**,
**+0.705 ms/cycle**). That residual is useful for ranking secondary leaves, but
it is not the headline target while the active safe row is dominated by verifier
replay.
The approximate
ms/output column divides by the old unsafe full-suite `llama-compat` row's
**2.64 visible outputs/cycle** (264 outputs / 100 cycles), so use it as
attribution, not as a replacement for the full-suite timing rows.

Non-sync GPU-event split for the same active route:

| GPU event bucket | ms/cycle | approx ms/output | reading |
| --- | ---: | ---: | --- |
| `draft_device_chain_drain` (host timer) | **4.864** | **1.842** | Async drain waiting for queued draft GPU work. |
| `draft_topk_readback` (host timer) | **5.501** | **2.083** | Drain plus tiny D2H and host accounting; compare through `draft_initial`. |
| `draft_topk_d2h` | 0.075 | 0.028 | D2H is not the draft gap. |
| `draft_gpu_decode_initial` | **2.684** | **1.017** | Depth-0 queued draft work; includes the MTP block and lm-head/top-1. |
| `draft_gpu_decode_next` | **2.636** | **0.998** | Depth-1 queued draft work; effectively same cost as initial. |
| `draft_gpu_run_lm_head` | **3.716** | **1.407** | Dominant event bucket; matches the kernel trace's Q6 top-1 stage1 dominance. |
| `draft_gpu_run_attention` | 0.157 | 0.059 | Now small after the parallel dense-attention body. |
| `draft_gpu_run_qkv_kvwrite` | 0.249 | 0.094 | Q/gate, K/V, RoPE, and KV-write work. |
| `draft_gpu_run_ffn_selected_gate_up` | 0.451 | 0.171 | Largest non-Q6 FFN leaf. |
| `draft_gpu_run_moe_down_combine` | 0.382 | 0.145 | Selected down plus combine. |
| `draft_gpu_run_project` | 0.201 | 0.076 | E/H projection path. |
| `draft_gpu_run_ffn_up_shared` | 0.084 | 0.032 | Shared expert path is no longer a top target. |
| `draft_gpu_run_ffn_router_select` | 0.075 | 0.028 | Router path is small after row-parallel logits. |
| `draft_gpu_device_topk_gather` | 0.005 | 0.002 | Device gather is effectively gone. |

| kernel-family bucket | calls/cycle | ms/cycle | approx ms/output | kernel share | next action |
| --- | ---: | ---: | ---: | ---: | --- |
| `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1` | 2.0 | **3.571** | **1.353** | **70.7%** | Largest individual draft kernel family, but the ROCTX llama.cpp comparison shows per-call parity: hipEngine **1.786 ms/call** vs llama.cpp Q6_K `mul_mat_vec_q` **1.781 ms/call** (**+0.005 ms/call**). Treat this as a guardrail unless a same-protocol rerun reopens the parent draft row. |
| `top1_stage2_gather` | 2.0 | 0.097 | 0.037 | 1.9% | Not the missing cost; prior stage2/gather work already made this small. |
| `gguf_q4_k_selected_dual_prefill_out` | 2.0 | 0.441 | 0.167 | 8.7% | Selected gate/up target, but prior X8/raw attempts regressed. |
| `gguf_k_selected_prefill_out` | 2.0 | 0.330 | 0.125 | 6.5% | Selected down remains secondary. |
| `q8_0_dp4a_gemv` | 6.0 | 0.179 | 0.068 | 3.5% | Draft-only dense-Q8 singleton leaves; replaces slower `gguf_k_prefill_out` calls in this route. |
| `q8_0_dp4a_triple_split_rowtile_gemv` | 2.0 | 0.176 | 0.067 | 3.5% | Draft Q/K/V dense-Q8 triple leaf. |
| `hipengine_mtp_rmsnorm_f32` | 14.0 | 0.067 | 0.026 | 1.3% | Prep cost; not the retained parent limiter. |
| `hipengine_mtp_dense_attn_f32` | 2.0 | 0.036 | 0.014 | 0.7% | Closed by the parallel dense-attention body; no longer a top target. |
| `qwen35_router_logits` | 4.0 | 0.027 | 0.010 | 0.5% | Covers both router logits and the shared-gate scalar dot after the fix; no longer a top draft target. |
| `q8_0_dp4a_dual_split_rowtile_gemv` | 2.0 | 0.025 | 0.009 | 0.5% | Draft dense-Q8 dual leaf; small after the draft-only selector. |

Fine-sync draft leaf attribution for the same active X8 route:
`scripts/gguf_mtp_draft_rocprof.py --steps 4 --warmup 2 --q6-top1-dp4a
--q6-top1-stage1-shape x8 --selected-down-x8-repack q6
--record-stage-timings --sync-stage-timings --require-cached`. The profile is
attribution-only because every leaf inserts a sync, but it makes the non-Q6
draft-body target order explicit. The router-row refresh replaces the old generic
router linear with `qwen35_router_logits`, leaving the row below as a regression
guard. Keep this table complete for the current artifact so the next target can
be chosen from a measured leaf, then validated by moving the async/full-suite
parent bucket.

| sync-stage leaf | ms/cycle | reading |
| --- | ---: | --- |
| `draft_run_lm_head_q6_top1_dp4a_x8_stage1` | **3.589** | Still the largest single draft leaf, but llama.cpp's matching Q6_K dispatch is per-call parity. Further work should first prove a proposal/acceptance or launch-amortization mismatch, not another isolated Q6 scale/layout variant. |
| `draft_run_ffn_selected_gate_up` | **0.465** | Largest remaining non-Q6 leaf. Prior raw/X8 verifier-shaped copies regressed, so any draft fix needs a different mechanism or full-suite proof. |
| `draft_run_qkv_q_gate` | **0.396** | Q/gate projection + split + Q norm. Candidate for a draft-specific dense-Q8/q8_1 path only if F32 output precision is handled. |
| `draft_run_moe_selected_down` | **0.382** | Selected Q5_K down body. |
| `draft_run_attention_core` | **0.371** | Attention core; visible but smaller than selected gate/up and Q6 top-1. |
| `draft_run_attention_out` | 0.217 | Gate/out projection/residual after attention core. |
| `draft_run_project_eh_proj` | 0.204 | E/H projection inside draft input project. |
| `draft_run_qkv_k_rope` | 0.127 | K RoPE/write sub-leaf; below the main GEMV targets. |
| `draft_run_lm_head_q6_top1_dp4a_stage2_gather` | 0.121 | Final block-winner reduction plus optional embedding gather remains small. |
| `draft_prepare_inputs` | 0.086 | Input setup; not large enough to explain the parent draft gap. |
| `draft_run_project_norm_concat` | 0.078 | Draft input norm/concat prep. |
| `draft_run_qkv_v_kvwrite` | 0.065 | V projection/write sub-leaf. |
| `draft_run_ffn_shared_gate_up` | 0.064 | Shared gate/up dual launch; retained dual path keeps this small. |
| `draft_run_ffn_shared_down` | 0.052 | Shared down projection after shared SiLU. |
| `draft_run_ffn_router_linear` | 0.047 | Fixed by row-parallel router logits; keep as a regression guard, not an active target. |
| `draft_run_moe_combine_cast_inputs` | 0.044 | Device-MoE combine cast/setup leaf. |
| `draft_run_moe_weighted_combine` | 0.039 | Final weighted combine leaf. |
| `draft_run_ffn_router_select_only` | 0.035 | Router top-k selection alone; row-parallel logits removed the router-linear leaf. |
| `draft_run_ffn_post_norm` | 0.033 | FFN RMSNorm leaf. |
| `draft_run_project_attn_norm` | 0.033 | Attention norm inside draft input projection. |
| `draft_run_lm_head_norm` | 0.034 | Draft lm-head RMSNorm leaf. |
| `draft_run_moe_selected_silu` | 0.031 | Selected expert SiLU leaf. |
| `draft_seed_upload` | 0.029 | Seed upload/setup; too small to drive the draft gap. |
| `draft_topk_readback` | 0.028 | Readback timer inside the sync profile; parent `draft_topk_readback` remains the async rollup target. |
| `draft_run_ffn_post_norm_cast_bf16` | 0.027 | FFN post-norm cast. |
| `draft_run_lm_head_quant_q8_1` | 0.027 | q8_1 activation quantization is not the missing Q6 top-1 cost. |
| `draft_run_ffn_shared_gate_linear` | 0.027 | Fixed by row-parallel scalar dot; keep as a regression guard, not an active target. |
| `draft_topk_d2h` | 0.026 | D2H copy is not the source of the draft drain gap. |
| `draft_run_lm_head_cast_bf16` | 0.025 | Draft lm-head BF16 cast. |
| `draft_run_ffn_shared_silu` | 0.024 | Shared SiLU/multiply. |
| `draft_device_topk_gather` | 0.001 | Device gather is effectively gone as a target. |

Continuity rollups from the same sync artifact: `draft_mtp_layer_forward`
**6.579 ms/cycle**, `draft_run_lm_head` **3.799**,
`draft_run_lm_head_q6_top1_dp4a_x8_gather` **3.711**,
`draft_run_ffn_up_shared` **0.780**, `draft_run_qkv_kvwrite` **0.589**,
`draft_run_attention` **0.589**, `draft_run_moe_down_combine` **0.498**,
`draft_run_project` **0.308**, and `draft_run_ffn_router_select` **0.146**.
These rollups intentionally overlap the leaf rows above and are for continuity
with older artifacts, not for summing a new total.

Pack16 diagnostic result: `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack16`
is rejected. It halves the number of top-1 stage1 blocks, but the larger
per-block register/weight body loses more than the q8/reduce traffic saves:
same-session denseq8all smoke is **71.74 -> 71.72 tok/s** and draft rocprof
reports `gguf_q6_k_pack16_gemv_q8_1_dp4a_top1_stage1` at
**3.684 ms/cycle** vs the retained pack8 **3.603 ms/cycle**. Do not rerun
pack-width-only variants unless the Q6 top-1 body/layout changes materially.

X8 dscale diagnostic result:
`HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8_dscale` is rejected. It keeps the
retained X8 Q6_K lm-head layout but adds a precomputed FP32 `d*scale` sidecar
for every X8 tile. The measured draft-chain rocprof row regresses against the
retained X8 artifact: **6.805 -> 8.023 ms/cycle host wall**,
**6.427 -> 7.615 ms/cycle kernel time**, and `draft_lm_head_q6_top1`
**3.648 -> 4.859 ms/cycle**. This rules out simple scale precompute as the
missing draft fix. After the ROCTX per-call comparison, another Q6 top-1 change
needs evidence of a proposal/acceptance or launch-amortization mismatch first,
or a materially different fusion that moves async `draft_initial`.

#### Current all-sync leaf attribution

Attribution-only artifact:
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-nocopy-allsync-smoke.json`.
Do not use this row for headline tok/s: extra sync barriers slow the diagnostic
run. Its purpose is to rank leaf kernels inside the retained no-copy full-suite
lane.

| sub-bucket | ms/output | source | interpretation |
| --- | ---: | --- | --- |
| `target_block_linear_attn_norm_qkv_gate` | 2.049 | no-copy verifier all-sync | Aggregate retained for continuity. |
| `target_block_linear_attn_attn_qkv_gate_dense_q8_dp4a` | **1.661** | no-copy verifier all-sync | Largest linear-attention dense projection leaf after the GDN copy fix. |
| `target_block_linear_attn_ffn_moe_expert_gate_up` | **1.677** | no-copy verifier all-sync | Main selected-MoE leaf. |
| `target_block_linear_attn_ffn_moe_expert_down` | **1.194** | no-copy verifier all-sync | Still non-trivial but no longer part of a llama.cpp wall deficit. |
| `target_block_linear_attn_prefill_gdn_state_rows` | **0.785** | no-copy verifier all-sync | Closed copy bottleneck: prior copied-state all-sync was **2.913 ms/output**. |
| `target_block_full_attn_layers` | **3.091** | no-copy verifier all-sync | Secondary verifier layer family; compare only through retained verifier total. |
| `target_block_lm_head_sample` | **1.028** | no-copy verifier all-sync | Verifier-side lm-head/sample cost. |
| `draft_run_lm_head` | **1.309** | no-copy all-sync | Aggregate draft lm-head bucket from the finer split. |
| `draft_run_lm_head_q6_top1_dp4a_x8_stage1` | **1.245** | no-copy all-sync | Draft Q6_K top-1 stage1 compute/layout. |
| `draft_run_lm_head_q6_top1_dp4a_x8_gather` | **1.285** | no-copy all-sync | Stage1 plus final gather aggregate. |
| `draft_run_lm_head_q6_top1_dp4a_stage2_gather` | **0.040** | no-copy all-sync | Final block-winner reduction plus optional embedding gather is not material. |

#### Current block-verifier rocprof attribution

This table profiles the verifier shape inherited by the active
`denseq8all-x8top1-f32ssm-routerrow` `llama-compat` lane directly:
`scripts/gguf_mtp_verifier_rocprof.py --mode block-verify --verify-dp4a
--selected-down-x8-repack q6 --verify-dense-q8-dp4a-all
--verify-dense-q8-dp4a-f32 --record-stage-timings --steps 4 --warmup 1`.
It is diagnostic-only and does not replace the full-suite speed rows above. Its
job is to keep the verifier target order mechanical when `llama-compat` already
matches llama.cpp's B2/no-probe structure.

Summary for the B2-shaped block profile refresh: **30.936 ms/block host wall**,
**22.881 ms/block kernel time**, **74.0% kernel share**, and
**1069.0 kernel calls/block**. The measured host residual is
**8.055 ms/block**, but the dominant remaining work is still kernel time.

| kernel-family bucket | calls/block | ms/block | kernel share | next action |
| --- | ---: | ---: | ---: | --- |
| `dense_q8_0_gemv` | 160.0 | **8.319** | **36.4%** | F32 `ssm_out` removes most of the remaining dense singleton T16 cost; dense remains large but no longer explains the parent gap alone. |
| `moe_selected_gemv` | 77.0 | **6.647** | **29.1%** | First remaining verifier leaf target: compare selected gate/up/down body and scheduling against llama.cpp `mul_mat_vec_q_moe`. |
| `lm_head` | 3.0 | **2.683** | **11.7%** | Second remaining target after selected-MoE; includes `q6_k_t16_gemv_rowtile`. |
| `gdn_linear_attn` | 60.0 | 1.665 | 7.3% | Track, but smaller than selected/dense/lm-head GEMV. |
| `moe_router` | 120.0 | 1.049 | 4.6% | Guard against regressions; not first-order gap. |
| `rmsnorm_rope` | 92.0 | 0.557 | 2.4% | Not a priority. |
| `memcpy_fill` | 169.0 | 0.306 | 1.3% | Not enough to close the verifier gap. |
| `moe_combine_silu` | 120.0 | 0.266 | 1.2% | Earlier fused-SiLU route regressed async smoke; keep secondary. |
| `attn_core` | 30.0 | 0.168 | 0.7% | Not the current limiter. |

Top individual kernel families in the same trace:

| kernel family | calls/block | ms/block | kernel share | reading |
| --- | ---: | ---: | ---: | --- |
| `q8_0_dp4a_dual_split_rowtile_gemv` | 30.0 | **4.013** | 17.5% | Raw-Q8 dp4a pair body replacing the old `q8_0_t16_dual_split_gemv`. |
| `q4_k_t16_selected_dual_q8_1_dp4a_direct_gemv` | 40.0 | **4.010** | 17.5% | Main selected gate/up body; now tied with dense pair as the largest single verifier body. |
| `q6_k_t16_gemv_rowtile` | 1.0 | **2.657** | 11.6% | Verifier lm-head bucket. |
| `qk_t16_selected_q8_1_dp4a_direct_gemv` | 37.0 | **2.637** | 11.5% | Selected down body; q6-only X8 helped but did not erase it. |
| `qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop` | 30.0 | 1.501 | 6.6% | GDN recurrent work, below GEMV priorities. |
| `q8_0_dp4a_rowtile_gemv` | 30.0 | **1.418** | 6.2% | New F32 `ssm_out` replacement body; full-suite positive despite the extra quantize launch. |
| `q8_0_t16_gemv` | 50.0 | **1.158** | 5.1% | Dense singleton T16 leftovers after F32 `ssm_out` replacement. |
| `q8_0_dp4a_triple_split_rowtile_gemv` | 10.0 | **1.035** | 4.5% | Raw-Q8 dp4a triple body is faster than the old T16 triple body. |

The F32 `ssm_out` verifier diagnostic (`--verify-dense-q8-dp4a-f32`) remains
retained inside the active router-row llama-compat lane. It adds a GGML-compatible F32 q8_1 activation
quantizer and routes direct-state `ssm_out` through the raw-Q8 dp4a singleton
body. The block profile improves **32.470 -> 30.936 ms/block host wall** and
**23.893 -> 22.881 ms/block kernel time**. Same-session smoke improves
**70.74 -> 71.43 tok/s**, cycle **14.160 -> 14.023 ms/output**, and verifier
drain **11.359 -> 11.230 ms/output** with identical smoke acceptance. Full-suite
B2 improves **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**,
verifier drain **12.662 -> 12.158 ms/output**, acc/output **0.567 -> 0.578**,
and target rows/output **1.299 -> 1.266**. Keep it in the accuracy-traded
`llama-compat` route under the newer router-row lane; it is not exact-default
eligible.

The shared-Q8 verifier diagnostic (`--verify-dense-q8-dp4a-shared`) is rejected
despite the isolated profile looking slightly better. It routes shared-expert
gate/up/down through the raw-Q8 q8_1/dp4a helpers, which changes the block
profile to **31.631 ms/block host wall**, **23.648 ms/block kernel time**, and
**1119.0 kernel calls/block**. Dense singleton T16 calls drop (`q8_0_t16_gemv`
**80 -> 40 calls/block**, **3.187 -> 2.682 ms/block**), but q8_1 quantize rises
(`120 -> 200 calls/block`) and the async full-suite route regresses. Same-session
smoke improved **70.64 -> 71.66 tok/s**, cycle **14.181 -> 13.978 ms/output**,
and verifier drain **11.377 -> 11.183 ms/output** with identical smoke
acceptance; full-suite B2 regressed **61.31 -> 59.63 tok/s**, cycle
**16.331 -> 16.793 ms/output**, verifier drain **12.662 -> 13.038 ms/output**,
and target rows/output **1.299 -> 1.333**. Keep the route as diagnostic evidence
only; the active parity lane is now `denseq8all-x8top1-f32ssm`.

The refreshed llama.cpp HIP verifier-shaped pp4 proxy confirms the source-level
contrast. Command:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/llamacpp-hip-pp4-rocprof-20260701 \
  --output-file pp4 -- \
  /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -dev ROCm0 -fa 1 -p 4 -n 0 -r 1 -b 4 -ub 4
```

Summarized by `scripts/llamacpp_kernel_trace_summary.py` into
`benchmarks/results/2026-07-01-llamacpp-hip-pp4-kernel-summary.json`
(`performance_claim=false`). This proxy is not the MTP denominator, but it shows
what llama.cpp's verifier-shaped HIP forward spends GPU time on:

| llama.cpp pp4 kernel bucket | dispatches | ms/pp4 trace | kernel share | parity reading |
| --- | ---: | ---: | ---: | --- |
| `llama_mmvq_moe` (`mul_mat_vec_q_moe`) | 240 | **21.814** | **40.2%** | llama's selected-MoE verifier proxy is one unified MMVQ family; hipEngine's analog is split across selected gate/up/down bodies. |
| `llama_mmvq` (`mul_mat_vec_q`) | 502 | **18.411** | **34.0%** | llama's dense/quantized projections and lm-head use the same MMVQ family; hipEngine's analog is specialized Q8T16/Q6T16 kernels. |
| `llama_mmvf` | 280 | 2.246 | 4.1% | F32 matvecs are secondary. |
| `llama_quantize_q8_1` | 742 | 1.037 | 1.9% | Activation quantization is not the dominant llama cost, matching hipEngine's selected-MoE split where q8_1 quantize is also secondary. |
| `llama_topk_argsort` | 78 | 0.610 | 1.1% | Top-k kernels are small in the verifier proxy; the MTP draft drain is still tracked through the traced `llama_draft_sample_topk` stage. |

The llama.cpp HIP MTP whole-run rocprof proxy now gives a direct MTP
kernel-family view, but still not a stage-window kernel attribution. Command:

```bash
PYTHONPATH=. HIP_VISIBLE_DEVICES=0 python3 scripts/llamacpp_mtp_rocprof.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen36-35b --port 8021 --ctx-size 8192 --gpu-layers 99 \
  --draft-max 2 --token-repeat --prompt-tokens 32 --max-tokens 8 \
  --profiler-finalize-timeout 90 \
  --label llamacpp-hip-mtp-token32-gen8-whole-run \
  --raw-root /tmp/hipengine-llamacpp-mtp-rocprof-token32-gen8 \
  --out benchmarks/results/2026-07-02-llamacpp-mtp-rocprof-token32-gen8-whole-run.json
```

Artifact:
`benchmarks/results/2026-07-02-llamacpp-mtp-rocprof-token32-gen8-whole-run.json`
(`performance_claim=false`, `status=diagnostic_retained`). The profiled
request completed and wrote stage timings plus a kernel CSV, but the rocprof
wrapper had to send `SIGKILL` after the profiler finalize timeout
(`terminate_status=killed_after_finalize_timeout`). Treat the bucket table below
as a whole-process source/kernel proxy, not as a headline speed row and not as a
draft-window-only split.

| llama.cpp MTP whole-run metric | value | reading |
| --- | ---: | --- |
| request predicted tokens | 8 | Bounded diagnostic request only. |
| request reported tok/s | 70.197 tok/s | Close to the traced llama.cpp HIP B2 row, but not retained as a benchmark because this was profiled. |
| stage rows / visible outputs | 2 cycles / 6 outputs | Stage JSONL records MTP cycles, not the full request denominator. |
| stage accepted / drafts | 4 / 4 | This prompt hit 100% draft acceptance across the two measured cycles. |
| stage cycle wall | 14.983 ms/output | Same order as the retained traced llama.cpp stage row; useful for sanity only. |
| `draft_initial` | 2.134 ms/output | Matches the llama.cpp target row used for the compatibility comparison. |
| `llama_draft_sample_topk` | 1.702 ms/output | The llama.cpp draft sampler/lm-head path remains the closest analog for hipEngine's Q6 top-1 draft drain. |
| `target_block_verify_total` | 12.834 ms/output | Whole-run profiled request is slightly slower than the retained traced row; do not replace the dashboard value. |
| `mtp_context_replay_append` | 9.127 ms/output | llama.cpp process/replay batch construction plus target decode parent. |

| llama.cpp MTP whole-run kernel bucket | dispatches | total ms | share | parity reading |
| --- | ---: | ---: | ---: | --- |
| `llama_mmvq` | 1628 | **73.924** | **29.1%** | Main dense quantized GEMV family for draft/verify/lm-head; source anchor remains `mul_mat_vec_q`. |
| `other` | 3193 | **56.871** | **22.4%** | Includes rocBLAS/Tensile MMQ, SSM conv, top-k MoE helper, and elementwise bodies; needs ROCTX windows before assigning to draft vs verify. |
| `llama_copy_layout` | 2253 | **51.570** | **20.3%** | Copy/layout traffic is a large whole-process cost in llama.cpp; do not assume it maps to hipEngine's hot-window gap without marker correlation. |
| `llama_mmvq_moe` | 612 | **40.686** | **16.0%** | Unified selected-MoE MMVQ family; compare against hipEngine selected gate/up/down leaves when targeting verifier or draft MoE work. |
| `llama_mmvf` | 857 | 7.221 | 2.8% | Secondary F32 matvec family. |
| `llama_elementwise` | 2169 | 6.104 | 2.4% | Small per-dispatch, high-count helper work. |
| `llama_gdn` | 210 | 5.392 | 2.1% | Gated delta net kernels. |
| `llama_norm` | 1407 | 3.986 | 1.6% | Norm kernels remain small relative to GEMV/copy. |
| `llama_quantize_q8_1` | 2240 | 3.529 | 1.4% | Activation quantization is still not the dominant cost in llama.cpp. |
| `llama_flash_attn` | 160 | 2.107 | 0.8% | Attention core is small at this decode shape. |
| `llama_topk_argsort` | 164 | 1.341 | 0.5% | GPU sort itself is small; the larger draft row is the surrounding sample/lm-head path. |

The follow-up ROCTX range proxy closes that instrumentation gap enough for the
next source comparison. llama.cpp commit `dd7ec418c` adds env-gated
`LLAMA_MTP_ROCTX=1` ranges around the existing MTP stage timers without adding a
hard ROCTX link dependency. The refreshed wrapper command adds
`--roctx-ranges`, which sets `LLAMA_MTP_ROCTX=1`, collects
`rocprofv3 --marker-trace`, and joins kernel dispatches to marker windows by
timestamp. Artifact:
`benchmarks/results/2026-07-02-llamacpp-mtp-rocprof-token32-gen8-roctx-ranges.json`
(`performance_claim=false`, `status=diagnostic_retained`,
`terminate_status=killed_after_finalize_timeout`, llama.cpp clean at
`dd7ec418c`).

Stage sanity for this profiled request: reported request throughput
**72.57 tok/s**, stage JSONL **2 cycles / 6 visible outputs**, cycle wall
**14.522 ms/output**, `draft_initial` **2.068 ms/output**, and
`target_block_verify_total` **12.440 ms/output**. These are profiler-run sanity
numbers, not replacements for the canonical traced row.

Whole-process aggregate range-name buckets from the same artifact:

| ROCTX range | calls | kernel ms | range ms | top kernel-family buckets |
| --- | ---: | ---: | ---: | --- |
| `mtp_context_replay_append` | 5 | **137.128** | 181.598 | `other` 46.249, `llama_mmvq` 36.833, `llama_mmvq_moe` 21.504, `llama_copy_layout` 14.981 |
| `llama_process_build_draft_batch` | 5 | **135.360** | 156.705 | `other` 45.345, `llama_mmvq` 36.372, `llama_mmvq_moe` 21.447, `llama_copy_layout` 14.893 |
| `target_block_forward` | 5 | **22.891** | 93.174 | `other` 9.355, `llama_mmvq` 4.658, `llama_copy_layout` 3.444, `llama_mmvq_moe` 2.271 |
| `draft_initial` | 5 | **9.028** | 12.423 | `llama_mmvq` 8.315, `llama_copy_layout` 0.174, `llama_norm` 0.130, `other` 0.098 |
| `llama_draft_sample_topk` | 4 | **8.679** | 10.211 | `llama_mmvq` 8.142, `other` 0.098, `llama_flash_attn` 0.079, `llama_norm` 0.076 |
| `llama_process_decode_ctx_dft` | 5 | 1.756 | 24.734 | `other` 0.893, `llama_mmvq` 0.461, `llama_flash_attn` 0.090, `llama_copy_layout` 0.088 |
| `llama_draft_decode_initial` | 2 | 0.305 | 1.084 | `llama_mmvq` 0.174, `llama_copy_layout` 0.070, `llama_norm` 0.045, `llama_quantize_q8_1` 0.016 |
| `llama_draft_decode_next` | 2 | 0.044 | 1.076 | `llama_copy_layout` 0.034, `llama_norm` 0.009 |

Reading: the draft gap comparison now has a real llama.cpp stage-window analog.
Inside the whole-process `draft_initial` windows, llama.cpp's kernel work is
almost entirely `llama_mmvq`, and the nested `llama_draft_sample_topk` window is
also dominated by `llama_mmvq`. That reinforces the current hipEngine target:
our remaining draft drain should be compared against llama.cpp's Q6/Dense
`mul_mat_vec_q` path and sampler/lm-head wiring, not against attention,
activation quantization, or standalone top-k sort. The `mtp_context_replay_append`
rows are still polluted by warmup/prompt/server work because this is a
whole-process trace; the next instrumentation refinement, if needed, is to
filter marker ranges to the measured JSONL cycles or use selected-region
profiling after request warmup.

This makes the next verifier implementation target more precise: do not repeat
the rejected rowtile-all or pair-only raw-sidecar experiments blindly. The
retained raw-Q8 all-sidecar route has captured part of llama.cpp's
`mul_mat_vec_q` economy, so the next retainable verifier win must either recover
row economy while keeping that dense speed, attack selected-MoE
`mul_mat_vec_q_moe`, or reduce the number of dense/selected GEMV calls that roll
into `target_block_verify_total`.

Stage timings from the child agree with the all-sync ledger: per block,
`target_block_layer_total` is **29.890 ms**, split into
`target_block_linear_attn_layers` **22.077 ms** and
`target_block_full_attn_layers` **7.812 ms**, with
`target_block_lm_head_sample` **2.801 ms**. This confirms the current verifier
work queue: dense Q8T16 projection kernels, selected-MoE GEMV bodies, then
verifier lm-head.

The broader Q8T16 rowtile-all diagnostic is rejected as a route but retained as
evidence. `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` routes qwen35 rows>1 Q8T16
singleton, pair, and triple verifier projections through rowtile4 where the
runtime can do so; the suite route
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall` records this env
explicitly. Correctness passed against the existing exact singleton/pair/triple
wrappers, and the block verifier profile showed the intended isolated movement:

| block profile row | retained `x8q6` | q8rowtileall | delta |
| --- | ---: | ---: | ---: |
| host wall / block | 33.959 ms | **32.599 ms** | **-1.360 ms** |
| kernel time / block | 26.053 ms | **25.276 ms** | **-0.777 ms** |
| dense Q8 GEMV bucket | 11.420 ms | **10.811 ms** | **-0.609 ms** |
| `q8_0_t16_dual_split*` pair body | 6.025 ms | **5.316 ms** | **-0.709 ms** |
| `q8_0_t16_triple_split*` body | **1.537 ms** | 1.608 ms | +0.071 ms |
| `q8_0_t16_gemv` singleton body | **3.172 ms** | 3.188 ms | +0.016 ms |

Async smoke rejected promotion before a full-suite run: same-session retained
`x8q6` reached **68.78 tok/s**, **14.561 ms/output** cycle wall, and
**11.755 ms/output** target verifier, while q8rowtileall reached **68.54 tok/s**,
**14.614 ms/output**, and **11.790 ms/output** with identical acceptance
(`acc/output=0.667`, draft acceptance `1.000`, target rows/output `1.000`).
The rowtile-all route trimmed `target_block_layer_total` by **0.105 ms/output**
but lost that back in setup/replay/lm-head noise. Conclusion: more exact
row-amortization over the current T16 layout is not the retained fix; the next
dense-Q8 attempt should compare and port llama.cpp's actual `mul_mat_vec_q`
layout/scheduler rather than expanding T16 rowtile coverage again.

The broader raw-Q8 dp4a all-sidecar diagnostic is now promoted for the
llama-replication lane.
`--verify-dense-q8-dp4a-all` / `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL=1` retains raw
Q8_0 sidecars for every dense Q8T16 tensor and routes supported rows>1 verifier
projections through q8_1+dp4a rowtile wrappers: the existing pair wrapper plus
new singleton and Q/K/V triple wrappers. Correctness passes the q8_1 oracle and
KL/top-1 gate; `rocprofv3` confirms both
`q8_0_dp4a_rowtile_gemv_kernel<unsigned short, 4>` and
`q8_0_dp4a_triple_split_rowtile_gemv_kernel<unsigned short, 4>` launch.

| block profile row | retained `x8q6` | denseq8all | delta |
| --- | ---: | ---: | ---: |
| host wall / block | 33.959 ms | **31.133 ms** | **-2.826 ms** |
| kernel time / block | 26.053 ms | **23.427 ms** | **-2.626 ms** |
| dense Q8 GEMV bucket | 11.420 ms | **8.902 ms** | **-2.518 ms** |
| `q8_0_dp4a_dual_split_rowtile*` pair body | n/a | 3.996 ms | n/a |
| `q8_0_dp4a_triple_split_rowtile*` body | n/a | 1.030 ms | n/a |

Async smoke looked good: same-session retained control reached **68.44 tok/s**,
cycle **14.635 ms/output**, and verifier drain **11.827 ms/output**; denseq8all
reached **71.44 tok/s**, cycle **14.021 ms/output**, and verifier drain
**11.184 ms/output** with identical smoke acceptance. The original full-suite
row was speed-positive but acceptance-regressing: retained `x8q6` was
**60.36 tok/s**, cycle **16.587 ms/output**, acceptance **0.583**, draft
acceptance **0.700**, verifier drain **13.023 ms/output**; denseq8all reached
**60.89 tok/s**, cycle **16.446 ms/output**, acceptance **0.567**, draft
acceptance **0.655**, verifier drain **12.742 ms/output**. The refreshed rowhist
rerun confirmed the speed result at **60.96 tok/s**, cycle **16.427 ms/output**,
and verifier drain **12.727 ms/output**, with the same acceptance/economy
regression. The follow-up default-on resident Q8 shared gate/up dual GEMV
retains the same acceptance/economy and moves the active full-suite lane again to
**61.19 tok/s**, cycle **16.364 ms/output**, and verifier drain
**12.666 ms/output**. For the llama-replication lane this is now retained
because it copies llama.cpp's dense MMVQ precision/scheduler class more closely
and moves total wall; the row-economy regression is tracked as the next
secondary target.

The Q6 top-1 stage1 thread-count diagnostic is rejected. llama.cpp's RDNA3
Q6_K MMVQ selects a two-warp single-column shape, so hipEngine added
`--resident-mtp-draft-q6-top1-stage1-threads 64` and route
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64` to A/B the analogous
smaller launch. On the real all-sync route, 128 threads beat 64 threads:
stage1 **1.218 vs 1.246 ms/output**, Q6 top-1 aggregate
**1.260 vs 1.286 ms/output**, and cycle wall **19.252 vs 19.335 ms/output**.
Same-session async smoke also favored 128 threads:
**69.06 tok/s / 14.501 ms** vs t64 **68.79 tok/s / 14.557 ms**, with identical
acceptance (`acc/output=0.667`, draft acceptance `1.000`). Keep the t64 route
as a diagnostic only.

The closer llama.cpp row-shape diagnostic is also rejected. hipEngine added
`--resident-mtp-draft-q6-top1-stage1-shape row` plus routes
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row` and
`...-row-allsync` to launch one output row per block with a llama.cpp-like
Q6_K MMVQ signed `__vsubss4`/dot4 body. This copies the shape more faithfully
than the t64 check, but it does not move the route: all-sync row stage1 is only
slightly faster than pack8 (**1.202 vs 1.218 ms/output**) while row stage2 is
much larger (**0.252 vs 0.041 ms/output**) because it reduces over `vocab`
instead of `vocab/8`. Aggregate Q6 top-1 is therefore worse
(**1.454 vs 1.260 ms/output**), and async smoke regresses
**69.06 tok/s / 14.501 ms** pack8 to **66.95 tok/s / 14.958 ms** row with
identical acceptance. The next draft fix is not a mechanical copy of llama.cpp's
row scheduler; it needs a Q6_K stage1 body/layout or fused top-1 reduce that
keeps the small final-reduce economy.

The Q6 top-1 pack8 scale-hoist diagnostic is also rejected. hipEngine added
`--resident-mtp-draft-q6-top1-stage1-shape pack8_scalehoist` plus routes
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist` and
`...-scalehoist-allsync` to keep pack8's `vocab/8` final reduce while hoisting
each Q6_K block's `d*scale[16]` values into shared memory. Correctness matches
the q8_1/Q6_K oracle and `rocprofv3` confirms
`gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_scalehoist_stage1_kernel` launches at both
128 and 64 thread shapes in the fixture. Same-session async smoke rejected the
route before all-sync/full-suite was justified:

| smoke route | B2 tok/s | cycle wall | draft_initial | draft_topk_readback | target verify | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| retained `x8q6` rerun | **68.65** | **14.589 ms/output** | **2.482 ms/output** | **2.323 ms/output** | **11.776 ms/output** | 0.667 acc/output, 1.000 draft |
| `x8q6-scalehoist` | 68.54 | 14.610 ms/output | 2.485 ms/output | 2.341 ms/output | 11.797 ms/output | 0.667 acc/output, 1.000 draft |

Conclusion: repeated Q6 scale loads are not the missing draft-stage cost in the
current pack8 body. Keep `pack8_scalehoist` as evidence only; the active compat
lane remains `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`, and the next
draft-side attempt needs a different Q6_K body/layout or a broader fused
sampler path that moves the async `draft_initial` row.

The Q6 top-1 pack8-with-llama-vecdot-body diagnostic is also rejected. hipEngine
added `--resident-mtp-draft-q6-top1-stage1-shape pack8_llama` plus routes
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama` and
`...-pack8llama-allsync` to keep the retained `vocab/8` final-reduce economy
while swapping the stage1 inner body to the llama.cpp Q6_K MMVQ vecdot
decomposition. Correctness passes the existing q8_1/Q6_K oracle for fused and
split stage1+stage2 paths. The all-sync B2 leaf moved in the intended direction:
Q6 stage1 **1.220 -> 1.205 ms/output** and Q6 aggregate
**1.261 -> 1.247 ms/output**. The async B2 parent row still rejected the route:

| smoke route | B2 tok/s | cycle wall | draft_initial | target verify | acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| retained `x8q6` control | **68.88** | **14.541 ms/output** | **2.487 ms/output** | **11.722 ms/output** | 0.667 acc/output, 1.000 draft |
| `x8q6-pack8llama` | 67.92 | 14.747 ms/output | 2.493 ms/output | 11.920 ms/output | 0.667 acc/output, 1.000 draft |

Conclusion: copying llama.cpp's Q6_K vecdot decomposition into hipEngine's pack8
stage1 is not enough to move the real async draft bucket; the tiny all-sync
stage1 gain is swallowed by route-level noise and verifier drift. Keep
`pack8_llama` as evidence only. The active compat lane remains
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`; do not retry mechanical Q6
body copies unless they change the parent `draft_initial` row.

The pair-dispatch cache is a small host-side cleanup, not the missing mechanism:
full-suite compat B2 moved **55.410 -> 55.453 tok/s** and
`target_block_verify_total` **14.044 -> 14.025 ms/output** with unchanged
acceptance. Its main value is that the new split proves the remaining
`norm_qkv_gate` bucket is the actual Q8T16 pair projection, not RMSNorm or
fallback dispatch.

The selected T16 dp4a thread-count change is the first selected-MoE scheduler
tweak that survived async validation. The old default launched Q4 gate/up and Q5
down selected dp4a kernels with 128 threads. A 64-thread launch cut the Q4
microbench dot **0.0518 -> 0.0405 ms** and Q5 selected-down dot
**0.0531 -> 0.0369 ms**. Full-suite `llama-compat-device-chain-dp4a` B2 moved
**55.45 -> 58.83 tok/s**, `cycle_wall_ms_per_output` **18.057 -> 17.019**, and
`target_block_verify_total` **14.025 -> 13.134 ms/output** with similar
acceptance (`acc/output` **0.561 -> 0.578**, draft acceptance
**0.640 -> 0.685**). Default for selected T16 dp4a is now 64 threads; set
`HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS=128` only for rollback diagnostics.

The Q5 selected-down one-wave scheduler copy is rejected. A q5-only override
(`HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS=32`) keeps Q4 gate/up at the
retained 64-thread launch while testing Q5 selected-down with one wave/token,
closer to llama.cpp's MoE MMVQ shape. The isolated Q5 microbench improved:
prequantized dot **0.03608 -> 0.03305 ms**, quantize+dot
**0.04031 -> 0.03685 ms**, with KL mean **0.00398**, KL max **0.03093**, and
top-1 **0.9375** vs the T16 float path. A Q4 control confirmed Q4 still used
64 threads (`t16_dp4a_dot_prequantized` **0.04007 ms**). The async compat smoke
still regressed the retained route: q5t32 **68.14 tok/s / 14.776 ms/output**
vs current pack8/q6 smoke around **69.06 tok/s / 14.501 ms/output**, with the
same smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). Keep the
override as a diagnostic only; the active compat lane stays on 64-thread Q5 T16
plus q6-only X8 selected-down.

The q8_1/dp4a Q6_K draft top-1 lm-head is the first draft-side approximation
that survived full-suite validation. It keeps the exact Q6 top-1/gather path as
the default but adds `--resident-mtp-draft-q6-top1-dp4a` for the llama-compat
route, matching llama.cpp's quantized matvec economy more closely. Full-suite
`llama-compat-device-chain-dp4a-q6top1dp4a` B2 moved **58.83 -> 59.63 tok/s**,
`cycle_wall_ms_per_output` **17.019 -> 16.793**, `draft_initial`
**3.564 -> 3.293 ms/output**, and `draft_topk_readback`
**3.390 -> 3.114 ms/output**. Acceptance did not change (`acc/output`
**0.578**, draft acceptance **0.685**). The all-sync split confirms the intended
bucket: `draft_run_lm_head` **1.471 -> 1.253 ms/output**. The remaining
compat gap vs llama.cpp HIP was still **+2.562 ms/output**, split roughly between
draft drain (**+1.153 ms/output**) and verifier drain (**+1.095 ms/output**).
The q6-only X8 selected-down run below is the current tracker row.

**Retained 2026-07-01 q6-only X8 selected-down compat win:** q6-only X8
selected-down repack is now the active `llama-compat` comparison lane, still
accuracy-traded/default-off outside that route. The materializer gate is
`--selected-down-x8-repack q6`, which routes Q6_K selected-down experts through
the X8 q8_1/dp4a replacement layout while leaving Q5_K selected-down on T16.
The isolated microbench explains why q6 is the only promoted family:
Q6 selected-down moved from production T16 **0.0610 ms** to X8
quantize+dot **0.0337 ms** (`1.81x`), while Q5 moved only
**0.0702 -> 0.0628 ms** and the q5/both smoke route regressed vs q6-only.

Full-suite `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` B2 moved
**59.625 -> 60.362 tok/s**,
cycle wall **16.7928 -> 16.5868 ms/output**, and
`target_block_verify_total` **13.1776 -> 13.0228 ms/output**. Acceptance moved
slightly up (`acc/output` **0.578 -> 0.583**, draft acceptance
**0.685 -> 0.700**) and target rows/output improved **1.266 -> 1.250**. At that
point the compat gap vs llama.cpp HIP was **+2.356 ms/output**, split across
draft drain (**+1.104 ms/output**), verifier drain (**+0.940 ms/output**), and
row economy (**+0.102 target rows/output**). q5/both remains rejected for this
route: the q6-only smoke reached **69.03 tok/s**, while q5+q6 X8 smoke reached
only **64.81 tok/s**.

Full rowhist rerun of the same retained lane
(`benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rowhist-full.json`)
is within noise at **60.28 tok/s**, cycle wall **16.610 ms/output**, and
`target_block_verify_total` **13.038 ms/output** with unchanged acceptance. The
then-active gap table tracked **+2.379 ms/output**, split into
draft drain **+1.108 ms/output**, verifier drain **+0.955 ms/output**, and row
economy **+0.102 target rows/output**.

**Rejected 2026-07-01 direct-F32 q8_1 draft lm-head input check:** a dirty-tree
diagnostic changed the compat draft Q6_K dp4a top-1 path from
`rmsnorm_f32 -> f32_to_bf16 -> q8_1 quantize -> q6 dp4a top1` to direct
`rmsnorm_f32 -> f32 q8_1 quantize -> q6 dp4a top1`, matching llama.cpp's
activation-quantization shape more closely and removing one BF16 cast launch.
The idea did not survive full-suite validation. Smoke moved only
**68.33 -> 68.42 tok/s** and cycle wall **14.656 -> 14.639 ms/output**.
All-sync showed no intended leaf win: `draft_run_lm_head`
**1.253 -> 1.261 ms/output**. Full-suite B2 was neutral/slightly negative:
**59.625 -> 59.621 tok/s**, cycle wall **16.7928 -> 16.7943 ms/output**,
with unchanged acceptance (`acc/output` **0.578**, draft acceptance **0.685**).
The small draft-side full-suite movement (`draft_initial`
**3.2928 -> 3.2853 ms/output**) was offset by verifier noise
(`target_block_verify_total` **13.1776 -> 13.1859 ms/output**). The code was
backed out; do not retry direct-F32 q8_1 as a standalone draft fix. If revisited,
it needs a fused RMSNorm+q8_1 path or a broader lm-head scheduler change.

**Rejected 2026-07-01 Q8T16 pair thread-count check:** the Q8T16 verifier
`attn_qkv+attn_gate` split pair kernel now has a diagnostic
`HIPENGINE_GGUF_Q8_T16_THREADS` launch-width override so the llama-compat lane can
A/B the exact hot shape without changing the default. The default remains the
existing 128-thread launch. A focused microbench at the qwen35 linear-attention
pair shape (`in=2048`, `out=(8192,4096)`) rejected 64 threads:

| rows | 64-thread pair | 128-thread pair | result |
| ---: | ---: | ---: | --- |
| 2 | 197.77 us | **179.26 us** | 64 is 10.3% slower. |
| 3 | 224.80 us | **207.05 us** | 64 is 8.6% slower. |
| 4 | 251.96 us | **237.02 us** | 64 is 6.3% slower. |

`rocprofv3 --kernel-trace` on the 64-thread correctness fixture confirmed the
new path really launched `q8_0_t16_dual_split_gemv_kernel<unsigned short,
unsigned short>` with `Workgroup_Size_X=64`, `Grid_Size_X=768`,
`Grid_Size_Y=3`, and `End-Start=5645 ns`. No async smoke/full-suite run is
justified because the isolated hot pair is already slower. The Q8T16 pair gap is
therefore not a simple 64-thread scheduler mismatch; the next comparison needs a
different kernel body/schedule against llama.cpp's `mul_mat_vec_q`/mmvq shape, not
just a smaller workgroup.

**Rejected 2026-07-01 Q8T16 q8_1/dp4a pair-body check:** implemented a
diagnostic `gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out` wrapper that
keeps the current Q8T16 replacement layout but consumes GGML q8_1 activation
blocks and uses `sudot4`, matching llama.cpp's Q8_0×Q8_1 arithmetic more closely.
Correctness passed against a q8_1 CPU oracle plus the KL/top-1 quality gate, and
`rocprofv3` confirmed the fixture launched
`q8_0_t16_dual_split_q8_1_dp4a_kernel<unsigned short>` with
`Workgroup_Size_X=128`, `Grid_Size_X=1536`, `Grid_Size_Y=8`. The performance
result is a clear rejection: on the qwen35 linear-attention pair shape
(`in=2048`, `out=(8192,4096)`), even pre-quantized q8_1 is much slower than the
exact T16 pair.

| rows | exact 128-thread pair | q8_1 quantize + dp4a | prequantized q8_1 + dp4a | result |
| ---: | ---: | ---: | ---: | --- |
| 2 | **181.50 us** | 304.78 us | 303.05 us | dp4a is 1.68x slower. |
| 3 | **207.98 us** | 448.32 us | 452.51 us | dp4a is 2.16x slower. |
| 4 | **236.26 us** | 558.14 us | 566.29 us | dp4a is 2.36x slower. |

This explains why the earlier raw-Q8 sidecar also lost: the problem is not only
the extra quantize launch. The current T16 layout is byte-neutral and exact, but
its `[32 K lanes, 16 cols]` payload makes four adjacent K bytes for one output
column strided by 16 bytes, so the dp4a body has to pack scattered bytes before
every dot4. The Q8 verifier gap now points at a true llama-style layout/schedule
port or a row-amortized verifier kernel, not q8_1/dp4a over the existing T16
tile.

**Rejected 2026-07-01 Q8T16 exact rowtile pair check:** implemented diagnostic
exact row-amortized rowtile2/rowtile4 split-output wrappers for the actual
`attn_qkv+attn_gate` shape. The rowtile kernels compute multiple verifier rows
per `(out_tile16)` block and reuse the T16 weight tile across rows, matching the
llama.cpp `mul_mat_vec_q` row-amortization idea while preserving BF16-input FP32
accumulation and BF16 output. Correctness passed bit-for-bit vs the existing
exact pair on the qwen35 pair shape (`rows=3`, `in=2048`, `out=(8192,4096)`),
and a cached `rocprofv3 --kernel-trace` confirmed the runtime-intended
`q8_0_t16_dual_split_rowtile_gemv_kernel<unsigned short, unsigned short, 4>`
launch with `Workgroup_Size_X=64`.

The isolated pair microbench is positive, and explains why this was worth a
full-suite test:

| rows | exact 128-thread pair | rowtile4 64-thread pair | isolated result |
| ---: | ---: | ---: | --- |
| 2 | 179.75 us | **154.05 us** | rowtile4 is 14.3% faster. |
| 3 | 207.70 us | **170.55 us** | rowtile4 is 17.9% faster. |
| 4 | 236.41 us | **191.16 us** | rowtile4 is 19.1% faster. |
| 5 | 265.87 us | **254.19 us** | rowtile4 is 4.4% faster. |
| 6 | 298.97 us | **271.06 us** | rowtile4 is 9.3% faster. |

Same-code smoke also looked positive: `llama-compat-device-chain-dp4a-q6top1dp4a`
B2 rowtile-on moved **66.00 -> 67.14 tok/s**, cycle wall
**15.180 -> 14.925 ms/output**, and `target_block_verify_total`
**12.285 -> 12.054 ms/output** with unchanged smoke acceptance. All-sync smoke
confirmed the intended leaf moved:
`target_block_linear_attn_attn_qkv_gate_pair` **2.224 -> 2.049 ms/output**.

The full suite rejected the route, so it is default-off:

| full-suite B2 row | MTP tok/s | cycle wall | target verify | acceptance |
| --- | ---: | ---: | ---: | ---: |
| retained q6top1dp4a | **59.63** | **16.793 ms/output** | **13.178 ms/output** | 0.578 acc/output, 0.685 draft |
| rowtilepair opt-in | 57.25 | 17.488 ms/output | 13.697 ms/output | 0.556 acc/output, 0.625 draft |

Conclusion: exact row-amortization over the current T16 pair layout is not a
retainable llama-compat fix despite the isolated pair win. Keep
`HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=1` only as a diagnostic A/B hook. The
default path and `llama-compat` path stay on the existing exact pair wrapper.

**Rejected 2026-07-01 Q8T16 rowtile-all expansion:** implemented default-off
`HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1`, single/triple rowtile4 wrappers, and the
suite route `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall` to
test whether broad exact row amortization across singleton, pair, and triple
Q8T16 verifier projections could keep the pair-rowtile isolated win while
avoiding the earlier full-suite regression. Correctness passed vs the existing
exact wrappers. The block-verifier profile improved the isolated dense-Q8 bucket
(**11.420 -> 10.811 ms/block**) and total kernel time
(**26.053 -> 25.276 ms/block**), mostly from pair body
`q8_0_t16_dual_split*` **6.025 -> 5.316 ms/block**; triple rowtile regressed
slightly (**1.537 -> 1.608 ms/block**) and singleton did not move.

Same-session async smoke rejected the route before a full-suite run:

| smoke route | B2 tok/s | cycle wall | target verify | layer total | acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| retained `x8q6` control | **68.78** | **14.561 ms/output** | **11.755 ms/output** | 9.436 ms/output | 0.667 acc/output, 1.000 draft |
| `x8q6-q8rowtileall` | 68.54 | 14.614 ms/output | 11.790 ms/output | **9.331 ms/output** | 0.667 acc/output, 1.000 draft |

The diagnostic proves the exact rowtile pair body is locally useful, but the
async route cannot spend that isolated win down into wall time. The retained lane
stays `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`. Do not retry broader T16
rowtile as the next dense-Q8 fix; inspect/copy llama.cpp's Q8_0×Q8_1 MMVQ
layout and scheduling instead.

Latest selected-MoE inner split after q6-only X8 selected-down
(`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync`, all-sync smoke,
extra sync points):

| selected-MoE bucket | aggregate ms/output | q8_1 quantize | GEMV body / SiLU | interpretation |
| --- | ---: | ---: | ---: | --- |
| `target_block_linear_attn_ffn_moe_expert_gate_up` | **1.655** | 0.199 | 1.408 GEMV + 0.189 SiLU | Gate/up remains a selected-MoE body target. |
| `target_block_linear_attn_ffn_moe_expert_down` | **1.145** | 0.174 | 0.941 GEMV | q6-only X8 trims selected-down, but the body is still non-trivial. |
| `target_block_full_attn_ffn_moe_expert_gate_up` | **0.538** | 0.065 | 0.457 GEMV + 0.060 SiLU | Same shape at lower weight because there are fewer full-attention layers. |
| `target_block_full_attn_ffn_moe_expert_down` | **0.371** | 0.058 | 0.302 GEMV | Same conclusion: optimize selected GEMV body/scheduler, not just quantize. |

Read this split together with the full-suite ledger above: the current
`llama-compat` verifier still has two leading operation-cost targets,
Q8T16 `attn_qkv+attn_gate` pair projection and selected-MoE GEMV bodies. The
new selected-MoE buckets rule out q8_1 quantization overhead as the primary
cause; copying llama.cpp more closely means comparing its `mul_mat_vec_q_moe`
schedule/body against these selected GEMV launches, not adding another raw-Q8
sidecar quantize path.

The Q4 X8 selected gate/up retry was rejected on the then-retained route.
It uses the existing `HIPENGINE_GGUF_SELECTED_GATE_UP_X8=1` materializer path,
now exposed by `--selected-gate-up-x8` and suite route
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup`. Same-session async
smoke kept acceptance identical but regressed B2 **67.62 -> 59.08 tok/s**,
cycle wall **14.810 -> 16.948 ms/output**, and target verifier drain
**12.005 -> 14.117 ms/output**. The all-sync split pinpoints the added cost:

| selected gate/up row | retained `x8q6` all-sync | `x8q6-x8gateup` all-sync | delta |
| --- | ---: | ---: | ---: |
| linear-attn gate/up aggregate | **1.655 ms/output** | 3.291 ms/output | +1.637 |
| linear-attn gate/up q8_1 quantize | 0.198 ms/output | **0.194 ms/output** | -0.004 |
| linear-attn gate/up GEMV | **1.408 ms/output** | 3.050 ms/output | +1.641 |
| full-attn gate/up aggregate | **0.548 ms/output** | 1.093 ms/output | +0.545 |
| full-attn gate/up GEMV | **0.462 ms/output** | 1.015 ms/output | +0.553 |

Conclusion: the Q4 X8 replacement layout is not a retainable gate/up fix for
the B2 verifier shape. The selected-MoE work queue stays on a llama.cpp
`mul_mat_vec_q_moe` body/scheduler comparison, not on broadening X8 gate/up.

The raw GGUF selected gate/up copy is also rejected. This route is exposed as
`--selected-gate-up-raw` and
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup`; it keeps Q4_K
selected gate/up experts in raw GGUF layout under decode-repack so
`--verify-dp4a` routes them through the raw selected-dual q8_1/dp4a body, which
is the closest in-tree analogue to llama.cpp's `mul_mat_vec_q_moe`. Same-session
async smoke regressed B2 **68.55 -> 62.04 tok/s**, cycle wall
**14.612 -> 16.142 ms/output**, and target verifier drain
**11.792 -> 13.328 ms/output**, with identical acceptance
(`acc/output=0.667`, draft acceptance `1.000`, target rows/output `1.000`).
The all-sync split attributes the loss to the raw selected gate/up GEMV body,
not q8_1 quantization:

| selected gate/up row | retained `x8q6` all-sync | `x8q6-rawgateup` all-sync | delta |
| --- | ---: | ---: | ---: |
| linear-attn gate/up aggregate | **1.658 ms/output** | 2.404 ms/output | +0.746 |
| linear-attn gate/up q8_1 quantize | 0.189 ms/output | 0.199 ms/output | +0.010 |
| linear-attn gate/up GEMV | **1.422 ms/output** | 2.153 ms/output | +0.731 |
| full-attn gate/up aggregate | **0.542 ms/output** | 0.825 ms/output | +0.284 |
| full-attn gate/up q8_1 quantize | 0.064 ms/output | 0.076 ms/output | +0.012 |
| full-attn gate/up GEMV | **0.461 ms/output** | 0.729 ms/output | +0.268 |

Conclusion: a mechanical raw-GGUF body/layout copy of llama.cpp's MoE MMVQ is
not the missing selected-MoE win for the current B2 verifier shape. Keep
selected gate/up on the retained T16 dp4a body; future selected-MoE work should
target a new scheduler/body over the T16 layout or a different fused verifier
shape, not raw GGUF gate/up.

**Rejected 2026-07-01 fused-SiLU q8_1/dp4a selected gate/up check:** a
microbench made the fused rows>1 selected gate/up idea look plausible:
`t16_silu_dp4a_quantize_plus_dot` was **0.057 ms** vs split
`t16_selected_dual_silu` **0.077 ms** at `x_rows=2`, `rows=16`. The all-sync
smoke also looked directionally positive: `target_block_verify_total`
**16.298 -> 15.811 ms/output** and `target_block_linear_attn_layers`
**10.710 -> 10.366 ms/output**, mostly by removing the separate selected-MoE
SiLU timing bucket. The async smoke rejected it: compat B2 moved
**66.66 -> 66.30 tok/s**, `cycle_wall_ms_per_output` **15.023 -> 15.105**, and
`target_block_verify_total` **11.960 -> 12.060 ms/output**. The runtime wiring
was backed out. Lesson for this tracker: all-sync sub-buckets can overstate a
launch-removal win; only the async/full-suite row updates the headline gap.

**Rejected 2026-07-01 target-block WMMA prefill re-check:** on the same B2
all-sync smoke prompt, `--target-block-wmma-prefill` regressed **52.10 -> 34.04
tok/s** and `target_block_verify_total` **16.092 -> 26.285 ms/output**. The
Q8T16 attention aggregate barely improved (`target_block_linear_attn_norm_qkv_gate`
**2.574 -> 2.529 ms/output**, only **-0.045 ms/output**), while selected-MoE
compact WMMA became the dominant regression (`target_block_linear_attn_ffn_moe_compact_wmma`
**6.484 ms/output**, plus `target_block_full_attn_ffn_moe_compact_wmma`
**2.075 ms/output**). Conclusion: the global target-block WMMA flag is the
wrong shape for the llama-compat B2 verifier and should remain off; it does not
justify a full-suite run.

**Rejected 2026-07-01 dense-Q8 q8_1/dp4a raw-sidecar check:** a new default-off
diagnostic, `--verify-dense-q8-dp4a`, retained raw Q8_0 sidecars for
linear-attention `attn_qkv`/`attn_gate` T16 weights and routed rows>1 verifier
blocks through the existing q8_1/dp4a dense Q8 GEMV. It fired correctly, but did
not close the gap. Async smoke moved **66.66 -> 66.19 tok/s** and
`target_block_verify_total` **11.960 -> 12.072 ms/output**. All-sync smoke showed
the named Q8 pair bucket regressing:
`target_block_linear_attn_attn_qkv_gate_pair` **2.217 ms/output** became
`target_block_linear_attn_attn_qkv_gate_dense_q8_dp4a` **2.622 ms/output**. The
reason is now concrete: llama.cpp's q8_1/dp4a dense matvec economy does not
transfer through hipEngine's raw-sidecar two-GEMV route; the extra q8_1 quantize
launch plus two per-output GEMV launches lose to the current Q8T16 dual pair
kernel. No full-suite run is justified until a T16-native fused q8_1/dp4a pair
kernel or a broader llama-style mmvq scheduler changes that cost model. Artifacts:
`benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-denseq8-smoke.json`
and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-denseq8-allsync-smoke.json`.

**Rejected 2026-07-01 raw-Q8 dp4a rowtile-pair sidecar retry:** the next
diagnostic replaced the two singleton raw-Q8 dp4a launches with one
`q8_0_dp4a_dual_split_rowtile_gemv_kernel<unsigned short, 4>` launch after a
single q8_1 activation quantize. That better matches llama.cpp's row economy:
one wave computes one output column across up to four verifier rows and reuses
the raw Q8_0 weight row bytes. Correctness passed against the q8_1 oracle plus
the KL/top-1 quality gate, and `rocprofv3` confirmed the 32-wide rowtile kernel.
Smoke/all-sync looked directionally useful (`target_block_verify_total`
**11.697 -> 11.488 ms/output** on async smoke and **15.947 -> 15.431
ms/output** on all-sync smoke), but the full suite rejected it:
`llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` B2 **60.36 -> 59.42 tok/s**,
cycle wall **16.587 -> 16.852 ms/output**, acceptance **0.583 -> 0.559**, draft
acceptance **0.700 -> 0.635**, target rows/output **1.250 -> 1.322**, and
`target_block_verify_total` **13.023 -> 13.093 ms/output**. Conclusion: a
llama-style Q8 verifier fix must improve the full acceptance/row economy as well
as the isolated projection body; this raw-sidecar pair route stays diagnostic.
Artifacts: `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-smoke.json`,
`benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-allsync-smoke.json`,
and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-rowtilepair-full.json`.

**There are two separate comparisons:**
1. **HIP-vs-HIP parity (the 67.3 tok/s row):** hipEngine's base decode is not behind:
   AR is **54.95 vs llama HIP 51.38 tok/s**. The remaining HIP-vs-HIP gap is MTP
   uplift/economics: llama's pipeline uses dp4a/q8_1 verify and can run no-probe
   full-block speculation at **0.402 target passes/output**; hipEngine's exact route
   needs a B1 probe and spends **0.567 passes/output**. Copying dp4a into hipEngine
   reaches only **61.3-61.6 tok/s** and fails the ja correctness gate.
2. **Best llama.cpp parity (Vulkan rows):** Vulkan adds a separate backend factor on
   Strix Halo: llama Vulkan AR is **62.65 tok/s** vs hipEngine HIP **54.95 tok/s**.
   The large lm-head is equally BW-efficient (566 vs 550 GFLOPS), but Vulkan's driver
   and fused ggml op shapes are stronger on the smaller ops. hipEngine is HIP-only, so
   matching llama Vulkan is a backend project, not an MTP-policy fix.

**Correctness-preserving levers (exact precision) — TESTED 2026-06-30, both already captured:**
- **Fusion: already done** — qkv is a single fused `attn_qkv` GEMV; selected-expert
  MoE is pack8-consolidated with gate+up+silu fused. Mirrors Vulkan's qkv/`MUL_MAT_ID`.
- **Verify vec-rowtile: built+bit-exact but REFUTED** (0.93× vs the existing
  `grid.y`-occupancy rows-kernel; reverted). The dense verify GEMV is already
  occupancy-amortized at rows>1. Rowtile is the right tool only for the lm-head
  (already shipped).
- A **Vulkan backend for hipEngine** would directly capture the (now-dominant)
  backend factor, but is a large architectural undertaking.

=> No remaining correctness-preserving HIP-kernel lever for AR/verify; the residual
gap is the **Vulkan-vs-HIP backend** + llama's dp4a precision.

### FINAL STAGE LEDGER — hipEngine GGUF HIP vs llama.cpp HIP

This is the current authoritative stage-by-stage attribution. Older historical
sections below are retained for archaeology; where they conflict with this table, this
table wins.

| Stage | hipEngine GGUF HIP | llama.cpp HIP | What it means |
| --- | --- | --- | --- |
| AR wall | **54.95 tok/s** (~18.2 ms/tok) | 51.38 tok/s (~19.5 ms wall; 17.26 ms GPU + host exposed) | hipEngine wins base decode. |
| AR launch shape | **762 launches/tok**, larger exact kernels, host mostly hidden | **1632 launches/tok**, `mul_mat_vec_q` dp4a dominates, ~2.2 ms host exposed | llama's dp4a kernel is good, but HIP launch shape costs it. |
| AR kernel mix | q8_0 attention proj **42%**, q4_K MoE **21%**, q6_K lm-head **9.6%**, GDN **8%** | `mul_mat_vec_q` dp4a **76.5%**, `mul_mat_vec_f` **5.8%**, `quantize_q8_1` **2.2%**, GDN **1.4%** | No hidden AR-stage deficit in hipEngine. |
| Large lm-head bandwidth | q6_K lm-head ~1850 us, **~550 GFLOPS** | Vulkan comparison: 1794 us, **566 GFLOPS** | Large contiguous GEMV is already at parity-class BW. |
| Current exact block verify | rows=4: **42.40 ms wall**, **38.08 ms GPU**, **875 launches**, only **10.2% host exposed** | llama MTP rocprof deadlocks at finalize; 4-row `llama-bench -p 4 -b 4` proxy shows dp4a matmuls dominate | The old host/graph hypothesis is dead; hipEngine verify is GPU-bound. |
| hipEngine verify GPU mix | q8_0 attention **32.7%**, GDN **16.1%**, q4/qK MoE selected **25.9%**, rowtile lm-head **5.9%**, router/norm/misc **19.4%** | proxy: `mul_mat_vec_q_moe` **40.5%** + `mul_mat_vec_q` **33.8%** | llama's advantage is cheaper dp4a/q8_1 verify, not missing hipEngine fusion. |
| Exact MTP economics | B5 **60.78 tok/s**, **1.1134x**, acc/out **0.535**, passes/out **0.567** | B2 **67.3 tok/s**, ~**1.31x**, acc/out **0.598**, passes/out **0.402** | hipEngine does **41% more target-pass work/output**. |
| dp4a transplant | B5 **61.61 tok/s**, **1.1322x**, +1.3% E2E; block verify **42.9 -> 41.2 ms** (-3.9%) | native llama HIP still **67.3 tok/s** | dp4a helps, but does not close the gap. |
| no-probe llama recipe | B5 **56.42 tok/s**, acc/out **0.324** | llama succeeds with no-probe economy | The recipe does not transfer; hipEngine needs the B1 probe. |
| Correctness | exact path passes; dp4a ja top-1 **0.700 < 0.90** gate | llama speed row uses dp4a/q8_1 | Matching llama's precision regime violates hipEngine's guard. |

**Deal:** every stage has now been accounted for. hipEngine is not missing a secret
llama.cpp HIP kernel stage. It has a faster exact AR pipeline, an exact verifier that
is already GPU-bound and already has the useful fusion/amortization, and a speculative
policy that needs one extra cheap probe because exact failed rows are expensive. llama
HIP's remaining advantage is a whole-pipeline dp4a/no-probe economy; reproducing only
the dp4a kernel in hipEngine gives ~61.6 tok/s, not 67.3, and fails Japanese.

### FINAL RESULT — the MTP gap vs llama HIP is the dp4a verify, with an exact accuracy cost

**Bottom line:** hipEngine's GGUF AR decode is *faster* than llama.cpp HIP's
(54.95 vs 51.38 tok/s) — our exact HIP kernels are genuinely good, and fusion /
verify-amortization are already captured. The **only** place we trail llama HIP is
the **MTP verify loop**, and the entire deficit is llama's **dp4a verify pass**,
which **does not pass hipEngine's correctness gate**.

**Exact performance cost (what dp4a buys, what it doesn't):**

| HIP-vs-HIP, full suite | hipEngine (exact) | llama HIP (dp4a) |
| --- | --- | --- |
| AR tok/s | **54.95** | 51.38 |
| MTP tok/s | 60.8 | **67.3** |
| ms / output token | 16.4 | 14.9 |
| MTP uplift over own AR | 1.114× | **1.31×** |
| target-verify passes / output | **0.567** | **0.402** |
| acc / output | 0.535 | 0.598 |

- We do **41% more target-verify work per output token** (0.567 vs 0.402). llama runs
  **1 verify pass/cycle** (passes/out = `1 − acc/out` = 0.402); hipEngine runs ~1.22
  (a cheap B1-probe pass + the block pass).
- **Why:** llama's dp4a verify is cheaper per row, so it (a) pays less per pass and
  (b) can speculate full blocks with **no probe** (wasted rows are cheap). Our *exact*
  verify is pricier per row, so a wasted block-row is costly → the B1-probe is the best
  route (removing it regressed to **1.069×**, acc/out collapsing 0.535→0.379).
- **Swapping only the verify to dp4a on hipEngine buys +1.3% E2E** (60.8 → **61.61
  tok/s**, 1.114×→**1.1322×**, `results/2026-06-30-ar-mtp-suite-full-dp4a-verify-diagnostic.json`)
  — still **8.5% behind llama HIP (67.3)**, because our exact AR is already fast (a
  fixed verify saving is a *smaller ratio* over a fast AR) and the no-probe structure
  needs whole-pipeline dp4a. On the GPU-bound block verify the wall barely moves
  (exact 42.9 ms → all-dp4a 41.2 ms = **−3.9%**); dp4a does **not** speed AR at all
  (54.97 ≈ 54.95). The isolated MoE-GEMV dp4a is ~2–3× but does not translate E2E
  (GPU-bound + added per-layer `quantize_q8_1` launches).

**Exact accuracy cost — llama's dp4a verify FAILS hipEngine's correctness gate:**

- Gate (`AGENTS.md`/`docs/TESTING.md`): **KL ≤ 0.05 AND top-1 agreement ≥ 90%** vs
  `kernels/cpu_reference/` on fixture inputs.
- Measured greedy top-1 agreement of the dp4a (q8_1) verify vs the exact path
  (`scratchpad/dp4a_correctness.py`, flag `HIPENGINE_GGUF_T16_SELECTED_DP4A=1`, real
  ja+code context, 30 tokens):

  | category | dp4a greedy top-1 agreement | gate ≥ 0.90 | first divergence |
  | --- | --- | --- | --- |
  | code | **1.000** (30/30) | PASS | none |
  | **general_ja** | **0.700** (21/30) | **FAIL** | token 20 |

- **dp4a is a hard FAIL on Japanese: 0.700 < 0.90** (q8_1 activation quantization
  loses CJK precision; the greedy path diverges from exact at token 20 and compounds).
  Code is unaffected (1.000). So llama's MTP speed advantage is bought with an accuracy
  loss that violates hipEngine's stated correctness guard — it is **not** a free win.

**Conclusion:** within the correctness gate, hipEngine's MTP (1.114× / 60.8 tok/s) is
at its exact-precision optimum and **already beats llama HIP on AR and on accuracy**.
Matching llama HIP's MTP tok/s requires its dp4a verify, which fails our ja gate
(0.700 top-1) and even then only reaches ~61.6 tok/s here (still < 67.3). The two
honest paths to actually exceed llama remain: relax the ja accuracy gate for dp4a
(not recommended — fails CJK, and insufficient alone), or add a **Vulkan backend**
(beats llama on both AR and MTP on this APU). The exact-precision HIP design point is
documented as closed.

### COFFIN NAIL — dp4a is NECESSARY but NOT SUFFICIENT to match llama HIP MTP

A default-off opt-in **`--verify-dp4a`** mode (bench flag + suite route
`resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a`) was added so anyone who
accepts llama's precision loss can get the max accuracy-traded perf. Measured, full
suite, gfx1151 (artifact `results/2026-06-30-ar-mtp-suite-full-dp4a-verify-diagnostic.json`):

| config | B3 | B4 | **B5 (best)** | vs llama HIP 67.3 |
| --- | --- | --- | --- | --- |
| dp4a + b1-probe (`--verify-dp4a`, the mode) | 59.85 | 60.10 | **61.3–61.6** (1.13×) | **−8.5%** |
| dp4a + no-probe (the "1 pass/cycle" recipe) | 55.71 | 56.34 | 56.42 (1.04×) | −16% |
| exact default (shipped) | 58.83 | 59.53 | 60.76 (1.114×) | −9.7% |

**Two findings nail the claim:**
1. **The "dp4a + 1 verify pass/cycle" hypothesis is FALSE on hipEngine.** No-probe is
   *worse* (56.4, acc/out collapses to 0.324) — our exact/dp4a draft + adaptive-AR-
   fallback latches to AR on a rejected block; the b1-probe is essential. dp4a's
   slightly-less-accurate drafts make no-probe slightly worse, not better.
2. **dp4a + b1-probe (best dp4a) reaches only ~61.6 tok/s — still 8.5% short of llama
   HIP 67.3.** So dp4a is *necessary but not sufficient*: llama's 1.31× uplift also
   needs its **slower AR baseline** (51.38 — a fixed verify saving is a bigger *ratio*
   over a slower AR) and its **no-probe acceptance economy** (which doesn't transfer
   to our fast-AR setup). hipEngine's faster exact AR (54.95) structurally caps the
   uplift ratio even with dp4a.

**Accuracy cost of using the mode** (unchanged): ja greedy top-1 **0.700 < 0.90 gate
FAIL** (first divergence token 20), code 1.000. So `--verify-dp4a` is correctly
default-off and labelled accuracy-degrading; it buys ~+1.3% over the exact default at
the cost of failing the ja gate, and does **not** reach llama HIP. The mode exists for
users who explicitly accept that trade; the shipped default stays exact (1.114×).

### MEASURED CYCLE-STAGE BUCKETS — same buckets on hipEngine and llama.cpp HIP

The deeper instrumentation is now in place on both sides:

- hipEngine: `--record-cycle-stage-timings` on `scripts/gguf_ar_mtp_suite.py`.
- llama.cpp HIP: local diagnostic patch in
  `/home/lhl/llama.cpp/llama.cpp-hip/tools/server/server-context.cpp` plus
  `/home/lhl/llama.cpp/llama.cpp-hip/common/speculative.cpp`; set
  `LLAMA_MTP_STAGE_TIMINGS=/path/file.jsonl` to emit one JSONL record per MTP verify
  cycle. The hipEngine harness summarizes it via `--stage-timings-jsonl`.

Measured setup: Qwen3.6-35B-A3B-UD-Q4_K_M GGUF, `gfx1151` / Radeon 8060S, prompt
suite `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy sampling,
reasoning off. These are **diagnostic timing runs**, not replacement headline rows:
hipEngine timing adds bookkeeping overhead, and the llama natural-24 server trace
measures a slightly faster protocol than the retained 67.3 tok/s HIP row. Use the
stage buckets for attribution; keep the retained non-instrumented rows for the
official tok/s ladder.

Artifacts:

- hipEngine exact B5 deep: `benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-exact-deep.json`
- hipEngine dp4a+B1 B5 deep: `benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-dp4a-deep.json`
- hipEngine llama-compat dp4a B2 after top-1 diagnostic fix:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2-top1-deep.json`
- hipEngine llama-compat dp4a device-chain smoke:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2-devicechain-smoke.json`
- hipEngine llama-compat dp4a prewarmed device-chain split:
  `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json`
- hipEngine llama-compat dp4a prewarmed device-chain sync-stage draft attribution:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after exact Q6_K top-1/gather
  specialization:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-full.json`
  and same-tree disabled control
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-control-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after Q6_K top-1/gather,
  sync-stage draft attribution:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-draftsync-full.json`
- hipEngine llama-compat dp4a prewarmed device-chain after verifier direct-state
  cleanup:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-full.json`
- hipEngine llama-compat dp4a all-sync fine-grained verifier attribution after
  verifier direct-state cleanup:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-allsync-smoke.json`
  and full-suite attribution
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-allsync-full.json`
- hipEngine llama-compat row-compact selected-MoE GEMV rejected smoke:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-rowcompact-allsync-smoke.json`
- hipEngine llama-compat after GGUF pair-dispatch cache:
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-paircache-full.json`
  plus all-sync split attribution
  `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-paircache-allsync-full.json`
- hipEngine llama-compat target-block WMMA prefill rejected smoke:
  `benchmarks/results/2026-07-01-mtp-llama-compat-device-chain-dp4a-allsync-wmma-smoke-mtp.json`
  and `.md`
- hipEngine fused-B1 block probe B5:
  `benchmarks/results/2026-06-30-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-b5-full.json`
  and non-stage check
  `benchmarks/results/2026-06-30-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-b5-full-nostage.json`
- hipEngine fused-B1 block probe smoke after non-llama direct-state snapshot-skip
  carryover:
  `benchmarks/results/2026-07-01-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-snapshot-skip-smoke.json`
- llama.cpp HIP B2 deep:
  `benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.json`
  and `.jsonl`

#### Instrumented economics

| config | AR tok/s | MTP tok/s | uplift | cycle wall / output | accepted / output | draft acceptance | target passes / output | target rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine exact B5 | 54.56 | 59.61 | 1.093× | 16.800 ms | 0.535 | 0.723 | 0.567 | 1.163 |
| hipEngine dp4a+B1 B5 | 54.60 | 60.01 | 1.099× | 16.690 ms | 0.533 | 0.735 | 0.570 | 1.154 |
| llama.cpp HIP B2 | 52.13 | 72.12 | 1.383× | 14.231 ms traced / 13.866 ms server | 0.567 server / 0.610 traced | 0.805 | 0.390 | 1.148 |

Denominator note: llama's server summary reports accepted/output over 240 predicted
tokens (`0.567`); the per-cycle trace excludes the first warmup task and reports 223
visible traced tokens (`0.610`). Stage ms/output uses the traced denominator.

#### Stage ms / output token

| bucket | hipEngine exact B5 | hipEngine dp4a+B1 B5 | llama.cpp HIP B2 | interpretation |
| --- | ---: | ---: | ---: | --- |
| `cycle_wall_ms_per_output` | 16.800 | 16.690 | 14.231 | Instrumented wall. dp4a closes only 0.110 ms/output in this run. |
| `draft_initial` | 1.937 | 1.943 | 2.140 | Draft is not the retained B5 gap; hipEngine is slightly faster here. |
| `draft_topk_readback` | 1.158 | 1.134 | n/a | hipEngine name is a synchronization drain + top-k readback, not pure top-k kernel time. |
| `llama_draft_sample_topk` | n/a | n/a | 1.886 | llama draft is sampler/top-k dominated; MTP decode itself is small (`0.118 + 0.134`). |
| `target_serial_verify_step` | **6.682** | **6.647** | 0.000 | This is the hipEngine B1 probe / serial verifier cost. llama has no equivalent bucket. |
| `target_block_verify_total` | 8.157 | 8.073 | 12.083 | Compare verifier total, not raw `target_block_forward` alone. |
| `target_block_layer_total` | 7.022 | 6.864 | n/a | hipEngine block verifier is GPU layer work: mostly linear-attn layers. |
| `target_block_linear_attn_layers` | 5.195 | 5.049 | n/a | Biggest hipEngine block sub-bucket. |
| `target_block_full_attn_layers` | 1.827 | 1.816 | n/a | Secondary hipEngine block sub-bucket. |
| `target_block_lm_head_sample` | 0.573 | 0.586 | n/a | Not the gap. |
| `target_block_forward` | 8.065 | 7.985 | 0.549 | llama's raw `llama_decode(ctx_tgt)` is async; its GPU drain lands below. |
| `mtp_context_replay_append` | 0.000 | 0.000 | **11.348** | llama's `common_speculative_process()` cost; this is part of verifier total. |
| `llama_process_build_draft_batch` | n/a | n/a | **11.235** | This is the newly split llama bucket. It effectively includes target decode drain + target nextn embedding handoff. |
| `llama_process_decode_ctx_dft` | n/a | n/a | 0.112 | Draft-context catch-up decode is not the big llama cost. |
| `target_block_snapshot` | 0.060 | 0.056 | 0.001 | Not the gap. |
| `target_block_acceptance_accounting` | 0.001 | 0.001 | 0.181 | Visible in llama, still too small to explain the delta. |
| `target_block_replay_or_commit` | 0.029 | 0.029 | 0.004 | Not the gap. |
| `accept_policy_and_seed` | 0.002 | 0.002 | 0.002 | Not the gap. |
| `cycle_wall_over_legacy_ms_per_output` | 0.026 | 0.026 | n/a | hipEngine has no hidden wall outside the legacy timing denominator. |

**Answer:** after adopting dp4a, the measured gap is not draft, snapshot, commit,
policy bookkeeping, or hidden host wall. The gap is the extra hipEngine verification
economy:

- hipEngine dp4a verifier work = `target_serial_verify_step + target_block_verify_total`
  = **14.720 ms/output**.
- llama verifier work = `target_block_verify_total` = **12.083 ms/output**.
- Difference = **+2.637 ms/output** for hipEngine, mostly the B1 serial probe.
- hipEngine draft is **0.197 ms/output faster**, so the net instrumented wall gap is
  ~**2.46 ms/output** (16.690 - 14.231), which is fully explained by verifier
  economics.

This is the fine-grained version of the earlier retained-row conclusion. The retained
non-instrumented gap is smaller (**~1.37 ms/output**, 61.61 vs 67.3 tok/s) because the
diagnostic protocols and instrumentation overhead differ, but the attribution is the
same: llama gets its speedup by avoiding the hipEngine B1 serial probe and spending
fewer target passes/output (`0.390` traced here, `~0.402` retained) while maintaining
higher draft acceptance. Directly copying `target_block_forward` is the wrong target;
in llama most verifier time is under `mtp_context_replay_append`, and the deep split
puts that cost specifically in `llama_process_build_draft_batch` (target decode drain
and nextn embedding handoff), not in the draft-context decode.

#### Compat draft split: prewarm fixes initialization; steady-state draft drain remains

The first compat-dp4a deep split showed `draft_initial ~= 4.03 ms/output` and
`draft_topk_readback ~= 3.80 ms/output`. A top-1 diagnostic fix was implemented so
`--llama-compat` no longer forces top-10 proposal readback, but the full-suite result
was flat:

| config | MTP tok/s | cycle wall / output | `draft_initial` | `draft_topk_readback` | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: |
| compat dp4a B2, old top-10 diagnostic | 52.42 | 19.096 ms | 4.031 | 3.799 | 14.749 |
| compat dp4a B2, top-1 diagnostic | 52.48 | 19.074 ms | 4.043 | 3.833 | 14.715 |
| compat dp4a B2 + prewarmed device-chain | **52.79** | **18.963 ms** | 4.028 | 3.839 | **14.620** |
| compat dp4a B2 + prewarmed device-seed-chain | 52.53 | 19.065 ms | 4.020 | 3.827 | 14.724 |

So the draft-side slowdown is not top-k width by itself. The first device-chain smoke
also showed the wrong bottleneck because it measured the lazy 268 MB full-vocab
embedding-table upload inside the short run:

| probe | MTP tok/s | cycle wall / output | `draft_initial` | `draft_device_chain_ensure_embed_table` | result |
| --- | ---: | ---: | ---: | ---: | --- |
| compat dp4a B2 + device-chain, smoke | 36.12 | 27.704 ms | 14.855 | 11.888 | full-vocab embed-table upload dominates the short run |
| compat dp4a B2 + prewarmed device-chain, full | **52.79** | **18.963 ms** | 4.028 | 0.000 | upload removed; steady-state still slow |

The prewarm/cache fix removes that initialization artifact, but it does **not** close
the llama gap. A split-bucket rerun of the same device-chain route shows why:

| split bucket, compat dp4a B2 + device-chain | ms/output |
| --- | ---: |
| `draft_initial` | 4.033 |
| `draft_topk_readback` | 3.839 |
| `draft_device_chain_drain` | **3.830** |
| `draft_topk_d2h` | **0.008** |

The "readback" bucket is therefore almost entirely a GPU drain, not host copy time.
This is the draft-side target for replication: hipEngine is draining roughly
**3.83 ms/output** of resident draft GPU work where llama's draft sampler/top-k bucket
is **1.886 ms/output** and total `draft_initial` is **2.140 ms/output**.
Persistent/prewarmed device-chain and resident `pending_h` semantics are now explicit
routes, but the remaining win requires reducing the actual device draft work/drain
or fusing it differently; avoiding D2H alone cannot produce the missing tokens.

#### Sync-stage draft attribution: where that GPU drain actually goes

Follow-up diagnostic route:
`llama-compat-device-chain-dp4a-draftsync` =
`--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --verify-dp4a`.
This route inserts `hipDeviceSynchronize()` boundaries inside each resident MTP draft
layer section, so it changes timing and is **not** a retained performance route. Its
only purpose is attribution of the previous `draft_device_chain_drain` bucket.

Command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-draftsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json
```

Measured full-suite result, Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151/Radeon 8060S,
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy, reasoning off, 10 prompts:

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | passes / output | rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine compat device-chain dp4a B2, sync-stage | 54.72 | **52.37** | 0.957x | 19.122 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| llama.cpp HIP B2 deep trace | 52.13 | **72.12** | 1.383x | 14.231 ms | 0.610 traced | 0.805 | 0.390 | 1.148 |

Draft-side split, ms/output:

| bucket | hipEngine compat device-chain dp4a B2 sync-stage | llama.cpp HIP B2 deep | delta / meaning |
| --- | ---: | ---: | --- |
| `draft_initial` | **4.084** | **2.140** | hipEngine draft costs **+1.944 ms/output**. |
| `draft_mtp_layer_forward` | 3.639 | n/a | Sum of the synchronized hipEngine draft layer sections. |
| `draft_run_project` | 0.101 | n/a | Not the gap. |
| `draft_run_qkv_kvwrite` | 0.211 | n/a | Not the gap. |
| `draft_run_attention` | 0.718 | n/a | Material, but smaller than lm-head. |
| `draft_run_ffn_up_shared` | 0.557 | n/a | Material, secondary. |
| `draft_run_moe_down_combine` | 0.166 | n/a | Not the gap. |
| `draft_run_lm_head` | **1.882** | n/a | Biggest hipEngine draft section; roughly equals llama's whole `llama_draft_sample_topk` bucket. |
| `draft_device_topk_gather` | 0.357 | n/a | Device top-k + embedding gather for the next draft depth. |
| `draft_topk_readback` | 0.007 | n/a | D2H remains tiny after sync splitting. |
| `llama_draft_decode_initial` | n/a | 0.118 | llama MTP decode itself is small. |
| `llama_draft_decode_next` | n/a | 0.134 | llama MTP decode itself is small. |
| `llama_draft_sample_topk` | n/a | **1.886** | llama draft is sampler/top-k dominated. |

Verifier-side split, ms/output:

| bucket | hipEngine compat device-chain dp4a B2 sync-stage | llama.cpp HIP B2 deep | delta / meaning |
| --- | ---: | ---: | --- |
| `target_block_verify_total` | **14.715** | **12.083** | hipEngine verifier costs **+2.632 ms/output**. |
| `target_block_layer_total` | 12.827 | n/a | hipEngine's block verifier is still real target-layer work. |
| `target_block_linear_attn_layers` | **9.451** | n/a | Biggest hipEngine verifier sub-bucket. |
| `target_block_full_attn_layers` | 3.375 | n/a | Secondary hipEngine verifier sub-bucket. |
| `target_block_lm_head_sample` | 1.198 | n/a | Material, but not the biggest verifier delta. |
| `mtp_device_kv_commit` | 0.297 | n/a | Small compat lifecycle overhead. |
| `target_block_forward` | 14.573 | 0.549 | Raw bucket is async-misaligned across engines. |
| `mtp_context_replay_append` | 0.009 | 11.348 | llama's target decode drain and nextn embedding handoff live here. |
| `llama_process_build_draft_batch` | n/a | 11.235 | Main llama verifier drain is in process/build, not draft decode. |
| `llama_process_decode_ctx_dft` | n/a | 0.112 | Draft-context catch-up is not the big llama cost. |

This answers the current parity question precisely:

- hipEngine now matches the **observable llama.cpp MTP semantics** in the compat lane:
  B2, no B1 probe, p_min 0, shifted MTP context replay, device MTP KV, resident
  device-chain drafting, and optional dp4a verify.
- hipEngine does **not** yet match llama.cpp's **operation cost**. The measured gap is
  still about **4.89 ms/output** in the diagnostic trace (`19.122 - 14.231`):
  **+1.94 ms/output draft** and **+2.63 ms/output verifier**, with the rest from
  acceptance/pass economy and small lifecycle/accounting differences.
- The draft gap is no longer a black box. Inside the prior GPU drain, the largest
  section is the full-vocab draft LM head (**1.882 ms/output**), followed by draft
  attention (**0.718**), FFN/up/shared (**0.557**), and device top-k/gather
  (**0.357**). D2H is still negligible.
- The verifier gap is target-layer work, especially hipEngine's B2 linear-attention
  layer bucket (**9.451 ms/output**) and full-attention layers (**3.375 ms/output**),
  not a missing `pending_h` handoff or a hidden host copy.

#### First gap-closing fix: exact Q6_K top-1 + embedding gather for compat draft

Implemented an exact Q6_K lm-head specialization for the llama-compat device-chain
draft path:

- New kernel: `hipengine_gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32`.
- It preserves the same per-output Q6_K dot-product reduction and top-1 tie-break as
  `gguf_q6_k_pack8_gemv_decode_bf16_f32_out -> topk_f32_rows_i32`, but writes one
  winner per pack8 block, reduces those winners, and optionally gathers the selected
  FP32 embedding row for the next draft depth.
- Runtime flag: `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER=1` by default, scoped to
  resident MTP draft `top_k == 1`. Set it to `0` for same-tree A/B.

Validation:

```bash
python3 -m py_compile \
  hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_pack8_gemv.py \
  hipengine/speculative/mtp_resident_draft.py \
  tests/test_gguf_q6_k_pack8_gemv_decode.py

PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 pytest -q \
  tests/test_gguf_q6_k_pack8_gemv_decode.py

PYTHONPATH=. pytest -q \
  tests/test_mtp_resident_draft_device_commit.py \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py
```

The new unit gate compares the fused kernel against the old logits -> top-k ->
gather chain and requires identical selected id, selected value, and embedding row.

Same-tree full-suite A/B, Qwen3.6-35B-A3B-UD-Q4_K_M, gfx1151/Radeon 8060S,
`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, greedy, reasoning off,
`--scope full --mtp-route llama-compat-device-chain-dp4a --record-cycle-stage-timings
--require-cached-build`:

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | `draft_initial` | `draft_topk_readback` | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q6 top-1/gather disabled | 54.74 | 52.60 | 0.961x | 19.033 ms | 0.561 | 0.640 | 4.033 ms | 3.838 ms | 14.682 ms |
| Q6 top-1/gather enabled | 54.75 | **53.34** | **0.974x** | **18.772 ms** | 0.561 | 0.640 | **3.712 ms** | **3.518 ms** | 14.737 ms |

Measured effect:

- Headline: **52.60 -> 53.34 tok/s** on the llama-compat dp4a B2 diagnostic route
  (**+1.4%**), with unchanged acceptance.
- Cycle wall: **-0.261 ms/output**.
- Draft drain: **-0.321 ms/output** (`draft_initial`), almost exactly the section this
  fix targeted.
- Verifier: unchanged within noise (**+0.056 ms/output** in this A/B).

Sync-stage rerun after the fix:

| bucket | before Q6 top-1/gather | after Q6 top-1/gather | delta |
| --- | ---: | ---: | ---: |
| MTP tok/s | 52.37 | **53.43** | +2.0% |
| `cycle_wall_ms_per_output` | 19.122 | **18.737** | **-0.385 ms** |
| `draft_initial` | 4.084 | **3.758** | **-0.326 ms** |
| `draft_run_lm_head` | 1.882 | 1.916 | +0.034 ms |
| `draft_device_topk_gather` | 0.357 | **0.001** | **-0.356 ms** |
| `draft_topk_readback` | 0.007 | 0.007 | flat |
| `target_block_verify_total` | 14.715 | **14.661** | -0.055 ms |
| `target_block_linear_attn_layers` | 9.451 | **9.422** | -0.029 ms |
| `target_block_full_attn_layers` | 3.375 | **3.367** | -0.008 ms |

This closes the obvious top-k/gather waste, but it does **not** close the llama.cpp
gap. After the fix, the sync-stage diagnostic is still **18.737 ms/output** vs
llama.cpp HIP B2 trace **14.231 ms/output**, a remaining **+4.51 ms/output**:

- draft side: `draft_initial` **3.758** vs llama **2.140** = **+1.62 ms/output**;
- verifier side: `target_block_verify_total` **14.661** vs llama **12.083** =
  **+2.58 ms/output**;
- the rest is small lifecycle/accounting plus acceptance/pass economy.

The next compat-lane target is therefore no longer device top-k/gather. It is the
actual target verifier layer cost, especially `target_block_linear_attn_layers`
(still **9.42 ms/output**) and `target_block_full_attn_layers` (**3.37 ms/output**),
plus any remaining draft lm-head/attention/FFN work that differs from llama.cpp's
MTP draft decode shape.

#### Second gap-closing fix: defer exact direct-state writes and skip unnecessary snapshots

The next cleanup targeted verifier lifecycle overhead that hipEngine was still
paying even though the block verifier already captures per-row linear states:

- In the direct-state block verifier, the linear-attention direct branch no longer
  runs the BF16-to-F32 QKV conversion used only by the non-direct prefill conv path.
- `verify_target_block(..., defer_linear_state_commit=True)` no longer copies the
  final captured Conv/GDN state back into the resident state when the caller will
  immediately commit an accepted captured row or restore/replay.
- Shared block-verifier callers now skip `_linear_state_snapshot()` when direct
  commit is exact for the block (`bulk` verifier with
  `start_position + rows < 1024`, which covers the measured B2 suite). Rollback
  still keeps the snapshot on non-exact paths.
- New diagnostic flag `--target-block-sync-stage-timings` and suite route
  `llama-compat-device-chain-dp4a-allsync` add verifier-internal sync buckets for
  attribution only.

This is not llama-only. The retained non-llama `can_block_verify` path already
uses the shared snapshot policy. A follow-up carried the same policy into the
non-llama B1 branch-safe/fused-B1 block verifier: exact direct-commit outcomes
commit captured row 1 on strict B1 accept and row 0 on reject/root-topK branch,
so the rollback snapshot is unnecessary there as well. Smoke validation for
`resident-fused-b1-block-direct-cap32k-minrows2-pmin05` passed on 2026-07-01
(`benchmarks/results/2026-07-01-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-snapshot-skip-smoke.json`);
that route remains diagnostic and below AR/default, but it now uses the same
direct-state waste policy.

Validation:

```bash
python3 -m py_compile \
  hipengine/runtime/qwen35_gguf_runner.py \
  scripts/gguf_mtp_bench.py \
  scripts/gguf_ar_mtp_suite.py \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py

PYTHONPATH=. pytest -q \
  tests/test_gguf_mtp_bench_metrics.py \
  tests/test_gguf_ar_mtp_suite.py \
  tests/test_mtp_resident_draft_device_commit.py
```

Full-suite A/B against the prior Q6 top-1/gather row, same command family:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-full.json
```

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | `target_block_verify_total` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after Q6 top-1/gather | 54.75 | 53.34 | 0.974x | 18.772 ms | 0.561 | 0.640 | 14.737 ms |
| + direct-state cleanup | 54.67 | **55.41** | **1.014x** | **18.069 ms** | 0.561 | 0.640 | **14.044 ms** |

Measured effect:

- Headline: **53.34 -> 55.41 tok/s** on the llama-compat dp4a B2 route
  (**+3.9%**), with unchanged acceptance.
- Cycle wall: **18.772 -> 18.069 ms/output** (**-0.702 ms/output**).
- Verifier: `target_block_verify_total` **14.737 -> 14.044 ms/output**
  (**-0.694 ms/output**).
- The fixed cost was not draft-side: `draft_initial` stayed flat
  (**3.712 -> 3.708 ms/output**).

Stage deltas vs the Q6 top-1/gather row:

| bucket | before | after | delta |
| --- | ---: | ---: | ---: |
| `target_block_verify_total` | 14.737 | **14.044** | **-0.694 ms** |
| `target_block_forward` | 14.590 | **13.997** | **-0.593 ms** |
| `target_block_layer_total` | 12.847 | **12.477** | **-0.370 ms** |
| `target_block_linear_attn_layers` | 9.467 | **9.185** | **-0.282 ms** |
| `target_block_full_attn_layers` | 3.380 | **3.292** | **-0.088 ms** |
| `target_block_setup` | 0.270 | **0.049** | **-0.221 ms** |
| `target_block_snapshot` | 0.090 | **0.000** | **-0.090 ms** |
| `target_block_replay_or_commit` | 0.051 | **0.042** | -0.009 ms |

Final all-sync full-suite attribution after this cleanup, diagnostic-only.  The
retained performance row remains the async full-suite `55.41 tok/s` result above;
this run intentionally synchronizes inside draft and verifier buckets, so its
`44.34 tok/s` throughput is not a performance comparison.

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-allsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-skip-snapshot-allsync-full.json
```

Full-suite synchronized buckets, ms/output:

| bucket | ms/output |
| --- | ---: |
| `target_block_verify_total` | 18.495 |
| `target_block_layer_total` | 17.083 |
| `target_block_linear_attn_layers` | **12.961** |
| `target_block_full_attn_layers` | 4.121 |
| `draft_initial` | 3.761 |
| `draft_mtp_layer_forward` | 3.664 |
| `target_block_linear_attn_norm_qkv_gate` | **3.061** |
| `target_block_linear_attn_ffn_moe_expert_gate_up` | **2.051** |
| `draft_run_lm_head` | 1.916 |
| `target_block_linear_attn_ffn_moe_expert_down` | **1.627** |
| `target_block_lm_head_sample` | 1.197 |
| `target_block_linear_attn_ssm_out` | 1.109 |
| `target_block_linear_attn_chain_gdn` | 0.937 |
| `target_block_full_attn_norm_qkv_split` | 0.914 |
| `target_block_linear_attn_ffn_moe_router` | 0.740 |

This is now the clearest operation-level target list: after semantic replication and
direct-state cleanup, the remaining verifier cost is dominated by target linear
attention projection (`norm_qkv_gate`) plus selected-MoE expert gate/up/down in the
linear-attention layers. The remaining draft cost is still mainly the MTP lm-head.

Rejected follow-up: enabling the existing row-compact selected-MoE GEMV route
(`HIPENGINE_GGUF_ROW_COMPACT_GEMV=1`) made the same all-sync smoke materially
slower: B2 **36.05 tok/s**, cycle **27.765 ms/output**,
`target_block_verify_total` **24.277 ms/output**, with the new
`target_block_linear_attn_ffn_moe_compact_gemv` bucket alone at
**8.977 ms/output**. This confirms the current selected-MoE split path is faster
for the compat B2 verifier shape; row-compact GEMV is not the missing llama.cpp
mechanism.

Updated remaining gap vs llama.cpp HIP B2 deep trace after q6top1dp4a plus
q6-only X8 selected-down:

| bucket | hipEngine compat B2 after cleanup | llama.cpp HIP B2 deep | remaining delta |
| --- | ---: | ---: | ---: |
| cycle wall / output | 16.610 ms | 14.231 ms | **+2.379 ms** |
| `draft_initial` | 3.248 ms | 2.140 ms | **+1.108 ms** |
| `target_block_verify_total` | 13.038 ms | 12.083 ms | **+0.955 ms** |

So the remaining replication work is concrete: reduce the resident draft LM-head /
top-k section and the B2 target block layer time. Simply copying the llama.cpp
high-level no-probe lifecycle has already been tested and does not make the speed
match.

The next optimization question is therefore specific: keep the no-probe
`llama-compat` semantics and reduce the measured operation costs. The older
approximate no-probe route collapsed acceptance (`56.42 tok/s`, acc/output `0.324`).
The true `llama-compat-device-chain-dp4a-q6top1dp4a` route with q6-only X8
selected-down now keeps acceptance near llama's retained row (`0.583`
acc/output) and beats its same-run AR baseline (`60.36 tok/s`, `1.101x`). This
paragraph is historical: the later resident initial-KV and shared-gate scalar-dot
fixes first moved the active compat row to `71.84 tok/s`; the later parallel
MTP attention fix moves it again to `74.39 tok/s` on the clean current-HEAD
rerun, and the draft-only dense-Q8 selector moved the unsafe direct-state row to
`75.15 tok/s`. The later direct-state lifecycle comparator supersedes that
performance row as an exact-state claim. The active apples-to-apples
llama-replication compat lane is now the no-copy directcommit natural24
cyclecap24 row: **71.52 tok/s** / **14.005 ms/output**, with zero replay rows.
The artifact filename contains `f32head`, but that retained route did not enable
the verifier-head flag; the measured verifier-head route is rejected at
**66.45 tok/s** / **15.072 ms/output**. The active route is still
slightly below the llama.cpp HIP rerun request headline (**71.91 tok/s**) even
though its stage wall is faster (**14.005 vs 14.269 ms/output**). The fixed-cycle
provenance row for the same route remains **72.23 tok/s** / **13.865 ms/output**. The serial
state-only row remains the exact
semantic-safe control at **51.85 tok/s** / **19.308 ms/output**.

#### Queued fixes, ordered by expected impact

| priority | fix | why this is next | success gate |
| ---: | --- | --- | --- |
| 1 | **Fused B1/block verifier path** | Current dp4a B5 pays `target_serial_verify_step` **6.647 ms/output** plus block verify **8.073 ms/output**. A useful implementation must preserve the B1 probe's acceptance economy while avoiding a separate full serial target pass. | **Implemented and rejected for promotion 2026-06-30.** It cuts B1 serial work but moves too much work into 2-row blocks; exact B5 is **60.40 tok/s**, below the retained exact **60.78** and dp4a **61.61** rows. |
| 2 | Proposal / row-economy comparison | Cyclecap24 shows the live economic gap: hipEngine emits **2.474 visible outputs/cycle** vs llama **2.563**, acc/output **0.596** vs **0.610**, and target rows/output **1.171** vs **1.148**. The focused trace now labels the first mismatch `bonus_token_after_full_accept`: both engines accept draft `[11, 567]`, then hipEngine samples bonus `8940` while llama.cpp samples `668`. Row-2 target probes show llama has `668` ahead of `8940` by **0.0096 logits**, while hipEngine bulk/serial/F32 diagnostics keep `8940` ahead by **0.413-0.519 logits**. | Capture llama.cpp tensor rows for this exact cycle/row and compare layer-boundary/pre-output taps; improve only with full-suite category evidence. |
| 3 | Verifier and draft regression guards | Current natural24 cyclecap24 draft drain is **2.101 ms/output** vs llama **2.141**, and verifier drain is **11.436 ms/output** vs llama **12.120**. These are closed speed buckets, not active deficits. The actual verifier-head diagnostic is rejected because it raises verifier drain to **12.501 ms/output**. | Keep all-sync/rocprof splits available after each acceptance-policy or verifier change; do not chase these unless a new run reopens a positive gap. |
| 4 | Confidence-gated no-probe policy | Historical pre-resident-initial-KV note: no-probe acc/output was **0.578**. The current natural24 directcommit compat row is **0.596** and no longer pays serial replay; confidence gating is now an acceptance/row-economy question, not a verifier wall question. | Revisit only with full-suite category evidence and proposal-trace comparison against llama.cpp. |
| 5 | Keep llama.cpp deep instrumentation aligned | The current split proved llama's verifier drain lives in `llama_process_build_draft_batch`, not raw `target_block_forward`. Keep this patch available for A/B after every major hipEngine verifier change. | Re-run llama deep trace when upstream or local diagnostic patch changes; do not compare raw async buckets. |

**Fused-B1 implementation result (2026-06-30):** added default-off
`--fused-b1-block-probe` and suite route
`resident-fused-b1-block-direct-cap32k-minrows2-pmin05`. The flag lets adaptive B1
probe cycles use one strict two-row block over `[prev, draft0]` instead of entering
the serial verifier loop. Row-state commit uses the existing exact direct-commit
block path.

| route / artifact | MTP tok/s | vs AR | acc / output | passes / output | rows / output | `target_serial_verify_step` | `target_block_verify_total` | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| retained exact B5, non-stage rowtile confirm | **60.78** | **1.113×** | 0.535 | n/a | n/a | n/a | n/a | current exact default |
| fused-B1 B5, non-stage | **60.40** | **1.107×** | 0.535 | n/a | n/a | n/a | n/a | **do not promote** |
| retained exact B5, stage-timed | 59.61 | 1.093× | 0.535 | 0.567 | 1.163 | **6.682 ms/out** | **8.157 ms/out** | baseline attribution |
| fused-B1 B5, stage-timed | 60.40 | 1.107× | 0.535 | 0.465 | 1.205 | **2.095 ms/out** | **12.447 ms/out** | serial mostly removed, block cost rises |
| retained dp4a B5, non-stage | **61.61** | **1.132×** | 0.533 | n/a | n/a | n/a | n/a | accuracy-traded ceiling still higher |

Why it fails the promotion gate:

- It does what it says mechanically: stage-timed serial rows fall from **78** to
  **25** on B5, and those remaining serial rows are p_min zero-draft AR cycles
  (`linear_draft_tokens=0`), not missed fused B1 probes.
- But it turns many B1 probes into two-row target blocks: block passes rise from
  **44** to **75**, block rows from **172** to **234**, and
  `target_block_verify_total` rises by **+4.29 ms/output**. The serial bucket falls
  by **-4.59 ms/output**, so the verifier bucket only improves by about
  **0.30 ms/output** in the instrumented run.
- The non-stage full-suite row is **60.40 tok/s**, below the retained exact
  **60.78 tok/s** and far below the accuracy-traded dp4a **61.61 tok/s**; it is not
  a retained speed win.

**Replication-lane next unit:** keep the llama-compatible no-probe B2 shape and cut
the measured compat costs directly: resident draft GPU drain first, then B2 block
verifier layer time. Confidence-gated no-probe may still be useful for the default
hipEngine policy, but it is not the current llama.cpp replication task.

#### Can we adopt a true llama.cpp mode?

Yes, as an explicit opt-in **llama-compat / accuracy-traded mode**. No, not as the
shipped exact default. The current "no-probe" hipEngine experiments should not be
over-read as a full llama.cpp clone: they tested the high-level idea (one block pass,
no B1 probe) inside hipEngine's existing policy stack, not every llama.cpp semantic.

What a true llama mode needs to replicate:

| llama.cpp piece | why it matters | current hipEngine status |
| --- | --- | --- |
| `--spec-draft-n-max 2`, `--spec-draft-p-min 0.0` lifecycle | llama drafts every cycle up to B2 unless the MTP sampler itself stops; no hipEngine p_min gate. | Implemented in opt-in `--llama-compat`; suite routes are fixed to B2 to avoid mislabeled artifacts. |
| No B1 probe / one target block verify per cycle | This removes the `target_serial_verify_step` bucket entirely. | Implemented in `--llama-compat`: disables adaptive B1 probe/fallback and forces block verify with `--target-block-min-rows 2`. |
| llama MTP context handoff (`common_speculative_process` / `pending_h` / `verify_h`) | Draft quality depends on how target verify hidden rows seed the next MTP draft. | Shifted prompt catch-up via `--mtp-context-replay` plus device-resident MTP KV is implemented in `--llama-compat`. Explicit subroutes now add prewarmed resident device-chain drafting and optional resident device seed (`pending_h`) starts. |
| llama accept/checkpoint semantics | Partial accepts restore/commit through llama's checkpoint and `common_speculative_accept` path. | hipEngine has rollback/direct-commit paths, but they are not mechanically identical. |
| q8_1 / dp4a verify + draft lm-head economy | This is part of llama's speed/acceptance economics, and it fails hipEngine's ja gate when used broadly. | Exact compat route stays precision-preserving; `llama-compat-dp4a` adds default-off `--verify-dp4a` for selected-expert verify, `llama-compat-device-chain-dp4a-q6top1dp4a` adds q8_1/dp4a Q6_K draft top-1 lm-head, and the current best `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` row also sets `--selected-down-x8-repack q6` for Q6_K selected-down. |

Implemented opt-in routes (2026-06-30):

| route | extra args | budget | purpose |
| --- | --- | ---: | --- |
| `llama-compat` | `--llama-compat` | B2 fixed | Precision-preserving closest semantic replica: B2, p_min 0, full draft vocab, shifted context replay + device KV, no B1 probe/fallback, one block verify/cycle. |
| `llama-compat-dp4a` | `--llama-compat --verify-dp4a` | B2 fixed | Same semantics plus llama-style q8_1/dp4a selected-expert verify. Accuracy-traded; ja gate failure remains expected until proven otherwise. |
| `llama-compat-device-chain` | `--llama-compat --resident-mtp-device-chain` | B2 fixed | Adds prewarmed resident device-chain drafting, mirroring llama's resident `ctx_dft` lifecycle more closely than per-depth host embedding handoff. |
| `llama-compat-device-chain-dp4a` | `--llama-compat --resident-mtp-device-chain --verify-dp4a` | B2 fixed | Accuracy-traded device-chain route; best measured compat replication row so far. |
| `llama-compat-device-chain-dp4a-q6top1dp4a` | `--llama-compat --resident-mtp-device-chain --verify-dp4a --resident-mtp-draft-q6-top1-dp4a` | B2 fixed | Base q6top1dp4a compat row. Accuracy-traded; not the exact default. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` | `--llama-compat --resident-mtp-device-chain --verify-dp4a --resident-mtp-draft-q6-top1-dp4a --selected-down-x8-repack q6` | B2 fixed | Current best compat replication row. Q6_K selected-down uses the X8 q8_1/dp4a replacement layout; Q5_K stays on T16 because q5/both smoke regressed. |
| `llama-compat-device-chain-dp4a-draftsync` | `--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --verify-dp4a` | B2 fixed | Diagnostic-only sync-stage route that attributes the resident draft GPU drain. Not a performance route. |
| `llama-compat-device-chain-dp4a-allsync` | `--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --target-block-sync-stage-timings --verify-dp4a` | B2 fixed | Diagnostic-only route that sync-splits both resident draft and target block verifier sections. Not a performance route. |
| `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync` | `--llama-compat --resident-mtp-device-chain --resident-mtp-draft-sync-stage-timings --target-block-sync-stage-timings --verify-dp4a --resident-mtp-draft-q6-top1-dp4a --selected-down-x8-repack q6` | B2 fixed | Diagnostic-only route that attributes the current q6top1dp4a+x8q6 draft and verifier sections. Not a performance route. |
| `llama-compat-device-seed-chain` | `--llama-compat --resident-mtp-device-seed --resident-mtp-device-chain` | B2 fixed | Also starts each draft from resident target `pending_h` rather than a host-copied seed. |
| `llama-compat-device-seed-chain-dp4a` | `--llama-compat --resident-mtp-device-seed --resident-mtp-device-chain --verify-dp4a` | B2 fixed | Full llama-lifecycle diagnostic: B2 no-probe, context replay + device KV, resident device seed, prewarmed device chain, and dp4a verify. |

`--llama-compat` is deliberately an override flag: if a wrapper passes conflicting
draft/policy knobs first, the bench normalizes them after parsing. The suite also
refuses non-B2 budget overrides for these routes because the child bench would force
`draft_n_max=2`; allowing a `B5` label would make the artifact misleading.

Exact route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-b2.json
```

Accuracy-traded dp4a route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-dp4a-b2.json
```

Accuracy-traded prewarmed device-chain route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full.json
```

Accuracy-traded resident device-seed + device-chain route command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-seed-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-seed-chain-dp4a-b2-full.json
```

Split-bucket attribution rerun for `draft_device_chain_drain` / `draft_topk_d2h`:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json
```

Sync-stage attribution rerun for the inside of `draft_device_chain_drain`:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-draftsync \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json
```

Measured full-suite result (2026-06-30, same model/gfx1151, stage timings enabled):

| config | AR tok/s | MTP tok/s | vs AR | cycle wall / output | acc / output | draft acceptance | passes / output | rows / output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama-compat` exact B2 | 54.76 | **51.16** | 0.934× | 19.570 ms | 0.559 | 0.635 | 0.441 | 1.322 |
| `llama-compat-dp4a` B2 (top-1 diagnostic) | 54.77 | **52.48** | 0.958× | 19.074 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-chain-dp4a` B2 | 54.71 | **52.79** | 0.965× | 18.963 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-chain-dp4a-draftsync` B2 | 54.72 | **52.37** | 0.957× | 19.122 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| `llama-compat-device-seed-chain-dp4a` B2 | 54.74 | **52.53** | 0.960× | 19.065 ms | 0.561 | 0.640 | 0.439 | 1.316 |
| prior dp4a+B1-probe B5 | 54.60 | **60.01** | 1.099× | 16.690 ms | 0.533 | 0.735 | 0.570 | 1.154 |
| llama.cpp HIP B2 trace | 52.13 | **72.12** | 1.383× | 14.231 ms | 0.610 traced | 0.805 | 0.390 | 1.148 |

Stage ms/output:

| bucket | compat exact B2 | compat dp4a B2 | compat device-chain dp4a B2 | prior dp4a+B1 B5 | llama.cpp HIP B2 | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `draft_initial` | 4.084 | 4.043 | 4.033 split / 4.028 headline | 1.943 | 2.140 | hipEngine compat's shifted-context/full-vocab B2 draft is expensive; prewarmed device-chain does not reduce steady-state draft drain. |
| `target_serial_verify_step` | 0.000 | 0.000 | 0.000 | 6.660 | 0.000 | compat successfully removes the B1 serial probe. |
| `draft_topk_readback` | n/a | 3.833 | 3.839 | 1.134 | n/a | now split: this is almost all GPU drain, not copy time. |
| `draft_device_chain_drain` | n/a | n/a | **3.830** | n/a | n/a | resident device-chain still waits on the full draft GPU work at chain end. |
| `draft_topk_d2h` | n/a | n/a | **0.008** | n/a | n/a | D2H is too small to be the missing llama gap. |
| `target_block_verify_total` | 15.164 | 14.715 | 14.714 split / 14.620 headline | 8.073 | 12.083 | the saved serial probe is paid back by a much more expensive B2 block verifier. |
| `target_block_forward` | 15.021 | 14.585 | 14.581 | 7.985 | 0.549 | llama's raw forward bucket is not comparable; most llama verify/state work is in `mtp_context_replay_append`. |
| `mtp_context_replay_append` | 0.008 | 0.008 | 0.009 | 0.000 | 11.348 | hipEngine's context replay cost is not in this bucket; its cost manifests in draft/block wall. |
| `mtp_device_kv_commit` | 0.299 | 0.296 | 0.297 | 0.000 | n/a | small but nonzero compat lifecycle overhead. |
| `cycle_wall_ms_per_output` | 19.570 | 19.074 | 19.066 split / 18.963 headline | 16.690 | 14.231 | best compat replication is still ~4.73 ms/output slower than llama's traced path. |

**Result:** copying the observable llama policy is not sufficient. Adding the next
llama lifecycle pieces also does not close the gap: prewarmed device-chain improves
the compat dp4a headline only **52.48 -> 52.79 tok/s**, and resident device seed is
slightly worse (**52.53 tok/s**). The route does remove the B1 probe and preserves
decent full-suite acceptance, but the hipEngine realization of that lifecycle is
slower than the retained B1-probe path. Compared with prior dp4a+B1 B5, compat saves
**6.65 ms/output** of serial verify, but adds roughly **+6.64 ms/output** in block
verify and **+2.09 ms/output** in draft work. Versus llama.cpp HIP B2, the best
replication row is still slower by about **4.73 ms/output**: **~1.89 ms/output** in
draft, **~2.54 ms/output** in block verify, and the rest in small lifecycle/accounting
differences plus weaker acceptance/pass economy. The residual gap is now even more
concrete: reduce the actual resident draft GPU drain and B2 block verifier cost, not
just the llama no-probe policy flag or host readbacks.

So the real answer is: there is no architectural reason we cannot add a true
`llama-compat` mode while keeping exact mode as default. The reasons not to promote
it by default are the known correctness tradeoff (dp4a ja top-1 **0.700 < 0.90**) and
the fact that the current approximate no-probe routes did not reproduce llama's
economics. The clean experiment is now implemented and measured: the exact route lands
at **51.16 tok/s**, the dp4a route lands at **52.48 tok/s**, and the best prewarmed
device-chain dp4a replication row lands at **52.79 tok/s**, all below hipEngine AR and
well below llama HIP. Semantic parity alone was not the missing piece; the gap is
implementation/backend cost in the compat draft/verifier lifecycle.

Commands used:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --budgets 5 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-exact-deep.json

PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a \
  --budgets 5 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/2026-06-30-ar-mtp-stage-timing-b5-dp4a-deep.json

PYTHONPATH=. python3 scripts/llamacpp_mtp_bench.py \
  --server-bin /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen36-35b \
  --port 8013 \
  --ctx-size 8192 \
  --gpu-layers 99 \
  --draft-max 2 \
  --mode both \
  --protocol natural \
  --max-tokens 24 \
  --server-extra-arg=--reasoning \
  --server-extra-arg=off \
  --stage-timings-jsonl benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.jsonl \
  --output benchmarks/results/2026-06-30-llamacpp-mtp-stage-timing-b2-natural24-deep.json \
  --log-dir /tmp/llamacpp-mtp-stage-timing-b2-natural24-deep-logs
```

Important: these stage windows are **diagnostic**, not a new retained benchmark
denominator. Some fields are nested by design (`target_block_verify_total` includes
snapshot/forward/accounting/replay sub-windows), so totals should be used for
attribution and ranking, not summed as disjoint wall time. The retained tok/s still
uses the existing suite protocol; `cycle_wall_*` is there to expose hidden overhead
that the legacy counters may miss.

Profiling harness (both engines, reproducible): llama HIP via `rocprofv3
--kernel-trace` on `llama-bench`/`llama-cli` (MTP path deadlocks rocprof at finalize
→ use the batched-forward proxy `llama-bench -p 4 -b 4`); llama Vulkan via
`GGML_VK_PERF_LOGGER=1` (per-op GFLOPS); hipEngine via the AR/MTP suite + rocprof.


| same model, same prompt where noted | AR (tg128) | MTP B2 (llama-cli, explain_concept prompt) |
| --- | --- | --- |
| llama.cpp **HIP/ROCm** | 51.38 | 75.4 |
| llama.cpp **Vulkan** | **62.65** | **84.6** |
| hipEngine (HIP/ROCm only) | 54.95 | 60.8 (full suite; same-prompt TBD) |

Two attributions:
1. **hipEngine's HIP kernels are FASTER than llama's HIP** (AR 54.95 > 51.38). The
   earlier "hipEngine wins AR" holds only against llama's *slower* (HIP) backend.
2. **llama's Vulkan AR (62.65) beats hipEngine's best HIP (54.95) by ~14%**, and
   Vulkan MTP (84.6) vs hipEngine (60.8). On this RDNA3.5 APU the **Vulkan shader
   compiler/driver is materially more efficient for these GEMVs than ROCm/HIP**. So
   a large part of "where we lose" is the **ROCm-vs-Vulkan backend gap**, which is
   SEPARATE from the MTP algorithm. hipEngine is HIP-only, i.e. structurally on the
   disadvantaged backend for this hardware. Closing it means either (a) a hipEngine
   Vulkan backend, or (b) raising the HIP GEMV efficiency toward Vulkan's.

The gap therefore decomposes into **(backend: HIP vs Vulkan) + (MTP uplift)** — not
uplift alone. The AR/verify analysis below is within the HIP/ROCm backend.

### Why Vulkan is faster: kernel FUSION, not raw BW (lm-head is equal on both)

Vulkan per-op timing (`GGML_VK_PERF_LOGGER=1`, AR decode) gives effective GFLOPS,
which for these memory-bound GEMVs tracks effective bandwidth:

| op (AR decode) | Vulkan | hipEngine | note |
| --- | --- | --- | --- |
| lm-head `q6_K m=248320 k=2048` | 1794 µs, **566 GFLOPS** | ~1850 µs, **~550 GFLOPS** | **EQUAL** — both saturate BW on a large contiguous GEMV |
| attn proj `q8_0 m=8192 k=2048` | 90.8 µs, 369 GFLOPS (**qkv FUSED into one m=8192 op**) | current audit: qkv already fused as one `attn_qkv` GEMV | no missing qkv-fusion lever remains |
| MoE `MUL_MAT_ID_VEC q4_K m=512 k=2048 n=8 n_expert=256` | 24.9 µs, **674 GFLOPS** (**all 8 selected experts in ONE call**) | current audit: pack8 selected MoE already consolidated with gate+up+silu fused | high-level MoE consolidation already captured |

**Finding:** on the *large* op (lm-head) the two backends are **equally BW-efficient
(566 vs 550 GFLOPS)** — Vulkan has no magic raw-bandwidth edge. The initial read was
that hipEngine lacked Vulkan's qkv/selected-expert fusion. The follow-up audit below
closed that: hipEngine already has fused `attn_qkv` and pack8 selected-MoE. So the
remaining Vulkan advantage is backend/compiler/op-shape efficiency on the small ops,
not a missing high-level fusion item in the HIP path.

**Concrete implication:** a hipEngine Vulkan backend is the clean way to capture this
backend factor. More HIP-side qkv/MoE fusion is not an available correctness-preserving
lever for the current path.

### 2026-06-30 IMPLEMENTED+TESTED: both levers are ALREADY captured by existing kernels

Acted on the two levers above (build/test, not just propose). Result: **both are
already implemented in hipEngine's HIP path; neither has remaining headroom.**

- **Fusion — already done.** Attention qkv is a single fused `attn_qkv` weight/GEMV
  (`qwen35_gguf_runner.py:1753`), not split q/k/v. The selected-expert MoE is already
  consolidated (`_launch_selected_expert_pack8_moe_pair` ids-GEMV) with gate+up fused
  (`dual`) and silu fused (`q4_k_t16_selected_dual_silu_direct`). Mirrors Vulkan's
  fused qkv + `MUL_MAT_ID`. No new fusion to add.
- **Verify amortization — built, bit-exact, but REFUTED (reverted).** Wrote a q8_0
  t16 **rowtile** (read each weight tile once, accumulate ROW_TILE rows — the lm-head
  rowtile pattern). Bit-exact vs per-row decode (rows 2-6). A/B vs the runner's
  *actual* verify kernel (`q8_0_t16_gemv_kernel` at rows=R, single launch `grid.y=R`):
  rowtile **0.93-0.94×** (rows=4: 31.0 vs 28.7 µs; rows=6: 40.3 vs 38.0 µs). The
  existing `grid.y` kernel already amortizes via **occupancy** — at rows=R it launches
  R× more blocks (better GPU utilization on these small weights) than the rowtile's
  single block/tile. The rowtile only beats *naive 4×-separate-launches* (1.36-1.58×),
  which the runner doesn't do. Right tool only for the huge lm-head (already shipped).
  Not landed.

**Net:** hipEngine's HIP kernels already capture both fusion and verify amortization
(the dense GEMV at rows>1 is already occupancy-amortized: `q8_0_t16_dual_split` 141 µs
at rows=1 → 220 µs at rows=4 = 1.56× for 4× rows = 2.5× cheaper per row). The residual
MTP gap is therefore the **Vulkan-vs-HIP backend** (which hipEngine can't close without
a Vulkan backend) plus llama's dp4a precision. No remaining correctness-preserving
HIP-kernel lever for the AR/verify dense path.

### AR decode (within HIP): hipEngine beats llama HIP; gap to Vulkan is backend

| AR decode (single-token), same model | llama.cpp HIP | hipEngine GGUF |
| --- | --- | --- |
| Wall tok/s (`llama-bench tg128` / hipEngine suite) | **51.38** | **54.95** |
| GPU kernel ms/token (rocprof kernel-trace) | 17.26 | ~18 |
| Kernel launches / token | **1632** | **762** |
| Dominant kernel(s) | `mul_mat_vec_q` (dp4a) **76.5%**, one unified GEMV for attn+MoE+lm-head; `quantize_q8_1` 2.2% | specialized EXACT kernels: q8_0 attn proj ~42% (`q8_0_t16_dual_split`+`_gemv`+`_triple`), q4_k MoE ~21%, q6_k lm-head ~9.6%, GDN ~8% |
| Bound by | host launch overhead (1632 small dp4a launches → ~2.2ms host exposed on top of 17.26ms GPU = ~19.5ms wall) | GPU-bound, host hidden (762 larger exact launches; 18.2ms wall ≈ GPU time) |

**Finding:** hipEngine's AR decode is **faster** than llama's (54.95 vs 51.38).
hipEngine uses fewer, larger *exact* kernels that are GPU-bound (host hidden);
llama uses a single highly-optimized *dp4a* `mul_mat_vec_q` for 76.5% of decode but
issues 2× the launches, exposing ~2.2ms/token of host overhead. So the base decode
is not where hipEngine loses — **the entire MTP gap (60.8 vs 67.3 tok/s) is in the
speculative machinery / uplift** (llama 1.342× vs hipEngine 1.114× over their own
AR). This redirects the investigation from AR GEMVs (we win) to the verify/draft
economics. (Method: `rocprofv3 --kernel-trace` on `llama-bench -p 0 -n 64 -r 1`
and the hipEngine AR step loop; normalized per-token.)

**Next:** profile llama's MTP verify (batched B+1 rows) — hypothesis: llama's
batched dp4a `mul_mat_q`/`mul_mat_vec_q` amortizes weight reads across verify rows
more cheaply *relative to its slower AR* than hipEngine's per-row exact verify does
relative to its faster AR. That relative-amortization is the suspected uplift lever.

### Verify mechanism: initial hypothesis, then refuted by direct test

llama-cli MTP B2 on the explain_concept prompt = **75–78 tok/s** (uplift ~1.47–1.52×
over its 51 AR; even higher than the server-suite 67.3 because this is a favorable
English prompt). rocprof of the MTP path itself DEADLOCKS at finalize (the draft-mtp
second-context/queue setup; not size- or graph-dependent), so the verify was profiled
via its equivalent **batched B+1-row forward** (`llama-bench -p 4 -b 4 -ub 4`, which
finalizes cleanly):

| verify-shape (4-row) forward | llama.cpp HIP | hipEngine initial read |
| --- | --- | --- |
| matmul kernels | `mul_mat_vec_q_moe` 40.5% + `mul_mat_vec_q` 33.8% (dp4a vec; batch is a grid dim → **each block reads a weight tile ONCE, computes all B+1 rows** = weight-read amortized across verify rows) | `q8_0_t16_dual_split` etc. with `blockIdx.y = row` → **a separate block per row, weight re-read B+1×** (NO cross-row amortization). The only amortized hipEngine path (WMMA prefill) is SLOWER at rows=4 (56.9 vs 42.3 ms) due to tile-setup overhead. |

**2026-06-30 correction:** this was the right hypothesis to test, but it is no longer
an open lever. The exact q8_0 rowtile was built and bit-exact for rows 2-6, then
lost to the existing rows kernel: rows=4 **31.0 vs 28.7 us** (0.93x), rows=6 **40.3
vs 38.0 us** (0.94x). The current `grid.y=R` kernel already gets the useful
multi-row benefit through occupancy; rowtile underutilizes the small q8_0 weights.
Only the huge shared lm-head benefits, and that rowtile is already shipped.

**Current attribution:** llama's verify is cheaper because it runs the whole
speculative economy in dp4a/q8_1 and can afford no-probe full-block attempts.
hipEngine's exact verifier is GPU-bound and already at its exact-precision floor.

---

# GGUF MTP llama.cpp Parity Trace (history)

- Date: 2026-06-29 (Goal — Part 1 set: target-verify amortization is the sole remaining gap; acceptance shown already at llama.cpp parity via cap32k-recover; shootout order inverted, verify-wall promoted to P0; llama.cpp parity shootout matrix update; B1-probe/block-direct/cap32k AR-beating route retained; bulk row-1 direct-commit exactness diagnostic; native row-1 direct-commit diagnostic; context replay + device-seed route rejection; device-seed + draft-KV route rejection; resident draft p_min strict-block rejection; direct verifier row-state commit diagnostic; resident device hidden-seed diagnostic; hybrid strict-block cap32k rejection; cap32k recovery full-suite diagnostic; strict-context route added; deferred hidden-copy rejection; device top-k40 rejection; resident top-k40 full-suite update; production verifier/full-suite update; systemic workbench update; performance-path update 2026-06-27; correctness-solved update 2026-06-26; original trace 2026-06-25)
- Branch: `mtp-gguf`
- Hardware for all runtime numbers below: **gfx1151 / AMD Radeon 8060S (Ryzen AI Max+ 395)**, not the default W7900. Numbers state their scope; the current authoritative MTP numbers are full-suite AR/MTP suite rows.
- hipEngine source baseline for the current performance review: `cfb584615b801ce0be7f622ea695327950018f74`
- llama.cpp checkout used for source/runtime evidence: `6e9007ae61f4e994c27484759caac6ef2aa32b30`

## 2026-06-29 — HANDOFF: current state, per-stage gap, tried levers, how to continue

This section is the current, authoritative snapshot. The dated sections below it
are the historical record of how we got here; where they conflict with this
section, **this section wins** (several older numbers were measured with stale
tooling or a since-corrected methodology — flagged inline below).

### TL;DR

- **Correctness is solved.** Target AR first-token + 12-token greedy trace and
  strict B3 draft acceptance match llama.cpp on the merge-sort prompt.
- **AR decode (no MTP) is already FASTER than llama.cpp's AR.** Current eager
  resident path measures **~55 tok/s** (54.65 tok/s, code prompt, gfx1151, this
  session, `scripts/gguf_ar_mtp_suite.py --scope smoke`). llama.cpp's retained
  full-suite HIP AR reference is **50.13 tok/s**.
- **MTP is still the parity gap, but the same-protocol AR-beat gate is now
  closed.** The retained default full-suite route
  `resident-b1-probe-block-direct-cap32k` measures **AR 54.59 tok/s; best MTP B3
  56.54 tok/s = 1.0356× AR**, `apple_to_apple_ok=true`, `mtp_beats_ar=true`.
  It combines a cheap strict B1 cap32k probe with direct-commit B3 block
  verification after a full B1 accept. B3 acceptance is **40/140 = 0.286
  accepted/output**, draft acceptance is **0.645**, target layer passes drop to
  **0.779/output**, direct commit rows are **15**, and replay rows are **0**.
  The previous best retained diagnostic was B1 **52.08 tok/s = 0.9540× AR** via
  resident device hidden seed. llama.cpp's retained full-suite reference is
  still **67.29 tok/s at B2 = 1.342× its AR**. hipEngine now beats its own AR,
  but needs about **+19% relative tok/s** from 56.54 tok/s to match that
  llama.cpp row.
- **Current next goal:** close the llama.cpp MTP parity gap by improving the
  retained `resident-b1-probe-block-direct-cap32k` family, not by reworking AR
  kernels. `Qwen35GGUFMTPContext` already covers the
  `process_verifier_rows()`/`draft()`/`accept()` seed lifecycle shape, and direct
  row-state commit has proven exact enough to retain a B3 speed route. The
  remaining llama.cpp patterns to adopt are the target/draft memory economics:
  keep `pending_h`/`verify_h`-style rows resident across the target batch,
  promote B2/B3 block verification without serial fallback waste, and lift
  accepted/output while keeping target layer passes below the current
  **0.779/output**. Success for the next goal is a full-suite artifact that
  approaches or beats llama.cpp's **67.29 tok/s B2** row under the same
  no-gaming category protocol.
- **There is no single bandwidth-starved GEMV to fix.** Measured cold-DRAM
  (MALL-defeated): dense Q8_0 c=1 GEMV ~51–70% of peak, selected-MoE GEMV
  ~70–80%. Every kernel micro-lever (dp4a, split-K, fusion, MoE-graph, cache
  hints) is real in isolation and **flat e2e** (table below).
- **Verifier host-vs-GPU split is resolved for the current suite route.** Fresh
  GGUF serial-target rocprof (`scripts/gguf_mtp_verifier_rocprof.py`, 12
  measured target steps, post no-logits cleanup) shows **18.63 ms host wall /
  16.56 ms kernel time per target step = 89% kernel time**, ~709 launches/step.
  A same-day rerun after the capped/short-block probes remains the same shape:
  **19.37 ms host / 16.95 ms kernel = 87.5% kernel time**, **708.9
  launches/step** (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-rerun.json`).
  A current 8-step rerun is unchanged: **19.03 ms host / 16.76 ms kernel =
  88.0% kernel time**, **708.5 launches/step**, with dense Q8_0 GEMV **48.9%**
  and selected-MoE GEMV **24.0%** of kernel time
  (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-current.json`).
  A post bulk-row1-exactness rerun remains the same shape: **18.65 ms host /
  16.75 ms kernel = 89.8% kernel time**, **708.6 launches/step**, dense Q8_0
  GEMV **49.0%** and selected-MoE GEMV **24.6%**
  (`benchmarks/results/2026-06-29-gguf-mtp-verifier-rocprof-post-bulk-row1.json`).
  The retained
  `resident-serial-fallback` route is GPU/weight-streaming bound, not
  host-launch-bound.
- **New standard measurement:** `scripts/gguf_ar_mtp_suite.py` produces ONE
  apple-to-apple AR-vs-MTP artifact under an enforced config (see "How to
  continue").

### Where hipEngine still falls short vs llama.cpp

The milestone is real: hipEngine GGUF MTP now beats the same-run hipEngine AR
baseline. The remaining parity target is llama.cpp's MTP uplift and category
coverage, not hipEngine AR speed.

| Dimension | hipEngine current default | llama.cpp retained reference | Gap / interpretation |
| --- | --- | --- | --- |
| Best total MTP throughput | B3 **56.54 tok/s** | B2 **67.29 tok/s** | llama.cpp is **+10.75 tok/s / +19.0%** faster in absolute decode throughput. |
| AR-normalized uplift | **1.0356x AR** | **1.3423x AR** | llama.cpp gets **+29.6%** more uplift relative to its own AR. |
| Accepted/output at speed winner | B3 **0.286** (`40/140`) | B2 **0.598** (`3064/5120`) | hipEngine accepts about **2.1x fewer** draft tokens per visible output. |
| Draft acceptance at comparable B3 | B3 **0.645** | B3 **0.660** | Per-attempt B3 quality is close; the bigger miss is how often useful B2/B3 drafting is attempted and retained. |
| Target pass amortization | B3 measured **0.779 target layer passes/output** | B2 inferred **0.402 target batches/output** from `1 - accepted/output` | hipEngine still streams target layers about **1.9x** more often per output token. llama.cpp does not expose layer-pass counters in the retained artifact, so this is an inference from accepted/output. |
| Category coverage | Code B3 wins (**57.55 tok/s**, **0.500 accepted/output**); `general_en`, `general_ja`, and `mixed_ja_en` have **0 accepted drafts** under the retained route | Code winner B3 **72.59 tok/s**, non-code winners are B2: `general_en` **63.83**, `general_ja` **62.25**, `mixed_ja_en` **67.27 tok/s** with **0.56-0.60 accepted/output** | hipEngine's current route is effectively a code-category win plus near-AR fallback elsewhere. llama.cpp's B2 works across all categories. |
| Budget shape | B1/B2 remain below AR; B3 is the only winning budget; B4/B5 regress | B1/B2/B3 all beat AR strongly; B2 is fastest, B5 maximizes acceptance | The next target is a robust B2/B3 policy, not deeper B4/B5 drafting. |

Bottom line (corrected by the artifact evidence in "Goal — Part 1" below):
**acceptance is already at llama.cpp parity on every category; the only
remaining gap is target-verify amortization.** The retained default route's
non-code zeros are a *policy artifact* (`--adaptive-ar-fallback` stops drafting
after one miss), not a draft-quality deficit — `cap32k-recover` already matches
llama.cpp's per-category acceptance. The current route pays too many target
layer passes per visible token, and so does every high-acceptance route we have.

### Goal — Part 1 (HIGHEST PRIORITY): close the target-verify amortization gap

**STATUS 2026-06-30 — CLOSED (banked) by owner decision.** After an exhaustive,
measurement-backed investigation (every uplift lever tried/refuted, AR multiplier
profiled, dp4a verify ~1.13x AND dp4a AR == exact AR measured, verify on its fast
path, correctness validated bottom-up incl. a new mtp_dense_attn_f32 gate, baseline
audited), llama's absolute **67.3 tok/s was shown unreachable on hipEngine in ANY
precision regime** within the correctness guard: it is a property of llama's
slower-AR (50.1) x higher-uplift (1.342x) profile, which hipEngine's faster-AR
(54.95) x exact-precision (1.114x) profile cannot reproduce. The retained, shipped
result is the bit-exact Q6_K T16 rowtile lm-head kernel: GGUF MTP **1.0534x ->
1.1134x AR (60.8 tok/s = 90.3% of llama's 67.3)**, beating llama on AR and on the ja
correctness gate. **Owner chose to bank this exact-precision win** rather than relax
the ja gate for dp4a (which reaches only ~62 tok/s anyway) or fund a speculative,
multi-session AR-decode kernel-R&D project (the only correctness-preserving path that
could raise the absolute number, with no high-confidence optimization identified, on
a path where hipEngine already beats llama). The parity goal is therefore closed as
**structurally bounded at the exact-precision design point**, not as an open
engineering gap. See the 2026-06-30 entries in the "Bottom line" section and WORKLOG.

The original P0 framing below is retained for history.

This is the first part of the llama.cpp-parity goal. Resolve these P0
determinations **before** running any S1-S3 policy probe. The evidence below
re-frames the gap and is the reason the shootout order in the next section is
inverted from how it was first written.

**Evidence that re-frames the gap (from the 2026-06-29 full-suite artifacts):**

1. The retained default route's non-code "0 accepted" is drafting being switched
   off, not bad drafts. Per-category for
   `2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`: code B3
   `drafts=56, accepted=40`; `general_en`/`general_ja`/`mixed_ja_en` each
   `drafts=2, accepted=0` — i.e. two attempts, then `--adaptive-ar-fallback`
   runs pure AR for the rest. The headline **1.0356x AR is a code-only win
   averaged up**, not a cross-category win.
2. **Acceptance is already solved.** A route that keeps drafting
   (`cap32k-recover`, child of
   `2026-06-29-ar-mtp-suite-full-cap32k-recover.json`) matches llama.cpp B2
   per category:

   | Category | hipEngine `cap32k-recover` acc/out | llama.cpp B2 acc/out |
   | --- | ---: | ---: |
   | general_en | 0.608 | 0.576 |
   | general_ja | 0.459 | 0.563 |
   | mixed_ja_en | 0.615 | 0.599 |

   Yet `cap32k-recover` measures **~0.95x AR (below AR)**. So hipEngine already
   drafts as well as llama.cpp on every category and **still cannot turn that
   acceptance into a speedup**. The bottleneck is cost per visible token, not
   acceptance.
3. The cost is target-verify amortization. hipEngine's best is **0.779 target
   layer passes/output** (one near-full target weight stream per ~1.3 visible
   tokens); llama.cpp's fused 4-token verify graph (~9 ms) implies **~0.25-0.40
   passes/output**. AR is already faster than llama.cpp's AR, GEMV is near-peak
   BW, draft acceptance is matched — **target-verify amortization is the only
   structural advantage llama.cpp has left.**

**P0 determinations (the highest-priority list):**

| ID | Determination | What to measure / decide | Done when |
| --- | --- | --- | --- |
| P0.1 | Amortization ceiling | Compute the tok/s a single fused B-token target verify would yield at `cap32k-recover` acceptance: hold acc/out fixed, drop `target_verify_layer_passes_per_output` from 0.779 to ~0.25-0.40, and project total tok/s. Confirms the lever is sufficient to reach ~67 tok/s before building it. | A back-of-envelope + 1 measured block-verify route row showing projected tok/s ≥ llama.cpp B2 at matched acceptance. |
| P0.2 | ~~Unblock the fused multi-token block verifier~~ **CLOSED 2026-06-30 — REFUTED, do not build.** | The premise (host-launch floor → collapse 875 launches via graph capture / GDN-fix / C-loop) was the OLD serial route. The current block verify is **GPU-kernel-BOUND (38.1 ms GPU / 42.4 ms wall, only 10.2% host exposed; see 2026-06-30 correction below)**. Graph capture / C-loop cap at ≤10% and ROCm 7.x re-pays per-node (M12.1). The GDN-corruption fix would be wasted effort. | n/a — closed. Remaining gap is GPU compute, only cuttable by dp4a (fails ja gate) or FLOP/quality loss. |
| P0.3 | Re-baseline the verify work on a keep-drafting route | Stop using the code-only `b1-probe-block-direct-cap32k` as the input for verify-wall work; use `cap32k-recover` (already high acc/out, all categories, ~0.95x AR). It already satisfies the old S5 precondition ("good acc/out, poor tok/s"). | The shootout scoreboard records `cap32k-recover` as the verify-wall starting baseline with per-category acc/out. |

Success for Goal — Part 1: a full-suite artifact whose **best budget keeps
`cap32k-recover`-class per-category acceptance AND drives target layer
passes/output toward ~0.4 or below**, lifting non-code budgets above AR. Only
after that lands do the S1-S3 acceptance/policy probes below become worth
running — until then they will reproduce `cap32k-recover` (acceptance up, tok/s
pinned at ~0.95x AR).

#### P0 RESULTS (2026-06-30, gfx1151) — measured, and they reframe the lever

**P0.1 block-verify cost model (measured).** `scratchpad/p01_block_cost_probe.py`
times `verify_target_block` at fixed sequence position (snapshot/restore), bulk
mode + repack, realistic tokens:

| call | rows | ms | x c1 |
| --- | --- | ---: | ---: |
| c1 step (AR) | 1 | 18.9 | 1.00 |
| block B1 | 2 | 30.0 | 1.58 |
| block B2 | 3 | 36.6 | 1.93 |
| block B3 | 4 | 43.5 | 2.30 |
| block B5 | 6 | 57.3 | 3.03 |

Fit: `block(rows) ≈ 16.7 + 6.82·rows ms`. The block verifier **already does true
single-weight-stream amortization** (one Python layer loop; dense weights read
once) — it does **not** need graph capture. But only ~60% of per-step cost
amortizes: the marginal **6.82 ms/row** decomposes (via `advance_state_only`,
which skips lm-head) into **5.60 ms MoE expert over-read + attn compute** and
**1.23 ms lm-head**. This matches the decode rocprof split (dense GEMV 47%
amortizes; MoE 26% + lm-head 10% are paid per row). **The per-row cost is paid on
every *attempted* row, including rejected drafts** — that waste, not the pass
count, is the bottleneck.

**P0 acceptance (measured, decisive).** Route
`resident-strict-block-direct-nofallback` (strict top-1 + block verify + direct
commit, **no AR fallback so it keeps drafting**), `--scope full`
(`scratchpad/p01-strict-block-nofallback-full.json`), per-category best-budget
acc/out vs llama.cpp B2:

| Category | hipEngine strict-top-1 | llama.cpp B2 | Verdict |
| --- | ---: | ---: | --- |
| code | 0.64 (B3) | 0.627 | match |
| general_en | 0.60 (B3) / 0.556 (B2) | 0.576 | match |
| mixed_ja_en | 0.592 (B2) | 0.599 | match |
| general_ja | 0.394 (B3) | 0.563 | lags (Japanese only) |

**The "0 accepted on non-code" in the retained default was entirely
`--adaptive-ar-fallback` quitting after 2 drafts — not draft quality.** Under
identical strict-top-1 greedy (which is exactly what llama.cpp uses; its
`accept()` only reseeds `pending_h`), hipEngine matches llama.cpp acceptance on 3
of 4 categories. Confirmed: llama.cpp's root acceptance is strict argmax, so
hipEngine's `--root-topk-accept 40` relaxation is **not** apple-to-apple greedy
and is not the parity path; strict top-1 is.

**Why the strict-keep-drafting route is still 0.77× AR at B3** (and the reframed
levers): target time/output = `passes/out 0.418 × 43.5 ms ≈ 18.2 ms` ≈ a full AR
step, plus draft. Two structural costs, both fixable:

1. **Block verify is gated to B≥3** (`can_block_verify` needs
   `len(draft_tokens)+1 ≥ ssm_conv_kernel = 4`), so B1/B2 fall to serial
   (passes/out = 1.0, no amortization), and B3 must attempt **4 rows** even when
   ~2.4 are accepted — paying the 6.82 ms/row over-read on ~1.6 wasted rows/cycle.
   Lever: enable block verify at B1/B2 (2–3 rows).
2. **Draft cost ≈ 4.4 ms/depth** (backed out: total 23.7 ms/out − 18.2 ms target
   = 5.5 ms/out ÷ ... ≈ 4.4 ms/draft step), vs llama.cpp's NextN head ~1.5 ms.
   At B3 that is ~13 ms/cycle of draft. Lever: cut draft cost.
3. **general_ja draft quality** (0.39 vs 0.56) — the one true acceptance gap.

**Reframed P0.2:** the lever is **not** a new fused verifier or graph capture
(amortization already works). It is: (a) allow block verify at B1/B2, (b) reduce
the per-row block over-read (MoE+lm-head) and/or the draft cost, (c) replace
`--adaptive-ar-fallback` with a keep-drafting policy now that acceptance is known
good, (d) close general_ja draft quality. The cost model says: at the measured
block structure with cheap drafts and the measured acceptance, B2 block verify
reaches ≈ AR–1.1× today and clears llama parity once the per-row over-read or
draft cost drops.

#### P0 RETAINED WIN + settled conclusion (2026-06-30)

**Retained:** `--target-block-min-rows 2` promoted to the default route
(`resident-b1-probe-block-direct-cap32k-minrows2`). Full suite: best **B2 56.8
tok/s = 1.0399× AR** (confirm 1.0385×), beating the prior B3 1.0356× default. B2
moved from serial **0.9845×** to block-amortized **1.0399×** (+5.6%); B3
unchanged. Exact (bit-exact vs serial-exact rows 2–3), `apple_to_apple_ok=true`.

**Policy space is now exhausted — acceptance is not the lever.** Two further
full-suite diagnostics settle it:
- Keep-drafting (no fallback, cheap B1 probe) **restored** non-code acceptance
  (en/mixed 0.41, B2 acc/out 0.482 vs default 0.265) yet tok/s **fell to 1.007×**.
- A larger draft cap (98304) was **worse** (1.0276×): costlier drafts, fallback
  still latches.

This confirms P0.1 at the route level: **raising acceptance does not raise tok/s
while per-token verify cost (block over-read 6.82 ms/row + probe) eats the gain.**
The selective-fallback default is faster because it skips verify work on low-yield
prompts. The aggregate is dragged below the amortization threshold by
**general_ja** (draft acc 0.167 capped / ~0.5 full vs llama **0.563**).

**The remaining gap to llama 1.34× is now kernel/model work, not benchmark policy:**
1. **general_ja draft quality + coverage** (highest leverage) — cap32k halves ja
   draft acc (Japanese token IDs > 32768); full vocab ~0.5 (vs llama 0.56) but
   ~4 ms/depth. Needs a cheap full-vocab or CJK-covering draft.
2. **Draft lm-head cost** — full vocab ~3 ms (reads 638 MB Q6_K lm-head);
   cap32k ~0.7 ms but drops CJK. A smaller-quant/shortlist draft lm-head that
   preserves CJK lets ja/mixed use full coverage cheaply.
3. **Per-row block over-read** 5.6 ms/row (MoE distinct-expert, top-8/256) — the
   same constraint llama faces; only a more BW-efficient small-batch MoE verify
   GEMV reduces it.

The S1–S3 acceptance/policy probes in the shootout below are therefore
**confirmed dead-ends** for tok/s (acceptance restored, speed flat/down); skip
them and go straight to the three kernel/model levers above.

**Quantified roadmap (where each lever lands).** Cost model fit to the measured
block structure (`block(rows) ≈ 16.7 + 6.82·rows ms`, c1 = 18.9 ms) and llama's
own B2 (block(3)≈34 ms, draft ~1.5 ms, acc/out 0.598 → 14.86 ms/out = 1.34×):
- **Today:** B2 1.0399× (code-only contribution; en/mixed/ja ≈ AR).
- **+ general_ja draft quality to ~llama (0.56) realized cheaply:** aggregate
  acc/out ≈ 0.58 → B2 ≈ (36.6 + ~1.4)/2.4 ≈ 15.8 ms/out ≈ **1.16×**. This is the
  single highest-leverage lever. Blocker: ja full-vocab draft is ~0.5 draft_acc
  but only 0.13 acc/out when escalated (the chain collapses) AND full-vocab draft
  costs ~4 ms — so it needs BOTH cheaper full-vocab draft AND a draft-quality fix.
- **+ block ~llama (34 vs 36.6 ms) via a more BW-efficient small-batch MoE verify
  GEMV:** closes the rest toward **~1.34×**. (WMMA confirmed slower than
  gemv-decode here; gemv-decode is already near-peak, so this is hard.)
Net: ~1.16× is reachable with the draft levers; the last ~1.16→1.34× is the
hardware-limited MoE verify GEMV. Each is a correctness-gated kernel/model
sub-project (new kernel ⇒ RED test + `kernels/cpu_reference/` gate), not a
benchmark-policy change.

### Next shootout matrix

> **Order note (2026-06-29):** the S5 precondition ("a route with good
> accepted/output but poor tok/s") is **already met** by `cap32k-recover`, so
> the target-verify-wall work in S5 is promoted into "Goal — Part 1" above and
> runs **first**. S1-S3 are demoted: they are acceptance/policy reshuffles
> inside a Pareto frontier already known to sit at ≤ AR on non-code, and should
> only run after the verify wall drops.

Every row below is a full-suite shootout candidate, not a single-prompt probe.
Run `scripts/gguf_ar_mtp_suite.py --scope full` (with a named route for variants)
and compare against both the current hipEngine default artifact and the retained
llama.cpp matrix.

For this shootout, the retained evidence must include category rows. If the
compact suite artifact still records only aggregate `mtp_by_budget`, either copy
the `child_artifacts.mtp_category` summary into `benchmarks/results/` or extend
the suite artifact before promoting a result.

Current hipEngine baseline:
`benchmarks/results/2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`
(B3 **56.54 tok/s**, **1.0356x AR**, **0.286 accepted/output**,
**0.779 target layer passes/output**).

llama.cpp target:
`benchmarks/results/2026-06-22-llamacpp-35b-mtp-category-off-b1-b5-gfx1151.json`
(B2 **67.29 tok/s**, **1.3423x AR**, **0.598 accepted/output**).

| ID | Candidate | Priority | Hypothesis | Required evidence | Promote / reject rule |
| --- | --- | --- | --- | --- | --- |
| S5 | Target verifier wall reduction (fused B-token verify) | **P0 — do first (see Goal — Part 1)** | One target weight stream verifies the whole `[prev]+drafts` block, cutting target layer passes/output from 0.779 toward llama.cpp's ~0.25-0.40 at unchanged acceptance. | rocprof + full-suite row for a `cap32k-recover`-based block-verify route, reporting per-category acc/out and target passes/output. | Promote if full-suite best beats AR on non-code budgets while keeping `cap32k-recover`-class acceptance; this is the parity lever. |
| S0 | Current default rerun | After P0 | Establish noise band for `resident-b1-probe-block-direct-cap32k` before changing policy. | Full-suite total and category rows for B1-B5; confirm B3 stays around **56.5 tok/s** and **1.03x AR**. | Baseline only. Do not retune from a single rerun unless it reproduces the retained shape. |
| S1 | Non-code rescue after zero strict probe | After P0 (will reproduce `cap32k-recover` until the wall drops) | Keep the code-path B1 probe + B3 direct block, but when a category/prompt gets zero strict accepts, fall back to a cheap root-topK/cap32k B1/B2 route instead of pure AR. | `general_en`, `general_ja`, and `mixed_ja_en` accepted/output must move from **0.000** toward llama.cpp B2's **0.56-0.60** without lowering code B3 below current. | Promote only if full-suite best beats **56.54 tok/s** and no non-code category remains at zero accepted drafts. |
| S2 | B2 direct-block promotion | After P0 | llama.cpp is fastest at B2, so test a direct-commit B2 verifier after the cheap B1 probe rather than jumping to B3. | B2 total tok/s, accepted/output, discarded rows, direct commit rows, target layer passes/output. | Promote if B2 beats the current B3 row or materially raises accepted/output with no total tok/s regression. |
| S3 | B3 promotion threshold sweep | After P0 | The current route's B3 win is code-heavy; try stricter/looser promotion criteria that preserve cheap cap32k drafting but increase safe block use on non-code prompts. | Per-category accepted/output and target passes/output, not just aggregate tok/s. | Keep only if non-code accepted/output rises and aggregate B3 remains above AR and above the current baseline. |
| S4 | llama.cpp lifecycle parity route | After P0 | Re-test context replay + device MTP KV + resident `pending_h`/`verify_h` only with exact row-state commit and prompt catch-up aligned; earlier dense-KV routes collapsed acceptance. | One artifact with per-category acceptance plus a narrow trace showing draft context parity on at least one non-code prompt. | Promote only after full-suite acceptance improves; otherwise record as rejected lifecycle evidence. |

Fill the shootout scoreboard with these columns for every retained or rejected
attempt:

| Candidate | Best budget | Total tok/s | vs AR | Code acc/out | General EN acc/out | General JA acc/out | Mixed acc/out | Target passes/output | Direct commits | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default + min-rows 2 (2026-06-30) | B2 | 56.8 | 1.0399 | — | — | — | — | 0.794 | — | **RETAINED default** (B2 0.9845→1.0399 via 3-row block) |
| Prior default (cap32k) | B3 | 56.54 | 1.0356 | 0.500 | 0.000 | 0.000 | 0.000 | 0.779 | 15 | superseded (code-only win) |
| `cap32k-recover` (P0.3 verify-wall input) | B1 | ~51.7 | ~0.948 | 0.640 | 0.608 | 0.459 | 0.615 | ~1.0 | 0 | acceptance solved, serial (no block) |
| llama.cpp reference | B2 | 67.29 | 1.3423 | 0.627 | 0.576 | 0.563 | 0.599 | inferred 0.402 | n/a | target |

### Measurement reset — what to distrust in the history below

1. **"1.9× = selected-GEMV bandwidth" is RETRACTED.** It rested on a microbench
   that reported dense Q8_0 at ~20% of peak. That was an 8× byte-count bug
   (Q8_0 T16 block spans 32 k-values, not the 256 K-quant super-block) compounded
   by the 32 MB MALL caching the looped weight buffer. Corrected
   (`scripts/gguf_q8_0_dense_bw_microbench.py`, >2×-MALL weight pool): dense Q8_0
   is ~51–70% of peak. See `docs/ROOFLINE-gfx1151.md` §6.6.
2. **The "verifier is ~50/50 host-dispatch-bound (875 launches / ~54 ms host
   floor)" diagnostic is superseded.** It predates #9 and the current suite
   route. Re-measurement on current code with
   `scripts/gguf_mtp_verifier_rocprof.py` shows the retained
   `resident-serial-fallback` target verifier is GPU-bound after the no-logits
   cleanup: 18.63 ms host wall / 16.56 ms kernel time per target step (89%
   kernel share), ~709 launches/step. A 2026-06-29 rerun measured 19.37 ms host /
   16.95 ms kernel (87.5% kernel share), 708.9 calls/step, with dense Q8_0 GEMV
   48.8% and selected MoE GEMV 24.7% of kernel time. A current artifact measured
   19.03 ms host / 16.76 ms kernel (88.0% kernel share), 708.5 calls/step, with
   dense Q8_0 GEMV 48.9% and selected MoE GEMV 24.0% of kernel time. A post
   bulk-row1-exactness artifact measured 18.65 ms host / 16.75 ms kernel (89.8%
   kernel share), 708.6 calls/step, with dense Q8_0 GEMV 49.0% and selected MoE
   GEMV 24.6% of kernel time. The
   pre-cleanup call-site profile was 18.99 ms host / 16.68 ms kernel with unused
   full-logits D2H.
3. **The `--true-ar-baseline-json` apple-to-apple path is BROKEN.** Since #8
   retired the HIP decode graph, the production AR path emits `decode_path:
   eager_step`, but `gguf_mtp_category_bench.py`'s `TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   (and a parallel speed-claim contract + tests) still demand the retired
   `graph_replay`. So that attach rejects every current AR baseline. The new
   suite sidesteps it (computes the ratio itself); the contracts need a proper
   eager-path fix — tracked in `docs/REFACTOR.md`.

### Per-stage gap vs llama.cpp (AR + MTP pipeline)

Superseded by the final stage ledger at the top, but retained here in the historical
section with current numbers instead of stale single-prompt estimates.

| Pipeline stage | hipEngine | llama.cpp HIP | Gap / status |
| --- | --- | --- | --- |
| Target AR decode (c=1) | **54.95 tok/s**, ~18.2 ms/tok, 762 launches/tok | 51.38 tok/s, 17.26 ms GPU plus ~2.2 ms exposed host, 1632 launches/tok | **hipEngine faster**; AR is not the MTP gap. |
| AR kernel mix | q8_0 attention **42%**, q4_K MoE **21%**, q6_K lm-head **9.6%**, GDN **8%** | `mul_mat_vec_q` dp4a **76.5%**, `mul_mat_vec_f` **5.8%**, `quantize_q8_1` **2.2%** | Different precision/layout regimes; no hidden hipEngine AR deficit. |
| Large lm-head | ~1850 us, **~550 GFLOPS** | Vulkan comparison: 1794 us, **566 GFLOPS** | Large GEMV bandwidth is parity-class. |
| Current block verify | rows=4 **42.40 ms wall**, **38.08 ms GPU**, **875 launches**, **10.2% host exposed** | MTP path deadlocks `rocprofv3` finalize; 4-row proxy matmuls: `mul_mat_vec_q_moe` **40.5%** + `mul_mat_vec_q` **33.8%** | hipEngine verify is GPU-bound; graph/launch collapse is not the lever. |
| Verify GPU breakdown | q8_0 attention **32.7%**, GDN **16.1%**, selected MoE **25.9%**, rowtile lm-head **5.9%**, misc **19.4%** | dp4a/q8_1 matmuls dominate proxy | Exact components are already optimized, unquantizable, or dp4a-gated. |
| Exact MTP throughput | B5 **60.78 tok/s**, **1.1134x**, acc/out **0.535**, passes/out **0.567** | B2 **67.3 tok/s**, ~**1.31x**, acc/out **0.598**, passes/out **0.402** | llama spends fewer target passes/output and uses cheaper dp4a rows. |
| Accuracy-traded dp4a transplant | B5 **61.61 tok/s**, **1.1322x**; ja top-1 **0.700** gate fail | native llama HIP still **67.3 tok/s** | dp4a is not sufficient and not correctness-retainable. |

### Everything we tried — expected vs actual

| Lever | Hypothesis / expected | Actual measured | Verdict |
| --- | --- | --- | --- |
| dp4a q8_1+sudot4, selected MoE | 2.6× isolated kernel | flat e2e (BW already saturated) | diagnostic only |
| dp4a dense Q8_0 attention | faster verify | 1.2× isolated, flat e2e | not promoted |
| target-block WMMA prefill in `llama-compat` B2 | use the existing batched WMMA path to amortize small verifier rows | all-sync smoke regressed **52.10 -> 34.04 tok/s**; `target_block_verify_total` **16.092 -> 26.285 ms/output**. Q8T16 attention saved only **0.045 ms/output**, while selected-MoE compact WMMA added **6.484 ms/output** in linear-attn layers plus **2.075 ms/output** in full-attn layers | rejected; global target-block WMMA remains off |
| split-K dense Q8_0 (c=1) | more MLP → more BW | **0.74× (negative)** | rejected |
| non-temporal weight loads (c=1) | +14% via cache-bypass | +14% isolated, **flat/worse e2e** | not promoted, reverted |
| MoE-FFN HIP graph (launch cut) | fewer launches | −0.84% e2e (slight regress) | not promoted |
| dense small-B rowtile (verify) | 3× microbench at B=4 | flat e2e | kept (kernel-level win) |
| device-chain resident draft (#3) | cut per-depth host sync | bit-exact, flat e2e | kept default-off (clean arch) |
| partial-accept LM-head skip (#4) | cut discardable replay work | **+3.5% B5, bit-exact** | **kept, default-on** |
| serial verifier no-logits cleanup | remove unused full-logits D2H | **+0.7% B1 full-suite, acceptance unchanged** | **kept, default-on** |
| deferred serial hidden-seed D2H copies | avoid copying intermediate verifier hidden rows that production route does not consume | full-suite flat/noise: B1 **50.18 → 50.19 tok/s**, ratio **0.9206 → 0.9202x AR** | rejected/reverted |
| resident top-k40 draft route | avoid full legacy draft fallback for root top-k40 | **+2.9% B1 full-suite, acceptance unchanged** | **kept, default-on** |
| one-block device top-k40 | avoid resident root-K40 host logits readback + NumPy top-k | correctness passed, but smoke B3 **45.58 → 24.74 tok/s** at identical acceptance | rejected/reverted; serial K40 merge dominates |
| strict-context route | existing llama.cpp-style prompt replay + device MTP KV with root/sibling top-1 | smoke B3 **42.81 tok/s = 0.780x AR**; partial best B1 **48.69 tok/s = 0.889x AR**, B3 **45.16 = 0.825x AR** | route is a valid diagnostic but not production-competitive; build resident lifecycle abstraction |
| adaptive full-vocab recovery after capped miss | keep cheap capped-vocab draft normally, switch to full vocab after a generic capped zero-accept miss instead of permanent AR fallback | partial route `resident-cap32k-recover`: AR **54.76 tok/s**, best B1 **52.45 tok/s = 0.958x AR**, accepted/output **19/39 = 0.487**; full suite: AR **54.55 tok/s**, best B1 **51.71 tok/s = 0.9478x AR**, accepted/output **78/178 = 0.438**; cap sweep B1 diagnostics peaked around cap18k/24k at **~52.6 tok/s** but still below AR | diagnostic only; B1 throughput improves, but acceptance regresses vs resident top-k40 and the serial verifier route remains bounded by target wall + draft overhead |
| short B1 target block verify with confidence gate | use 2-row target block verify for high-confidence exact B1 drafts, rollback to serial/root-topK on mismatch | direct rows=2 block probe was exact and faster than two serial steps (**32.8 ms vs 39.7 ms**), but partial B1 p=0.8 had 15 attempts/14 hits/1 rollback and regressed to **50.07 tok/s**; p=0.9 had 11/11 hits but still **51.84 tok/s**, below capped recovery **52.45 tok/s** | rejected; savings per hit too small and rollback/noise erases it |
| branch-safe B1 root-topK block verifier | batch `[prev, draft0]`; use row 1 only on strict draft top-1 accept; for root-topK branch/reject restore and replay row 0 unless a captured row-0 direct commit is requested | original restore/replay route smoke `resident-b1-branch-safe-block-cap32k-device-seed` B1 measured AR **54.93 tok/s**, MTP **31.11 tok/s = 0.566x AR**, accepted/output **0.400**; after fixing captured row-0 FP32 `ssm_out` exactness, direct row-0 branch route smoke `resident-b1-branch-safe-direct-cap32k-device-seed` B1 measured AR **54.97 tok/s**, MTP **26.66 tok/s = 0.4849x AR**, accepted/output **0.400** | rejected/default-off diagnostic; direct row-0 commit is now serial-exact, but row 1+ still needs replay and the route is slower than restore/replay, so it is not the amortization path |
| serial-exact verifier row baseline | use the normal token-serial decode scheduler to stage per-row `h_nextn` plus Conv/GDN state and prove direct row commits are exact before optimizing the batched path | focused wrong-branch gate passes: direct row-0 commit after `[prev, wrong_child]` matches serial hidden/state bit-for-bit and the corrective next step remains exact | correctness oracle/scaffold only; it does not amortize target weight loads and is not a speed route |
| hybrid strict-block/cap32k route | begin with strict top-1 block-promotion probe, then fall back generically to root-topK B1 + cap32k recovery if probe acceptance is weak | smoke B3 **48.94 tok/s = 0.890x AR**; partial best B3 **54.63 tok/s = 0.9973x AR** looked close, but full suite dropped to AR **54.58 tok/s**, best B3 **50.91 tok/s = 0.9328x AR**, B4 **48.94 = 0.8967x**, B5 **48.52 = 0.8890x**, accepted/output **94/194 = 0.485** | rejected/default-off diagnostic; partial was not predictive, and the route is worse than cap32k recovery B1 full-suite **51.71 tok/s = 0.9478x AR** |
| strict-context/block `draft_p_min=0.8` selector | suppress weak resident drafts before expensive strict block verification | smoke route `resident-strict-context-block-pmin08`: AR **55.00 tok/s**, B3 **38.44 tok/s = 0.6991x AR**, accepted/output **0.571** | rejected/default-off diagnostic; probability gating cannot fix strict block economics when low-accept cycles still pay target block work |
| direct verifier row-state commit | adopt llama.cpp-style verifier-row materialization for strict block verification: capture per-row GGUF linear-attention Conv/GDN state and commit the accepted row without rollback replay | row-0 wrong-branch commit is serial-exact after aligning captured `ssm_out` to the serial FP32 activation path; row 1 is serial-exact in `target-block-verify-mode=native` after fixing the native row-serial full-attention verifier to use absolute continuation positions and capture row states. Default `bulk` row 1 is now exact for short verifier blocks (`end < 1024`) after replacing the drifting suffix full-attention prefill reduction with a c1-exact row-batch decode context path and fixing the batch context kernel to honor shared physical block IDs. Standalone smoke remained negative: bulk hybrid direct B3 **49.01 tok/s = 0.893x AR**; native hybrid direct B3 **48.17 tok/s = 0.875x AR**; old pure strict B3 **37.20 tok/s = 0.678x AR**; B1 branch-safe direct **26.66 tok/s = 0.4849x AR**. | exactness scaffold retained; direct commit by itself was not a speed route, but it is required by the later B1-probe/block-direct/cap32k route that beats AR |
| resident device hidden seed | adopt llama.cpp-style resident `pending_h` and avoid target hidden-seed D2H/H2D before resident draft | full suite route `resident-cap32k-device-seed`: AR **54.59 tok/s**, best B1 **52.08 tok/s = 0.9540x AR**, accepted/output **78/178 = 0.438**; cap32k recovery control was B1 **51.71 tok/s = 0.9478x AR** with the same acceptance | retained default-off structural diagnostic; +0.7% over cap32k recovery, not enough to beat AR; confirms lifecycle direction but remaining lever must cut target verifier work per visible token |
| B1-probe/block-direct/cap32k route | use a cheap strict B1 cap32k probe to avoid non-code B3 block waste, then promote to direct-commit B3 block verification after a full B1 accept | full suite route `resident-b1-probe-block-direct-cap32k`: AR **54.59 tok/s**, best B3 **56.54 tok/s = 1.0356x AR**, accepted/output **40/140 = 0.286**, draft acceptance **0.645**, target layer passes **0.779/output**, direct commit rows **15**, replay rows **0** | retained default route; closes the same-protocol AR-beat gate, but still trails llama.cpp B2 **67.29 tok/s** by ~19% relative tok/s |
| resident device seed + dense draft KV | combine resident `pending_h` with device-resident draft KV and commit accepted verifier rows from staged device hidden rows instead of host hidden arrays | route `resident-cap32k-device-seed-kv`: B3 smoke AR **54.66 tok/s**, MTP **38.94 tok/s = 0.7124x AR**, draft_acceptance **0.032**; B1 smoke AR **54.92 tok/s**, MTP **39.73 tok/s = 0.7235x AR**, draft_acceptance **0.017** | rejected/default-off route; keep verifier-row staging + device-base KV commit primitives, but dense draft KV without llama.cpp prompt/context catch-up changes drafts and collapses acceptance |
| context replay + resident device seed | combine llama.cpp shifted prompt catch-up, device MTP KV, resident `pending_h`, staged verifier rows, and cap32k recovery | route `resident-context-cap32k-device-seed`: B1 smoke AR **54.93 tok/s**, MTP **50.84 tok/s = 0.9257x AR**, accepted/output **0.400**; B3 smoke AR **54.87 tok/s**, MTP **46.97 tok/s = 0.856x AR**, accepted/output **0.571** | rejected/default-off structural diagnostic; prompt/context lifecycle is now wired, but serial target verification still runs one target pass per visible token and draft overhead keeps it below true AR |
| dispatch-resolve cache (#9) | ~15 µs/launch host | landed | kept |
| X8 selected-down repack (Q5/Q6) | sidecar-free dp4a layout | mixed; ≤ default B3 | diagnostic |
| T16 Q4/Q5 selected dp4a variants | faster MoE GEMV | 1.04–1.10× iso, flat/regress B3 | diagnostic gates |
| 32k draft vocab cap | ~5 ms/cycle draft | prompt-sensitive | diagnostic |
| adaptive AR fallback after zero-accept | avoid catastrophic block replay | robust full-suite route | **kept (production selector)** |
| HIP graph capture of verify | collapse the ~875 launches | **refuted 2026-06-30: block verify is GPU-bound (10.2% host exposed); ROCm 7.x re-pays per-node (M12.1)** | **rejected — not a lever** |

Pattern: **every GPU/kernel/launch micro-lever is real in isolation and flat at
e2e.** The retained e2e wins are route/amortization cleanups (#4 LM-head skip,
serial no-logits, resident top-k40, adaptive fallback), not raw kernel
micro-optimization. That is the signal to stop optimizing kernels and work the
amortization.

### Decode-wall composition (rocprof, current code, c=1, this session)

`scripts/gguf_decode_rocprof.py`: dense_q8_0_gemv **47%**, selected-MoE GEMV
**26%**, lm-head Q6_K **10%**, GDN linear-attn **6%**, router **4%**, rmsnorm/rope
**3%**, rest <2%. Both dominant GEMV families are near-peak BW, so this wall is
mostly irreducible weight streaming — consistent with AR already beating
llama.cpp's AR.

### The new validation suite (`scripts/gguf_ar_mtp_suite.py`)

One entry point, one artifact, apple-to-apple enforced:

- Pins ONE canonical decode config on both AR and MTP: `HIPENGINE_GGUF_DECODE_REPACK=1`,
  `--decode-repack --use-gemv-decode --use-wmma-prefill`, eager decode, greedy,
  `--prompt-reasoning off` forced on both sides.
- Runs the true no-MTP AR baseline (`gguf_true_ar_category_bench.py`) and the MTP
  category suite (`gguf_mtp_category_bench.py`) — reusing the validated
  measurement code — then **computes the MTP/AR ratio itself** (does not rely on
  the stale `--true-ar-baseline-json` attach).
- **Enforces** the apple-to-apple invariants and records every problem: same
  decode protocol (`timing_protocol`), same prompt-set hashes; fails loudly with
  `apple_to_apple_ok=false` otherwise.
- Emits one artifact: `shared_config`, full provenance (git commit, hardware,
  host), the AR row, per-budget MTP rows with `vs_ar_ratio`, and a `verdict`
  (`best_mtp_budget`, `best_mtp_vs_ar_ratio`, `mtp_beats_ar`).
- Scope presets: `smoke` (1 prompt / 3 cycles / B3), `partial`
  (4 prompts / 5 cycles / B1,B3,B5), `full` (all 10 prompts / 10 cycles / B1–B5).
  The MTP suite loads the model **once** and loops all (prompt × budget)
  in-process (opt-in resident-session cache + per-prompt `reset()`; bit-exact
  validated vs the per-subprocess path — identical acceptance/token metrics, 1.89×
  faster on 2 prompts). So `full` runs in ~2–3 min instead of ~40+ min of repeated
  ~50 s model loads. The AR baseline already loads once.

```bash
# Quick directional check during development (1 prompt, ~1 min after first load):
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py --scope smoke

# Authoritative real-world number before retaining any change (~3-4 min, load-once):
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
    --scope full --output benchmarks/results/<date>-ar-mtp-suite-full.json
```

### Validation protocol — run the suite for EVERY change (mandatory)

**The only number that counts is the full-suite apple-to-apple result. Microbenches
and partials routinely do NOT translate to real-world e2e.** This session is the
proof: dp4a (2.6× isolated), split-K, dense rowtile (3× at B=4), and non-temporal
loads (+14% cold-DRAM) were all real wins in isolation and **flat or negative at
e2e** (see the tried-levers table). A kernel/host/launch microbench is a hypothesis,
not a result. So:

1. **Every GGUF AR/MTP optimization is gated by `scripts/gguf_ar_mtp_suite.py`,
   not by a microbench.** A change is a "win" only if `--scope full` improves AR
   tok/s and/or the MTP `vs_ar_ratio` **without regressing acceptance**
   (`accepted_per_output`), measured against the committed baseline.
2. **Cadence:** `--scope smoke` for a fast directional read while iterating →
   `--scope full` before promoting/committing anything as a win or making it
   default. Never retain a speed claim off a microbench, a single prompt, or a
   `partial` run alone.
3. **Compare to the committed hipEngine baseline and the llama.cpp target:**
   hipEngine current default is
   `benchmarks/results/2026-06-29-ar-mtp-suite-full-b1-probe-block-direct-cap32k.json`
   (AR 54.59 tok/s; MTP B3 56.54 tok/s = 1.0356× AR;
   `mtp_beats_ar=true`). The external target is
   `benchmarks/results/2026-06-22-llamacpp-35b-mtp-category-off-b1-b5-gfx1151.json`
   (llama.cpp B2 67.29 tok/s = 1.3423× AR). Diff the `verdict`,
   per-budget `vs_ar_ratio`/`accepted_per_output`, and the per-category
   accepted/output shootout columns. The suite asserts `apple_to_apple_ok=true`
   (same decode protocol + prompt-set hashes) — if it is false, the comparison is
   invalid, full stop.
4. **Record it** per the evidence policy: drop the artifact under
   `benchmarks/results/`, update `benchmarks/README.md` + `benchmarks/CHANGELOG.md`,
   and note the before→after `vs_ar_ratio` in `WORKLOG.md`. A flat/negative e2e
   result is a *retained finding* too (it tells the next person not to re-chase it).
5. **Anti-gaming:** the suite runs the full multi-prompt category suite (code /
   general_en / general_ja / mixed_ja_en), never the single merge-sort prompt, and
   the true-AR denominator comes from the **same run** under the same config. Do
   not tune to one prompt.

This is the gate that stops the recurring trap of shipping an isolated win that
disappears at e2e.

**Scope:** `gguf_ar_mtp_suite.py` covers the **GGUF Q4_K_M path only**
(`Qwen35GGUFResidentSession`). The **PARO path** (BF16 / W4-PARO safetensors) is a
separate MTP/AR codepath with its own harnesses (`qwen35_paro_bench.py` AR;
`mtp_chain_e2e_bench.py` / `mtp_verifier_economics.py` MTP) and is **not** covered
by this suite — a PARO change needs e2e validation there. See `docs/BENCHMARK.md`
"Honest native GGUF-MTP category diagnostics" for the cross-path scope note.

### How to continue (ordered, all gated by the suite)

1. **Done: verifier host-vs-GPU split is settled for current code.**
   `scripts/gguf_mtp_verifier_rocprof.py` shows the retained
   `resident-serial-fallback` target verifier is GPU-bound (18.63 ms host /
   16.56 ms kernel per target step, 89% kernel share; latest current rerun
   19.03 ms host / 16.76 ms kernel, 88.0% kernel share). Do not start with a
   launch-collapse project unless a new route/profile proves host residual is
   back on the critical path.
2. **Treat strict-context as a diagnostic baseline, not the next optimization
   target.** The existing `resident-strict-context` route records
   `--resident-mtp-draft --root-topk-accept 1 --sibling-topk-accept 1
   --mtp-context-replay --mtp-device-kv-cache --no-target-block-verify`.
   Initial evidence: smoke B3 is **42.81 tok/s = 0.780× AR**; partial best is B1
   **48.69 tok/s = 0.889× AR** with B3 accepted/output **0.697** but only
   **0.825× AR**. A full run is useful after lifecycle changes, but the existing
   diagnostic hooks do **not** generalize into a competitive route by
   themselves.
3. **Port the llama.cpp target-memory pattern, not another micro-lever.**
   `Qwen35GGUFMTPContext` already owns the seed lifecycle (`pending_h` /
   verifier hidden rows / `accept()` reseed). The missing adoption target is a
   branch-safe transactional target verifier with llama.cpp-like recurrent
   rollback slots: run `[prev]+drafts` through scratch target state, materialize
   exact per-row `h_nextn` plus GGUF Conv/GDN state, and advance the resident
   target to the accepted row without serial restore/replay. In llama.cpp terms,
   this is the `llama_memory_recurrent::seq_rm()` / bounded `n_rs_seq` behavior,
   not merely a renamed draft context. Success is lower target passes per
   visible token on the full category suite, not a wider candidate-rank or
   confidence diagnostic.
   Source anchor: llama.cpp commit `6e9007ae61f4e994c27484759caac6ef2aa32b30`
   defines this lifecycle in `common/speculative.h` (`common_speculative_process`,
   `common_speculative_draft`, `common_speculative_accept`), implements the MTP
   state in `common/speculative.cpp::common_speculative_impl_draft_mtp`
   (`pending_h`, `verify_h`, `last_n_drafted`), and invokes it from
   `tools/server/server-context.cpp` (`draft()` before target batch construction,
   `process()` after target decode, `accept()` after accepted-row sampling).
   The same checkout builds Qwen3.5/Qwen3.6 MTP as a first-class
   `qwen35moe::graph_mtp` and exports target/draft `t_h_nextn`; the retained
   llama.cpp speed row is plain `--spec-type draft-mtp --spec-draft-n-max N`, not
   an ngram-stack or prompt-history trick.
   The capped-vocab recovery, hybrid strict-block, direct row-state commit, and
   device hidden-seed probes confirm this direction. The payoff became positive
   only after combining them as `resident-b1-probe-block-direct-cap32k`: full
   suite B3 **56.54 tok/s = 1.0356× AR**, target layer passes
   **0.779/output**, replay rows **0**. The route is now the baseline to improve,
   not a reason to restart from micro-kernel work.
4. **Close the llama.cpp parity gap from the retained route.** The remaining
   lever is better verifier amortization and acceptance economics: make B2/B3
   block promotion pay on more prompts, keep direct commits exact without serial
   fallback waste, and preserve the cheap cap32k draft cost. The concrete target
   is moving from B3 **56.54 tok/s** toward llama.cpp's B2 **67.29 tok/s** on
   the same full category suite. **Order superseded by "Goal — Part 1" above:**
   the verify-wall reduction (old S5) runs first because `cap32k-recover`
   already meets its precondition; non-code rescue / B2 / B3 policy sweeps
   (S1-S3) run only after the wall drops.
5. **~~The fused multi-token target verifier is now P0~~ — CLOSED/REFUTED
   2026-06-30.** The "collapse 875 launches into one weight stream" lever assumed
   a host-launch floor. Measured: the block verify is **GPU-kernel-bound** (38.1 ms
   GPU / 42.4 ms wall, 10.2% host exposed). HIP graph capture / C-loop / GDN-fix
   are **not levers** (≤10% ceiling; ROCm 7.x re-pays per-node, M12.1). llama's
   "~9 ms fused graph" advantage is its **dp4a/q8_1 cheaper kernels**, not graph
   topology — and dp4a fails hipEngine's ja correctness gate (top-1 0.700). The
   exact-precision GPU-compute ceiling is reached at `1.1134×`. See P0.2 + the
   2026-06-30 correction.
6. **Use llama.cpp parity, not AR-beat, as the next retained speed gate.** The
   current route already satisfies `mtp_beats_ar=true` on `--scope full`. The
   next retained claim should either move materially toward llama.cpp's
   **67.29 tok/s B2** row with the same no-gaming full-suite protocol, or record
   why a directly adopted llama.cpp pattern fails in hipEngine.
7. **Fix the stale AR-baseline contracts** (`TRUE_AR_PRODUCTION_TIMING_REQUIRED`
   + speed-claim contract + tests) to the eager path so the category bench's own
   `--true-ar-baseline-json` comparison works again (REFACTOR.md).

### Don't re-chase (closed lines of work)

GEMV instruction efficiency (dp4a/rowtile), split-K, MoE-FFN graph, cache
hints, deferred hidden-seed D2H copies, the one-block device top-k40 extension,
cap-only/rootK sweeps, resident device hidden-seed copy avoidance by itself,
device-seed + dense draft-KV without context catch-up, context replay + device
seed under serial target verification, strict block `draft_p_min` gating, short
B1 confidence-gated target block verify, and branch-safe B1 root-topK block
verify are all measured too small, acceptance-regressive, or negative e2e and
are not the lever. Direct row-state commit by itself was not a speed win, but it
is now part of the retained `resident-b1-probe-block-direct-cap32k` route; do not
re-test it as isolated exactness scaffolding unless a correctness regression
appears.
The per-kernel GEMV bandwidth is already near-peak. Kernel micro-optimization is
exhausted; the gap is amortization.

## Production verifier status (2026-06-28)

### Update 2026-06-28 (later) — graph replay retired; AR denominator corrected; bandwidth-bound

The "AR denominator blocked by graph replay token divergence" framing **below is
superseded**. The GGUF decode-graph machinery (the divergent `--graph-replay-decode`
path) was **retired** (task #8). The current no-MTP AR path is the **eager** resident
`step()` loop with `HIPENGINE_GGUF_DECODE_REPACK=1` + `--use-gemv-decode`, with no graph
on the hot path. Measured this session (35B-A3B Q4_K_M, gfx1151, prompt-12 + 32 steps,
short-context diagnostic):

| Path | tok/s | Notes |
| --- | ---: | --- |
| **Eager AR (repack + gemv-decode), current production** | **~55.1** | no graph; the ~55.5 "divergent graph AR" row below was the now-retired graph path |
| MoE-FFN graph replay (`HIPENGINE_GGUF_MOE_GRAPH`, default off) | ~54.7 | bit-exact (KL=0, 40 cap / 3800 replay / 0 reject) but **−0.84% wall** — launch-count is not the bottleneck |

**Today's decisive finding: the decode/verify wall is weight-bandwidth bound, and every
kernel-compute/launch lever is flat.** A one-model-load AR flag sweep toggling every gated
path — `RAW`/`Q4K`/`T16` selected dp4a, `FUSED_MOE_FFN`, `COMPACT_MOE_C1`, `MOE_GRAPH`,
all-dp4a — moved AR tok/s within **−0.9%..+0.0% with bit-identical tokens** (baseline 55.15).
Bandwidth arithmetic: ~1.6–1.7 GB active Q4_K weights/token at 18.1 ms/token ≈ **~90 GB/s
achieved on ~256 GB/s peak LPDDR5X ≈ ~35% of peak**; llama.cpp's 1.9× implies ~68% of peak.
**The 1.9× gap is a memory-bandwidth-efficiency gap, not compute or launch count.** dp4a
(compute), fusion (launches), and graph (launches) are therefore exhausted as levers and
not promotable (matches the prior full-B3 dp4a −0.4% e2e; the "1.31x verifier" was an
env-toggle dispatch-thrash artifact). Artifacts:
`benchmarks/results/2026-06-28-ar-flag-sweep-bandwidth-bound.json`,
`benchmarks/results/2026-06-28-moe-graph-rows1-ab.json`.

**Open denominator question (task #5, in progress):** the honest fast eager AR is ~55 tok/s,
NOT the 19.67 "exact eager" slow control quoted below. The MTP ratio must be recomputed on the
**same protocol** with this eager-repack denominator: if AR is ~55 and resident-serial MTP is
~47.6, MTP is currently **~0.86× AR (not winning)** rather than the 2.42× implied by the 19.67
denominator. Settling this same-protocol (true-AR category bench with repack + gemv-decode vs the
MTP category bench) is the #5 deliverable. Caveat: the raw (`repack=0`) eager path is currently
**broken** by the committed `ssm_out` f32-activation fusion (`a12d8c4c`) — no `(raw_gguf, f32,
bf16)` dispatch — so the exact reference must come via the T16-repack path, and a clean eager
token-trace re-validation vs the established llama.cpp reference is part of #5.

**Re-pointed next work:** (1) #10 raise the selected-expert GEMV's *achieved* bandwidth toward
peak (coalesced/vectorized Q4_K block loads, occupancy, llama.cpp `mul_mat_vec_q` RDNA3 layout)
— the actual 1.9×; (2) #4/#3 speculative amortization (cut the ~303 ms partial-accept rollback,
keep the draft chain on-device) — fewer weight-read passes per output token. Kernel-compute and
launch-count micro-optimization is closed as a line of work.

---

_Historical (superseded above):_

**Full-suite broad verifier path exists, but the production AR denominator is
currently blocked by graph replay token divergence.**

The most robust measured route is the resident GGUF MTP draft chain with serial
target graph probing and adaptive AR fallback after zero-accept cycles:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_mtp_category_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --budgets 3 --cycles 5 \
  --raw-root /tmp/hipengine-gguf-mtp-parity-workbench/2026-06-28-resident-serial-fallback-category-b3-c5/category/resident-serial-fallback \
  --output benchmarks/results/2026-06-28-resident-serial-fallback-category-b3-c5-eager-ar-summary.json \
  --true-ar-baseline-json benchmarks/results/2026-06-28-true-ar-eager-b3-c5.json \
  --reuse-existing \
  --extra-arg=--prompt-reasoning --extra-arg=off \
  --extra-arg=--root-topk-accept --extra-arg=1 \
  --extra-arg=--mtp-context-replay --extra-arg=--mtp-device-kv-cache \
  --extra-arg=--target-block-verify --extra-arg=--mtp-draft-vocab-cap \
  --extra-arg=32768 --extra-arg=--resident-mtp-draft \
  --extra-arg=--adaptive-ar-fallback --extra-arg=--no-target-block-verify
```

Result on the full default 10-prompt `mtpbench-code-general-ja.jsonl` suite,
B3/C5, gfx1151:

| Route / baseline | tok/s | Ratio | accepted/output | draft accept | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact no-MTP eager AR | 19.67 | 1.00x exact eager control | n/a | n/a | `--no-graph-replay-decode`; token-correct, but not the production speed denominator |
| Production graph no-MTP AR | ~55.5 | invalid denominator | n/a | n/a | graph replay settings; currently token-divergent |
| Resident serial-fallback MTP | 47.62 | 2.42x exact eager / 0.858x divergent graph AR | 0.438 | 0.542 | best robust full-suite MTP route measured |
| Always-block resident MTP | 16.60 | 0.84x exact eager | 0.597 | 0.493 | partial-accept block replay is too expensive |

The exact eager artifact is useful because it emits the expected merge-sort AR
trace. It is not evidence that production AR regressed to 19.67 tok/s; it is the
slow non-graph decode path. Artifacts:
`benchmarks/results/2026-06-28-true-ar-eager-b3-c5.json` and
`benchmarks/results/2026-06-28-resident-serial-fallback-category-b3-c5-eager-ar-summary.json`.

Important caveat: the faster graph-replay true-AR baseline measured about
`55.5 tok/s`, but it is currently token-divergent from exact eager AR on the
merge-sort diagnostic. It is not a valid speed denominator until graph replay
correctness is fixed. This is a graph correctness bug/denominator issue, not a
ROCm regression and not evidence that AR is actually 19.67 tok/s in production.

Rejected verifier routes from this update:

- Always-block resident draft is not production-safe: it reaches high acceptance
  but falls to `16.60 tok/s` full-suite because every partial accept triggers
  expensive block rollback/replay.
- B5 block promotion after a full B3 serial probe failed on the merge-sort smoke:
  `38.40 tok/s`, with two B5 partial cycles costing `~137-141 ms`. Do not make
  B5 block promotion a default without a stronger predictor and rollback fix.

Next performance work is now unambiguous: fix graph replay correctness so the
fast AR path is eligible as the denominator, then continue reducing verifier
GEMV cost and improving draft acceptance. The current full-suite route is a
useful robust MTP baseline, but it is not yet faster than the production graph
AR path and remains far from llama.cpp's ~90 tok/s MTP diagnostic.

## Executive summary (2026-06-27)

**Correctness is solved. The remaining gap is GGUF quantized GEMV performance,
roughly 1.9x on the single-prompt gfx1151 diagnostic.**

| Milestone | Status |
| --- | --- |
| Target AR first-token parity | ✅ `71093` matches llama.cpp (Qwen3.5 GDN K-head broadcast fix) |
| Target AR 12-token greedy trace | ✅ identical sequence `[71093,12305,198,727,10562,17885,10620,25,1103,8,1411,1103]` |
| Strict B3 draft acceptance | ✅ `2/9` → `9/9`, and `15/15` over 5 cycles (context replay + device MTP KV) |
| F32 router/alpha/beta retention | ✅ landed (registry-dispatched mixed kernels) |

The earlier blocker — hipEngine's target autoregressive stream diverging from
llama.cpp at the first sampled token — is fixed. The root cause was Qwen3.5
linear-attention Gated-Delta-Net K-head mapping: GGML maps value head `v_head` to
key head `v_head % num_k_heads`, while hipEngine inherited grouped `v_head /
repeat`. With the interleaved mapping, target AR and strict B3 acceptance both
match llama.cpp on the merge-sort prompt.

### Performance: current numbers (single-prompt diagnostic, gfx1151)

llama.cpp B3 MTP on the same reasoning-off 12-token trace:
**`eval time = 89.55 tok/s`** (`134.01 ms / 12 tokens`), 100% strict draft
acceptance, from `/tmp/hipengine-llamacpp-mtp-cli-reasoning-off-debug.log:3813`.

hipEngine best diagnostic configs (all `15/15` strict accepts, B3/C5, merge-sort
prompt):

| Configuration | tok/s | vs AR | verify ms/cycle | draft ms/cycle | accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Block verify GEMV prefill + dense rowtile + 32k draft cap | 48.8 | 0.80x | ~61 | ~17 | 15/15 |
| Block verify GEMV prefill + 32k draft cap, pre-rowtile | 48.1 | 0.80x | ~61–66 | ~17 | 15/15 |
| One-step graph + 32k draft cap | 44.5 | 0.81x | ~72 | ~17 | 15/15 |
| One-step graph, full vocab | 42.3 | 0.77x | ~73 | ~22 | 15/15 |

Gap to llama.cpp: **~48.8 vs ~89.6 tok/s ≈ 1.8-1.9x slower**, and it is almost entirely
target verification overhead, not acceptance and not draft quality.

### Where the time goes (per B3 cycle)

| Stage | hipEngine | llama.cpp | Gap |
| --- | --- | --- | --- |
| Target verify (4 tokens) | ~64 ms (block GEMV) / ~73 ms (graph) | ~8.9 ms (`dur(g)=26.7 ms / 3 calls`) | 7–8x |
| MTP draft (3 tokens) | ~17 ms (32k cap) / ~22 ms (full vocab) | included in `dur(g)` | ~2x |
| Commit / bookkeeping | ~1.6 ms | negligible | minor |

A synchronized per-layer probe over the first B3 verifier block showed most time
inside the 30 linear-attention layers, but a later sync-free rocprof trace
narrowed the actual hot bucket: selected-expert MoE GEMV is ~54% of verifier GPU
time (`gguf_q4_k_selected_dual_prefill_out_kernel` gate+up ~36% plus
`gguf_k_selected_pack8_prefill_out_kernel` down ~18%). Dense rowtile kernels are
now default-on and are ~3x faster on their microbench share, but end-to-end is
flat because dense projections are only ~11-17% of the verifier after clean
profiling.

**dp4a POC result (2026-06-27): positive, not runtime-default.** A bounded
q8_1+sudot4 selected-dual Q4_K variant now exists as a diagnostic wrapper. At
the qwen35moe verifier shape (`x_rows=4`, `rows=32`, `experts=256`, `in=2048`,
`out=512`, gfx1151), the existing raw selected-dual kernel measured `0.946 ms`
vs q8_1 quantize+dp4a at `0.357 ms` (**2.65x**). q8_1 quantization alone was
`0.0025 ms`. Correctness vs the existing float-dequant kernel on that diagnostic
was `KL_mean=0.0031`, top-1 `1.0` for both gate/up outputs. Disassembly confirms
`v_dot4_i32_iu8` emission, and `rocprofv3 --kernel-trace` shows
`gguf_q4_k_selected_dual_q8_1_dp4a_prefill_out_kernel` averaging `~338 us` vs
`~1007 us` for `gguf_q4_k_selected_dual_prefill_out_kernel` in the same short
trace. Artifact:
`benchmarks/results/2026-06-27-hipengine-gguf-q4-k-selected-dual-dp4a-poc.json`.

**Verifier integration diagnostic (2026-06-27): exact, but not the production
hot path.** The rows>1 verifier now has a default-off
`HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A=1` path with caller-owned q8_1 workspace.
B3/C5 merge-sort smoke with the production decode-repack route stayed exact
(`15/15`) and measured `50.44 tok/s` (`50.73 tok/s` warm), but rocprof showed
no q8_1/dp4a kernels in that production trace. The active selected-MoE verifier
route is T16 decode-repack (`q4_k_t16_selected_dual_*` and
`qk_t16_selected_direct_gemv_kernel`), not the raw Q4_K fallback. With
`--no-decode-repack`, the raw fallback does launch `40` q8_1 quantize calls and
`40` `gguf_q4_k_selected_dual_q8_1_dp4a_prefill_out_kernel` calls, but that mode
is much slower overall (`35.66 tok/s`, verifier `96.2 ms`) because it disables
the production T16 materialization.

**T16 selected-dual dp4a diagnostic (2026-06-27): launches in production, but
too small to promote.** The same env gate now also has a T16 Q4_K selected-dual
q8_1+sudot4 variant for the rows>1 split gate/up path. The isolated T16
microbench at the verifier shape measured current T16 split dual `0.198 ms` vs
q8_1 quantize+dp4a `0.191 ms` (**1.04x**), with gate/up `KL_mean=9.25e-05` and
top-1 `1.0`; disassembly confirms `v_dot4_i32_iu8`. The callable fused-SiLU
T16 dp4a variant is retained as a diagnostic but is **not routed** in production
because the c1 profile regressed it. Split-only B3/C5 smoke stayed exact
(`15/15`) but remained flat (`49.31 tok/s`, warm `50.60 tok/s`). A short
production trace confirms only the row-bulk split path uses dp4a: `80`
`q4_k_t16_selected_dual_q8_1_dp4a_direct_gemv_kernel<unsigned short,false>`
calls at `141.8 us` avg plus `80` q8_1 quantize calls at `3.35 us`; c1 fused
stays on `q4_k_t16_selected_dual_silu_direct_gemv_kernel` at `62.5 us` avg. The
next material bucket is still selected-down Q5_K T16 (`851` calls, `51.6 us`
avg, `43.9 ms` in the same two-cycle trace). Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-q4-k-t16-selected-dual-dp4a-poc.json`
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-q4k-t16-dp4a-verifier-diagnostic.json`.

**T16 selected-down Q5_K dp4a diagnostic (2026-06-27): kernel-positive, not a
runtime win.** The next bucket was ported under a new default-off broad env gate:
`HIPENGINE_GGUF_T16_SELECTED_DP4A=1`. The Q5T16 selected-down microbench at the
c1-like down shape (`rows=8`, `E=256`, `in=512`, `out=2048`, gfx1151) measured
current T16 `0.0335 ms` vs q8_1 quantize+dp4a `0.0306 ms` (**1.10x**),
`KL_mean=0.00678`, `KL_max=0.03093`, but only `0.875` top-1 on that small
synthetic fixture. `rocprofv3 --kernel-trace` confirms
`qk_t16_selected_q8_1_dp4a_direct_gemv_kernel<unsigned short>` launches, and
extracted device ISA contains `v_dot4_i32_iu8`. B3/C5 merge-sort smoke remained
exact (`15/15`) but regressed to `47.62 tok/s` (warm `48.44`), so the Q5 path is
kept diagnostic/default-off. Q6_K was not routed: a synthetic probe had
acceptable KL but only `0.75` top-1 vs the T16 float path. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-q5-k-t16-selected-down-dp4a-poc.json`
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-q5-t16-dp4a-verifier-diagnostic.json`.

**Raw selected-down Q5_K/Q6_K dp4a diagnostic (2026-06-27): broad raw layout
is promising, but not enough yet.** The raw no-decode-repack selected-down path
now has Q5_K and Q6_K q8_1+sudot4 variants under the default-off
`HIPENGINE_GGUF_RAW_SELECTED_DP4A=1` gate. On the selected-down microshape
(`rows=8`, `E=256`, `in=512`, `out=2048`, gfx1151), Q5_K measured raw
float-dequant `0.0916 ms` vs q8_1 quantize+dp4a `0.0395 ms` (**2.32x**),
and Q6_K measured `0.0419 ms` vs `0.0259 ms` (**1.62x**). Correctness vs the
existing float-dequant path cleared the project gate on the diagnostic:
Q5_K `KL_mean=0.00011`, top-1 `1.0`; Q6_K `KL_mean=0.00512`, top-1 `1.0`.
A cached `rocprofv3 --kernel-trace` microbench confirms
`gguf_k_selected_pack8_q8_1_dp4a_prefill_out_kernel<unsigned short,5/6>`
launches, with q8_1 quantization at `~2.1 us` average and dp4a dot kernels at
`~44.7 us` (Q5) / `~19.5 us` (Q6) in the short trace. B3/C5 raw-layout smoke
stayed exact (`15/15`) and improved no-decode-repack from `31.63 tok/s` to
`39.61 tok/s` (warm `31.86 -> 40.29`), but the production decode-repack
baseline on the same short smoke was still `51.31 tok/s` (warm `52.00`). Keep
this as a diagnostic proof that GGML-style raw q8_1 vector-dot is worth a broad
layout port; do not promote the raw env as a runtime default yet. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-raw-q5-q6-selected-pack8-dp4a-poc.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-raw-selected-dp4a-verifier-diagnostic.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-raw-selected-float-verifier-baseline.json`,
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-default-verifier-baseline-for-raw-dp4a.json`.

**X8 selected-down production-layout slice (2026-06-27): correct and
sidecar-free, not default yet.** The first broad-port slice now has a
byte-neutral X8 replacement layout for selected-down Q5_K/Q6_K experts:
`tiles[expert, out_pack8, k_block, 8 * block_bytes]`. It preserves the raw GGUF
block bytes while giving the production decode-repack materializer the same
eight-output q8_1+sudot4 dot shape as the raw sidecar diagnostic. It is opt-in
via `HIPENGINE_GGUF_SELECTED_X8_REPACK=1`; gate/up remains on the current T16
Q4_K path. On the selected-down microshape (`rows=8`, `E=256`, `in=512`,
`out=2048`, gfx1151), X8 matched raw dp4a outputs exactly and cleared the
quality gate versus production T16 float, but the timing is mixed: Q5_K
production T16 `0.03352 ms` vs X8 q8_1 quantize+dot `0.03864 ms` (**0.87x**),
while Q6_K production T16 `0.03206 ms` vs X8 q8_1 quantize+dot `0.02602 ms`
(**1.23x**). A cached `rocprofv3 --kernel-trace` microbench confirms
`gguf_x8_selected_q8_1_dp4a_gemv_kernel<unsigned short,5/6>` launches; the
short trace averaged `~37.2 us` for Q5 X8, `~22.9 us` for Q6 X8, and `~1.9 us`
for q8_1 quantization. B3/C5 merge-sort smoke with X8 materialization stayed
exact (`15/15`) but was slower than the same-tree default control:
`49.74 tok/s` (`50.65` warm) vs default `51.43 tok/s` (`53.09` warm). Keep X8
default-off until the Q5 path beats T16 or a quant-selective production route
improves the same B3/full-suite protocol. Artifacts:
`benchmarks/results/2026-06-27-hipengine-gguf-x8-selected-down-dp4a-poc.json`,
`benchmarks/results/2026-06-27-hipengine-mtp-b3-x8-selected-down-verifier-diagnostic.json`,
and
`benchmarks/results/2026-06-27-hipengine-mtp-b3-default-verifier-control-for-x8.json`.

**X8 Q5 tuning / quant-selective route (2026-06-28): useful diagnostic, still
not a default.** Reducing X8 selected-down launches from 128 to 64 threads helps
the synthetic small-B shape: Q5_K X8 dot moved to `0.03026 ms` and q8_1
quantize+dot to `0.03378 ms` versus production T16 `0.03364 ms` (roughly
break-even), while Q6_K X8 quantize+dot moved to `0.02014 ms` versus T16
`0.03304 ms` (**1.64x**). The materializer now accepts
`HIPENGINE_GGUF_SELECTED_X8_REPACK=q5|q6|both`; `=1` remains `both`. This lets
diagnostics route by quant family, but the B3 verifier still does not improve:
full X8 with the 64-thread body measured `49.08 tok/s` (`49.41` warm), q6-only
X8 measured `50.32 tok/s` (`51.07` warm), and same-tree default T16 measured
`51.77 tok/s` (`52.56` warm), all exact `15/15`. Keep the selector opt-in and
do not promote X8 until the production verifier, not just the microshape, wins.
Artifacts:
`benchmarks/results/2026-06-28-hipengine-gguf-x8-selected-down-t64-dp4a-poc.json`,
`benchmarks/results/2026-06-28-hipengine-mtp-b3-x8-t64-selected-down-verifier-diagnostic.json`,
`benchmarks/results/2026-06-28-hipengine-mtp-b3-x8-q6-only-selected-down-verifier-diagnostic.json`,
and
`benchmarks/results/2026-06-28-hipengine-mtp-b3-default-verifier-control-for-x8-t64.json`.

**Superseding llama-compat note (2026-07-01):** the production B3 route above
still stays on T16, but q6-only X8 is now retained for the accuracy-traded
`llama-compat-device-chain-dp4a-q6top1dp4a` B2 lane. That full-suite row moves
**59.63 -> 60.36 tok/s** and `target_block_verify_total`
**13.178 -> 13.023 ms/output** with `HIPENGINE_GGUF_SELECTED_X8_REPACK=q6`.
q5/both remains rejected for this route.

**Systemic E2E/per-piece workbench (2026-06-28): landed.**
`scripts/gguf_mtp_parity_workbench.py` is now the standard local gate for the
GGML-style broad port. It runs the same B3/C5 E2E command shape across named
runtime candidates (`default`, `x8-q5`, `x8-q6`, `x8-both`, `t16-dp4a`,
`q4-t16-dp4a`, `raw-dp4a`), runs the selected-MoE per-piece microbenches
(`Q4_K` gate/up, raw `Q5_K/Q6_K` down, X8 `Q5_K/Q6_K` down), and can optionally
run rocprof bucket summaries and category-suite diagnostics. The first smoke
validated the wrapper on gfx1151 with one default B3 cycle plus low-iteration
piece runs:
`PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_mtp_parity_workbench.py --tag 2026-06-28-gguf-mtp-parity-workbench-smoke --raw-root /tmp/hipengine-gguf-mtp-parity-workbench --output benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-smoke.json --stages e2e,pieces --candidates default --cycles 1 --draft-n-max 3 --piece-iters 4 --piece-warmup 1`.
That smoke measured default E2E `49.3 tok/s`, AR baseline `60.62 tok/s`, exact
`3/3` accepts for the one cycle. Treat the piece timings in this smoke as
harness validation only because `--piece-iters 4` is intentionally noisy; use the
full default `--piece-iters 80`/`--cycles 5` workbench or a higher-iteration run
before making kernel decisions. Artifact:
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-smoke.json`.
The first full B3/C5 workbench matrix then showed why same-protocol repeats are
required before routing decisions: `default,x8-q6,x8-both` measured `46.19`,
`49.74`, and `50.49 tok/s`, but the reversed-order E2E repeat measured
`x8-both=48.07 tok/s` and `default=51.33 tok/s`, all exact `15/15`. This keeps
X8 diagnostic/default-off and confirms the workbench should be used as a
multi-run gate, not a single-run promotion oracle. Artifacts:
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-b3-current.json`
and
`benchmarks/results/2026-06-28-hipengine-gguf-mtp-parity-workbench-b3-repeat.json`.

### Next steps, ordered by impact

1. **Do not promote the current straight dp4a diagnostics.** Raw Q4_K/Q5_K/Q6_K
   q8_1+sudot4 is strong in isolation and improves the raw no-decode-repack
   verifier, but production B3 still uses T16 and remains faster. The first
   production-compatible X8 selected-down slice removes the raw sidecar and the
   64-thread body helps the isolated Q5/Q6 microshape, but full-X8 and q6-only
   X8 still trail default B3. T16 Q4 split is only `1.04x` in its small
   row-bulk bucket, T16 Q5 selected-down is only `1.10x` in isolation while
   regressing B3, and raw selected-down still trails default decode-repack at
   the verifier level. Keep q6-only X8 only for the active accuracy-traded
   llama-compat lane; otherwise keep
   `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` and
   `HIPENGINE_GGUF_T16_SELECTED_DP4A` / `HIPENGINE_GGUF_RAW_SELECTED_DP4A` /
   `HIPENGINE_GGUF_SELECTED_X8_REPACK` as diagnostic gates only.
2. **Broad port target: match GGML's q8_1/x4 vector-dot layout, gated through
   the workbench.** The next implementation should make the production verifier consume a GGML-like
   q8_1 activation plus x4 packed K-quant dot path for the selected-MoE and dense
   GGUF GEMVs, instead of continuing one-off T16 ports. The raw Q4/Q5/Q6 and X8
   results prove the instruction path and a sidecar-free materialization route;
   the missing piece is making the Q5 selected-down body and the remaining hot
   GGUF GEMVs faster than T16 on the same production verifier protocol. Use
   `scripts/gguf_mtp_parity_workbench.py --stages e2e,pieces,rocprof` for local
   acceptance of each broad-port slice before promoting any default.
3. **Extend only proven GGUF GEMVs into defaults.** Carry q8_1+sudot4 into
   dense/raw Q4_K/Q5_K/Q6_K/Q8_0 GEMVs when the local shape clears the quality
   gate and improves the same B3/full-suite protocol. The existing small-B
   rowtile dense kernels are complementary and should be combined with dp4a where
   rows 2..8 share an activation tile.
4. **MTP draft resident path.** Keep all MTP intermediates (embeddings,
   projections, KV, hidden seeds) on device across draft depths; only D2H the
   final top-1 token ID. Chain the B draft steps in one call instead of B separate
   `run_draft()` calls with full alloc/copy per depth. Validate the 32k draft
   vocab cap on the full suite before promoting (saved ~5 ms/cycle here but is
   prompt-sensitive).
5. **Partial-accept rollback is catastrophic (~303 ms for a B5 partial cycle).**
   Track which linear-attention buffers were modified and copy-on-write only
   those, or replay only the accepted prefix instead of full target decodes. Or
   just keep B3 (100% accept on this prompt) and skip B5 until rollback is cheap.
6. **Full-suite validation before any retained speed claim.** Everything above is
   single-prompt merge-sort diagnostics. Need the full
   `mtpbench-code-general-ja.jsonl` category suite, category heldouts, a true
   no-MTP AR baseline from the same protocol, and the draft vocab cap validated
   for non-regressive acceptance across prompts.
7. **Longer-term: match llama.cpp's architecture.** Both target verification and
   MTP drafting run through one optimized GGML compute graph in a single process
   with shared weight memory. C-level dispatch or HIP graph capture remains a
   later layer, after the hot GEMV kernels stop wasting instruction issue on
   float dequant-then-FMA.

The historical trace evidence below is retained as the record of how correctness
parity was reached.

## Source evidence: what llama.cpp does

All llama.cpp source links below point to commit
`6e9007ae61f4e994c27484759caac6ef2aa32b30`.

### 1. Qwen35MoE MTP graph

The Qwen35MoE MTP graph is built as a one-layer decoder graph:
[`src/models/qwen35moe.cpp#L550-L736`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L550-L736).
Important details:

- It requires one NextN/MTP block.
- It chooses `nextn.embed_tokens` when present, otherwise `model.tok_embd`.
- It takes a separate hidden-state input tensor named `mtp_h_input`.
- It calls `build_attn_inp_kv()`, so the MTP block has its own draft-context K/V state.
- It computes:
  1. `h_norm = RMSNorm(h_input, nextn.hnorm)`
  2. `e_norm = RMSNorm(token_embedding, nextn.enorm)`
  3. `concat = [e_norm, h_norm]`
  4. `eh_proj`
  5. attention + gated output projection + residual
  6. MoE/shared-expert FFN + residual
  7. shared-head norm, then LM head fallback to `model.output`.

This graph shape matches our Python/GPU wrapper at a high level.  The gap is in
**state lifecycle and numerical/runtime parity**, not the obvious concat order or
which head/embedding tensors are chosen.

### 1b. GGUF GEMV inner loop

The current performance-path delta is below the graph shape: llama.cpp/GGML
quantizes activations to q8_1 and runs quantized weight x q8_1 dot products,
while hipEngine's raw GGUF kernels dequantize weights to float and then FMA.
Local source evidence in `/home/lhl/llama.cpp/llama.cpp-hip/ggml/src`:

- `ggml-common.h` defines `block_q8_1` as 32 signed int8 activation quants plus
  `d` and `s` fp16 metadata.
- `ggml-cuda/mmvq.cu` dispatches `GGML_TYPE_Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`
  through `vec_dot_*_q8_1` functions and allocates/quantizes `src1_q8_1` before
  `mul_mat_vec_q_switch_type(...)`.
- `ggml-cuda/vecdotq.cuh` uses repeated `ggml_cuda_dp4a(...)` calls in those
  vector-dot functions.
- `ggml-cuda/common.cuh` maps ROCm `ggml_cuda_dp4a(...)` to
  `__builtin_amdgcn_sudot4(...)` on AMD targets.

hipEngine's corresponding hot raw kernels are in
`hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip` and
`gguf_k_gemv.hip`; they currently unpack scales/mins/nibbles and accumulate in
float. This is why the bounded POC targets q8_1 activation quantization plus
sudot4 inside the raw selected Q4_K dual gate+up kernel before any broad port.

### 2. MTP state maintained by llama.cpp

The MTP speculative implementation stores per-sequence state in
[`common/speculative.cpp#L816-L918`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L816-L918):

- `pending_h`: hidden row used to seed the next MTP draft.
- `verify_h`: hidden rows captured from the target verifier batch.
- `verify_h_rows`: how many verifier hidden rows are available.
- `last_n_drafted`: last draft length, used for recurrent/rollback bookkeeping.

This is the critical lifecycle we only partially approximate today.

### 3. `process()` mirrors target verifier rows into the draft/MTP context

llama.cpp's MTP `process()` is in
[`common/speculative.cpp#L955-L1045`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L955-L1045).
The important behavior:

- It copies target `h_nextn` rows from the target context.
- It builds an MTP batch with token/hidden pairs.
- It calls `llama_decode(ctx_dft, batch)` on the draft/MTP context.
- That decode advances the MTP graph and its K/V state, not just a single isolated
  row.
- It stashes verifier hidden rows in `verify_h` and refreshes `pending_h`.

This is what our old no-context path lacked.  Our new `--mtp-device-kv-cache`
implements a first B1 approximation of the K/V portion, but not the full
llama.cpp process lifecycle or B>1 rollback/transactional semantics.

### 4. `draft()` seeds from `pending_h`, samples from `ctx_dft`, and chains `h_nextn`

llama.cpp's MTP `draft()` is in
[`common/speculative.cpp#L1048-L1168`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1048-L1168):

- It adds the last accepted token `dp.id_last` at `dp.n_past`.
- It overwrites the draft batch embedding with `pending_h`.
- It calls `llama_decode(ctx_dft, batch)`.
- It samples a draft token from the draft/MTP logits.
- It reads `llama_get_embeddings_nextn_ith(ctx_dft, i_batch)` and uses that as
  the hidden seed for the next draft step.
- It repeats up to `n_max`, respecting `p_min`.

This is where llama.cpp gets an actual predictive draft chain.  hipEngine's
`run_draft()` also chains `return_hidden_seed`, but our state before/around that
chain has not matched llama.cpp's `process()`/draft context yet.

### 5. `accept()` chooses the verifier hidden row for the next seed

llama.cpp's MTP `accept()` is in
[`common/speculative.cpp#L1171-L1184`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1171-L1184):

- It chooses `i_h = min(n_accepted, n_rows - 1)`.
- It copies `verify_h[i_h]` into `pending_h`.

This matches our conceptual `pending_hidden_row_index = accepted` logic, but we
must still validate that our captured row is numerically the same row at the same
point in the graph.

### 6. Runtime stats are reported by common speculative stats

The aggregate counters are printed by
[`common/speculative.cpp#L2079-L2103`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L2079-L2103):

- `#gen drafts`
- `#acc drafts`
- `#gen tokens`
- `#acc tokens`
- begin/draft/accept durations

These counters are the cleanest runtime evidence we have without editing the
read-only llama.cpp checkout.

## Source evidence: what hipEngine currently does

All hipEngine source links below point to commit
`98df03ddd00ae682c07e302721343040373e1b55`.

### 1. Acceptance accounting

hipEngine's benchmark implements llama.cpp-style strict acceptance in
[`scripts/gguf_mtp_bench.py#L259-L297`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L259-L297):

- The target samples `[last_token] + accepted_draft_prefix`.
- The first mismatch emits a corrective target token.
- Visible output tokens are accepted draft targets plus the corrective token.

The benchmark also has root/sibling top-K acceptance diagnostics; those are useful
for measuring whether the target is somewhere in the draft distribution, but they
are **not** evidence that the draft chain matches llama.cpp.

### 2. Device-resident MTP KV cache, default off

The new opt-in dense device cache is in
[`hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L636-L760),
with the device-to-device write and dense attention read in
[`mtp_nextn.py#L975-L1002`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L975-L1002).

Accepted-row cheap commit is handled via `kv_write_only` in
[`mtp_nextn.py#L880-L930`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/hipengine/kernels/hip_gfx1100/speculative/mtp_nextn.py#L880-L930),
and the benchmark uses it in
[`scripts/gguf_mtp_bench.py#L1126-L1155`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/scripts/gguf_mtp_bench.py#L1126-L1155).

The fixture proving sequential cache writes match two-row dense attention is
[`tests/test_mtp_dense_device_kv_cache.py#L1-L120`](https://github.com/shisa-ai/hipEngine/blob/98df03ddd00ae682c07e302721343040373e1b55/tests/test_mtp_dense_device_kv_cache.py#L1-L120).

This is useful infrastructure, but it remains default-off because it has not yet
improved same-suite speed/acceptance.

## Runtime trace commands and artifacts

### llama.cpp CLI MTP debug trace

Command:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 12 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --log-file /tmp/hipengine-llamacpp-mtp-cli-debug.log \
  --log-verbosity 5
```

Artifact: `/tmp/hipengine-llamacpp-mtp-cli-debug.log`.

Caveat: `llama-cli --no-conversation` is not supported by this binary.  The
working CLI path is server/chat-style.  The debug trace had `task.n_tokens = 19`.
A `--no-jinja` probe used `task.n_tokens = 17` and still had 100% draft
acceptance, but generation timing collapsed to 0.88 tok/s, so it is not used for
performance comparison.

Aggregate llama.cpp result for the debug trace:

```text
draft acceptance = 1.00000 (8 accepted / 8 generated)
statistics draft-mtp: #calls(b,g,a) = 1 3 3,
  #gen drafts = 3, #acc drafts = 3,
  #gen tokens = 8, #acc tokens = 8,
  dur(b,g,a) = 0.004, 26.710, 0.001 ms
```

Per-draft-call table parsed from the debug log:

| call | history size before draft | drafted | accepted | top-1 draft IDs | corrective / sampled token | new token count |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 19 | 3 | 3 | `[579, 264, 7047]` | 1817 | 23 |
| 2 | 23 | 3 | 3 | `[25, 271, 16]` | 13 | 27 |
| 3 | 27 | 2 | 2 | `[220, 2972, 15771]` | 15771 | 30 |

Interpretation:

- `accepted == drafted` for every MTP call in the trace.
- The verifier call commits `accepted_draft_tokens + 1` visible tokens: 4, 4, and
  3 respectively.
- Visible output / verifier call is therefore `11 / 3 = 3.67`.
- Accepted draft tokens / verifier call is `8 / 3 = 2.67`.

### Target-AR parity trace (new primary blocker)

The cleanest apples-to-apples prompt mode is llama.cpp `--reasoning off`, which
renders the same 21-token text as hipEngine's retained `reasoning='off'` prompt:

```text
<|im_start|>user
Write a Python function that implements merge sort:<|im_end|>
<|im_start|>assistant
<think>

</think>

```

llama.cpp verbose prompt evidence:

```text
common_sampler_init prefill tail:
  248045 <|im_start|>, 74455 assistant, 198 \n,
  248068 <think>, 271 \n\n, 248069 </think>, 271 \n\n
task.n_tokens = 21
next token: 71093 '```'
```

Command/artifact:

```bash
/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-cli \
  -m /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  -p 'Write a Python function that implements merge sort:' \
  -n 1 \
  -ngl 99 \
  --spec-draft-ngl 99 \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  --simple-io \
  --reasoning off \
  --verbose-prompt \
  --log-file /tmp/hipengine-llamacpp-reasoning-off-verbose-prompt.log \
  --log-verbosity 5
```

hipEngine target traces for the same 21-token prompt:

| hipEngine mode | First token after prefill | Next verifier target | Notes |
| --- | --- | --- | --- |
| retained default (`WMMA prefill + GEMV + graph`) | `760` = `The` | `198` = `\n` | `/tmp/hipengine-mtp-target-parity-off-default.json` |
| no WMMA prefill | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_wmma.json` |
| no WMMA/GEMV/graph/decode-repack | `248069` = `</think>` | `271, 16` = `\n\n1` | `/tmp/hipengine-mtp-target-parity-off-no_fast.json` |
| true token-serial `prefill(..., use_bulk=False)` probe | `1919` = `This` | n/a | top-1 from direct session probe |

None match llama.cpp's `71093` code-fence first token.  Therefore the first
confirmed divergence is **target AR prefill/decode/logit parity**, before MTP
draft acceptance.  The MTP acceptance gap is downstream of this target mismatch.

### hipEngine strict B3 trace

Command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --cycles 3 \
  --draft-n-max 3 \
  --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-b3-strict-trace.json
```

Artifact: `/tmp/hipengine-mtp-b3-strict-trace.json`.

Caveat: the hipEngine benchmark applies the Qwen chat prompt wrapper used by its
GGUF harness and reported `Prompt tokens: 21`; this is close but not byte-for-byte
identical to the llama.cpp CLI trace (`19` chat/server tokens).  The strict B3
numbers are still useful because the acceptance gap is large and consistent with
full-suite behavior.

Metrics:

```text
accept_per_draft     = 0.2222
accepted_per_output  = 0.4000
visible/cycle        = 1.6667
tokens_per_sec       = 33.38
speedup_vs_ar_visible= 0.598x
total_accepted       = 2 / 9 draft tokens
```

Per-cycle table:

| cycle | accepted / drafted | target samples | draft IDs | target rank in draft top-10 | visible output | target verify ms | MTP draft ms |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | 0/3 | `[198]` | `[803, 328, 760]` | `[None]` | 1 | 17.94 | 20.31 |
| 1 | 0/3 | `[17]` | `[760, 21397, 25]` | `[2]` | 1 | 18.00 | 19.51 |
| 2 | 2/3 | `[15, 15, 15]` | `[15, 15, 248046]` | `[1, 1, 2]` | 3 | 53.60 | 20.42 |

Interpretation:

- hipEngine's MTP top-1 is often wrong even when the target is near the top of
  the distribution (`target_rank_in_draft_top10 = 2` in cycles 1 and 2).
- This is exactly why root-top40 raised `accepted_per_output` while strict
  `draft_acceptance` stayed extremely low: the target is often in the top-K but
  not the actual draft token.
- B3 strict verification currently commits only `5/3 = 1.67` visible tokens per
  verifier call, far below llama.cpp's `3.67` in the debug trace.

### hipEngine retained/default and device-KV smoke context

Retained root-top40 B1 smoke artifact: `/tmp/hipengine-mtp-with-attn-smoke.json`

```text
accept_per_draft    = 0.0225
accepted_per_output = 0.4737
visible/cycle       = 1.9
tokens_per_sec      = 46.6
total_accepted      = 9 / 400 candidate-count denominator
```

Device-KV B1 smoke artifact:
`/tmp/hipengine-mtp-device-kv-smoke-fastcommit.json`

```text
accept_per_draft    = 0.0187
accepted_per_output = 0.4286
visible/cycle       = 1.75
tokens_per_sec      = 43.68
total_accepted      = 3 / 160 candidate-count denominator
KV rows             = 7 / 12
commit cost         = ~1.2-1.9 ms per accepted-row KV write
```

The device-KV path is much faster than prior host replay/prefix diagnostics, but
it did not reproduce llama.cpp's high B3 acceptance and remains default-off.

## What llama.cpp is doing that hipEngine is not yet doing

### 0. Target AR parity before speculation

llama.cpp and hipEngine must first agree on the target model's greedy token after
the prompt.  They currently do not.  For the same reasoning-off prompt tail,
llama.cpp picks code fence token `71093`; hipEngine picks `760`, `248069`, or
`1919` depending on prefill path.  This points to a target runtime issue, not an
MTP model-quality issue.

Likely places to investigate in order:

1. Prompt/output-row scheduling: llama.cpp decodes the 21-token prompt as a 17-row
   cached prefix plus a 4-row tail; hipEngine bulk/serial row selection may be
   sampling the wrong hidden row.
2. Qwen3.6 hybrid recurrent/Gated Delta Net state: fastpath toggles change the
   first sampled token, which means recurrent/prefill state is affecting target
   semantics.
3. LM-head/argmax parity: direct token-serial hipEngine top-10 does not contain
   llama.cpp's code fence token, so verify output logits against llama.cpp after
   the prompt.
4. Logit processors/biases: llama.cpp biases EOG tokens to `-inf`; confirm
   hipEngine has equivalent generation-time biasing.  This is unlikely to explain
   `71093` vs `760`, but should be checked.

Until this stage matches, MTP token acceptance is not the primary bug.

### A. Full draft-context lifecycle, not just K/V rows

llama.cpp's `process()` decodes verifier rows through `ctx_dft` and updates all
relevant draft-model state.  For Qwen35MoE MTP this primarily means attention K/V,
but it also means the exact graph scheduling, output IDs, and hidden-row selection
are controlled by the same decode path as `draft()`.

hipEngine now has device K/V row writes, but still drives MTP from a Python wrapper
that repeatedly uploads/downloads intermediates and manually chooses which rows to
commit.  It does not yet have the same transactional draft context abstraction.

**Roadmap item:** add an in-tree `GGUFMTPDraftContext` owning device K/V, position,
pending hidden row, accepted verifier rows, and rollback/commit state.  The
benchmark should call this object rather than open-coding row bookkeeping.

### B. B>1 transactional semantics

llama.cpp B3 drafts can be generated, verified, accepted, and rolled forward while
preserving draft context.  hipEngine's `--mtp-device-kv-cache` intentionally
rejects `--draft-n-max != 1` today because we do not yet have safe rollback for
unaccepted draft rows.

**Roadmap item:** implement draft transaction:

1. Save `kv_len_before_draft`.
2. Append draft rows while generating B tokens.
3. Verify target batch.
4. Roll back unaccepted draft rows.
5. Commit accepted target rows and the corrective pending hidden row exactly like
   llama.cpp's `accept()`.

### C. Numeric parity of MTP logits has not been proven

The largest unexplained delta is that llama.cpp's top-1 MTP tokens are accepted
in the debug trace, while hipEngine's top-1 tokens often miss even when the target
is rank 2.  That could be due to:

- hidden seed captured at the wrong point,
- RoPE position/context count mismatch,
- missing or stale MTP K/V context,
- output ID / row selection mismatch,
- quantized GEMV/layout differences in attention, FFN, or shared head,
- sampler/logit post-processing differences.

**Roadmap item:** create a one-step parity harness that records, for the same
prompt/token position:

- token ID entering MTP,
- `pending_h` checksum/norm,
- K/V cache length,
- MTP top-10 logits/tokens,
- `h_nextn` checksum/norm,
- accepted prefix length.

Without editing the read-only llama.cpp checkout, we can only get aggregate and
some debug candidate logs.  For true tensor parity we need either a temporary
instrumented llama.cpp worktree/copy or a local patch that is not committed to the
reference repo.

### D. hipEngine wrapper overhead is still high

Even when B1 device K/V is active, hipEngine draft time is ~8.5 ms/cycle on the
smoke.  The source-level issue is that the correctness-first Python wrapper still
allocates/copies many intermediates.  The WORKLOG follow-up already identified:

- remove Q/gate D2H split,
- avoid Q6_K temporary H2D uploads in attention,
- keep more MTP intermediates resident,
- move from Python orchestration to one or a few persistent launch wrappers.

**Roadmap item:** after numeric parity, port MTP attention+FFN+head into a real
resident path.  Do not optimize the wrong math first.

### E. Root-topK is not a substitute for draft quality

Root-top40 showed the target is frequently *near* the draft distribution, but the
speculative algorithm commits actual draft tokens.  llama.cpp's debug trace has
true top-1 acceptance.  hipEngine's root-topK acceptance is therefore a diagnostic
for rank quality, not a path to B3/B5 break-even.

**Roadmap item:** keep root-topK as diagnostic only.  Promote only changes that
raise strict top-1 chain acceptance and committed tokens/verifier call.

## What we can adopt from llama.cpp

| llama.cpp behavior | Adopt in hipEngine? | Notes |
| --- | --- | --- |
| `pending_h` / `verify_h` lifecycle | Yes | We already use a similar concept; needs parity checksum tests. |
| Draft context with persistent MTP K/V | Yes | Started with default-off B1 dense device cache; must become transactional and resident. |
| `process()` verifier-row mirroring | Yes | Need a resident `process_verifier_rows()` equivalent. |
| B>1 rollback/commit semantics | Yes | Required before meaningful MTP speedups. |
| `p_min` early stop | Yes, diagnostic first | We already have `--draft-p-min`; tune after top-1 parity. |
| Backend sampling | Maybe | llama.cpp logs backend TOP_K support missing on ROCm in this run; hipEngine top-k is already explicit. |
| Chat/server prompt handling | No as-is | hipEngine benchmark prompt protocol must stay fixed and anti-gaming compliant. |
| Loading full model twice for MTP | No | Must keep hipEngine torch-free/lean and use in-model MTP weights only. |

## Prioritized roadmap to effective MTP

### Phase 0 — target AR parity on one prompt

1. Reproduce llama.cpp's 21-token reasoning-off prompt exactly.
2. Add a hipEngine target-only trace that emits:
   - prompt token IDs,
   - chunking/prefill schedule,
   - final hidden-row index sampled,
   - top-20 target logits after prefill,
   - first generated token.
3. Instrument a temporary llama.cpp copy or use verbose prompt + a small tensor
   dump to get the same target top-20 logits.
4. Fix target parity before changing MTP acceptance logic.

Success criterion: hipEngine target prefill chooses `71093` for the documented
reasoning-off prompt, matching llama.cpp, under the narrowest correctness-first
path.  Then optimize back toward the retained fast path.

**2026-06-25 status:** achieved for both correctness-first and retained fast
paths.  The blocker was Qwen3.5 linear-attention GDN K-head broadcast semantics:
llama.cpp/GGML maps value head `v_head` to key head `v_head % num_k_heads`, while
hipEngine inherited the grouped `v_head / repeat` mapping.  After switching the
GDN decode/prefill kernels and CPU replay oracles to the interleaved mapping, the
same 21-token reasoning-off prompt has `initial_prev_token=71093`.  A follow-up
12-token greedy target trace also matches llama.cpp exactly:
`[71093, 12305, 198, 727, 10562, 17885, 10620, 25, 1103, 8, 1411, 1103]`
(decoded as a Python code fence followed by `def merge_sort(arr: list) -> list`).
The single-prompt B3 smoke improves from
the prior `2/9` accepted drafts / `5` visible output tokens to `7/9` accepted
drafts / `10` visible output tokens.

Evidence command:

```bash
python3 scripts/gguf_mtp_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt "Write a Python function that implements merge sort:" \
  --prompt-reasoning off --cycles 3 --draft-n-max 3 --root-topk-accept 1 \
  --output /tmp/hipengine-mtp-target-parity-final-c3.json
```

### Phase 1 — exact MTP trace parity on one prompt

1. Add a hipEngine trace mode that emits per-step JSON:
   - prompt token IDs,
   - previous token,
   - position,
   - pending hidden norm/checksum,
   - MTP KV length,
   - MTP top-10 IDs/logits/probs,
   - target samples,
   - accepted prefix length,
   - committed output tokens.
2. Produce a temporary instrumented llama.cpp copy or local patch that emits the
   same fields from `common_speculative_impl_draft_mtp`.
3. Compare the first divergence.
4. Fix math/state mismatches before doing more performance work.

Success criterion: on the same prompt/token positions, hipEngine and llama.cpp
produce the same MTP top-1/top-K tokens for at least the first several draft
steps, or we can explain every difference.

### Phase 2 — B3 transactional device KV

1. Promote the B1 device cache into a draft-context object.
2. Add rollback/commit around B>1 draft rows.
3. Validate with a CPU/synthetic fixture and then a GGUF smoke.
4. Run strict B3, no root-topK, same prompt.

Success criterion: strict B3 `accepted_draft_tokens / generated_draft_tokens`
substantially improves over the old `2/9 = 22.2%` smoke and approaches the
llama.cpp debug trace on the same prompt.

**2026-06-25 status:** achieved for the diagnostic llama.cpp-lifecycle path.  The
missing piece after target parity was the draft model context lifecycle: replay
the shifted prompt rows into a device-resident MTP KV cache, keep the cycle-start
row, roll back rejected speculative rows, and commit accepted rows with
verifier-derived target hidden seeds.  With `--mtp-context-replay`,
`--mtp-device-kv-cache`, `--draft-n-max 3`, and `--root-topk-accept 1`, the same
single-prompt smoke reaches `9/9 = 100%` accepted drafts and `12` visible output
tokens over three verifier calls.

### Phase 3 — full-suite strict acceptance before speed claims

Run `mtpbench-code-general-ja.jsonl` in strict mode and record:

- accepted draft tokens / verifier call,
- visible output tokens / verifier call,
- strict draft acceptance,
- rank histogram for target token in MTP top-K,
- raw tok/s.

Success criterion: committed tokens/verifier call rises enough that speed work is
worthwhile.  If strict acceptance remains low, return to Phase 1.

### Phase 4 — performance optimization only after parity

Once strict acceptance is credible:

- fuse resident MTP attention/FFN/head launches,
- eliminate host-side intermediate copies,
- pre-upload/cache Q6_K weights and scratch buffers,
- replace sequential target verification with a rollback-safe block verifier,
- profile verifier MoE grouping/budgeting to reduce `eta`,
- revisit B2/B3/B5 economics.

**2026-06-25 status:** first draft-side performance wins landed, and a
rollback-safe target continuation block verifier now exists, but performance
parity is still blocked by verifier kernel shape.  Batching accepted-row MTP KV
commit into one `kv_write_only` pass improved the corrected B3 merge-sort smoke
from `41.7` to `42.3 tok/s` (`15/15` strict accepts over five cycles).  A
hot-token draft LM-head cap of `32768` improved the same one-step-graph smoke to
`44.5 tok/s` with unchanged `15/15`, but it is prompt-sensitive and remains
diagnostic until full-suite validation.  The new `--target-block-verify` path
snapshots linear recurrent state, runs the target over `[prev]+drafts` as a
continuation block, records target IDs + FP32 hidden seeds, and restores/replays
the consumed prefix on partial accepts.  Its first version was exact (`15/15`) but
slow on the B3+32k smoke (`37.8 tok/s`, verifier `~90 ms/cycle`) because the
selected/WMMA prefill kernels are the wrong shape for tiny B.  The verifier now
defaults to the GEMV prefill fallback internally (`--no-target-block-wmma-prefill`)
while leaving normal prompt prefill WMMA enabled; that lifts the same B3+32k
smoke to `48.1 tok/s` with unchanged `15/15` and verifier `~61-66 ms/cycle`
(except variance on late cycles).  B5 remains unattractive because a partial
rollback cycle costs hundreds of ms in the generic restore/replay path.

**2026-06-26 profiling — the verifier is WORK-bound, not launch-bound.**  Two
single-process diagnostics overturn the earlier "captured HIP graph / C-level
dispatch loop" hypothesis for the #1 verifier fix:

- *Row-scaling* (`verify_rowscale.py`): `verify_target_block` GEMV wall-time is
  ~flat per row (`24 ms/row`, fit `23 ms + 24 ms·rows`); rows=128 costs **26× rows=4**.
  If launch-overhead-bound, rows=4 and rows=128 would cost nearly the same
  (~420 launches either way).  WMMA per-row falls `31.5 → 8.86 ms/row` (amortizes
  but high fixed cost at B=4).
- *Per-family* (`verify_family.py`, rows=4 GEMV): dense Q4_K projections
  (`launch_gguf_linear`) **44%**, MoE selected-expert GEMV **28%**, GDN 6%,
  router 7%, Q6_K lm-head sample 5%.  72% is quantized matmuls run per-row.
  Cross-check: `launch_gguf_linear` ≈ 89 µs/call vs ~20 µs B=4 weight-bandwidth
  floor ⇒ **~4× over floor**, i.e. the Q4_K weight is reloaded once per row.

Initial root cause: at rows>1 with WMMA off, `launch_gguf_linear` uses the decode-shaped
`dense_gemv:prefill_out` = `dense_gemv_out_kernel`
(`hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip:122`), grid `(out_col, row)`
— one block per (column,row), so the column is re-dequantized per row.  This is
exactly llama.cpp's advantage: GGML batches the 4 rows into one weight-load-
amortized matmul (~8.9 ms total ≈ 2.2 ms/row).

**2026-06-27 update: dense rowtile landed, but the bottleneck moved.**
The small-B rowtile idea is implemented for raw Q4_K and raw K-family
Q8_0/Q5_K/Q6_K dense GEMVs, bit-exact against the per-row kernels, and default-on
for rows 2..8 when WMMA is off. Microbench speedups at B=4 are ~3x on dense
projection shapes, and a B3 verifier smoke with the 32k draft cap stayed exact at
`48.77 tok/s` (`15/15`, verifier ~61 ms/cycle), flat vs the pre-rowtile `48.1`
within run noise.

A clean sync-free rocprof pass corrected the family attribution: selected-expert
MoE GEMV is the real top bucket, not dense projection row reload. The hot verifier
GPU-time shares are:

| Kernel family | Share |
| --- | ---: |
| `gguf_q4_k_selected_dual_prefill_out_kernel` (MoE gate+up) | ~36% |
| `gguf_k_selected_pack8_prefill_out_kernel` (MoE down, Q5_K) | ~18% |
| residual per-row dense `gguf_k_prefill_out_kernel` | ~17% |
| dense rowtile `gguf_k_prefill_out_rowtile_kernel` | ~11% |
| GDN recurrent/rmsnorm-gate | ~8% |
| Q6_K lm-head pack8 | ~6% |

Two cheap MoE ideas are now ruled out:

- Row amortization/group-by-expert does not apply at B=4. A microbench with
  qwen35moe shapes showed 32 same-expert rows at `0.567 ms` vs 32 distinct
  experts at `0.882 ms`; B=4/top_k=8 selects ~30 distinct experts, so there is
  essentially no expert overlap to reuse.
- `expert_sidecar`/pack8 gate+up for the verifier is ~15x slower (`103.4 ms`
  raw vs `1588.4 ms` sidecar) because per-layer H2D movement dominates.

**Current #1 verifier task:** selected-MoE remains the verifier bottleneck, but
the straightforward T16 dp4a ports are not retainable defaults. The raw
selected-dual Q4_K POC is positive (`0.946 ms -> 0.357 ms` at the qwen35moe
verifier shape), but production B3 uses T16 decode-repack. The T16 Q4_K split
gate/up port launches under `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A=1` and cuts
the row-bulk split kernel in the short trace (`~172 us -> ~142 us`), but B3
stays flat. The T16 Q5_K selected-down port launches under
`HIPENGINE_GGUF_T16_SELECTED_DP4A=1` and is `1.10x` faster in isolation, but B3
regresses (`47.62 tok/s`, warm `48.44`) and the c1 synthetic top-1 is marginal
(`0.875`). Next work should either adapt the layout closer to GGML's q8_1/x4
vector-dot path or find a selected-down reduction/layout change that improves
B3 without top-1 drift; do not keep porting Q6/dense dp4a as a default path
without that gate.

Captured-graph/C-loop work is deprioritized to a later launch-overhead layer
after GEMV instruction efficiency improves. Cheaper partial-accept rollback
remains important for B5, but it does not address the full-accept B3 verifier
hot path.

**2026-06-28 correction — the verifier is ~50/50 HOST-dispatch-bound; the
deprioritization above was wrong.**  A warm `verify_target_block` (rows=4)
issues **875 kernel launches** (~22/layer × 40 layers); the pure host launch
dispatch is **~54 ms** (~52% of the wall).  A dp4a A/B under `rocprofv3` shows
dp4a genuinely cuts GPU kernel time −35% (MoE dual `1256→400 ms`, 3.14×) yet the
E2E wall stays flat/worse because dp4a *adds* launches (per-layer q8_1 quantize)
and the host-dispatch floor dominates.  So GEMV instruction efficiency (dp4a,
rowtile) cannot move E2E until the ~54 ms host-launch floor is removed.  The
**primary lever is collapsing the 875 launches** — HIP graph capture (gated by
the 3rd-relaunch GDN corruption, see WORKLOG 2026-06-28) or a C-level multi-layer
dispatch loop — exactly the original plan.  dp4a/rowtile are complementary GPU
wins that materialize *after* the launch floor is cut.  llama.cpp runs the whole
4-token verifier as one fused GGML graph (~9 ms); the 875-launch host floor is
the core of the gap.

**2026-06-30 correction — the 2026-06-28 "host-bound" claim was the OLD serial
per-row route; the CURRENT block verify is GPU-kernel-BOUND (~90%).**  Decisive
differential measurement (`scratchpad/launch_overhead_decomp_blockverify.py`,
wall = clean `perf_counter` over the block loop, GPU = `rocprofv3 --kernel-trace`
DurationNs sum, both differenced over N=8 vs N=32 to cancel prefill) on the
production `verify_target_block(rows=4, bulk)` path with the landed rowtile
lm-head:

| per-block (rows=4) | ms |
| --- | --- |
| wall (host+GPU overlapped+sync) | **42.40** |
| GPU kernel-sum | **38.08** |
| host EXPOSED (wall − GPU) | **4.33 (10.2%)** |

A standalone async-issue probe (`scratchpad/launch_overhead_decomp.py`) confirms
per-kernel-launch dispatch is **~12 µs**, so 875 launches ≈ **10.5 ms** of host
issue — fully overlapped behind the 38 ms of GPU work, leaving only ~4.3 ms
exposed.  The "~54 ms host floor" did not reproduce on the block path; it was the
serial route's per-row-synced dispatch.  **Consequences, all evidence-backed:**

- **Graph capture / fused draft+verify graph is REFUTED as a lever** (and the
  GDN-corruption fix it requires would be wasted effort): only 10.2% host is
  exposed, and ROCm 7.x `hipGraphLaunch` re-pays per-node overhead at ~1000-node
  DAGs (M12.1 `2026-05-22-...graph-capture-diagnostic.json`, L3/L13 in DFLASH).
  Best case — eliminating *all* exposed host — caps the verify at 38.1 ms (≈ +11%
  → ~1.22× absolute ceiling), and that is physically unreachable.
- **C-loop / Python dispatch memoization is REFUTED**: same ≤10% host ceiling.
- **The 38.1 ms GPU kernel-sum IS the wall.**  Only cheaper kernels cut it:
  pipeline-wide **dp4a/q8_1** (REFUTED — ja greedy top-1 0.700 < 0.90 gate, even
  MoE-selected, `scratchpad/dp4a-verify-full.json`) or fewer FLOPs (quality loss).
- The **lm-head rowtile** (landed, bit-exact, `1.0534×→1.1134×`) captured the
  only shared-weight GPU amortization (all verify rows read the same head).  The
  MoE is per-row **disjoint**-weight (top-8 of 256, rarely shared across rows) →
  no cross-row amortization (grouping de-risk: all-distinct only 1.40–1.54× of
  all-same, L2-served) → near its efficient exact point.

**Net:** hipEngine's GGUF block verify is at its **exact-precision GPU-compute
ceiling**.  The residual gap to llama 1.342× is purely llama's pipeline-wide
dp4a/q8_1 precision tradeoff, which violates hipEngine's ja correctness gate.
hipEngine reaches `1.1134×` (60.8 tok/s, 90.3% of llama's 67.3) while **beating
llama on AR** (54.6 vs 50.1 tok/s) and on precision (exact; passes the ja gate
llama's recipe fails).  Closing the rest is not a config/kernel/graph lever — it
requires accepting llama's precision loss, which the stated correctness guard
forbids.

Success criterion: same-protocol full-suite row improves all three: raw weighted
decode tok/s, accepted/output, and strict draft acceptance.

## Bottom line

**2026-06-30 FINAL — investigation complete, root cause confirmed bottom-up.**
Retained win: bit-exact Q6_K T16 rowtile lm-head kernel, GGUF MTP `1.0534x ->
1.1134x AR` (60.8 tok/s = 90.3% of llama.cpp's 67.3; hipEngine AR 54.6 > llama AR
50.1). Every llama MTP pipeline lever was implemented/tested and either shipped or
refuted with committed full-suite artifacts: dp4a (only -4% on the GPU-bound verify
AND fails the ja gate, top-1 0.700), HIP graph capture (verify is 90% GPU-bound,
ROCm re-pays per-node, M12.1), MoE grouping (L2-served), vocab-cap recover (-1.3%),
p_min 0/0.3/0.5 (0.5 optimal), probe/no-probe (probe optimal), budgets 1-8 (plateau
at B5), generation length (uplift stable), and context (validated CORRECT via a new
mtp_dense_attn_f32 gate; the model's NextN simply does not benefit from it). The
llama baseline was audited apples-to-apples (matching metric defs; gap is real).

**The absolute number 67.3 tok/s is unreachable on hipEngine in ANY precision
regime**, not merely the exact one. Matching it needs a `1.233x` uplift over
hipEngine's 54.6 AR; measured uplift ceilings are exact `1.114x` and dp4a `~1.13x`
(prior session) - both below 1.233x. llama reaches 67.3 via a *slower* AR (50.1) x
a *higher* uplift (1.342x); hipEngine's faster-AR / exact-precision profile has a
different optimum (higher AR, lower uplift). A cross-tool draft-logit comparison vs
a captured llama oracle (`benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json`)
confirmed the residual is the exact-vs-dp4a PRECISION REGIME manifesting through the
whole speculative economy (seed hiddens, draft logits, verification targets all
differ because hipEngine is exact and llama is dp4a) - NOT a hipEngine bug. Closing
to llama requires adopting llama's dp4a regime end-to-end, which fails hipEngine's
ja correctness gate (the stated guard) and still would not reach 67.3 given
hipEngine's already-faster exact AR. 1.1134x is the exact-precision optimum.

---


and MTP draft context with verifier-row processing, persistent draft K/V state,
hidden-row handoff, and B>1 accept/rollback semantics.  In the short debug trace
it commits `3.67` visible tokens per verifier call with `100%` strict draft
acceptance.

hipEngine now matches llama.cpp's documented reasoning-off target AR trace and,
with the llama.cpp-style context replay + device MTP KV lifecycle, reaches strict
B3 `9/9` (and `15/15` over five cycles) on the merge-sort smoke. Correctness
parity is therefore solved.

The remaining gap is performance: ~48.8 vs ~89.6 tok/s (~1.8-1.9x) on gfx1151.
The latest evidence says the q8_1+sudot4 recipe is valid, but the layout
decision matters more than the intrinsic itself: raw Q4_K selected-dual is
`~2.65x` faster in isolation, raw Q5_K/Q6_K selected-down is `~2.32x`/`~1.62x`
faster including q8_1 quantization, and the raw B3 verifier improves
`31.63 -> 39.61 tok/s`; meanwhile T16 Q4_K split gate/up is only `~1.04x`, T16
Q5_K selected-down is only `~1.10x`, and the production decode-repack smoke is
still faster at `51.31 tok/s`. Dense rowtile is already landed and retained as a
kernel-level win, but selected MoE dominates. Next: broad-port a GGML-like
q8_1/x4 vector-dot layout into the production GGUF verifier path, then promote
only the same-protocol B3/full-suite non-regressive pieces. Graph/C loop work,
resident MTP draft consolidation, and rollback improvements remain on the
roadmap after the GEMV instruction path is de-risked. These remain single-prompt
diagnostics, not benchmark rows.

## 2026-07-02 correction - prompt catch-up row parity is not solved

The latest fine-grained attention traces overturn the broad "correctness parity
is solved" statement above.  hipEngine and llama.cpp match the generated late
rows closely, but they do **not** match the early prompt rows in the MTP draft
KV cache.  The first concrete mismatch is prompt row 2, and that row remains
visible and high-impact at the position-49 draft divergence.

Diagnostic artifacts:

- hipEngine row-probe run:
  `/tmp/hipengine-mtp-rowprobe/hipengine-stage.json`
- llama.cpp row-probe run:
  `/tmp/hipengine-mtp-rowprobe2/llamacpp-stage.jsonl`
- compact artifact:
  `benchmarks/results/2026-07-02-mtp-rowprobe-diagnostic.json`

Comparable point: depth 0, input token `1103`, position `49`.

| Field | hipEngine llama-compat | llama.cpp HIP |
| --- | ---: | ---: |
| Drafts at this point | `[65342, 18078]` | `[8, 1411]` |
| Cache tokens allocated | `50` | `256` |
| Visible attention rows | `50` | `50` |
| Host recompute vs GPU attention | `4.1e-7` mean abs | `4.8e-4` mean abs |
| Top-row histogram | row 49: 9 heads, row 48: 7 heads | row 49: 8 heads, row 48: 7 heads, row 2: 1 head |
| Row 48/49 K/V first4 agreement | close (`<=~0.055` max abs) | close (`<=~0.055` max abs) |
| Row 2 K/V first4 agreement | bad | bad |

Row 2 is now labeled in llama.cpp via ubatch metadata: token `198` at prompt
position `2`.  hipEngine's shifted prompt replay intends the same row identity:
token `198` at position `2`, paired with the target hidden row from prompt
position `1`.

Representative row-2 probes:

| Head | hipEngine row-2 score / weight | llama row-2 score / weight |
| ---: | ---: | ---: |
| qh7 / kv0 | `8.1026 / 0.00002697` | `17.9259 / 0.2181` |
| qh10 / kv1 | `10.3185 / 0.001087` | `15.4524 / 0.1456` |
| qh12 / kv1 | `6.8666 / 0.002343` | `12.5809 / 0.3911` |
| qh13 / kv1 | `3.8032 / 0.000613` | `9.8557 / 0.1965` |

Representative row-2 K/V first4:

| KV head | hipEngine K first4 | llama.cpp K first4 | hipEngine V first4 | llama.cpp V first4 |
| ---: | --- | --- | --- | --- |
| kv0 | `[0.130057, 0.021326, 0.719520, 1.068448]` | `[0.047455, -0.057037, -0.009056, 0.105896]` | `[-1.019979, -0.621223, 0.708425, -1.369513]` | `[-0.107239, -0.257080, 4.664063, 1.229492]` |
| kv1 | `[0.607833, 1.223730, 0.277249, 2.120688]` | `[-0.002890, -0.013268, -0.006294, 0.018204]` | `[1.245373, 5.050378, 1.246049, -1.432447]` | `[7.609375, 5.277344, 4.574219, -9.156250]` |

Interpretation:

- This is **not** a flash-attention math bug.  llama.cpp FA matches a host dense
  recompute closely enough at the same row, and hipEngine's dense attention
  matches its own host recompute to `~1e-6`.
- This is **not** a late generated-row layout problem.  Rows 48 and 49 agree
  closely between the engines.
- This is **not** a mask visibility problem.  llama row 2 is explicitly visible
  (`mask = 0`, visible count 50).
- At the time of this diagnostic, the semantic split was the **initial MTP
  prompt catch-up KV source**.  hipEngine's active prompt replay was still
  feeding initial MTP device KV through the legacy writer, while llama.cpp
  populated the draft context through `llama_decode(ctx_dft, batch)` using the
  shifted target `h_nextn` rows.  Those were not numerically equivalent for
  early prompt row 2.

Resolution status: fixed by `resident_write_kv_rows`, not by the hidden-row tap
alone.

The all-row bulk `h_nextn`/post-output-norm tap is still required because
llama.cpp's MTP `process()` consumes all target prompt rows.  But rerunning the
row probe with only that tap did **not** move row 2: hipEngine still drafted
`[65342, 18078]`, and row 2 stayed on the old K/V values.  The actual semantic
split was that initial prompt MTP device KV still used the legacy
`run_draft(..., kv_write_only=True)` writer, while accepted/generated rows used
the resident writer that already matched llama.cpp on late rows.

After switching initial prompt catch-up KV to `resident_write_kv_rows`, the
same row-probe point now matches llama.cpp:

| Field | hipEngine before | hipEngine after | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Initial prompt KV writer | legacy `run_draft(..., kv_write_only=True)` | `resident_write_kv_rows` | `llama_decode(ctx_dft, batch)` |
| Drafts at token `1103`, position `49` | `[65342, 18078]` | **`[8, 1411]`** | **`[8, 1411]`** |
| Target at that point | `[65342, 18078, 28649]` | `[65342]` after rejection | `[65342]` after rejection |
| Row 2 qh7 weight | `2.697e-5` | **`0.227`** | **`0.218`** |
| Row 2 qh12 weight | `0.00234` | **`0.408`** | **`0.391`** |
| Row 2 K/V first4 max delta vs llama | bad | `<=0.0378` | reference |

Artifact: `benchmarks/results/2026-07-02-mtp-resident-initial-kv-diagnostic.json`.

Residual: row 0, the zero-pending prompt boundary row, still differs.  At the
position-49 decision it is not the governing row; row 2 and rows 48/49 now match
closely enough to produce the same draft decision.  Keep row 0 on the semantic
debt list, but do not block perf attribution on it unless a later trace shows it
changes acceptance.

Full-suite follow-up: the resident-init rerun completed in
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-residentinit-routerrow-full.json`
and the subsequent shared-gate scalar-dot rerun is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-sharedgate-routerrow-full.json`;
the later parallel-attention rerun is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-parallelattn-clean-rerun-full.json`;
the draft-only dense-Q8 rerun is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-draftonly-full.json`.
That row moved **64.41 -> 75.15 tok/s** and cycle wall **15.547 -> 13.325
ms/output**, but it is now superseded as unsafe direct-state evidence. The
semantic-safe follow-up is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directstate-prefillgdn-partialfix-full.json`:
B2 **50.96 tok/s**, **19.645 ms/output**, **0.9312x AR**, acc/output **0.606**,
draft acceptance **0.770**, target rows/output **1.331**, verifier drain
**17.222 ms/output**, replay/commit **2.775 ms/output**, **38** replay rows, and
**46** discarded rows. The current retained safe follow-up is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`:
B2 **51.85 tok/s**, **19.308 ms/output**, **0.9472x AR**, verifier drain
**16.891 ms/output**, and replay/commit **2.489 ms/output** with the same
acceptance and row economy. The current llama-replication follow-up is
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`:
B2 **60.56 tok/s**, **16.534 ms/output**, **1.1055x AR**, acc/output **0.609**,
draft acceptance **0.780**, target rows/output **1.172**, verifier drain
**14.071 ms/output**, replay/commit **0.043 ms/output**, **0** replay rows,
and **44** discarded rows. The active parity work is no longer prompt-context
semantics, draft wall, or serial replay; it is closing the remaining target
verifier forward gap against llama.cpp HIP.

## 2026-07-02 direct-state lifecycle comparator

The next mismatch is no longer the initial prompt MTP K/V source.  A new
diagnostic mode in `scripts/gguf_mtp_forced_target_probe.py`,
`--state-lifecycle-compare`, runs two target-verifier lifecycle policies through
the same proposal trace and hashes the post-cycle FP32 hidden seed plus every
linear-attention Conv/GDN resident state:

- `replay`: score the target block, restore on partial accept/reject, then
  serial-replay the accepted prefix.
- `direct`: score the target block with captured Conv/GDN rows, then commit the
  accepted verifier row directly with `_commit_verify_linear_state_row`.

Full-attention K/V is deliberately not hashed yet because reset does not zero
unused cache tails, so byte hashes would include irrelevant capacity.

Trace:
`/tmp/hipengine-mtp-proposal-trace/hipengine-active-draftdenseq8-draftonly-f32selectedintermediate-c32.json`,
cycle 12, `--target-block-verify-mode bulk --no-target-block-wmma-prefill`.

| Diagnostic | Extra env | Cycles compared | First mismatch | Visible tokens | State mismatch | Interpretation |
| --- | --- | ---: | --- | --- | ---: | --- |
| Base direct capture | none beyond the active F32 selected-intermediate route | 13 | cycle 0 | replay/direct both `[12305, 198, 727]` | 59 | Direct capture is not byte-identical to replay/block state even on the first full-accept cycle. |
| Prefill-shaped GDN capture | `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` | 13 | cycle 3 | replay/direct both `[65342]` | 61 | Cycles 0-2 become state-identical, then the first partial/reject cycle diverges. |

Artifacts:

- `benchmarks/results/2026-07-02-mtp-state-lifecycle-compare.json`
- `benchmarks/results/2026-07-02-mtp-state-lifecycle-prefillgdn-compare.json`

Conclusion: prefill-shaped Conv/GDN row capture fixes the full-accept state
equality problem for the early trace, but the current direct commit is still not
the replay/llama-style lifecycle when a block partially accepts or rejects.  The
next semantic target is accepted-row state after partial accept/reject: either
direct capture must become prefix-equivalent for the accepted row, or the
llama-compat path needs a narrow prefix-equivalent commit mechanism for rejected
blocks without falling back to full serial replay as the steady-state verifier.

Implementation update: `--llama-compat` now forces
`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`, separates full-block direct commit
from partial direct commit, and treats bulk partial/reject commits as not exact.
The path still commits captured rows directly on full-accept blocks, but restores
and serial-replays the accepted prefix for rejected bulk blocks.

Post-fix lifecycle artifact:
`benchmarks/results/2026-07-02-mtp-state-lifecycle-prefillgdn-partialfix-compare.json`.
Result: 13 cycles compared, `first_mismatch: null`.  At cycle 3, replay state
source is `serial_exact_accepted_prefix`, direct-policy state source is
`serial_exact_replay`, both emit `[65342]`, mismatch count is 0, and
`direct_partial_commit_exact=false`.

Real bench smoke artifact:
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directstate-prefillgdn-partialfix-smoke.json`.
This is semantic validation, not a retained speed row.  It confirms
`llama_compat=true`, `verify_capture_prefill_gdn_env=1`, total accepted
`24/26`, cycle 3 draft `[8, 1411]` rejects to target `[65342]`, and the verifier
transaction mix is 12 direct-commit rows plus 1 replay/serial row over the
13-cycle smoke.

Full-suite semantic-safe follow-up:
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directstate-prefillgdn-partialfix-full.json`.
Command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly \
  --budgets 2 \
  --record-cycle-stage-timings \
  --output benchmarks/results/2026-07-02-ar-mtp-llama-compat-directstate-prefillgdn-partialfix-full.json
```

Result: AR **54.73 tok/s**, semantic-safe B2 **50.96 tok/s** (**0.9312x AR**),
cycle wall **19.645 ms/output**, acc/output **0.606**, draft acceptance
**0.770**, target rows/output **1.331**, verifier drain **17.222 ms/output**,
and replay/commit **2.775 ms/output**.  The unsafe `75.15 tok/s` row hid this by
direct-committing rejected/partial bulk-block state.  At this point the
semantic-safe fix target was a prefix-equivalent partial commit path that kept
`first_mismatch: null`; the later llama-replication row instead adopts
llama.cpp-style direct commit and keeps this serial-state row as the exact
control.

Serial state-only replay follow-up:
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`.
Command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-serialstate \
  --budgets 2 \
  --record-cycle-stage-timings \
  --output benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json
```

Result: AR **54.74 tok/s**, semantic-safe B2 **51.85 tok/s** (**0.9472x AR**),
cycle wall **19.308 ms/output**, acc/output **0.606**, draft acceptance
**0.770**, target rows/output **1.331**, verifier drain **16.891 ms/output**,
and replay/commit **2.489 ms/output**.  This keeps the lifecycle comparator clean
while skipping only replay LM-head sampling; it is the exact-state control, not
the active llama-replication timing row.

Llama-style directcommit follow-up:
`benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`.
Command:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 \
  --record-cycle-stage-timings \
  --output benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json
```

Historical copied-state result: AR **54.78 tok/s**, active replication B2 **60.56 tok/s**
(**1.1055x AR**), cycle wall **16.534 ms/output**, acc/output **0.609**,
draft acceptance **0.780**, target rows/output **1.172**, verifier drain
**14.071 ms/output**, and replay/commit **0.043 ms/output**.  The corresponding
lifecycle diagnostic
`benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`
diverges from serial replay at cycle 3 while emitting the same visible token
`[65342]`; that divergence is expected for the llama-replication lane. The
initial no-copy natural24 follow-up was **71.42 tok/s** / **14.025 ms/output**,
but it used only 10 cycles and two low-accept prompts stopped before the
24-token cap. The corrected BF16-head cyclecap24 row superseded it at
**71.11 tok/s** / **14.087 ms/output** with all 10 prompts reaching 24 output
tokens. The refreshed active-route cyclecap24 row is **71.52 tok/s** /
**14.005 ms/output** with unchanged acceptance and row economy, but it did not
enable `--verify-lm-head-q6-top1-dp4a` despite the `f32head` artifact name. The
actual current-shape verifier-head diagnostic regresses to **66.45 tok/s** /
**15.072 ms/output**. The same route's fixed-cycle provenance row remains
**72.23 tok/s** and **13.865 ms/output**.
