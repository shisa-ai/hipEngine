# Debugging the gfx1151 128K prefill stall

**Status:** open, reproducible, intermittent, no production-safe workaround<br>
**Last updated:** 2026-07-16<br>
**Primary platform:** Framework Desktop, Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`)<br>
**Current publication decision:** hipEngine GGUF rows through 64K are retained;
the current repeated-128K row is blocked.

This document is the handoff and escalation record for a silent long-prefill
no-progress state observed in hipEngine's torch-free HIP runtime. It separates
what is measured from what is inferred, records the controls already run, and
defines what to capture and report next.

> **Important:** no named hipEngine kernel, HIP call, KFD queue, MES command, or
> firmware component has been proven faulty. The leading scheduler/queue-retirement
> interpretation is a hypothesis supported by elimination and symptom evidence,
> not root-cause proof.

## Executive summary

A single-process, single-GPU Qwen3.6 35B-A3B Q4_K_M 128K bulk prefill can stop
making device-visible progress after previously completing one or more identical
prefills. The process and machine remain alive, but the GPU remains indefinitely
at approximately **100% reported activity, 2.9 GHz, and only 42-59 W** instead of
the roughly 120 W working regime. Device memory remains allocated and stable.
The kernel journal remains clean. Terminating the process immediately returns
the GPU to idle without a reset.

The failure:

- occurs with one resident model and one HIP hardware queue;
- occurs under both tested HIP 7.13 and HIP 7.15 user-space stacks;
- occurs with SDMA disabled;
- occurs after restoring older router and metadata policies;
- occurs after removing repeated metadata H2D copies;
- occurs after removing compact-MoE scalar D2H reads;
- moves between chunks, layers, warmup/measured passes, and fresh processes;
- is not reliably prevented by same-stream layer completion markers;
- has produced no amdgpu/KFD fault, VM/page fault, ring timeout, reset, OOM, or
  thermal warning in the captured run windows.

A persistent same-stream flight recorder proves that host submission gets ahead
of retirement. In three layer-granularity failures, the final retired checkpoint
lags host submission by exactly two markers. The pending previous marker is
behind a linear-attention layer each time, but the exact layer and outer chunk
move. This narrows the window; it does **not** identify the linear-attention
kernel, marker kernel, or any other named dispatch as the cause.

The highest-value current-boot next step is to snapshot KFD's existing
`rls`, `mqds`, and `hqds` debugfs views during a healthy run and once after the
persistent state begins. The next traced run should force rocprofiler's legacy
queue interception. The next diagnostic boot should enable MES event logging.

## User-visible impact and scope

### Affected contract

| Field | Value |
| --- | --- |
| Engine | hipEngine torch-free GGUF runtime |
| Model | Qwen3.6-35B-A3B UD-Q4_K_M GGUF |
| Model size | 22,663,387,424 bytes |
| Sampled model SHA-256 | `936659d614707776d8e6ca1fb8595991159e78361bff2e3a3616aa91564c89fb` |
| Quant / KV | `gguf_q4_k_m` / BF16 KV |
| Prompt | 131,072 repetitions of token ID `9707` |
| Decode | 128 greedy graph-replay tokens after prefill |
| Lifecycle gate | one discarded warmup plus three measured resets in one resident process |
| Expected prefill time | about 258-262 seconds at approximately 500 tok/s |
| Expected final token | `9707` for every completed repetition |
| Tracked peak allocation | approximately 25.493 GiB |

The long gate is intentionally stronger than a one-shot throughput run. A
successful single 128K pass does not establish lifecycle safety.

### Known-good neighboring scope

- The same current production path completes right-sized 512, 1K, 4K, 32K, and
  64K warmup+3 processes with exact final IDs and finite logits.
- Multiple isolated 128K runs and complete warmup+3 processes have succeeded.
- A prior current-code decode-only 128K profile completes. The open failure is
  long/repeated **prefill lifecycle**, not an unconditional inability to address
  128K KV or decode state.
- llama.cpp HIP and Vulkan have completed 128K workloads on this machine under
  their own runtime and kernel paths. They are useful controls, not equivalent
  reproductions.

### Failure signature

A run is considered in the persistent stall state only when all of these hold:

1. no benchmark output or recorder retirement progress for several minutes;
2. reported GPU activity remains 100%;
3. GFX clock remains approximately 2.9 GHz;
4. package power collapses to a stable low band, historically 42-59 W;
5. allocated memory remains fixed;
6. the process and host remain responsive;
7. no relevant kernel fault/reset line appears;
8. process termination immediately restores idle without a GPU reset.

Do not classify a slow compiler probe, a long but advancing 128K chunk, profiler
startup, or a transient 40-second low-power interval as this failure.

## Reference system

The main evidence set uses:

| Component | Version / value |
| --- | --- |
| System | Framework Desktop / AMD Ryzen AI MAX+ 395 |
| GPU | AMD Radeon 8060S Graphics, `gfx1151`, PCI `0000:c1:00.0` |
| Kernel | `7.1.3-2-cachyos` |
| BIOS / VBIOS | BIOS 03.04 (2025-11-19) / `113-STRXLGEN-001` |
| MES firmware | `0x00000088` |
| MES KIQ firmware | `0x0000006f` |
| Main HIP stack | TheRock HIP `7.15.0-0000000` |
| Comparison HIP stack | HIP `7.13.60980-c76140fa27` |
| Main compiler | AMD clang 23, `aa451e1f...+PATCHED:440716f8...` |
| Queue policy | `GPU_MAX_HW_QUEUES=1` unless a row explicitly says default/four |
| amdgpu scheduler policy | `sched_policy=0` (hardware scheduling enabled) |
| CWSR | `cwsr_enable=1` |

Current relevant amdgpu values are:

```text
mes_log_enable=0
sched_policy=0
gpu_recovery=-1
send_sigterm=0
debug_evictions=N
halt_if_hws_hang=0
timeout_period=0
cwsr_enable=1
```

Record these again after every boot. Kernel/module behavior, not only HIP
user-space version, is part of the reproduction identity.

## Canonical reproduction

Prebuild every JIT `.so` and the compiler-version cache file outside the bounded
run. The direct command shape is:

```bash
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gfx1151-hipcc-version.txt
export HIPENGINE_HIPCC_VERSION_FILE=/tmp/hipengine-gfx1151-hipcc-version.txt
export GPU_MAX_HW_QUEUES=1

python3 scripts/qwen35_readme_sweep.py \
  --engine gguf \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --quant gguf_q4_k_m \
  --backend hip_gfx1151 \
  --workloads 128K/128 \
  --warmup-runs 1 --measured-runs 3 \
  --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
  --require-cached-build \
  --json /tmp/gfx1151-128k.json
```

Use a 1,800-second process-group bound and preserve stdout/stderr, ten-second
telemetry, kernel journal, task states, and debugfs snapshots. Do not wrap the
parent prompt-suite/economics harness in rocprof. Profile the final child only.

For least-perturbing retirement localization, add:

```bash
--prefill-flight-recorder /tmp/gfx1151-128k.flight \
--prefill-flight-recorder-granularity chunk
```

Inspect from another process without calling HIP:

```bash
python3 scripts/qwen35_prefill_flight_recorder.py \
  /tmp/gfx1151-128k.flight --entries 8 --watch-seconds 1
```

Escalate from `chunk` to `layer` only when the chunk capture is insufficient.
Layer mode adds 1,315 same-stream system-fence marker kernels per prefill and can
change incidence.

## Evidence chronology

| Date | Probe | Result | Interpretation |
| --- | --- | --- | --- |
| Jul 13 | Current 128K lifecycle repetitions | One retained 1+3 window completed; later attempts stalled on measurement 5 and measurement 2 | Intermittent lifecycle failure, not a deterministic first-pass kernel failure |
| Jul 15 | Default queue policy | First warmup stalled about 80 seconds into prefill at 100% / 2.9 GHz / 41-43 W | Multiple queues increase risk on this workload |
| Jul 15 | Same-command `GPU_MAX_HW_QUEUES=1` A/B | Complete exact warmup+3 at about 500 tok/s | Strong initial mitigation signal, later disproven as sufficient |
| Jul 15 | Final production with one queue | Warmup completed at 509.708 tok/s; measured pass 1 stalled | One queue is risk reduction only |
| Jul 15 | Router-512 + device-metadata-off control | Warmup completed at 503.455 tok/s; measured pass 1 stalled | New router geometry and scoped device metadata are not necessary triggers |
| Jul 15 | `HSA_ENABLE_SDMA=0` | One 1+1 screen passed; fresh full gate stalled after warmup | SDMA is not a reliable workaround |
| Jul 15 | HIP 7.13 vs 7.15 | HIP 7.13 passed 2/3 fresh processes then stalled; HIP 7.15 stalled 2/2 | Neither tested user-space stack is lifecycle-safe; incidence rates are not established |
| Jul 16 | Chunk flight recorder | Warmup passed; measured pass 1 last retired `[24576,28672)`, host entered layer 11 of `[28672,32768)` | Host submission ahead of same-stream retirement; unresolved interval localized |
| Jul 16 | Merged request/chunk metadata | First warmup last retired `[57344,61440)`, host entered layer 18 of next chunk | Repeated metadata copy frequency is not necessary; location moves |
| Jul 16 | Compact-MoE scalar no-read | Warmup passed; measured pass 1 last retired `[32768,36864)` while host queued all 40 next-chunk layers | Per-layer D2H reads expose host waiting but are not necessary for no-progress |
| Jul 16 | Layer markers, first process | Entire exact warmup+3 completed; cursor 5,392/5,392 | Instrumentation perturbs timing; one completion is not stability evidence |
| Jul 16 | Two independent layer-marker repeats | Both stalled with two pending checkpoints | Layer markers are not a reliable workaround; two-layer retirement window repeats |
| Jul 16 | rocprofv3 inline queue interception | First prefill stalled; injected profiler signal also stopped; no trace finalized | Ambiguous instrumentation result; no last user kernel recovered |
| Jul 16 | Current-boot KFD controls | Two independent chunk-recorder warmup+3 gates completed exactly; healthy MQD/sysfs snapshots captured | Establishes a healthy queue baseline but no stall/HQD comparison; `kfd/rls` is not a usable discriminator by itself |

## Flight-recorder localization

The layer recorder writes a host submission record before each layer and places a
tiny system-fence marker after the layer. Therefore:

- a retired marker proves all earlier same-stream work completed;
- a submitted next-layer record proves only that the host reached that call;
- a pending layer marker does not prove which preceding dispatch is stuck;
- the marker itself and profiler instrumentation can perturb timing.

The three failed layer captures are:

| Capture | Chunk | Last retired | Pending previous marker | Host reached | Lag |
| --- | --- | --- | --- | --- | ---: |
| Repeat 1 | `[28672,32768)` | layer 11, full attention | layer 12, linear attention | layer 13, linear attention | 2 |
| Repeat 2 | `[16384,20480)` | layer 33, linear attention | layer 34, linear attention | layer 35, full attention | 2 |
| rocprof inline | `[102400,106496)` | layer 15, full attention | layer 16, linear attention | layer 17, linear attention | 2 |

The common safe statement is:

> Retirement stops after the host has returned from a linear-attention layer and
> queued its completion marker, while the host is inside the following layer.
> The unresolved work includes the prior linear layer's post-scalar-read MoE
> tail and marker through the following layer's pre-read work.

It is **not** safe to say that a linear-attention kernel is hung. The previous
layer includes many kernels, the marker has not retired, the next layer has
already submitted additional work, and no completed-dispatch stream is yet
available.

## Tested hypotheses and what is actually eliminated

“Rejected” here means “not a necessary trigger or sufficient workaround under
the tested protocol.” It does not mean the component can never affect incidence.

| Hypothesis / proposed fix | Probe | Outcome | Evidence-bounded conclusion |
| --- | --- | --- | --- |
| ROCm's default four hardware queues cause the failure | Default vs one-queue matched A/B | Default failed; first one-queue gate passed; later one-queue gates failed | Queue count affects risk but one queue does not eliminate the bug |
| HIP 7.15 regression | Five-process HIP 7.13/7.15 lifecycle matrix | Both stacks reproduced | HIP 7.15 alone is not the root cause; user-space version can still affect incidence |
| SDMA copy engine deadlock | `HSA_ENABLE_SDMA=0` full gate | Reproduced after warmup | SDMA is not necessary and disabling it is not a safe workaround |
| Scoped stream-ordered metadata preparation | Explicit metadata-off control | Reproduced | LCP-M2 metadata path is not necessary |
| 128-thread router selection | Restore 512 threads | Reproduced | LCP-4B router geometry is not necessary |
| Repeated request/chunk metadata H2D allocation/copy | Premerge and reuse each request/chunk metadata once | Both reproduced at different points | Copy frequency and request/chunk slab churn are not necessary |
| Synchronous compact-MoE scalar D2H reads | Skip selected-row scalar reads through 32,768 rows | Reproduced and host submitted farther ahead | Reads determine where the host blocks; they are not necessary for device no-progress |
| Completion markers/heartbeat keep the queue alive | One layer-marker completion plus two repeats | Both repeats failed | A heartbeat is not a reliable production mitigation |
| One deterministic outer chunk/layer | Compare recorder failures | Chunks and layers move | No fixed chunk or layer is established |
| A prior graph-decode/reset corrupts the next prefill | Compare first-warmup and post-warmup failures | The first warmup can fail before any decode/reset lifecycle | Prior graph decode is not a necessary trigger |
| JIT compiler/cache stall | Prebuild plus `--require-cached-build` and compiler-version guards | Reproduces with no compiler child | Lazy compilation/version probing is not this failure |
| OOM / allocation churn | Stable 25.493 GiB tracked allocation and fixed residency during stalls | No OOM; memory releases on kill | No evidence of OOM; this does not exclude an unreported VM/queue fault |
| Thermal throttling | Stable low power, high clock, no thermal line | Repeated | Thermal limit is unsupported by current evidence |
| Ordinary DRM scheduler timeout will recover it | Persistent 24-30 minute runs with clean journal | No timeout/reset | Current KFD/MES user-queue failure is not being caught by the observed hangcheck path |
| A Python thread deadlock | Periodic stacks and task states | Host wait location moves with removed sync points; main thread can remain runnable under profiler | Python does not explain missing same-stream retirement; host stacks are observation points, not cause proof |
| rocprof can identify the last dispatch as currently configured | Inline kernel/HIP/HSA/copy trace | Profiler signal stalls and files do not finalize | Inline-interposition result is unusable for naming a user kernel |

## What remains possible

### Leading interpretation: gfx11 CP/MES/KFD queue-retirement failure

Evidence supporting this interpretation:

- queue count materially changes initial incidence;
- AMD has publicly described the gfx11 fix as still under development in
  [ROCm#5107](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4800268515),
  with a [nearby follow-up](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4847244516)
  noting that the related gfx10/gfx11 class also needs CP-logic changes rather
  than a firmware-only change;
- same-stream markers stop retiring while the host remains alive;
- no ordinary DRM scheduler timeout or reset occurs;
- the failure survives user-space HIP stack, SDMA, metadata, readback, and marker
  changes;
- process termination immediately clears the condition without resetting the
  host or GPU.

Missing proof:

- two healthy-active KFD MQD/sysfs baselines exist, but no active-stall
  runlist/MQD/HQD dump has been captured;
- MES event logging is currently disabled;
- no firmware-decoded MES trace has been collected;
- no KFD queue rptr/wptr comparison is available;
- no minimal standalone reproducer exists;
- no fixed-stack scheduler-policy A/B has been completed.

### Application dispatch sequence as a trigger

The workload may expose a driver/firmware bug through a particular long-running
sequence of linear attention, selected MoE, copies, and synchronization. A
movable trigger does not exonerate hipEngine. Until the last completed user
dispatch and queue state are known, a bad kernel access, missing dependency, or
HIP ABI misuse remains possible even without a reported VM fault.

### Profiler queue interception

The first long rocprofv3 run used the default inline queue-interposition path.
At stall onset, rocprofiler's own injected HSA completion signal stayed at value
`1` and its handler reached 153,092,096 polls. No CSV finalized. Upstream
[rocm-systems#7464](https://github.com/ROCm/rocm-systems/pull/7464) documents an
intermittent hang in this path and recommends
`ROCPROFILER_QUEUE_INTERPOSITION=0`; the inline path originated in
[rocm-systems#4276](https://github.com/ROCm/rocm-systems/pull/4276).
Consequently that incidence can be an observer of the original failure or an
instrumentation-induced failure. It is not counted as independent causality.

### VM, page-table, or wave-level fault without reporting

Clean journals make these less likely but do not eliminate them. MES logging,
KFD queue state, VM info, and an AMD-supported wave/queue debug path are needed
before excluding a silent fault or permanently running shader.

## Debugging plan

### Priority 0 result: healthy KFD baseline captured; stalled HQD pending

Two current-boot processes complete exact warmup+3 gates at merged commit
`babbc8c6`, so idle, healthy-active, and post-termination snapshots are now
preserved. Both healthy snapshots coincide with 97-99% activity, 128-129 W, and
advancing recorder markers. KFD sysfs/MQDs show two compute queue objects plus
one SDMA queue, zero fault/page-in/page-out counters, and 3-4 ms cumulative
eviction time. Nevertheless, `kfd/rls` reports `No active runlist` in both
healthy snapshots. Treat that view as unsupported or insufficient for this MES
configuration; never infer an idle user queue from it alone. No HQD dump was
captured because the predeclared protocol reserves that large, potentially
perturbing read for an established stall.

The kernel exposes:

```text
/sys/kernel/debug/kfd/rls
/sys/kernel/debug/kfd/mqds
/sys/kernel/debug/kfd/hqds
/sys/kernel/debug/kfd/proc
```

Kernel documentation/source describes these as HWS runlists, per-process
software MQDs, and hardware queue descriptor/register dumps. `amdgpu_fence_info`
is not a substitute: it does not expose the doorbell-driven KFD user queues.

Capture three snapshots: idle baseline, one healthy active prefill, and one
persistent stall. Read `rls` and `mqds` first. `hqds` is large and reads hardware
register state, so capture it once after the failure is already established and
treat it as potentially perturbing.

```bash
out=/tmp/gfx1151-kfd-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$out"

sudo timeout 10 cat /sys/kernel/debug/kfd/rls  >"$out/kfd-rls.txt"  2>"$out/kfd-rls.err"
sudo timeout 10 cat /sys/kernel/debug/kfd/mqds >"$out/kfd-mqds.txt" 2>"$out/kfd-mqds.err"
sudo find /sys/kernel/debug/kfd/proc -maxdepth 4 -printf '%y %p\n' \
  >"$out/kfd-debug-proc-tree.txt" 2>&1
sudo find /sys/class/kfd/kfd/proc -maxdepth 5 -printf '%y %p\n' \
  >"$out/kfd-sysfs-proc-tree.txt" 2>&1
while IFS= read -r file; do
  name=${file#/sys/class/kfd/kfd/proc/}
  name=${name//\//__}
  timeout 2 cat "$file" >"$out/kfd-sysfs-${name}.txt" 2>&1 || true
done < <(find /sys/class/kfd/kfd/proc -maxdepth 5 -type f | sort)
sudo timeout 30 cat /sys/kernel/debug/kfd/hqds >"$out/kfd-hqds.txt" 2>"$out/kfd-hqds.err"

sudo timeout 10 cat /sys/kernel/debug/dri/0000:c1:00.0/amdgpu_vm_info \
  >"$out/amdgpu-vm-info.txt" 2>&1
sudo timeout 10 cat /sys/kernel/debug/dri/0000:c1:00.0/amdgpu_fence_info \
  >"$out/amdgpu-fence-info.txt" 2>&1
```

Do **not** write to `/sys/kernel/debug/kfd/hang_hws`; it deliberately induces a
scheduler hang and is not an observation interface.

Questions to answer from the snapshots:

1. Is the hipEngine PASID still present and mapped in the runlist?
2. How many compute and SDMA queues exist despite `GPU_MAX_HW_QUEUES=1`?
3. Is the queue mapped into an HQD or only represented by an MQD?
4. Do rptr/wptr/doorbell fields stop with unread packets?
5. Does HWS consider the runlist active, evicted, or drained?
6. Does repeated observation change any pointer or queue state?

MQDs can be stale for mapped queues; correlate them with HQDs rather than
interpreting the software descriptor alone.

### Priority 1: retry tracing with legacy queue interception

Use the same cached command and add:

```bash
export ROCPROFILER_QUEUE_INTERPOSITION=0
```

Collect kernel, HIP, HSA, memory-copy, and KFD traces. Run only one bounded
attempt. A completed run still provides useful exact dispatch order; a failed
run is useful only if CSVs persist before finalization or the profiler exits
cleanly. If no files are produced again, stop using rocprofv3 as a flight
recorder and implement a minimal rocprofiler-sdk callback that streams completed
dispatch records to a preallocated mmap/file.

Do not infer stability if tracing suppresses the failure.

### Priority 2: diagnostic MES-logging boot

The current-boot KFD evidence is preserved and this boot is now prepared in
Limine, but the parameters are not active until reboot. `/boot/limine.conf` was
regenerated successfully; the previous source config is backed up at
`/etc/default/limine.pre-gfx1151-debug-20260716T054023Z`.

Test the prepared boot with:

```text
amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=1 amdgpu.send_sigterm=1
```

Rationale:

- `mes_log_enable=1` enables the debugfs MES event log and is the most directly
  relevant additional scheduler evidence;
- `gpu_recovery=1` explicitly enables recovery if the driver recognizes a
  timeout;
- `send_sigterm=1` requests SIGTERM delivery for recognized unhandled HSA
  exceptions instead of only logging them.

Verify the loaded values after reboot. Locate the new debugfs node and capture
it before the run, immediately after the persistent state, and after termination:

```bash
sudo find /sys/kernel/debug/dri -maxdepth 2 -name amdgpu_mes_event_log -print
sudo dd if=/sys/kernel/debug/dri/0000:c1:00.0/amdgpu_mes_event_log \
  of=/tmp/amdgpu-mes-event-log.bin status=none
```

The format may require an AMD decoder; preserve raw bytes and firmware identity
exactly. MES logging can perturb timing, so this is never a performance run.

### Priority 3: non-HWS scheduler-isolation boot

Use a separate boot:

```text
amdgpu.sched_policy=2 amdgpu.mes_log_enable=1
```

`sched_policy=2` disables HWS and statically assigns queues. If the exact 128K
warmup+3 gate becomes repeatedly reliable, that strongly implicates the HWS/MES
scheduling plane. If it still fails, capture HQD/MQD state and do not conclude
that firmware is exonerated. This policy is debug-only, system-wide, can affect
TTY responsiveness/power/performance, and is not a production hipEngine fix.

### Priority 4: lightweight eviction diagnostics

After the profiler run and before the next reproduction, enable:

```bash
echo 1 | sudo tee /sys/module/amdgpu/parameters/debug_evictions
```

Preserve the journal. This can reveal queue eviction/restore activity but will
not by itself explain a non-eviction hang.

### Priority 5: fixed-kernel/firmware matrix

Repeat one exact bounded gate on:

1. the current kernel with newer gfx11 firmware if AMD identifies a candidate;
2. an AMD-recommended supported kernel/driver combination;
3. the same kernel with only the firmware changed, where possible.

Record firmware hashes and `amdgpu_firmware_info`; version strings alone are
insufficient. Do not mix a kernel, firmware, HIP, compiler, and application
change in one causal row.

### Priority 6: reduce the reproducer

After a last-completed dispatch or KFD queue is identified:

1. retain only the outer chunk/layer range that precedes the failure;
2. replace model data with deterministic synthetic buffers while preserving
   allocation sizes, stream ordering, and dispatch geometry;
3. remove decode and graph capture if prefill-only still reproduces;
4. reduce layer count, then kernel families, one variable at a time;
5. preserve a full-scale control after every reduction.

Do not hardcode a particular observed chunk or layer as the reproducer trigger:
all recorded locations have moved.

## Kernel parameters to avoid initially

| Parameter / change | Why not yet |
| --- | --- |
| `halt_if_hws_hang=1` | Can preserve a detected HWS hang but may require a hard reboot; use only with remote capture and recovery plan |
| `vm_fault_stop`, `noretry`, `no_queue_eviction_on_vm_fault` | No VM fault is currently observed; these alter failure/recovery behavior before the relevant evidence exists |
| `timeout_period` | Controls SQ watchdog/fatal behavior and can be destructive; use only under AMD guidance |
| `lockup_timeout` tuning | The state already persists far beyond ordinary timeout values; the observed KFD/MES queue is apparently not covered by the normal DRM scheduler timeout |
| Disable CWSR, clock gating, or power gating | Large behavioral changes with no current causal signal; would obscure the smaller scheduler tests |
| `mes=0` alone | On affected RDNA generations, KFD queue handling may still use MES/HWS paths; verify actual queue state instead of trusting the parameter name |

## Existing public reports

### What we have already reported

We posted two comments to
[ROCm/ROCm#5107](https://github.com/ROCm/ROCm/issues/5107), an issue originally
about persistent 100% utilization with multiple models/queues:

1. [Initial gfx1151 report](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4976739824)
   - single gfx1151 / single process and model;
   - first-warmup failure under default queue count;
   - 100%, 2.9 GHz, 41-43 W, repeated host stacks at a synchronous metadata
     `hipMemcpy`, clean kernel log, and immediate recovery on process kill;
   - matched one-variable `GPU_MAX_HW_QUEUES=1` warmup+3 completion;
   - explicit warning that the tiny `#2625` reproducer did not trigger locally
     and that identical mechanism was not claimed.
2. [Correction/follow-up](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4979442043)
   - one queue is helpful but not sufficient;
   - current production one-queue measured-pass-1 failure;
   - router/metadata rollback failure;
   - `HSA_ENABLE_SDMA=0` screen success followed by full-gate failure;
   - 128K publication blocked and one queue retained only as risk reduction.

Those comments are accurate but incomplete relative to current evidence.

### Closest existing reports

| Issue | Similarity | Material difference |
| --- | --- | --- |
| [ROCm/ROCm#5107](https://github.com/ROCm/ROCm/issues/5107) | Queue-count sensitivity, 100% state, AMD says gfx11 MES/CP fix is under development | Primarily multi-model/idle utilization; our direct symptom is one-process long-prefill no-progress |
| [ROCm/ROCm#6165](https://github.com/ROCm/ROCm/issues/6165) | Same Framework gfx1151 platform, sustained long prefill, silent hang, no hangcheck/reset | Their MES is 0x86 and the whole host later freezes; ours uses MES 0x88, host remains responsive, and killing one process immediately recovers |
| [ROCm/ROCm#2625](https://github.com/ROCm/ROCm/issues/2625) / [ROCm/amdgpu#153](https://github.com/ROCm/amdgpu/issues/153) | RDNA hardware queues/MES, 100% activity, `sched_policy=2` workaround | Their minimal stream/memory reproducer does not trigger our symptom; their primary issue is persistent idle power, not halted prefill retirement |

The existing links support a scheduler-family relationship. They do not prove
all reports have the same root cause.

## What should be reported next

### Update the existing #5107 thread

Post one concise update after the committed artifacts are publicly reachable.
It should add only evidence not already in the correction:

- HIP 7.13 and 7.15 both reproduce;
- metadata slab reuse and scalar no-read controls reproduce;
- two independent layer-marker repeats fail with the same two-checkpoint lag but
  different chunks/layers;
- the pending prior marker follows a linear-attention layer in all three layer
  captures, without claiming that kernel family is faulty;
- inline rocprof tracing is ambiguous because its own completion signal stalls
  and no trace finalizes;
- KFD/MES state capture and a dedicated issue are now the tracking path.

Do not paste every benchmark number into #5107. Link this document and the
compact artifacts.

### File a dedicated issue

The threshold for a dedicated issue is already met. It should not wait for every
possible workaround because:

- the observed operation stops making forward progress, not merely reports a
  misleading utilization value;
- one process and one queue are sufficient;
- the issue survives both tested HIP stacks and multiple application controls;
- current 128K production is blocked;
- the symptom is related to, but materially different from, #5107 and #6165.

Recommended repository: **ROCm/ROCm**, so AMD can route it across HIP, KFD,
amdgpu, CP, and MES. Move or cross-file to drm/amd only if AMD requests it.

Suggested title:

```text
[gfx1151] Single-process 128K prefill queue stops retiring at 100%/2.9 GHz low power; no timeout or reset
```

Before filing:

1. push or otherwise make the exact reproducer/recorder commit reachable;
2. run the current-boot KFD `rls`/`mqds`/`hqds` capture once;
3. attach exact kernel cmdline, amdgpu parameter values, firmware hashes, and
   `amdgpu_firmware_info`;
4. redact hostnames and unrelated process data;
5. include the public model URL/fingerprint, never model weights.

Do **not** delay the issue for the MES-logging or `sched_policy=2` boots. Add those
as follow-up comments.

### Dedicated issue content checklist

#### Problem statement

- Expected: each 128K prefill completes in approximately 260 seconds and retires
  all same-stream markers.
- Actual: intermittent indefinite no-progress with the exact telemetry signature;
  no fault/reset; process kill recovers.
- Scope: one GPU, one process, one model, one hardware queue.

#### Exact environment

- system/GPU/PCI ID, BIOS/VBIOS;
- kernel build and full command line;
- amdgpu module values;
- MES, MES KIQ, MEC, RLC, and SMC firmware versions and hashes;
- HIP/HSA/compiler/rocprofiler versions and shared-library hashes;
- hipEngine commit and clean/dirty provenance;
- model URL, size, and sampled/full hash contract.

#### Reproduction

- exact command above;
- prebuild/cache steps;
- 1,800-second bound;
- expected pass output and typical time;
- note intermittent incidence and that one successful process is insufficient.

#### High-value evidence

- timeline from normal 120 W work to low-power no-progress;
- flight-recorder submitted/completed cursors;
- the three layer-capture rows;
- KFD runlist/MQD/HQD healthy-vs-stalled diff;
- process task states/stacks;
- kernel journal and `amdgpu_vm_info`;
- telemetry and immediate post-kill recovery;
- compact HIP 7.13/7.15 and workaround matrix.

#### Explicitly avoid overclaiming

State that:

- `hipMemcpy` is only where a synchronous host wait became visible;
- linear attention is the prior unretired layer, not a proven faulty kernel;
- `amdgpu_fence_info` is blind to KFD user queues;
- the rocprof inline run may be instrumentation-induced;
- issue-family similarity does not establish identical root cause.

#### Questions for AMD

1. Which KFD MQD/HQD fields should be decoded on gfx1151 to compare user-queue
   rptr/wptr, mapping state, and doorbell progress?
2. Is there an AMD-supported way to snapshot the active KFD user queue and last
   retired packet without stopping or evicting it?
3. Why does no user-queue hangcheck, timeout, or recovery fire during a
   20-30-minute persistent state?
4. Is MES `0x88` known to contain the gfx11 CP/MES fix discussed in #5107, or is
   a newer firmware/kernel pair required?
5. Is `amdgpu.mes_log_enable=1` sufficient for this path, and is a decoder
   available for `amdgpu_mes_event_log`?
6. Is `sched_policy=2` the preferred scheduler-isolation test on this kernel?
7. Are `gpu_recovery=1` and `send_sigterm=1` appropriate for this failure, or is
   a different KFD exception/recovery option recommended?
8. Can AMD provide a debug kernel/patch that logs HWS runlist progress, MES queue
   map/unmap events, and user-queue timeout state?

## Evidence index

| Artifact | Purpose |
| --- | --- |
| [`2026-07-15-gfx1151-hip-one-queue-stability-promotion.json`](../benchmarks/results/2026-07-15-gfx1151-hip-one-queue-stability-promotion.json) | Original default-vs-one-queue matched A/B and short-context non-regression |
| [`2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json`](../benchmarks/results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json) | Current 512-64K publication plus one-queue/router/SDMA 128K failures |
| [`2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`](../benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json) | Five-process HIP 7.13/7.15 lifecycle matrix |
| [`2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json`](../benchmarks/results/2026-07-16-gfx1151-128k-prefill-flight-recorder-stall.json) | First persistent chunk-retirement localization |
| [`2026-07-16-gfx1151-128k-merged-metadata-reuse-stall.json`](../benchmarks/results/2026-07-16-gfx1151-128k-merged-metadata-reuse-stall.json) | Request/chunk metadata reuse rejection |
| [`2026-07-16-gfx1151-128k-compact-no-read-stall.json`](../benchmarks/results/2026-07-16-gfx1151-128k-compact-no-read-stall.json) | Per-layer scalar D2H no-read rejection |
| [`2026-07-16-gfx1151-128k-layer-marker-completion.json`](../benchmarks/results/2026-07-16-gfx1151-128k-layer-marker-completion.json) | One complete instrumentation-sensitive layer-marker gate |
| [`2026-07-16-gfx1151-128k-layer-marker-repeat-stalls.json`](../benchmarks/results/2026-07-16-gfx1151-128k-layer-marker-repeat-stalls.json) | Two independent failed layer-marker repeats |
| [`2026-07-16-gfx1151-128k-rocprof-inline-interposition-stall.json`](../benchmarks/results/2026-07-16-gfx1151-128k-rocprof-inline-interposition-stall.json) | Ambiguous inline-profiler signal stall and missing trace finalization |
| [`2026-07-16-gfx1151-128k-kfd-healthy-controls.json`](../benchmarks/results/2026-07-16-gfx1151-128k-kfd-healthy-controls.json) | Two complete current-boot healthy MQD/sysfs controls and prepared MES-debug boot |

Raw telemetry, recorder mmaps, process stacks, fence samples, journals, and
profiler logs normally remain local under the `/tmp/gfx1151-*` directories named
and hashed by the compact artifacts. The two pre-reboot KFD bundles are also
compressed and checksum-preserved under
`/home/lhl/gfx1151-debug/2026-07-16-current-boot`; they are not public evidence
until an issue attachment bundle is created.

## Closure criteria

The bug is not closed by one successful 128K pass. Closure requires one of:

1. a fixed kernel/firmware/runtime combination completes at least three
   independent warmup+3 processes with exact output, finite logits, clean logs,
   and normal telemetry; or
2. a production-quality workaround does the same, remains exact and
   non-regressive at 512/4K/64K, and has a documented mechanism and rollback.

For an upstream fix, rerun both the original default-queue reproducer and the
one-queue production gate. Preserve the pre-fix KFD/MES evidence and report the
exact fixed kernel and firmware hashes.
