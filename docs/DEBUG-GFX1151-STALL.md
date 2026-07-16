# Debugging the gfx1151 128K prefill stall

**Status:** open, reproducible, intermittent, no production-safe workaround<br>
**Last updated:** 2026-07-16<br>
**Primary platform:** Framework Desktop, Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`)<br>
**Current publication decision:** hipEngine GGUF rows through 64K are retained;
the current repeated-128K row is blocked.<br>
**Dedicated upstream issue:** [ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437)<br>
**Immutable reproducer:** [`rocm-6437-reproducer-v1`](https://github.com/shisa-ai/hipEngine/tree/rocm-6437-reproducer-v1) (`a7b4fe4b213c5afcbe1be2b13cb33464f251a06e`)<br>
**Redacted evidence bundle:** [public gist](https://gist.github.com/lhl/dcdc0eb2e7a8f1bede6088130c383f72)

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
making device-visible progress either in a fresh first prefill or after one or
more identical prefills have completed. The process and machine remain alive,
but the GPU remains indefinitely at approximately **100% reported activity,
2.9 GHz, and only 41-59 W** instead of
the roughly 120 W working regime. Device memory remains allocated and stable.
No amdgpu/KFD fault, timeout, or reset is logged; one capture includes a single
coincident PCIe PME line. Terminating the process immediately returns the GPU to
idle without a reset.

The failure:

- occurs with one resident model and `GPU_MAX_HW_QUEUES=1`; KFD still exposes
  one primary compute queue, one auxiliary compute queue, and one SDMA queue;
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

The first run after rebooting with MES event logging reproduced during its first
128K prefill. Retirement stopped at sequence 339 after chunk `[32768,36864)`,
while the host submitted through sequence 389 at layer 6 of `[36864,40960)`.
One established-stall HQD snapshot found the 1 MiB AQL queue **active and
non-empty**, with `rptr=0x32250`, `wptr=0x32450` (**32 unread AQL packets** after
the gfx11 AQL pointer shift), and zero `CP_HQD_ERROR` and
`CP_HQD_DEQUEUE_REQUEST`. The exposed MES event-log bytes were identical in the
healthy-active, first-stall, and +30-second snapshots, then changed during
process teardown. This is direct evidence of a mapped user queue with unread
work; one HQD sample does not prove temporal pointer immobility, identify an
unread packet, or prove MES firmware is the faulty component.

The dedicated stack-wide report and redacted bundle are now public. The next
experiment is a separate `sched_policy=2` scheduler-isolation boot; its result
will be added to ROCm/ROCm#6437. A legacy-interposition or streaming rocprofiler
retry remains lower priority.

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
| Queue policy | `GPU_MAX_HW_QUEUES=1` unless a row explicitly says default/four; this does not mean only one KFD queue object |
| amdgpu scheduler policy | `sched_policy=0` (hardware scheduling enabled) |
| CWSR | `cwsr_enable=1` |

Current relevant values on the MES-debug boot are:

```text
mes_log_enable=1
sched_policy=0
gpu_recovery=1
send_sigterm=1
debug_evictions=N
halt_if_hws_hang=0
lockup_timeout=<unset>
timeout_period=0
cwsr_enable=1
noretry=-1
vm_fault_stop=0
no_queue_eviction_on_vm_fault=0
runpm=-1
```

The pre-reboot control differed only in `mes_log_enable=0`, `gpu_recovery=-1`,
and `send_sigterm=0`. Record the full set again after every boot. Kernel/module
behavior, not only HIP user-space version, is part of the reproduction identity.

### MES oversubscription timer and application stream topology

In response to AMD's
[#5107 question](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4981793551),
the installed kernel and exact failing route were audited before the
`sched_policy=2` reboot:

- hipEngine does not construct or submit MES packets. The kernel amdgpu driver
  constructs `SET_HW_RESOURCES` during MES initialization.
- The exact CachyOS `cachyos-7.1.3-1` source used by installed
  `linux-cachyos 7.1.3-2` sets
  [`mes_set_hw_res_pkt.oversubscription_timer = 50`](https://github.com/CachyOS/linux/blob/0e558f948dfe28b50d2eb9ddda58900d7de01aac/drivers/gpu/drm/amd/amdgpu/mes_v11_0.c#L717-L723),
  not zero. Package release `-2` is an NVIDIA rebuild of the same kernel source.
- There is no exposed amdgpu timer parameter, command-line/modprobe override,
  hipEngine reference, or local kernel patch. `mes_log_enable=1` sets adjacent
  event-log fields but does not alter the timer assignment.
- No available sysfs/debugfs view reads back the firmware's accepted live field.
  The bounded conclusion is that the kernel source submits **50** with no
  configured override, not that the firmware state was independently decoded.
- The failing README sweep calls `session.prefill()` from the main Python thread
  and queues the bulk prefill on HIP default stream 0. It does not use serving
  worker pools.
- The explicit event-linked AOTriton stream candidate is default-off on both
  backends after it produced severe intermittent 32K/128K stalls. The capture
  command did not enable it, the backend capability resolves false, and
  AOTriton receives the same caller stream in the current failure.
- ROCr helper/event threads and auxiliary KFD queue objects still exist. The
  audit establishes one application launch thread/stream, not an absence of
  runtime-internal queues or firmware scheduling.

The verified answer was posted in
[#5107 comment 4990476677](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990476677).

## Canonical reproduction

Use the immutable versioned tag, which resolves to the exact source commit used
for the MES/KFD/HQD capture:

```bash
git clone https://github.com/shisa-ai/hipEngine.git
cd hipEngine
git switch --detach rocm-6437-reproducer-v1
test "$(git rev-parse HEAD)" = a7b4fe4b213c5afcbe1be2b13cb33464f251a06e
```

Prebuild every JIT `.so` and the compiler-version cache file outside the bounded
run. On an empty cache, omit `--require-cached-build` for the cache-fill preflight
and restore it for the evidence run. The direct command shape is:

```bash
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
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
| Jul 16 | Current-boot KFD controls | Two independent chunk-recorder warmup+3 gates completed exactly; healthy MQD/sysfs snapshots captured | Establishes a healthy queue baseline; `kfd/rls` is not a usable discriminator by itself |
| Jul 16 | MES-log boot plus stalled HQD | First 128K prefill stops at cursor 389/339; 36 samples hold 100%/2.9 GHz/median 43 W; active 1 MiB HQD has 32 unread AQL packets and zero error/dequeue state; MES-log bytes change only during teardown | Direct mapped-queue backlog evidence; debug parameters are not a workaround, while one HQD sample and an undecoded MES buffer still do not name the failed packet/component |
| Jul 16 | AMD oversubscription-timer / stream-topology audit | Exact CachyOS source submits timer 50; no override; failing prefill uses one application thread and default stream 0; experimental AOTriton stream is off | Disabled timer and current application multistream submission are not supported as necessary triggers; firmware field still lacks live readback |

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
| MES oversubscription timer is disabled | Audit exact CachyOS source, module parameters, command line, modprobe state, and hipEngine source | Kernel path sets 50; no override or app MES packet path exists | Timer zero is unsupported by configured-source evidence; live firmware value is not independently readable |
| Multiple hipEngine threads/streams heavily oversubscribe CP | Audit exact README sweep and effective backend policy | Main Python thread submits bulk prefill on default stream 0; isolated AOTriton stream is off | Current failure does not require multiple application submission threads/streams; ROCr internal queues remain |
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
- the captured primary KFD compute HQD remains active and non-empty with 32
  unread AQL packets, zero HQD error state, and no dequeue request;
- the MES event-log view does not change between healthy-active and two stalled
  snapshots but does change when the process is terminated;
- no ordinary DRM scheduler timeout or reset occurs;
- `gpu_recovery=1` and `send_sigterm=1` do not act autonomously;
- the failure survives user-space HIP stack, SDMA, metadata, readback, and marker
  changes;
- process termination immediately clears the condition without resetting the
  host or GPU.

Missing proof:

- only one stalled HQD register sample exists, so temporal hardware-pointer
  immobility is not directly established;
- the MES event buffer has not been firmware-decoded;
- no last-retired AQL packet or named user dispatch has been recovered;
- no minimal standalone reproducer exists;
- no fixed-stack `sched_policy=2` A/B has been completed.

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

Clean journals, zero KFD fault/page counters, and the zero-error HQD make these
less likely but do not eliminate them. The MES buffer still needs decoding, and
an AMD-supported wave/last-retired-packet path is needed before excluding a
silent fault or permanently running shader.

## Debugging plan

### Priority 0 result: healthy controls and stalled HQD/MES capture complete

Two pre-reboot processes completed exact warmup+3 gates at merged commit
`babbc8c6`, preserving idle, healthy-active, and post-termination controls. Both
healthy snapshots coincided with 97-99% activity, 128-129 W, advancing recorder
markers, two compute queue objects, one SDMA queue, zero fault/page counters,
and only 3-4 ms cumulative eviction time.

The first MES-debug-boot attempt at public commit `a7b4fe4b` then reproduced in
its first 128K prefill:

- the cursor reached submitted/completed **389/339** and remained unchanged from
  17:11:20 through the final 17:14:21 check;
- sequence 339 proves completion through chunk `[32768,36864)`; sequence 389
  records host entry to linear-attention layer 6 of `[36864,40960)`;
- all 36 telemetry samples from 17:11:23 through 17:14:18 are **100% / 2.9 GHz**,
  with **41/43/49 W min/median/max** and fixed **26,662 MiB** residency;
- KFD still exposes two compute queues and one SDMA queue, with zero faults,
  page-ins, or page-outs and 10 ms cumulative eviction time;
- the primary 1 MiB queue is mapped at CP pipe 0 queue 2 with
  `ACTIVE=1`, `PQ_EMPTY=0`, `rptr=0x32250`, `wptr=0x32450`,
  `DEQUEUE_REQUEST=0`, and `ERROR=0`; gfx11 uses a four-bit AQL pointer shift,
  so the `0x200` gap represents **32 packets**;
- the auxiliary 4 KiB queue is active but empty at `rptr=wptr=0x140`;
- the MES event-log hashes are identical for healthy-active, first-stall, and
  +30-second snapshots (`b7a4abfb...`), then change after SIGTERM
  (`4a216fd8...`);
- the kernel records no amdgpu/KFD fault, timeout, or reset. One
  `PME: Spurious native interrupt!` appears 37 seconds after the last cursor
  change; it is a coincident signal, not an established trigger;
- the monitor sends SIGTERM only after all declared captures. The process exits,
  memory returns to 17 MiB, and no GPU reset occurs.

The HQD caveat matters: only `stalled_first` includes the large hardware-register
dump. The backlog is proven at that instant, but there is no second HQD sample
from which to claim that the hardware pointers stayed fixed. The byte-identical
software MQD is not a substitute because mapped MQDs can be stale. The decode
uses Linux's [56-register gfx11 HQD dump order](https://github.com/torvalds/linux/blob/37e2f878a7a660a216cc7a60459995fefd150f25/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v11.c#L313-L341),
[gfx11.5 CP register offsets](https://github.com/torvalds/linux/blob/37e2f878a7a660a216cc7a60459995fefd150f25/drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_5_0_offset.h#L3573-L3688),
and KFD's [four-bit AQL write-pointer shift](https://github.com/torvalds/linux/blob/37e2f878a7a660a216cc7a60459995fefd150f25/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L190-L208).
AMD should still confirm that this mainline map matches the running CachyOS
kernel and MES path.

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

Answers and remaining questions:

1. The PASID is present and two compute plus one SDMA software queue exist.
2. Both compute queues are mapped into active HQDs; the primary queue is
   non-empty with 32 unread AQL packets.
3. `kfd/rls` still says `No active runlist`, so it is not a trustworthy state
   discriminator on this MES configuration.
4. KFD reports no fault, paging, or meaningful eviction activity.
5. One HQD sample cannot answer whether the hardware rptr/wptr stopped changing;
   AMD guidance is needed for a safe second sample or last-retired-packet query.
6. The MES event-log bytes are static during the observation window but require
   an AMD decoder before interpreting operation/state fields.

MQDs can be stale for mapped queues; correlate them with HQDs rather than
interpreting the software descriptor alone.

### Priority 1 result: dedicated upstream issue filed

The report is public as [ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437),
with the text-only [redacted evidence bundle](https://gist.github.com/lhl/dcdc0eb2e7a8f1bede6088130c383f72).
The bundle contains:

- the compact artifact and exact public commit/command;
- `amdgpu_firmware_info`, decompressed firmware hashes, kernel version, command
  line, and complete amdgpu parameter values;
- recorder tail/cursor history, telemetry, process states, and filtered journal;
- the primary/auxiliary HQD excerpt and register-map source;
- one byte-identical healthy/stalled/+30-second MES event-log snapshot, the
  changed after-termination snapshot, and hashes for all original states;
- the healthy-vs-stalled matrix and an explicit evidence-boundary note.

Model weights, unrelated process listings, hostnames, root UUIDs, UFW/network
records, local user paths, and unredacted VM maps were excluded. The bundle has
15 text-only files, a verified `SHA256SUMS`, and no detected prohibited
identifier or secret pattern. The existing umbrella report was updated in
[#5107 comment 4990158250](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990158250).
Future evidence should be posted to #6437 first and cross-linked only when it
changes the broader #5107 scheduler-family picture.

### Priority 2: non-HWS scheduler-isolation boot

The separate boot is prepared but not yet loaded:

```text
amdgpu.sched_policy=2 amdgpu.mes_log_enable=1 \
  amdgpu.gpu_recovery=1 amdgpu.send_sigterm=1
```

`/etc/default/limine` and both current top-level CachyOS kernel entries contain
those four tokens exactly once. `limine-update` completed successfully. The
one-variable A/B rollback is
`/etc/default/limine.pre-gfx1151-sched-policy2-20260716T092026Z`; the full
pre-investigation backup remains
`/etc/default/limine.pre-gfx1151-debug-20260716T054023Z`. The current running
boot remains at default `sched_policy=0` until reboot. Preparation logs and
checksums are under
`/home/lhl/gfx1151-debug/2026-07-16-sched-policy2-boot-prep-20260716T092026Z`.

After reboot and before any GPU workload, verify `/proc/cmdline` plus loaded
`sched_policy=2`, `mes_log_enable=1`, `gpu_recovery=1`, `send_sigterm=1`, and
`cwsr_enable=1`. Keep the kernel, firmware, HIP stack, compiler, application
commit, one-queue environment, model, and capture protocol unchanged.
`sched_policy=2` disables HWS and statically assigns queues. If the exact 128K
warmup+3 gate becomes repeatedly reliable, that strongly implicates the HWS/MES
scheduling plane. If
it still fails, capture one HQD and all MES/KFD controls and do not conclude
that firmware is exonerated. This policy is debug-only, system-wide, can affect
TTY responsiveness/power/performance, and is not a production hipEngine fix.

### Priority 3: retry tracing with legacy queue interception

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

We posted three comments to
[ROCm/ROCm#5107](https://github.com/ROCm/ROCm/issues/5107), an issue originally
about persistent 100% utilization with multiple models/queues, and then opened a
dedicated report:

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
3. [Dedicated-issue cross-link](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990158250)
   - cursor 389/339 and the 100%/2.9 GHz/41-49 W signature;
   - active non-empty 1 MiB HQD with 32 unread AQL packets and zero
     error/dequeue state;
   - MES-log healthy/stall identity and teardown change;
   - explicit one-HQD-sample and undecoded-MES evidence boundaries;
   - link to the dedicated report and public redacted bundle.
4. [Oversubscription-timer and stream-topology answer](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990476677)
   - exact CachyOS source sets the MES oversubscription timer to 50;
   - no app/kernel-parameter/configuration override exists;
   - exact failing bulk prefill uses the main Python thread and default stream 0;
   - rejected isolated-AOTriton-stream experiment is disabled;
   - live firmware packet-field readback remains unavailable.

The dedicated [ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437)
contains the complete environment, reproducer, controls, HQD decode, raw MES
snapshot links, and questions for AMD.

### Closest existing reports

| Issue | Similarity | Material difference |
| --- | --- | --- |
| [ROCm/ROCm#5107](https://github.com/ROCm/ROCm/issues/5107) | Queue-count sensitivity, 100% state, AMD says gfx11 MES/CP fix is under development | Primarily multi-model/idle utilization; our direct symptom is one-process long-prefill no-progress |
| [ROCm/ROCm#6165](https://github.com/ROCm/ROCm/issues/6165) | Same Framework gfx1151 platform, sustained long prefill, silent hang, no hangcheck/reset | Their MES is 0x86 and the whole host later freezes; ours uses MES 0x88, host remains responsive, and killing one process immediately recovers |
| [ROCm/ROCm#2625](https://github.com/ROCm/ROCm/issues/2625) / [ROCm/amdgpu#153](https://github.com/ROCm/amdgpu/issues/153) | RDNA hardware queues/MES, 100% activity, `sched_policy=2` workaround | Their minimal stream/memory reproducer does not trigger our symptom; their primary issue is persistent idle power, not halted prefill retirement |

The existing links support a scheduler-family relationship. They do not prove
all reports have the same root cause.

### Tracker-selection research: ROCm/ROCm versus ROCm/TheRock

**Decision implemented: the dedicated report is
[ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437).** Cross-link TheRock
reports, but do not duplicate-file there unless an AMD maintainer requests it or
a later A/B establishes a TheRock-package regression.

Scope is the deciding factor:

- [ROCm/ROCm's contribution guide](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#issue-tracking)
  says that repository's issues track ROCm bugs across a stack described as
  drivers through end-user APIs and explicitly says to file when uncertain;
- [TheRock](https://github.com/ROCm/TheRock#therock) describes itself as an
  early-preview build platform providing daily user-space packages, source
  builds, and CI. Its
  [FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md#what-does-therock-provide-compared-to-more-traditional-rocm-releases)
  distinguishes those packages from traditional ROCm releases;
- our trigger uses a TheRock HIP 7.15 user stack, but it also reproduces on HIP
  7.13 and the strongest evidence is in kernel KFD/HQD, amdgpu CP registers, MES
  logging, firmware, and missing hang recovery. No TheRock build, wheel,
  packaging, or nightly-regression boundary is established;
- [TheRock#2655](https://github.com/ROCm/TheRock/issues/2655), the closest
  TheRock MES-scheduler issue, immediately points related gfx115x hangs to
  ROCm/ROCm#5724 and #5590 and a kernel patch. That is the clearest observed
  routing precedent.

Both trackers accept end-to-end reports, and response time alone does not select
a clear winner. This small, non-random snapshot was checked on 2026-07-16:

| Tracker / issue | First observable AMD response | Subsequent handling |
| --- | ---: | --- |
| [ROCm/ROCm#5107](https://github.com/ROCm/ROCm/issues/5107) | about 17 hours | Assigned, labeled **Under Investigation**, and still active nearly a year later |
| [ROCm/ROCm#5724](https://github.com/ROCm/ROCm/issues/5724) | about 9.4 days | AMD identified a MES firmware/KFD discrepancy and provided sustained firmware/kernel triage |
| [ROCm/ROCm#6273](https://github.com/ROCm/ROCm/issues/6273) | about 8.6 days | Assigned; AMD proposed CWSR isolation and followed the result |
| [ROCm/TheRock#1413](https://github.com/ROCm/TheRock/issues/1413) | about 8.9 days | AMD obtained hardware, attempted reproduction, and closed after the reporter confirmed recovery |
| [ROCm/TheRock#1271](https://github.com/ROCm/TheRock/issues/1271) | about 29.8 days | Slow first response but months of active driver/firmware and hardware-reproduction follow-up |
| [ROCm/TheRock#5581](https://github.com/ROCm/TheRock/issues/5581) | about 22.8 hours | Component-specific MIOpen report received a minimal reproducer and staging-build confirmation within a day |
| [ROCm/TheRock#5993](https://github.com/ROCm/TheRock/issues/5993) | no AMD response visible by Jul 16 | Detailed gfx1150 MES/devcoredump report remained open |

TheRock has useful `gfx1151` and `driver/fw update` labels and can be very fast
when a bug maps to a component or staging build. ROCm/ROCm can also respond
quickly, but some system issues take days or weeks. The sample is confounded by
issue specificity, reporter/maintainer identity, reproducer quality, and whether
hardware is available; it is evidence about handling patterns, not a service
level. For this cross-layer silent queue-retirement failure, correct routing to
the umbrella tracker is more important than the noisy timing difference.

## Upstream reporting status and follow-up

### Existing #5107 thread updated

The posted [cross-link comment](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990158250)
adds only evidence not already in the correction:

- HIP 7.13 and 7.15 both reproduce;
- metadata slab reuse and scalar no-read controls reproduce;
- two independent layer-marker repeats fail with the same two-checkpoint lag but
  different chunks/layers;
- the pending prior marker follows a linear-attention layer in all three layer
  captures, without claiming that kernel family is faulty;
- inline rocprof tracing is ambiguous because its own completion signal stalls
  and no trace finalizes;
- the MES-debug boot reproduces in the first prefill with cursor 389/339;
- the mapped 1 MiB HQD is active/non-empty with 32 unread AQL packets and zero
  HQD error/dequeue state;
- MES event-log bytes remain unchanged through the +30-second snapshot and
  change during teardown;
- a dedicated ROCm/ROCm issue is now the primary tracking path.

Future #5107 updates should remain concise and point detailed investigation to
#6437.

### Dedicated issue filed

The threshold was met without waiting for every possible workaround because:

- the observed operation stops making forward progress, not merely reports a
  misleading utilization value;
- one process with the one-hardware-queue runtime policy is sufficient, even
  though KFD exposes auxiliary compute and SDMA queue objects;
- the issue survives both tested HIP stacks and multiple application controls;
- current 128K production is blocked;
- the symptom is related to, but materially different from, #5107 and #6165.

The report was filed in **ROCm/ROCm** for routing across HIP, ROCr, KFD,
amdgpu, CP, MES, firmware, and hang recovery. TheRock was not primary because no
build/package/nightly regression is established; its closest MES report routes
related gfx115x cases back to ROCm/ROCm. Move or cross-file to TheRock or drm/amd
only if AMD requests it.

Actual issue title:

```text
[gfx1151] Single-process 128K prefill AQL queue stops retiring at 100%/2.9 GHz low power; no timeout or reset
```

Published package:

1. public exact run commit `a7b4fe4b` and capture/rollup commit `35d3d0e7`;
2. 15-file text-only redacted bundle with raw healthy/stall and teardown MES hex
   views, HQD dump/decode, recorder, telemetry, firmware, and summaries;
3. exact amdgpu values and decompressed firmware hashes;
4. explicit one-HQD-sample, stale-MQD, and undecoded-MES boundaries;
5. public model URL/fingerprint, never model weights.

The `sched_policy=2` and legacy-profiler results remain follow-up comments rather
than blockers for filing.

### Submitted issue content

#### Problem statement

- Expected: each 128K prefill completes in approximately 260 seconds and retires
  all same-stream markers.
- Actual: intermittent indefinite no-progress with the exact telemetry signature;
  no fault/reset; process kill recovers.
- Scope: one GPU, one process, one model, and `GPU_MAX_HW_QUEUES=1`; disclose
  the observed primary/auxiliary-compute/SDMA KFD queue objects.

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
- KFD runlist/MQD/HQD healthy-vs-stalled evidence, including the active
  non-empty HQD and the one-sample limitation;
- healthy/stalled/+30-second/teardown MES event-log bytes and hashes;
- process task states/stacks;
- kernel journal and a redacted `amdgpu_vm_info` summary;
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

1. Please confirm the gfx1151 HQD decode: why is the active 1 MiB AQL queue
   non-empty at `rptr=0x32250` / `wptr=0x32450` with zero error/dequeue state,
   and does the four-bit KFD AQL shift make this exactly 32 pending packets?
2. Is there an AMD-supported, low-perturbation way to take a second active KFD
   HQD sample and identify the last retired/unread AQL packet without stopping
   or evicting the queue?
3. Why does no user-queue hangcheck, timeout, recovery, or autonomous SIGTERM
   fire while this state persists despite `gpu_recovery=1` and
   `send_sigterm=1`?
4. Is MES `0x88` known to contain the gfx11 CP/MES fix discussed in #5107, or is
   a newer firmware/kernel pair required?
5. How should `amdgpu_mes_event_log` be decoded on MES `0x88`, and what does it
   imply that the exposed bytes are identical from healthy-active through two
   stalled snapshots but change during process teardown?
6. Is `sched_policy=2` the preferred scheduler-isolation test on this kernel?
7. Is the single temporally coincident `PME: Spurious native interrupt!` worth a
   PCIe/platform trace, or should it be treated as unrelated absent repetition?
8. Can AMD provide a debug kernel/patch that logs HWS runlist progress, MES queue
   map/unmap events, user-queue timeout state, and the last retired AQL packet?

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
| [`2026-07-16-gfx1151-128k-kfd-healthy-controls.json`](../benchmarks/results/2026-07-16-gfx1151-128k-kfd-healthy-controls.json) | Two complete pre-reboot healthy MQD/sysfs controls and prepared MES-debug boot |
| [`2026-07-16-gfx1151-128k-mes-kfd-stall-capture.json`](../benchmarks/results/2026-07-16-gfx1151-128k-mes-kfd-stall-capture.json) | First MES-debug-boot stall with recorder cursor, telemetry, active non-empty HQD decode, MES-log control, firmware hashes, and evidence boundaries |

Public reporting links:

- dedicated issue: [ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437);
- immutable source: [`rocm-6437-reproducer-v1`](https://github.com/shisa-ai/hipEngine/tree/rocm-6437-reproducer-v1) -> `a7b4fe4b213c5afcbe1be2b13cb33464f251a06e`;
- redacted raw bundle: [gist `dcdc0eb2e7a8f1bede6088130c383f72`](https://gist.github.com/lhl/dcdc0eb2e7a8f1bede6088130c383f72);
- #5107 evidence cross-link: [comment 4990158250](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990158250);
- #5107 timer/stream answer: [comment 4990476677](https://github.com/ROCm/ROCm/issues/5107#issuecomment-4990476677).

Raw telemetry, recorder mmaps, process stacks, fence samples, journals, and
profiler logs normally remain local under the `/tmp/gfx1151-*` directories named
and hashed by the compact artifacts. The two pre-reboot KFD bundles are also
compressed and checksum-preserved under
`/home/lhl/gfx1151-debug/2026-07-16-current-boot`. The MES-debug-boot preflight
and stalled capture are checksum-preserved under
`/home/lhl/gfx1151-debug/2026-07-16-mes-log-boot-b254b1d7`. The selected,
redacted subset listed above is public in the gist; excluded raw files remain
local and are not upstream evidence.

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
