# Redline Integration and GPU-Fault Dossier

This document is the durable diagnosis record for hipEngine's experimental
[Redline](https://github.com/warpfront/redline/) retained-PM4 integration. It
separates accepted performance evidence, genuine GPU faults, ordinary harness
failures, established facts, and untested hypotheses.

Redline remains **unvendored, explicit, and default-off**. Process isolation is
permitted to recover benchmark rows, but it is not an acceptable runtime fix.

## Current verdict

- Same-HSACO W7900 micro transport is accepted: 240/240 rows pass correctness;
  Redline beats HIP on 239 rows at median 2.792x and Vulkan on 208 rows at
  median 1.696x.
- The narrow production-sized Qwen3.6 GGUF p512/d128 graph diagnostic is also
  accepted: strict retained PM4 is bit-identical and improves decode
  92.812 -> 100.357 tok/s (+8.129%). A one-graph persistent soak passes 512 PM4
  launches with 0.095% range/median.
- Package-default integration is blocked by cross-device lifecycle failure. The
  W7900 reports an address-zero VM fault at row 49; the RX 7900 XTX reaches row
  173 and surfaces an HSA timeout, but its kernel journal confirms the same
  address-zero gfxhub fault followed by repeated full-device resets and VRAM
  loss.
- Ownership is not yet isolated between Redline's queue/IB lifetime,
  Hipfire's adapter lifetime, ROCr/CLR, and amdgpu. Both cards report the exact
  `0x00801431` / SQC-data / address-zero fault. An upstream search found a
  near-identical unrelated gfx11 fault chain and several real ROCm
  signal/queue-retirement bugs, which strengthens the resource-generation
  lifetime hypothesis without identifying the originating component.
- Production ROCm 7.14 and a return to the original 7.15 nightly each pass four
  fresh full matrices per card. That non-reproduction keeps runtime-version
  causality and exact-command repeatability unresolved; it does not clear the
  historical faults.

Machine-readable evidence:

- [`benchmarks/results/2026-07-28-gfx1100-redline-transport-spike.json`](../benchmarks/results/2026-07-28-gfx1100-redline-transport-spike.json)
- [`benchmarks/results/2026-07-28-gfx1100-redline-gguf-graph-spike.json`](../benchmarks/results/2026-07-28-gfx1100-redline-gguf-graph-spike.json)
- [`benchmarks/results/2026-07-28-gfx1100-redline-rx7900xtx-lifecycle-diagnostic.json`](../benchmarks/results/2026-07-28-gfx1100-redline-rx7900xtx-lifecycle-diagnostic.json)
- [`benchmarks/results/2026-07-28-gfx1100-redline-rocm714-715-lifecycle-diagnostic.json`](../benchmarks/results/2026-07-28-gfx1100-redline-rocm714-715-lifecycle-diagnostic.json)

## Pinned environment

| Component | Identity |
| --- | --- |
| GPU | AMD Radeon Pro W7900, gfx1100, 96 CUs, GPU ordinal 0 |
| Fault-time ROCm | TheRock `7.15.0a20260711`; HIP 7.15.0-0000000 |
| LLVM | AMD clang 23.0.0git `aa451e1f...e96` + TheRock patch |
| Host kernel | CachyOS `7.1.3-2-cachyos`, amdgpu srcversion `8FC7CD0E...27AE` |
| Scheduler controls | `cwsr_enable=1`, `mcbp=-1` (auto), kernel cmdline `amdgpu.runpm=0` |
| MES firmware | both cards: MES `0x00000088`, MES_KIQ `0x00000108`; `linux-firmware-amdgpu 1:20260622-1`; uncompressed `gc_11_0_0_mes_2.bin` SHA-256 `d29eaf0b...6c957` |
| Redline | clean `33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e` |
| Hipfire bridge | `455ffb9dfd6a5712889b504737f88fbbe87d3efe` |
| Same-HSACO binary | SHA-256 `655141e2e5eef7a1a31f08a9da7b6fb19cdc114b61a79152331ac1ee0a72a291` |
| Common controls | `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, gfx1100, Radiowave-tuned wave policy, default scheduler, certified-VMEM RMW, auto Redline queues, 3 warmups, 7 samples |

The primary table above describes GPU0. GPU1 is an AMD Radeon RX 7900 XTX,
gfx1100, physical PCI `0000:10:00.0`, unique ID `0xcc4d02090dc9c3ff`, card
model `0x744c`, revision `0xc8`. Its cross-device run uses the same binary and
TheRock stack.

The raw micro root is
`/tmp/hipengine-redline-w7900-20260728`; the raw graph root is
`/tmp/hipengine-redline-graph-gate-20260728`. Those paths are disposable. The
compact artifacts retain their commands and hashes.

## ROCm userspace-version matrix

The original faults occurred under the `therock` environment's
`rocm==7.15.0a20260711` nightly. This host previously had no production 7.14
SDK. Create an isolated `rocm-latest` environment from AMD's official
multi-architecture wheel index:

```bash
/home/lhl/mambaforge/bin/mamba create -y -n rocm-latest python=3.12 pip
/home/lhl/mambaforge/envs/rocm-latest/bin/python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[libraries,devel,device-gfx1100]==7.14.0'
/home/lhl/mambaforge/envs/rocm-latest/bin/rocm-sdk init
```

The result is production ROCm **7.14.0**, HIP `7.14.60850-0000000`, at
`/home/lhl/mambaforge/envs/rocm-latest`. Its counted-queue symbol and both GPUs
initialize correctly. Keep each run's `PATH` and `LD_LIBRARY_PATH` rooted only
at the selected SDK.

Using the same prebuilt binary and embedded HSACO on the same boot:

| Userspace | W7900 | RX 7900 XTX | New kernel faults |
| --- | ---: | ---: | ---: |
| Production ROCm 7.14.0 | 4/4 processes, 960/960 rows pass | 4/4 processes, 960/960 rows pass | 0 |
| Return to nightly 7.15.0a20260711 | 4/4 processes, 960/960 rows pass | 4/4 processes, 960/960 rows pass | 0 |
| 7.15 plus trailing `/opt/rocm/lib` control | 1/1 W7900 process, 240/240 rows pass | not run | 0 |

This is **not evidence that 7.14 fixes the defect**. Production 7.14 did not
reproduce it in this bounded window, but the exact original 7.15 stack also
stopped reproducing immediately afterward. The same-boot historical evidence
still contains two W7900 faults and one RX 7900 XTX fault. Therefore ROCm 7.15
alone is not sufficient on every run, a trailing system-library path alone is
not sufficient, and exact-command repeatability remains disproven. Version
bisection is not interpretable until a deterministic stress reproducer exists.

System ROCm 7.2.4 cannot run this exact pinned path because its ROCr lacks
`hsa_amd_counted_queue_acquire`; that failure occurs before GPU work. An
existing 7.13 nightly has the symbol, but testing it now would not provide a
causal bracket. See
[`2026-07-28-gfx1100-redline-rocm714-715-lifecycle-diagnostic.json`](../benchmarks/results/2026-07-28-gfx1100-redline-rocm714-715-lifecycle-diagnostic.json)
for every run and hash.

## Genuine GPU fault: canonical reproduction

The one-process same-HSACO command is:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_BENCH_ARCH=gfx1100 \
  /home/lhl/redline/target/hipengine-w7900-gfx1100/release/hipfire-6409-bench \
  --matrix hipengine --backends redline,hip,vulkan \
  --wave-policy radiowave --redline-queues auto \
  --scheduler-profile default --redline-rmw radiowave-vmem \
  --warmups 3 --samples 7 \
  --out /tmp/hipengine-redline-w7900-20260728/hipfire-same-hsaco/results.json
```

Observed behavior:

1. Rows 1-48 pass every selected backend's CPU oracle. They cover serial
   dispatch-grid (8 rows), geometry (8), reduction (24), and memory/waitcnt (8).
2. Row 49 is the first packed-dot row:
   `serial_latency/packed-dot/variant=q8_signed,groups=16,n=32768,body=64,wg=64`.
3. Its backend order is `redline -> hip -> vulkan`; no Redline timing or
   correctness row is emitted before the process aborts.
4. ROCr reports:

   ```text
   Memory access fault by GPU node-1 (...) on address (nil).
   Reason: Page not present or supervisor privilege.
   ```

5. The process exits through `SIGABRT`/exit 134 and never writes a complete
   results artifact.

Raw hashes:

- stdout: `bb6d1c0d45ece5a856f49237fac61c6f493e05ba6ce9cd249bedd83daf0ecf13`
- stderr: `8859d2d1883cc930568c1f2f5c7de449441cca0721dd12278f790f435039e73e`
- 25-second kernel journal: `380dfeed322d1f0143a34dff08ee2a1ed4b48892cbbfcaaf4fa8c3cecc4999df`

The kernel journal attributes ten address-zero gfxhub faults to the exact
`hipfire-6409-be` PID on ring 24 / vmid 8 / pasid 33067 / client 10. The first
fault is `GCVM_L2_PROTECTION_FAULT_STATUS=0x00801431`, `SQC (data) (0xa)`,
`MORE_FAULTS=1`, `PERMISSION_FAULTS=3`, `MAPPING_ERROR=0`, and read access.
MES queue removal and eviction then fail, and amdgpu performs two successful
full-device resets with VRAM loss. This is a real GPU VM fault, not a
correctness mismatch, timeout, comparator rejection, or Python exception. The
same signature was observed in the separate source-matched orchestration below,
but exact-command repeatability and a fixed failure threshold have not yet been
established.

## Controls already run

| Test | Result | What it establishes |
| --- | --- | --- |
| Fresh-process packed-dot, same binary/HSACO and controls, `--filter packed-dot` | 16/16 rows pass; Redline wins 14/16; empty stderr | Packed-dot code, arguments, launch shapes, and CPU oracle are valid from fresh state. Result SHA-256 `1c81b3bc...a3c`. |
| Ten family-isolated processes | Every process exits 0; deduplicated union passes 240/240 | Every family can complete under the same stack. Process isolation avoids the trigger but is only a benchmark workaround. Status SHA-256 `8c0a8943...7e36`. |
| Separate source-matched matrix runner using three warmups | A fresh family child faults entering serial q8 packed-dot after earlier matrix subprocesses; raw log SHA-256 `fef5b843...be74` | The same address-zero signature occurs in a second orchestration, but its process/code-object lifecycle differs from the canonical same-HSACO run. It is supporting evidence, not the attribution denominator or proof of the same trigger. |
| Independent simple families | Complete | Independent geometry/memory/VOPD/sampler work is not categorically broken. |
| Independent reduction families | Complete | Independent reduction and two-stage paths are not categorically broken. |
| Fresh independent tapes with safe preheat | Complete | The independent-tape setup itself is viable when started in a fresh process. |
| Strict two-kernel Redline hipGraph test | Four fresh-input BF16-bit-exact PM4 replays pass | Basic capture/instantiate/replay/destroy works. |
| Qwen3.6 p512/d128 ABBA | Two fresh Redline processes, 256 total strict PM4 launches, exact logits/IDs, no fallback | A production-sized 627-node graph is valid and faster in a narrow process lifecycle. |
| One-session Qwen persistent soak | One graph, four reset/prefill/warmup/d128 runs, 512 PM4 launches, exact output and memory recovery | Hundreds of replays and roughly 321k captured kernel dispatches on one retained graph do not reproduce the fault. |

### Background-run ledger

The `bg-*` labels are session-local, but recording them prevents ordinary
harness failures from being confused with the GPU fault.

| Runs | Outcome | Classification |
| --- | --- | --- |
| `bg-1` | Redline release build and 6/6 Rust tests pass; follow-up hash step expects a nonexistent library filename | Post-build harness/path failure. No GPU fault. |
| `bg-2` | Direct dispatch control completes 16/16 | Successful retained-PM4 control. |
| `bg-3`, `bg-4` | `three-backend hardware identities do not match` | Harness validation failure caused by RADV's parenthesized device suffix; fixed by normalization. Not a GPU fault. |
| `bg-5`, `bg-6`, `bg-7` | Missing measured code-object sidecar | Harness discovery failure; fixed by searching the actual hipEngine JIT cache. Not a GPU fault. |
| `bg-8`, `bg-9`, `bg-10` | Independent Redline/simple, native-peer, and reduction work completes | Successful component evidence used by the final matrix. |
| `bg-11` | Source-matched matrix orchestration reaches an address-zero GPU page fault in its serial q8 packed-dot family child after earlier matrix subprocesses; the journal confirms ring-24/client-10 faults plus two resets/VRAM loss | Genuine supporting reproduction of the signature, but not proof of the canonical same-process trigger. |
| `bg-12` | Fresh independent tapes complete | Successful isolation control. |
| `bg-13` | Canonical same-HSACO full process aborts at row 49; the journal confirms ring-24/client-10 address-zero faults, queue-removal failures, and two resets/VRAM loss | Genuine primary GPU-fault reproduction. |
| `bg-14` | Fresh packed-dot process passes 16/16 | Kernel/family isolation control. |
| `bg-15` | All ten family processes exit 0 | Final partitioned 240/240 evidence. |

A separate Python-feature Redline build failure (`build.rs` requests unavailable
mold) and the immediately repaired TheRock `ld.lld` truncation incident were
host-toolchain failures. The native HIP graph smoke passed after exact package
restoration and before Redline graph measurement; neither is evidence for this
GPU VM fault.

## RX 7900 XTX cross-device result

GPU1 had no prior retained Redline lifecycle result. Run the same 240-row binary
on the RX 7900 XTX with the pinned TheRock libraries. Because ROCr visibility
remaps physical GPU1 to visible ordinal 0, the correct filter pair is
`ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`; here `ROCM_ROOT` is
`/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel`:

```bash
PATH="$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:$LD_LIBRARY_PATH" \
ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 HIPFIRE_BENCH_ARCH=gfx1100 \
  /home/lhl/redline/target/hipengine-w7900-gfx1100/release/hipfire-6409-bench \
  --matrix hipengine --backends redline,hip,vulkan \
  --wave-policy radiowave --redline-queues auto \
  --scheduler-profile default --redline-rmw radiowave-vmem \
  --warmups 3 --samples 7 \
  --out /tmp/hipengine-redline-rx7900xtx-20260728/hipfire-same-hsaco-therock/results.json
```

The answer is **yes, GPU1 also fails**, but not identically:

- rows 1-172 pass correctness for all three backends;
- row 173 is independent packed-dot q6-zero, wg64, backend order
  `hip -> vulkan -> redline`; HIP and Vulkan pass before Redline times out on
  HSA signal `0x7f3de55f8b00` after 5 seconds;
- row 174 reports Vulkan device loss and HIP/Redline `HipError(719)`; the
  remaining 68 rows are rejected;
- stderr says RADV's command stream was cancelled because the context was lost
  and labels its context innocent;
- there is no **userspace** address-zero message, but an unprivileged
  `journalctl -k` correlation subsequently recovered the missing kernel record:
  ten gfxhub faults from the exact `hipfire-6409-be` PID at address zero,
  `ring:24`, `vmid:8`, `pasid:33335`, client 10, with the same
  `0x00801431` / `SQC (data)` / permission-3 / read tuple as GPU0;
- two seconds later amdgpu reports `MES failed to respond to msg=REMOVE_QUEUE`,
  queue eviction/destruction failures, and initiates three successful mode-1 GPU
  resets. Each reset reports VRAM lost and creates an AMDGPU device coredump;
- the harness returns 0 after recording errors, so its exit code is not a pass;
- a fresh post-fault process successfully performs HIP init, 4-KiB allocation,
  synchronization, and free on the RX 7900 XTX **after those resets**.

The incomplete 172-row prefix has **no performance-claim status**. Kernel
correlation upgrades this from merely the same broad lifecycle class to the
same **address-zero gfxhub VM-fault and GPU-reset class** as W7900. The different
row, profiled topology, and userspace manifestation still do not identify the
same originating resource. Raw result/stdout/stderr/kernel-journal hashes are
`e2216c2f...05f1`, `3b55f6c8...a133`, `1e69d736...577`, and
`d7a24b1c...71cea`.

Two setup attempts launched no GPU work and are not faults: applying both
visibility filters as ordinal 1 hides the ROCr-remapped sole device, and omitting
the pinned `LD_LIBRARY_PATH` loads a system ROCr missing
`hsa_amd_counted_queue_acquire`.

### What the cross-device difference changes

The second card materially narrows some claims while broadening the suspect
lifetime surface:

- A W7900-only board, PRO firmware, or 48-GiB-memory explanation is strongly
  disfavored. Both products are Navi 31/gfx1100, but they are distinct cards
  with different product configuration and VRAM capacity.
- Row 49, q8, serial timing, Redline-first backend order, and an address-zero
  **userspace** message are not universal trigger conditions. GPU1 first fails
  at row 173, q6, independent timing, after HIP and Vulkan have passed that row.
  The address-zero **kernel** signature is now common to both cards.
- The RX failure runs a profiled two-lane `MultiQueuePm4Ib`, so the defect is not
  confined to the profiled `SingleQueuePm4Ib` used by W7900 serial rows. Both
  paths still share a separate single-queue ownership acquire plus queue,
  completion, timestamp, indirect-IB, kernarg, and HIP-buffer lifecycle churn.
- HIP and Vulkan succeeding immediately before the RX Redline timeout shows
  that the process context was still useful on entry to that Redline
  measurement. It is evidence against a native backend having already reported
  the failure, but it does not prove Redline ownership: Vulkan had executed in
  the same process and the kernel journal identifies no originating PM4 packet
  or userspace resource.
- The recovered journal confirms that the HSA completion timeout was a
  userspace observation of a queue stopped by an address-zero VM fault. This
  materially favors a common failure class, but two paths could still submit
  different invalid address-zero work.
- Fresh-process recovery on GPU1 occurs only after repeated full-device resets
  and VRAM loss. It disfavors a permanent host-wide wedge, not a process-local
  event. Any colocated workload on that physical GPU would be at risk.
- `REMOVE_QUEUE`/destroy failures occur after the first logged VM fault. They
  strongly confirm teardown/recovery trouble but are not temporal proof that
  queue removal caused the original invalid access.

Therefore the retained statement is **same address-zero gfxhub VM-fault/reset
class, exact originating resource unproven**. There is no defensible fixed row,
fixed queue-owner count, or deterministic userspace-signature claim yet.

## Lifetime audit of the failing path

The audit uses clean Redline `33683f3d`:

- `examples/hipfire-6409/src/redline_backend.rs` — `RedlineBackend::new()` and
  `measure()`;
- `crates/redline-dispatch/src/aql/replay.rs` — `SingleQueuePm4Ib` construction,
  replay, field order, and `Drop`;
- `crates/redline-rocr/src/runtime.rs` — `AqlQueue`, `CompletionSignal`,
  `KernargBuffer`, and their teardown.

At that pinned source, `RedlineBackend` keeps these objects for the process:

- one initialized public-ROCr `Runtime` and selected `GpuDevice`;
- wave32/wave64 `Executable`s for every scheduler profile, loaded at backend
  construction rather than per row;
- one discovered `KernargPool`.

Each serial `measure()` creates and later drops:

- HIP input/output buffers;
- one or more kernarg buffers;
- one unprofiled single-queue ownership/acquire IB;
- one profiled single-queue retained IB, including a completion signal,
  indirect-command buffer, and 16-byte GPU timestamp buffer.

After output readback, the explicit order is profiled IB, ownership IB,
kernargs, then HIP buffers. `SingleQueuePm4Ib` declares its queue owner before
completion/indirect/timestamp pointees, so normal Rust field drop destroys the
queue before those buffers. `AqlQueue::drop()` attempts
`hsa_queue_inactivate()` and then `hsa_queue_destroy()`, but cleanup errors are
not surfaced to the benchmark.

For the 48 successful W7900 serial rows, source/protocol arithmetic implies:

- 96 single-queue retained-IB objects created and dropped (ownership + profiled);
- 960 successful queue submissions (two per warmup/sample, 10 repetitions/row);
- about 53,560 successful kernel dispatches inside the profiled IBs.

These counts are inferred from pinned source and the completed row protocol;
they were not emitted as runtime counters. The first W7900 failing row begins
the 97th/98th queue-owner creation pair, but current logs do not say whether the
fault occurs in its ownership IB, timestamp prologue, first kernel dispatch, or
timestamp epilogue. GPU1's later failure disproves that pair number as a
universal threshold.

The RX independent path has a different profiled topology: one full-device
single-queue ownership IB plus a profiled two-lane `MultiQueuePm4Ib`, with a
distinct kernarg for each independent operation. The exact cumulative owner,
signal, allocation, and submission counts through row 173 have not been audited
and must not be inferred from the W7900 serial arithmetic.

The Qwen persistent control executes more kernel dispatches while reusing one
graph. Combined with the fresh family controls and failures in both single- and
multi-queue profiled paths, this makes **shared queue/IB/signal or GPU-visible
allocation create/drop churn** the leading hypothesis rather than replay count,
packed-dot math, or one profiled queue topology. It does not prove that Redline
core owns the bug; ROCr queue destruction, HIP/ROCr interoperation, amdgpu, or
two distinct defects may be involved.

## Upstream issue and commit review

A GitHub issue/PR/commit search on 2026-07-28 covered gfx1100/gfx11, retained
PM4, HSA queue create/destroy, completion-signal reuse, MES `REMOVE_QUEUE`,
ring 24, and the full fault-status tuple. These are comparative signals, not
proof that unrelated reports share one root cause. In particular, ring 24 and
client 10 are broad gfx11 KFD/gfxhub attribution fields.

### Closest public fault signatures

| Reference | Match to this fault | Important distinction / conclusion |
| --- | --- | --- |
| [ROCm/ROCm#6273, follow-up log](https://github.com/ROCm/ROCm/issues/6273#issuecomment-4582576191) | Strongest match: unrelated gfx1103 processes repeatedly produce address zero, ring 24, client 10, `SQC (data)`, permission `3`, mapping `0`, read faults, then failed `REMOVE_QUEUE`, MODE2 reset, and VRAM loss. Its status payload is `...1430`; ours is `...1431`, the same payload plus `MORE_FAULTS=1`. | APU and ordinary Ollama/VS Code workloads, not Redline or retained PM4. It proves this complete symptom chain is not Redline-exclusive, but a null shader-data access can have many producers. The issue remains open; `cwsr_enable=0` did not prevent its later faults. |
| [ROCm/TheRock#6016, cancellation fault](https://github.com/ROCm/TheRock/issues/6016#issuecomment-5084608332) | Address-zero gfx11 shader fault followed two seconds later by failed `REMOVE_QUEUE`, failed KFD quiesce, coredump, and MODE2 reset while stopping an ongoing model workload. | gfx1103, ring 88, `SQC (inst)`, and cancellation after an already-hung workload. Relevant to teardown/cancellation ordering, not an exact client match. |
| [ROCm/ROCm#5616](https://github.com/ROCm/ROCm/issues/5616) | Address-zero gfx11 fault followed by failed `REMOVE_QUEUE` and reset. | gfx1103/ring 88/`SQC (inst)` under OpenCL, memory pressure, firmware, and sleep/resume discussion. The known gfx1150 MES `0x83` firmware problem discussed there is not our Navi31 MES `0x88`. |
| [ROCm/amdgpu#204](https://github.com/ROCm/amdgpu/issues/204) | Navi31 RX 7900 GRE and an RX 7900 XTX commenter report ring-24/client-10 faults; disabling pinned memory avoided one reproducer. | Fault addresses are nonzero and the decoded client is SDMA0, not SQC data. This is evidence that ring/client alone are insufficient, not an exact match. |
| [ROCm/ROCm#2205](https://github.com/ROCm/ROCm/issues/2205) | Repeated multi-card gfx11 bandwidth loops fault on ring 24/client 10 and then damage MES restore/reset. | 2023 report, nonzero address and SDMA0/1 clients. Structurally similar stress, different producer. |
| [ROCm/TheRock#1271](https://github.com/ROCm/TheRock/issues/1271), [#2655](https://github.com/ROCm/TheRock/issues/2655), and [#5993](https://github.com/ROCm/TheRock/issues/5993) | gfx11 VM faults can precede `REMOVE_QUEUE` failure, repeated reset, and device loss across HIP and Vulkan. | Mostly gfx115x/gfx1102, nonzero/memory-pressure or graphics faults, and known firmware/long-compute cases. Useful recovery-path precedents, not matches for our short address-zero SQC-data dispatch. |

No public GitHub result contained the exact complete `0x00801431` status at the
time of the search. Both hipEngine cards do.

### Concrete lifetime bugs and fixes

| Reference | Established upstream bug | Applicability here |
| --- | --- | --- |
| [ROCm/rocm-systems#8113](https://github.com/ROCm/rocm-systems/pull/8113) / superseded [#8107](https://github.com/ROCm/rocm-systems/pull/8107) (`ROCM-27035`) | HIP CLR cached a raw completion-signal handle after it could be recycled/destroyed, then used it to decide queue idleness. The fix requires queue-owned proof before hardware-queue release. | Highly relevant mechanism for the HIP rows sharing this process, but it is CLR rather than Redline's direct public-HSA queue. Inclusion in the opaque July 11 wheel was not proven from wheel metadata. |
| [ROCm/rocm-systems#6750](https://github.com/ROCm/rocm-systems/pull/6750) | rocprofiler destroyed and recycled an HSA barrier signal while a queued packet still referenced it; the replacement signal could deadlock unrelated work. | Direct proof that HSA handle reuse before packet retirement is a real ROCm failure mode. The affected profiler serializer is not intentionally loaded by this benchmark. |
| [ROCm/rocm-systems#8818](https://github.com/ROCm/rocm-systems/issues/8818) / [#8672](https://github.com/ROCm/rocm-systems/pull/8672) | rocJITsu recycled a destroyed queue's stale doorbell slot, intermittently losing or falsely completing the replacement queue's first dispatch. Retaining one queue passed; recreate failed. | The defect is in the simulator and its physical gfx942 HSA control passed 100 generations, so it is not evidence of a hardware/ROCr bug. Its public queue-generation reproducer and doorbell ledger are an unusually close diagnostic template. |
| [ROCm/rocm-systems#8854](https://github.com/ROCm/rocm-systems/pull/8854) | A timed-out dispatch left a queue alive while packet operands were freed; upstream changed test ownership to destroy the queue before signal/kernarg/device allocations/executable/reader on every exit. | Supports queue-first teardown. Redline already declares/drops the queue before packet pointees, so missing this basic ordering is disfavored; success semantics and delayed hardware access remain testable. |
| [tinygrad/tinygrad#3435](https://github.com/tinygrad/tinygrad/pull/3435) | tinygrad fixed an HSA GPU page fault by retaining code-object reader/storage through execution and synchronizing before destruction. | Redline already retains reader, executable, code bytes, and resolved kernels through every graph/IB, and Hipfire loads executables once for the process. This specific lifetime bug is disfavored. |
| [`a1fc9f584c4a`](https://github.com/ROCm/amdgpu/commit/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8), [`101025e94b53`](https://github.com/ROCm/amdgpu/commit/101025e94b537e8b5426c73a985b26fc95c199cb), [`7919b4cad554`](https://github.com/ROCm/amdgpu/commit/7919b4cad5545ed93778f11881ceee72e4dbed66) | amdkfd fixes for a queue-destroy buffer-access race, a missed queue reset during destroy, and destroy-vs-GPU-reset cleanup. | All are ancestors of upstream Linux v7.1, and there is no evidence the running 7.1.3 CachyOS kernel reverted them. They establish that this boundary has had real races, not a presently identifiable missing-upstream patch. |
| [`1fb710793ce2`](https://github.com/torvalds/linux/commit/1fb710793ce2619223adffaf981b1ff13cd48f17) | Enables the gfx11 MES long-running-compute workaround by default. | Also an ancestor of Linux v7.1. Our short-row faults and clean 321k-dispatch persistent graph do not resemble the long-compute trigger it addresses. |
| [`ade41558d610`](https://github.com/ROCm/amdgpu/commit/ade41558d610e2859df18ea01df5436f0c7711e4) | Flushes the MES queue during reset-time queue removal to align post-reset cleanup. | Newer than the running v7.1 kernel and absent from torvalds v7.1. It may improve recovery after a fault; its one-line reset path cannot explain or fix the first address-zero access. |
| [`f462df879c20`](https://github.com/ROCm/rocm-systems/commit/f462df879c20b63753d86ac07e0904c0b9c7cf6e) | Fixes a use-after-free in the new counted-queue release/query path. | Examined but not currently applicable: pinned Redline resolves the counted-queue symbols as an ABI requirement but creates/destroys its queues with `hsa_queue_create`/`hsa_queue_destroy`; no counted-queue call exists in its crates. |

### Theory update from the review

1. **The immediate event is a null scalar-data read by a live shader.** The
   exact SQC-data/read/permission tuple on both cards says the GPU executed a
   shader memory operation against VA zero. MES `REMOVE_QUEUE` begins two
   seconds later, so it remains recovery damage rather than the initial event.
2. **Resource retirement/reuse is now the leading concrete mechanism.** A late
   packet or prematurely released queue referring to a destroyed/recycled
   completion signal, kernarg, indirect IB, timestamp, or HIP buffer explains
   address zero and fits recreate failures plus persistent-reuse success.
   `#8113` and `#6750` show that this class of signal proof/reuse bug is real in
   adjacent ROCm components; they do not prove Redline or ROCr owns this case.
3. **Doorbell/queue-generation reuse moves up as a diagnostic, not as an
   attribution.** rocJITsu `#8818` reproduces the retained-queue-pass / queue-
   recreate-fail shape exactly, but only in its simulator. Log queue IDs, ring
   bases, doorbell handles/values, and generation numbers before testing that
   hypothesis on hardware.
4. **Code-object-reader lifetime and basic drop order move down.** tinygrad's
   known page-fault fix and ROCm's queue-first teardown rule are already present
   in Redline. The remaining question is whether successful queue destruction
   really retires every GPU reference before immediately freed addresses are
   reused.
5. **A mixed HIP/CLR predecessor remains plausible.** `#8113` gives a specific
   way a HIP queue can be released without queue-owned completion proof. A
   Redline-only lifecycle stress followed by HIP-added and Vulkan-added arms is
   required; row order alone cannot clear a backend.
6. **There is no upstream fix to claim yet.** The closest fault report is open,
   the old kernel race fixes and long-compute workaround are already present,
   and the newer MES flush patch affects recovery only. Disabling CWSR is not a
   defensible workaround because the closest report later reproduced the same
   address-zero SQC-data chain with CWSR disabled.

## What is ruled out, and what is not

### Ruled out or strongly disfavored

- **Packed-dot math or deterministic bad HSACO:** fresh packed-dot passes 16/16
  with the same binary and code-object policy.
- **A universal packed-dot launch-shape defect:** all four variants and both
  workgroup sizes pass in both timing modes when isolated.
- **W7900-only product configuration:** a distinct RX 7900 XTX reaches the same
  broad process-context-loss class.
- **A universal row, q-variant, timing mode, backend order, or userspace error
  string:** the two cards fail at row 49/serial-q8/Redline-first/address-zero
  versus row 173/independent-q6/Redline-last/signal-timeout.
- **The profiled single-queue topology alone:** GPU1 fails from a profiled
  two-lane multi-queue path. The separate single-queue ownership acquire remains
  common and is not ruled out.
- **Simple lack of warmup:** three warmups are present in the canonical repro;
  the separate preheated run also faults.
- **Replay-count exhaustion alone:** one Qwen graph survives 512 launches and
  substantially more captured kernel dispatches.
- **Per-row executable/module accumulation in the canonical Hipfire backend:**
  executables are loaded up front and retained, not loaded for each row.
- **A permanent machine-wide wedge after reset:** a fresh GPU1 process
  initializes, allocates, synchronizes, and frees after three successful
  full-device resets. This does not make the original failure process-local;
  amdgpu reports VRAM loss on every reset.

### Still open

1. Queue create/inactivate/destroy churn or a silent queue-destroy failure.
2. Completion-signal reuse/destruction ordering inside ROCr.
3. Reuse of freed kernarg, indirect-IB, timestamp, or other GPU-visible virtual
   addresses before ROCr/CP has fully stopped touching them.
4. The common separate ownership queue or interactions between it and the
   single-/multi-queue profiled topology.
5. GPU timestamp `COPY_DATA`/`RELEASE_MEM` prologue or epilogue.
6. HIP reset/allocation interoperation with direct public-ROCr submission.
7. Vulkan coexistence or accumulated mixed-backend state; neither failing row
   has a common backend order and Redline-only subtraction is still absent.
8. A specific predecessor family poisoning later state rather than a pure
   create/drop threshold.
9. One common invalid-address producer versus two user paths that independently
   submit address zero; both now have the same gfxhub/ring/client signature.
10. Redline adapter behavior versus an underlying ROCr, HIP, or amdgpu defect.
11. ROCm userspace-version sensitivity versus nondeterministic/address-layout
    or longer-lived driver state. Four fresh full processes per card pass on
    both production 7.14 and the original 7.15 nightly after the faults.

## Next diagnostic matrix

Do not rerun the full 240-row benchmark to diagnose this. Build a minimal
upstreamable stress reproducer and run the following reductions in order:

1. **Repeat one known-good tiny retained IB with recreate-per-cycle.** Run at
   least 128 create/replay/wait/drop cycles and report queue IDs, ring bases,
   doorbell handles/last values, read/write indices, destroy status, completion
   handles, indirect/timestamp/kernarg addresses, resource generation numbers,
   and the exact failing cycle. This tests churn without family changes and
   mirrors the useful rocJITsu `#8818` ledger without assuming its simulator bug.
2. **Reuse versus recreate.** Hold one queue/IB for the same number of replays,
   then recreate every cycle. A recreate-only fault sharply isolates lifetime
   handling.
3. **Create/drop without submit.** If this fails, queue lifecycle is sufficient;
   if it passes, execution/completion is required.
4. **Unprofiled versus profiled.** Remove GPU timestamp buffers and timestamp PM4
   while preserving the kernel tape. This isolates timestamp resources.
5. **One queue versus separate ownership/profile queues.** Fold the ownership
   acquire and dispatch into one retained queue as a diagnostic only. This is
   not a production workaround unless correctness and ordering are proven.
6. **Backend subtraction.** Run Redline-only; then add HIP reset/allocation;
   then add Vulkan construction/execution. This identifies interop requirements.
7. **Prefix/order bisection.** Run `packed-dot` after each predecessor family and
   after repeated copies of one family. Distinguish poison-family from count
   threshold.
8. **Teardown instrumentation and quarantine.** Fail loudly on every
   non-success `hsa_queue_inactivate`, `hsa_queue_destroy`, signal destroy, and
   memory free; sample queue read/write indices before destruction; never
   silently continue after an unproven teardown. As a diagnostic arm, destroy
   the queue first but quarantine its packet-referenced signals/kernargs/IBs/
   timestamps for several later generations. Recreate-only failure disappearing
   under quarantine would directly support delayed-reference/address-reuse.
9. **Fault/coredump/firmware correlation.** PID/PASID/VMID/ring/client and the
   exact `0x00801431` tuple are now confirmed on both cards. Before each stress,
   record MES/MES_KIQ firmware and scheduler parameters. On the next fault,
   capture the transient AMDGPU device coredump before it disappears and pair it
   with the userspace queue/resource-address ledger. Keep privileged collection
   outside the benchmark and record the exact host command and permissions.
10. **Runtime version A/B only after deterministic reproduction.** Run the same
    minimizer against 7.13, production 7.14, and 7.15 components. The current
    fresh-process 7.14/7.15 pass matrix cannot provide a causal version bracket.
11. **Stress the fix on both discrete cards.** Require at least 1,000
    create/replay/drop cycles, the original one-process 240-row matrix on W7900
    and RX 7900 XTX, the strict graph tests, and the Qwen persistent soak before
    calling the lifecycle issue fixed.

The reproducer should use pinned Redline public APIs and live in hipEngine or an
upstream PR branch; do not edit the external `/home/lhl/redline` checkout as
part of hipEngine work. Any temporary patch must be explicit, hashed, and
reported in provenance.

## Promotion policy

Redline cannot become a package default until all of the following are true:

1. the minimal reproducer has an owned root cause and upstream fix;
2. the create/drop stress and original broad process pass without isolation;
3. strict PM4 proof detects zero fallback;
4. natural-prompt/category and 4K-context correctness/performance pass;
5. concurrency, cancellation, server shutdown, and memory-recovery gates pass.

A page fault is fail-stop. No retry, subprocess restart, native-shadow fallback,
or family-specific process split may be presented as a production fix.
