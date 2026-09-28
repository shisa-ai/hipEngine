# Qwen4Exp scratch metadata and ordered pilot

This diagnostic confirms private-segment use in two active prefill kernels. It
does not close the scratch-throttle audit or establish a performance improvement.

On Framework machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S/gfx1151,
TheRock 10 / HIP 7.15.26333, the cached code objects report:

| Kernel | Private segment bytes | Static scratch instructions |
| --- | ---: | ---: |
| GDN prefill `<true,true>` | 36 | 11 |
| Dense-wide F16 input `<128,256,64,64,true>` | 92 | 38 |
| Dense-wide F32 input `<128,256,64,64,true>` | 92 | 40 |

These metadata values match the [current prefill trace](../2026-09-28-qwen4exp-current-attribution/README.md).
Instruction counts are static disassembly counts, not dynamic traffic.
The exact libraries, hashes, symbols and metadata are in [artifact.json](artifact.json).

## Ordered pilot

Four fresh processes ran serially in sham/scratch/scratch/sham order. Each
warmed only the scratch-free strict GDN victim, measured 21 victim calls,
launched one trigger, then measured 21 victim calls. Every call restored input
state outside the HIP-event timing. Shape: 64 tokens, Hk16/Hv48/Dk128/Dv128,
synthetic FP32 operands matching Qwen3.8-Flash-Next geometry; no model weights
or quantization were loaded. Scratch arms use wave-norm GDN as trigger; sham
arms use the victim itself. Outputs and recurrent state match byte-for-byte
across arms and before/after windows.

| Arm | Before median ms | After median ms | After / before |
| --- | ---: | ---: | ---: |
| sham | 2.592708 | 2.592933 | 1.000087 |
| scratch | 2.603521 | 2.599303 | 0.998380 |
| scratch | 2.598896 | 2.600032 | 1.000437 |
| sham | 2.591284 | 2.592768 | 1.000573 |

There is no large persistent penalty on this scratch-free victim in these runs.
This does **not** test a scratch-enabled victim, dense-wide as trigger, the
complete model queue, or clock/scratch-allocation causality. X-5 remains open.

## Reproduce

Use the environment in the current-attribution packet, with
`HIPENGINE_REQUIRE_CACHED_BUILD=1` and
`HIPENGINE_COMPILER_VERSION_FILE=/tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt`.
Run these commands serially with the environment's Python; each invocation is a
fresh process:

```bash
python scripts/qwen4exp_scratch_order_probe.py --arm sham --tokens 64 --repeats 21 --output /tmp/order-0.json
python scripts/qwen4exp_scratch_order_probe.py --arm scratch --tokens 64 --repeats 21 --output /tmp/order-1.json
python scripts/qwen4exp_scratch_order_probe.py --arm scratch --tokens 64 --repeats 21 --output /tmp/order-2.json
python scripts/qwen4exp_scratch_order_probe.py --arm sham --tokens 64 --repeats 21 --output /tmp/order-3.json
```

For metadata, copy the named cached library to a temporary directory and run
the same environment's `llvm-objdump --offloading <copy.so>`. On this toolchain
that extracts `<copy.so>.0.hipv4-amdgcn-amd-amdhsa--gfx1151`. Inspect it with
`llvm-readobj --notes <code-object>` and `llvm-objdump -d <code-object>`.
No recompilation is needed. Raw notes/disassembly stay in
`/tmp/qwen4exp-scratch-audit/`; their hashes are in the artifact.
