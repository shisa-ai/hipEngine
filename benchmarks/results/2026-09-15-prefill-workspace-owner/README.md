# Prefill-Only Workspace Owner

Source `fabd2da76`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon8060S/gfx1151, Flash-Next UD-Q4_K_XL/BF16, capacity4352.

`Qwen4ExpPrefillWorkspace` now owns prefill scratch/input storage separately
from decoder state and KV. Default allocation order and sizes are unchanged,
verified against the frozen allocator census at1024/2048/4096. Cleanup follows
owning workspaces rather than borrowed aliases.

The real-GPU check prepares a1024 workspace alongside native4096 storage:

| Allocation | Bytes |
| --- | ---: |
| Measured extra prefill-only owner | 1,633,867,320 |
| Full1024 runner at the same capacity, validated recipe | 1,897,478,408 |
| Derived duplicate decoder/state/KV bytes avoided | 263,611,088 |

No decoder/KV owner is replaced. The added bytes match the prefill-only
estimate exactly, reserve remains available, and tracked allocations return
to zero after close.

Code2052, Japanese4097 and mixed2049 inputs are reconstructed independently.
Native4096 final logits/full state match frozen pre-refactor isolated
references. Borrowed1024 execution matches all nine logits per case plus
prefill/final KV,index and recurrent state, with three repeats:
27 unique logit rows and81 repeated comparisons.

45 focused CPU tests also cover allocation order, ownership deduplication,
partial scratch/input failures and closing a runner while a foreign workspace
is borrowed. The foreign owner is not freed by the borrower.

**Automatic selection is not enabled.** This validates its storage mechanism,
not its admission policy, latency or new throughput. The default remains1024.

```bash
.venv/bin/python scripts/qwen4exp_owned_workspace_check.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
  --allocation-evidence /tmp/hipengine-journey-execute-20260914/resume-chunk4096-native-c2-accounted.json \
  --reference-capture /tmp/hipengine-journey-execute-20260914/resume-chunk4096-c2-deferred.json \
  --output /tmp/hipengine-journey-execute-20260914/resume-owned-workspace-check.json

.venv/bin/python benchmarks/results/2026-09-15-prefill-workspace-owner/assemble.py \
  --raw-root /tmp/hipengine-journey-execute-20260914
```

Use the cached-build ROCm environment recorded in the chunk4096 artifact.
The capture records command, source, host, model, manifest and reference hash.
