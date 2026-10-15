# Qwen4Exp X-10 production anchor

This packet establishes the current production timing and quality basis for the X-10 final-stack ablation protocol. It is not itself a complete X-10 ablation.

## Timing anchor

Protocol: canonical 12-case exact-token fixture, compact capture, production profile, chunk1024, warm PLE, one warmup and three measured repetitions per case. All 36 samples were deterministic.

Category-mean median rates:

| Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| p512 | 1,036.4 | 23.68 |
| p1024 | 1,317.7 | 22.72 |
| p4096 | 1,325.9 | 23.25 |

The artifact records host enqueue/device-drain telemetry and output digests for every case.

## Quality anchor

The restored production profile layer2 gate is measurement-valid and numerically/state/lifecycle passing. It remains `requires_task_review` with 13/18 strict-exact task outputs. This is the existing task-review condition; it is not a new numerical failure.

X-10 still requires five counterbalanced paired runs, contemporaneous thermal/load/memory telemetry, and leave-one-component-out ablations before any final-stack retention decision.

Artifacts: `production-anchor.json`, `production-quality.json`.
