# Qwen4Exp X-10 combined production quality

The current named production composition was run through the normative `qwen4exp_layer2_profile_gate.py --candidate production_baseline` gate with chunk1024 and three candidate repeats. The numerical, state-repeat, and lifecycle gates pass. The task gate remains review-required because only 13/18 free-output task continuations are strict-exact.

- 450 full-vocabulary teacher-forced rows.
- Mean KL 6.899e-4, p95 4.361e-3, p99 8.414e-3, max 1.463e-2, top-1 446/450 (99.111%).
- Repeat determinism passed for all three runs; state-repeat passed; lifecycle passed.
- The mixed_ja_en category mean KL is 1.245e-3 against its per-scope 1e-3 limit, so the artifact's numerical decision still reports the repository's calibrated/global numerical pass while preserving the scope detail for review.
- Task gate: 13/18 strict-exact, deterministic repeat outputs; status `requires_review`.
- 573 contemporaneous `rocm-smi` samples were preserved, with temperatures 40-84 C and no sampler errors.

Artifact: `production-final-quality.json`; telemetry: `telemetry.jsonl`.

Decision: do not close X-10 or claim final-stack qualification. The combined stack is numerically/state/lifecycle valid but remains blocked on task review and the documented scope review. No production composition change was made.