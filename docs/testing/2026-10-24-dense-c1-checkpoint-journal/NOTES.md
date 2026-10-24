---
status: closed
owns: Public device parity and recovery evidence for the dense GGUF single-request checkpoint-journal migration.
---
# Dense GGUF checkpoint journal validation

The shared `SpeculativeStateJournal` now owns the dense GGUF single-request
provider and target checkpoints. It restores target before provider and keeps
failed cleanup owned for retry. A retired checkpoint cannot restore old state
after commit, even if releasing its backup fails. Packed multi-request routes
keep their existing checkpoint sites because they do not expose the same
adapter-level target transaction.

This changes host ownership, not arithmetic, graph selection, or candidate
counts. It makes no performance or model-quality claim.

## Public checks

Qwen3.8-27B GGUF Q4_K_M on Radeon 8060S (`hip_gfx1151`), BF16 KV, prefix cache
off and a 2048-token allocation. Parent and candidate use frozen source trees;
[validation.json](validation.json) records their module hashes and exact commands.
[candidate.patch](candidate.patch) records the implementation against `e8ad5eaf2`.

- Twelve public `LLM.generate_detailed()` calls per arm cover lengths 37/257,
  one active request with greedy or temperature 0.7/seed 871 sampling, and two
  active requests with greedy sampling. Each case repeats, with 32 outputs per
  request. All 32 parent/candidate request outputs, candidate counts, acceptance
  counts and sampler routes match. No recovered-failure fallback runs.
- Candidate single-request calls capture both provider and target through the
  shared journal. Two-request calls continue to use their existing packed
  checkpoint path and make no journal captures.
- Four recovery cases inject failure or cancellation after the second provider
  repair, at lengths 37/257. The target position restores to its checkpoint,
  journal ownership releases, cancellation exposes only the committed prefix,
  and the next request matches the uninterrupted baseline.
- Every process closes with zero tracked allocations. The frozen candidate
  module hash still matches the working implementation after the tests.

The repeated-token prompts test ownership and determinism, not answer quality.
They are not throughput samples. Raw records are [parent.json](parent.json),
[candidate.json](candidate.json), and [recovery.json](recovery.json).

## CPU review and checks

Focused regressions cover failure during checkpoint capture, restore, commit,
release and request teardown. Failed release preserves the request and pending
cleanup; successful releases are not repeated. Restore eligibility is separate
from retained cleanup ownership. `BaseExceptionGroup` preserves interruptions
while ordinary errors remain catchable as `ExceptionGroup`.

Independent validation passed 227 focused tests before public device checks.
The default CPU unit suite, compileall, fixture checker, registry/CPU-fixture
and smoke-add-plan checks also passed with this implementation. They are shared
validation evidence with the concurrent PARO graph unit, not separate reruns.

## Reproduction

Use a ROCm Python environment and the model path in `validation.json`, or pass
another compatible dense model with `--model`. From the repository root:

```bash
R=docs/testing/2026-10-24-dense-c1-checkpoint-journal
PY="$HOME/miniforge3/envs/therock/bin/python"
ENV=benchmarks/results/2026-10-24-gfx1151-paro-graphs/environment.sh
bash "$ENV" env PYTHONPATH="$PWD" "$PY" "$R/public_dense_journal.py" \
  --candidate --output /tmp/dense-journal-candidate.json
bash "$ENV" env PYTHONPATH="$PWD" "$PY" "$R/public_dense_journal_recovery.py" \
  --output /tmp/dense-journal-recovery.json
```

Run the first driver without `--candidate` against the parent source to obtain
the control. The archived parent and candidate use the same driver hash.
