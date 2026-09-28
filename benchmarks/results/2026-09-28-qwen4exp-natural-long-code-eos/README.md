# Focused code completion after increasing the output cap

Both strict and `production_dense_q8_restore` reach EOS at **2424 tokens**
with the unchanged booking task and a 4096-token output cap. Outputs match
across arms and both repeats. Their first 2048 token IDs match the earlier
[truncated capture](../2026-09-28-qwen4exp-natural-long-dense-q8/README.md).
The completed implementation passes 11 generated tests and six independent
functional cases in a network/filesystem-isolated process.

This resolves code truncation only. It does not remove the prior language-task
failures, establish full task qualification, or promote the candidate.

## Evidence

[artifact.json](artifact.json) preserves the command, model/host identity,
source, manifests, prompt hashes, outputs and per-repeat hashes. Source is
`f0d151d32`, with a clean tracked tree. Host is Framework `gfx1151`, machine
`55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S, TheRock10/HIP7.15.26333.
Model is unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`.

Only `long_code_booking_merge` was rerun. Input length is438 tokens, chunk/input
budget1024, output cap4096, two repeats per arm. Candidate host observations
are398 calls per prefill, 796 total. Both arms close with zero tracked live
allocations. Host counts do not certify GPU role/layer engagement. No latency
or throughput was measured for a performance claim.

The generated out-of-order-revision test checks only booking A's acceptance,
so it cannot distinguish which interval won. The independent cases add a
competing booking that exposes wrong revision selection. They also check
last-arrival ties, cancellation before conflict processing, end/ID ordering,
rejected-booking isolation, room independence, adjacency, empty input and
immutability. The implementation satisfies these checks; the output still
uses Markdown fences despite the request for only executable code, and its
own revision test remains weak. Do not turn functional-check success into an
unqualified task-format or test-quality pass.

## Reproduction

Use the environment and command from the earlier capture, but use only
`--prompt-id long_code_booking_merge`, `--max-tokens 4096`, and output
`/tmp/qwen4exp-natural-long-dense-q8-code4096.json`. Keep all other arguments
unchanged. The complete argument vector is in the artifact.

The entire generated code was inspected before execution: it contains only
the booking function, assertions and a standard-library `copy` import. The
checker rejects any output whose extracted code hash differs from that
reviewed version. Run its checks in the isolated environment:

```bash
P="$PWD/benchmarks/results/2026-09-28-qwen4exp-natural-long-code-eos"
timeout 10s bwrap --unshare-all --die-with-parent --new-session \
  --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp --clearenv --setenv PATH /usr/bin \
  --ro-bind "$P/artifact.json" /capture.json \
  --ro-bind "$P/check_code.py" /check_code.py --chdir /tmp \
  /usr/bin/python3 -I -B /check_code.py
```

Result: `11 generated tests and 6 independent cases passed; input immutability checked`.
The sandbox has no network and does not mount the repository, home directory
or credentials. Only the two named files and system binaries/libraries are
readable. Do not use this checker as a generic untrusted-code runner.
