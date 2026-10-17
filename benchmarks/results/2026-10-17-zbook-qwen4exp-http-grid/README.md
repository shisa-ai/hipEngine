# Flash-Next HTTP feature audit on zbook

This is a functional audit, not a performance qualification. `grid.json`
retains the 2026-10-17 baseline at `e7eca661c` and focused verification after
publishing existing MTP counters through standard telemetry. It covers
50 baseline requests and six rechecks. HTTP 200 alone is never a pass for
MTP: the response's route, execution path, proposal counts and backend group
widths are inspected separately.

- Host: zbook, machine `87c566d30a5645cf8d12ed7ef6b6e1e8`, Ryzen AI Max+ Pro
  395, Radeon 8060S/gfx1151, 128 GiB unified memory.
- Model: Unsloth Flash-Next `UD-Q4_K_XL`, four shards at the path below;
  Q8_0 MTP sidecar, production profile, BF16 KV, 32K capacity, c2 admission,
  candidate budget 3, default batch window 0 ms.
- Workload: full ten-prompt `mtpbench-code-general-ja.jsonl` suite including
  category heldouts, up to 128 outputs with EOS respected; separate 256-output,
  control, stream, concurrent, and 29,912-token retrieval cases.
- Result: short explicit MTP really executes. Automatic MTP is off. Only 6/10
  AR/MTP output sequences match; that is a production diagnostic, not a failed
  numerical gate by itself. AR retrieval at approximately 30K succeeds, but
  MTP times out at the 30-second engine-service command boundary and the next
  AR request also fails. Hard thinking controls, logprobs, the generic
  speculative spelling under default scale metadata, C4 physical batching,
  and incremental MTP streaming are not complete.

See the [feature grid and prioritized gaps](../../../docs/QWEN4EXP-HTTP-COVERAGE.md).
The numerical/task qualification basis remains the source-pinned October 15
combined production packet; this functional audit cannot close its task review.
No rates from another physical gfx1151 host are compared to this run.

## Reproduce

```bash
uv run python -m hipengine.server \
  --model /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --backend hip_gfx1151 --quant gguf_ud_q4_k_xl \
  --execution-profile production --served-model-name qwen38fn-grid \
  --max-context-tokens 32768 --kv-storage bf16 --max-active-requests 2 \
  --speculative-provider qwen4_exp_mtp \
  --draft-model /models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --speculative-candidate-budget 3 --speculative-mtp-serving auto \
  --metrics prometheus --host 127.0.0.1 --port 18388
```

After `/ready` reports startup complete, in another terminal:

```bash
uv run python scripts/qwen4exp_http_grid.py \
  --output /tmp/he-qwen38fn-grid-20261017-baseline
```

The full baseline deliberately reaches the broken long-context MTP path.
Do not run it against a server serving other users. Stop the test server after
the audit; do not assume `/ready` establishes recovery after a timeout.

The collector writes raw requests/responses and stream timestamps under the
chosen output directory. The committed packet omits the repeated long prompt
and redundant stream deltas, but retains exact generated IDs, execution
metadata, errors, and SHA-256 hashes of the local raw request records. The
initial recovery response was non-JSON HTTP 500; its recorded parse failure
and status are preserved rather than inventing an OpenAI error body. The
collector now retains non-JSON bodies for future runs.

## Reporting repair

`Qwen4ExpMTPTextProvider` already emitted correct detailed diagnostics but
omitted the standard `mtp_generated_draft_tokens`, `mtp_accepted_draft_tokens`
and `mtp_cycles_count` keys consumed by the HTTP summary and usage accounting.
Those keys now carry the existing counts. The focused HTTP recheck reports
108 proposed, 91 accepted and 36 cycles, agrees with the detailed diagnostics,
and reproduces the pre-repair 128 token IDs. SSE includes owned counts and
accepted/rejected prediction usage. No inference arithmetic or dispatch policy
changed.

Checks:

```bash
uv run pytest -o addopts='' -q \
  tests/test_unit_qwen4_exp_mtp_provider.py \
  tests/test_unit_qwen4exp_mtp_phase_census.py \
  tests/test_unit_qwen4exp_http_grid.py \
  tests/test_integration_server_speculative_provider.py
```

Result: **27 passed**. This audit makes no claim that the failing feature cells
are repaired by those tests or by the telemetry change.
