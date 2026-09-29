# MT-5: the MTP prefill catch-up cost

**Measured at 1K: 8.00 MiB retained by the caller, 8.01 MiB created by the
catch-up itself, and 0.176 ms.** The cost is linear in prompt length at
**8 KiB/token per array**, and it is **memory, not time**.

This closes MT-5's open question ("measure our MTP prefill catch-up cost at 1K
first").

## Protocol

`python3 scripts/mtp_prefill_catchup_cost.py --output …/cost.json`. CPU-only,
host-side: it measures `_llama_cpp_mtp_catchup_rows` from
`hipengine/generation/qwen35_gguf.py`, which is the path the engine runs. Nine
repeats after two warmups; medians reported. `hidden_size = 2048`
(Qwen3.6-35B-A3B `text_config.hidden_size`).

The bench harness carries its own copy of this function; the script asserts the
two agree on every measured case, so the number cannot drift away from the
implementation the engine uses.

## Result

| prompt tokens | retained (caller) | catch-up peak | total | catch-up median |
|---:|---:|---:|---:|---:|
| 1024 | 8.00 MiB | 8.01 MiB | 16.01 MiB | **0.176 ms** |
| 4096 | 32.00 MiB | 32.01 MiB | 64.01 MiB | 2.503 ms |
| 16384 | 128.00 MiB | 128.01 MiB | 256.01 MiB | 9.973 ms |

Reference approach (one lagging row, as the ledger describes it): **8,192 bytes =
8.0 KiB**, constant in prompt length.

## Decomposition: what the catch-up actually costs

The two memory terms have different causes, and conflating them would overstate
the finding:

- **`retained_hidden_bytes` is not caused by the catch-up.** It is the prompt's
  hidden rows the caller already holds because the draft is primed with them
  (`draft.prime_prompt(token_ids, result.hidden_seeds)`). Any approach that primes
  the draft from the prompt's hidden states pays this.
- **`catchup_peak_bytes` is the catch-up's own marginal cost.** It is the
  right-shifted copy `concat([zero, hidden[:-1]])`, which exists only so that
  draft row `i` reads the target hidden from position `i-1`. This is avoidable in
  principle by having the consumer read with a one-row offset instead of
  materializing a shifted array.

So the catch-up's own cost at 1K is **8.01 MiB and 0.176 ms**, not 16 MiB.

## What this means

- **Time is not the issue.** 0.176 ms at 1K and 9.973 ms at 16K are small against
  a prefill of the same length.
- **Memory is the issue, and it scales.** At 8 KiB/token per array, a 128K context
  holds **1 GiB retained plus 1 GiB copied** for the catch-up alone. The reference
  approach holds 8 KiB.
- **The gap to the reference is structural, not a tuning miss.** The reference
  keeps one lagging row and replays the final chunk; we materialize the whole
  prompt's hidden state and a shifted copy of it. Closing it means changing how
  the draft is primed, not micro-optimizing the copy.

## Limits

- **Host-side only.** This excludes the prefill itself, the draft model, and all
  device work, so it is not an end-to-end MTP prefill cost.
- Bytes are exact and deterministic; the millisecond figures are medians over 9
  repeats on this host and are the noisy part.
- `hidden_size = 2048` for this model; the byte figures scale with it.
- **`performance_claim: false`.** This is a cost measurement, not a speedup.

## Next

The measured cost does not by itself justify a change. The concrete question it
raises is whether the shifted copy can be dropped by giving the consumer a
one-row offset, which would remove the 8 KiB/token second array without touching
the priming contract. That is a separate unit with its own correctness gate.
