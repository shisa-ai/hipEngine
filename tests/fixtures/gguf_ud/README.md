# UD codec fixtures and arithmetic contracts

These fixtures test seven GGML storage types against llama.cpp C decoders at
`17252c769a63c1cb650ce98ae309cf4de0da7778`. They do not establish model quality
or numerical qualification for another backend or execution mode.

## Independent decoded values

`synthetic.npz` pairs compressed blocks with bit-exact F32 C-decoder output,
including signed zero. The generator builds an exact Git archive of the pinned
source; it does not use hipEngine's decoder to generate expected values.
`manifest.json` records source hashes, the fixture hash, MIT license text,
independent block sizes and the generation command:

```bash
PYTHONPATH=. .venv/bin/python scripts/gguf_ud_codec_fixture.py
.venv/bin/python -m pytest -o addopts='' -q tests/test_gguf_ud_codecs.py
```

The tests assert codebook index, sign selector, packed high-bit and subscale
coverage at nonzero super-scale. Coverage enumerates each selector's domain,
not every Cartesian product of selectors. Super-scales include both zeros,
the smallest positive FP16 subnormal, smallest normal, unit/negative unit,
maximum finite FP16 and a nontrivial fractional value. Reshaping into 1/2/8/64
rows and rank-3 arrays checks row transitions and logical dimension order.

`real_rows.npz` contains first/last full rows from 20 tensors across the two
published files and all seven types, including K5120 and K17408. The same C
oracle produced their F32 values. `real_rows.json` records tensor identities,
model payload pins and the fixture hash. Full payload verification preceded
extraction; regeneration requires that verification again.

## Rounding and projection boundaries

Codec output is F32. No BF16/FP16 weight cast or activation conversion belongs
to codec equality. Comparing rounded values alone cannot pass this gate.

The strict raw dense leaves in `tests/test_gguf_ud_dense.py` use:

- BF16 input bits widened exactly to F32; decoded weights remain F32, without
  an intermediate BF16 weight rounding boundary.
- 128 strided F32 accumulators with separate multiply and add, not FMA;
  wave32 reductions at offsets 16/8/4/2/1; then four wave totals added serially
  to an F32 zero accumulator.
- Either the exact F32 result or a final BF16 round-to-nearest, ties-to-even
  cast. Q3_K embedding similarly rounds the decoded value once to BF16.
- No FP16 activation/output contract. The raw-leaf wrapper rejects FP16 output;
  adding FP16 support needs a separate conversion and numerical gate.

This is a declared raw-leaf contract, not parity with the previous BF16-expanded
resident route. The leaf suite independently compares against high-precision
NumPy dot products for its outer KL/top-1 floor and repeats its exact schedule
three times. It covers six raw dense types; IQ2_XS remains a separately budgeted
BF16 fallback at model integration. CPU oracle completion does not close that
compact-resident gap or the full-shape/model/backend gates.
