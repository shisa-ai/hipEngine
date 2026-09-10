# Qwen4Exp gfx1151 P12 validation report

- Date: `2026-09-02`
- Production manifest: `37d595645b376083c2c2045687b76fa51b3bdf3c74e511af4f7831a180e16779`
- Strict manifest: `ec1c7828611b055014df9f5e5eb8a6c814e283606227cdfc6b33bfcaeeb0109b`
- Focused tests: **268 passed**

## Canonical exact-token packet

| Profile | Context | Prompt tok/s | Decode tok/s | Max case CV |
| --- | ---: | ---: | ---: | ---: |
| strict | 512 | 61.049 | 13.517 | 4.42% |
| strict | 1,024 | 60.322 | 13.427 | 4.02% |
| strict | 4,096 | 52.561 | 9.472 | 2.62% |
| production | 512 | 83.352 | 14.180 | 0.50% |
| production | 1,024 | 82.933 | 14.164 | 0.71% |
| production | 4,096 | 69.200 | 12.160 | 0.25% |

## Binding status

- Short strict/production determinism, lifecycle, current-manifest quality/state/task/c2, and unlocked long-context gates pass.
- Final five-pair comparator windows remain open.
- Required 4K MTP remains blocked by the 1K provider and draft capacity.
- No final match, beat, or campaign-closure claim is made.
