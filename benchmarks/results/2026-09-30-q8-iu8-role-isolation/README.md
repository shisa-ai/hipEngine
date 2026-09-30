# Q8 iu8-WMMA dense route: role-isolated numerical sweep

**Verdict: the route is rejected. Every role family it can claim fails the
canonical numerical envelope on its own. No role set can be promoted.**

Host `gfx1151` (machine `55ea6c509d0b49eea8de7094a1023668`), source commit
`a9a33dd88553ce77b43ec0cc6ee3af9df19bc906`, model
`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL` (fingerprint
`fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53`), Flash-Next
UD-Q4_K_XL/BF16, 12 canonical 512/1K/4K cases, 64 shared-teacher decode
transitions, 3 repeats, **780 rows** per arm. Gate:
`scripts/qwen4exp_q8_repair_depth_gate.py`.

## Why this sweep exists

The route (`HIPENGINE_QWEN4_EXP_Q8_IU8_WMM`) was a single boolean claiming every
eligible Q8_0 prefill linear. Its two recorded rejections each closed an *arm*
rather than a role. `20260915T015835` rejected the unchanged generic switch and
said so explicitly ("not role-isolated non-GR work"); `20260915T022059` rejected
the non-GR arm while recording that it "does not implicate every projection" and
asked for "identical operands by non-GR role". `docs/REFACTOR.md` "Non-GR Q8
Diagnostic Scope" had already named the prerequisite.

A boolean cannot express role coverage, so this sweep first added the missing
control: `HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES` narrows the route to named roles,
matched as dotted-path prefixes against a weight's slot path after its
`layers.<n>.` prefix. The scope is part of the dispatch-resolution cache key,
because without that two roles of identical shape share a key and the first
resolution is replayed for the rest — the same omission that produced a
documented false pass for the WMMA layer scope.

## Result

Binding envelope: mean KL ≤ 0.001, p95 ≤ 0.005, p99 ≤ 0.02, max ≤ 0.05,
top-1 ≥ 0.99.

| arm | claims | mean | p95 | p99 | max | top1 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline (`production_conservative`) | 0 | 0.000448 | 0.002085 | 0.006351 | 0.017192 | 0.99103 | PASS |
| full route | 28656 | 0.001394 | 0.007240 | 0.015292 | 0.031511 | 0.98718 | **FAIL** mean/p95/top1 |
| `attn_gate` | 2592 | 0.001680 | 0.009228 | 0.022562 | 0.037380 | 0.98590 | FAIL |
| `attn_k` | 864 | 0.001219 | 0.006483 | 0.014633 | 0.027915 | 0.98462 | FAIL |
| `attn_output` | 864 | 0.001439 | 0.007642 | 0.017836 | 0.037103 | 0.99487 | FAIL mean/p95 |
| `attn_q` | 864 | 0.001859 | 0.008928 | 0.025821 | 0.106523 | 0.98333 | FAIL |
| `attn_qkv` | 2592 | 0.001702 | 0.009016 | 0.021932 | 0.099049 | 0.98974 | FAIL |
| `attn_v` | 864 | 0.001308 | 0.007258 | 0.014271 | 0.023857 | 0.98974 | FAIL |
| `hc_attn_down` | 3456 | 0.001490 | 0.008462 | 0.019720 | 0.045730 | 0.98846 | FAIL |
| `hc_ffn_down` | 3456 | 0.001439 | 0.007293 | 0.016456 | 0.032988 | 0.99103 | FAIL mean/p95 |
| `ple_key` | 72 | 0.001596 | 0.009174 | 0.018155 | 0.058100 | 0.98718 | FAIL |
| `ple_value` | 72 | 0.001659 | 0.008063 | 0.018750 | **0.115324** | 0.98462 | FAIL |
| `shared_down` | 3456 | 0.001447 | 0.007564 | 0.020310 | 0.037998 | 0.99103 | FAIL |
| `shared_gate` | 3456 | 0.001312 | 0.006389 | 0.018135 | 0.062865 | 0.98718 | FAIL |
| `shared_up` | 3456 | 0.001455 | 0.007162 | 0.019925 | 0.044718 | 0.99103 | FAIL mean/p95 |
| `ssm_out` | 2592 | 0.001667 | 0.008813 | 0.023085 | 0.062189 | 0.98974 | FAIL |
| `expert_down` | 0 | 0.000548 | 0.002744 | 0.007226 | 0.021451 | 0.99103 | inert |
| `hc_attn_up` | 0 | 0.000548 | 0.002744 | 0.007226 | 0.021451 | 0.99103 | inert |
| `hc_ffn_up` | 0 | 0.000548 | 0.002744 | 0.007226 | 0.021451 | 0.99103 | inert |

### What the sweep establishes

- **14 of 17 roles claim and every one of them fails alone.** All 14 exceed
  `kl_mean` and `kl_p95`; no role passes. The failure is a property of the iu8
  arithmetic, not of one projection.
- **The failure is not localizable.** A single role family reaching 12% of the
  calls (`shared_gate`, 3456) already reproduces the full route's top-1
  (0.98718, identical) while producing a *worse* max (0.062865 vs 0.031511).
  This is the non-monotonic tail `20260915T022059` warned forbids treating KL
  differences as additive causal contributions.
- **Two projections of 72 calls each still break the envelope.** `ple_key` and
  `ple_value` are one tensor each (0.25% of calls) and both fail; `ple_value`
  has the worst max in the sweep (0.115324, 2.3× the limit).
- **Three roles are unreachable by this route.** `expert_down`, `hc_attn_up` and
  `hc_ffn_up` claim zero calls. The first shares its `(640, 2560)` shape with
  `shared_down`, which does claim, so this is a parent-variant difference, not
  geometry. Their arms are not evidence about those roles and are reported as
  inert rather than as passes.

### Controls

- **Reproduction.** The full route reproduces `20260915T022059` on mean and p95
  (0.001394 vs 0.0013826; 0.007240 vs 0.0069630). Its max and top-1 differ
  (0.031511 vs 0.1051165; 0.98718 vs 0.99103) because this arm is broader — it
  includes eligible GR-down, and `candidate_dispatch_calls` is 28656 against the
  rejection's 21744 non-GR calls plus 6912 excluded GR reads.
- **Route-off control.** The three inert arms are one configuration (mutually
  identical to 1e-12 on mean, max and top-1) and **pass every binding
  threshold**. They are the correct control for "this route claims nothing":
  the candidate inherits from `production_gr_up_restore`, so it runs production
  defaults for every flag it does not name, unlike `production_conservative`
  which zeroes all conservative flags. Their pass isolates this route's
  arithmetic as the cause.

## Reproducing

```bash
source /home/lhl/tmp/hip1151-t6/env.sh
HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES=<role> \
  python3.12 scripts/qwen4exp_q8_repair_depth_gate.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
    --candidate production_dense_q8_restore \
    --output role_<role>.json
```

Each arm takes ~17 minutes. `artifact.json` carries each arm's envelope,
thresholds, dispatch counts and scope; the full per-arm gate output (per-case
and per-row detail, ~40 KB each) is not retained and is regenerable from the
command above. `python3 analyse.py` re-derives every verdict from the numbers in
`artifact.json` against the recorded thresholds and exits non-zero if its
re-derivation disagrees with any recorded `hard_gates_passed`, so a gate-revision
change that silently relaxed a bar shows up as a disagreement.

**Provenance note.** The role scope reaches the gate through the environment, and
the gate's artifact records only `candidate_spec.environment` — which does not
include the scope. Each arm's scope is therefore recorded in `artifact.json`
under `arms.<arm>.roles` and in `summary.jsonl`, not inside the gate's own
output.
