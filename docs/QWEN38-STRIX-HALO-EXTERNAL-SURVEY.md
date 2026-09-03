# Qwen3.8-27B on Strix Halo: external implementation survey

Many public Qwen3.8-27B implementations report impressive performance on Strix
Halo, but the reported numbers mix different models, quantizations, draft
methods, prompts, and timing boundaries ([overview][S1]). We independently
reproduced the major claims on one Ryzen AI MAX+ 395 system, checked the
outputs—not only the speed—and ran a shared multilingual prompt suite where the
implementations allowed it. For direct engine comparisons, we use the same
standard `Q4_K_M` file and compare each compatible route with both stock
llama.cpp and the current hipEngine implementation. Routes that require another
model format remain source-claim reproductions, not direct engine rankings.
This report separates reproducible, usable performance from narrow replay
results and invalid state-contaminated output.

## Conclusions

- **No engine wins every standardized workload:** in the pinned 2026-08-30/31
  matrix on standard `Q4_K_M`, Nathan leads prefill and AR at C1; Laurent
  leads prefill at C2-C3 and C6-C8, AR at C2, and fixed-K3 MTP at C1-C2 and
  C6; stock HIP leads prefill at C4-C5 and fixed-K3 MTP at C7-C8; mainline
  Vulkan leads fixed-K3 MTP at C5. The retained current-head overlays
  (2026-09-03, below) move the leadership: hipEngine now leads AR at C1 and
  C3-C8, prefill at C1 and C3-C7, and fixed-K3 MTP at C3-C6; Laurent keeps
  prefill C2/C8, AR C2, and MTP C1-C2; stock HIP keeps MTP C7-C8.
- **hipEngine AR now leads seven of eight widths:** the current-head overlay
  measures C1 11.511 (+3.1% versus Nathan's frozen 11.162), C2 19.249
  (-3.0% versus Laurent's 19.835), and C3-C8
  24.974/31.478/37.995/43.093/46.153/50.605, improving on the pinned row at
  every width. C2 remains the only AR deficit.
- **hipEngine prefill now leads six of eight widths at the current head:** the
  2026-09-03 refresh (after the B2 input-F16 and related retentions) measures
  **201.0/181.9/207.2/233.1/258.2/285.1/291.3/301.8 prefill-dominant tok/s at
  C1-C8**, leading the frozen external matrix at C1 (+31.0% versus Nathan)
  and C3-C7 (+7.7% to +16.3%) and trailing only at C2 (-14.2% versus
  Laurent) and C8 (-1.3% versus Laurent). The pinned baseline row remains in
  the table for provenance; the C2 dip and the marginal C8 gap keep their
  standing blockers.
- **hipEngine explicit K3 MTP beats own AR at C1-C4 and leads the fixed-K3
  matrix at C3-C6:** the one-group route lifts C5-C8 to
  **37.280/41.048/44.492/50.893 tok/s** (+33.2/+25.1/+34.4/+43.6% over the
  reviewed K3 row), and C2-C4 measure **30.094/32.919/37.985 tok/s
  (1.564x/1.342x/1.207x own AR)** with 10/10 AR equality. C8 is the first
  C2+ width where explicit MTP reaches own AR (**1.0057x**); C5-C7 remain
  slightly below. The retained singleton-target route resolves the C1
  capacity defect without reducing resident capacity: at capacity 3, C1 is
  **19.428 tok/s (1.687x AR)** with acceptance 0.7889 and 10/10 equality.
  Exactly one active request uses the transactional C1 target; C2+ keeps the
  packed target path ([L17]). Automatic production C2-C8 remains K0.
- **Laurent's ordinary built-in MTP path provides broad, usable gains:** it
  leads standardized MTP at C1-C2 in the pinned matrix. Stock HIP is stronger
  at C7-C8. This is separate from Laurent adaptive DFlash2, which remains
  unsafe across sequential requests.
- **Strongest reproduced specialized result:** `q38rocm` strict MTP K4 reached
  **38.85 decode tok/s** under its published protocol and **35.575 arithmetic /
  32.969 token-weighted decode tok/s** on our shared prompt suite. It requires
  the custom `ROCmFP4_FAST` model and exactly one server slot, so it is not part
  of the standard-`Q4_K_M` engine ranking and has no C2-C8 result.
- **Fastest valid single task:** Laurent adaptive DFlash2 reached **56.532 decode
  tok/s** for complete structured JSON in a fresh server process.
- **Laurent adaptive DFlash2 is not usable as a sequential server:** its 66.838
  tok/s JSON row repeated prose from the previous request. That result is
  invalid. Laurent's ordinary built-in MTP path did not show this defect.
- **The highest raw number is workload-specific:** Kyanite reached **167.64
  tok/s** by replaying a warm count-to-30 sequence, not by generating novel
  text.
- **MTP must be routed by concurrency:** Mike's Q8 result was 2.23x AR at C1,
  neutral at C3, and 0.84x AR at C4.
- **hipEngine is usable with the standard `Q4_K_M`:** the reviewed C1-C8 run
  passed 80/80 explicit-K3 exact/route/budget cells, the C6/C8 K1 pair passed
  all 40 control/candidate cells, and the current-head refresh passes
  mtp_self_exact at every re-measured width. At the current head it leads AR
  at C1 and C3-C8, prefill at C1 and C3-C7, and K3 MTP at C3-C6; C7/C8 MTP
  trail stock HIP. Production explicit C1 now uses the retained singleton
  target under a wider resident owner while preserving the physical C2+ path.
  Automatic production C2-C8 remains K0.

### Route decisions

**Yes** means the locally tested route produced valid output and showed no
server-lifecycle correctness blocker. **No** means the tested route exposed a
correctness blocker. A Yes does not mean that different quantizations have
equal model quality.

| Route | Usable? | Decision |
| --- | :---: | --- |
| hipEngine reviewed baseline `b768516f2`, with retained successors through `b58a70c82` | **Yes** | Current explicit K3 reaches **19.428/30.094/32.919/37.985/37.280/41.048/44.492/50.893 tok/s at C1-C8**; C1 is 1.687x own AR at resident capacity 3, and C8 is 1.0057x. The singleton target resolves the former capacity-gated C1 defect while retaining the wide provider and packed C2+ target; C3 collateral is +0.15% with unchanged acceptance and 10/10 equality. Prefill leads six of eight widths. Automatic C2-C8 remains K0. |
| `q38rocm` v1.5.2, `ROCmFP4_FAST`, strict MTP K4 | **Yes, C1 only** | Strong specialized result. Strict mode requires exactly one server slot and a custom model, so it is not ranked against standard-`Q4_K_M` engines. |
| Laurent built-in MTP K3, standard `Q4_K_M` | **Yes** | Broad reusable llama.cpp route; it leads standardized MTP at C1-C2 and C6. |
| Laurent adaptive DFlash2 fork `c28d538df` | **No** | Fast in a fresh process, but unsafe for sequential requests because speculative state leaks between requests. |
| `q38rocm` normal MTP K3, standard `Q4_K_M` | **Yes** | Supports C1-C8, but did not lead any standardized cell. |
| KyaniteLabs HIP MTP+ngram | **Yes** | Correct output. The 160+ tok/s result applies only to warm repetition replay. |
| PieBru recipes on Nathanw fork `0eb528051` | **Yes** | Q5/Q6/Q8 speed claims reproduced. Latest mainline is slightly faster in decode. |
| MikeVeerman stock llama.cpp pin `152d337fa`, Q8 MTP | **Yes** | Use MTP at low concurrency. Disable it for dense parallel work. |

### Source-claim reproduction

This table answers whether we could reproduce each source's result under its
own protocol. The rows use different models, workloads, output lengths, and
timing boundaries. **Do not use this table to rank engines.** The standardized
`mtp-bench` comparison below provides the apples-to-apples ranking.

| Route | Published claim | Local measurement |
| --- | --- | --- |
| `q38rocm` strict MTP K4 | 14.02 AR; 30.56-36.04 MTP decode tok/s | 14.31 AR; **38.85** MTP on the source protocol. Common suite: **35.575 arithmetic / 32.969 token-weighted decode tok/s**. |
| Laurent adaptive DFlash2 | 65.6 structured; 26.1 prose decode tok/s | Valid complete JSON: **56.532**. Prose: 25.618. Fresh-process common suite: **37.752 arithmetic / 34.483 token-weighted decode tok/s**. |
| Kyanite MTP+ngram | 59.7 cold; 148-163 warm count-to-30; 11-24 real traffic | 60.95 cold; **164.13-167.64** warm count-to-30. Common suite: **24.867 decode / 20.518 complete-wall tok/s**. |
| PieBru Q5/Q6/Q8 | About 23-24 / 17-21 / 15-18 served tok/s | **24.706 / 20.549 / 18.197 complete-wall tok/s** on Nathan. |
| MikeVeerman Q8 concurrency | MTP is 2.19x AR at C1 and 0.78x at C4 | **2.23x** at C1, 1.01x at C3, and **0.84x** at C4. |

### Standardized `Q4_K_M` comparison

These tables provide the apples-to-apples engine comparison. Every row uses the
same standard `Q4_K_M` file, ten `mtp-bench` prompts, greedy sampling, disabled
prompt caching, 24 generated tokens per request, and one physical host. Values
are aggregate complete-wall tok/s; higher is better. The comparison uses each
engine's production/default KV precision—BF16 for hipEngine and default F16 for
the llama.cpp routes—so it compares deployable engine configurations rather
than forcing identical internal arithmetic.

The prefill pass generated one token per request and reports prompt tokens divided
by barrier-to-last-completion wall time. It therefore includes one generated
token and API overhead. We use this common end-to-end boundary because llama.cpp
exposes internal prompt timing but hipEngine does not expose an equivalent field.
The external rows below remain the same-host measurements recorded on
2026-08-30 ([L6]). The full C1-C8 hipEngine rows were re-run from tracked-clean
`b768516f2` on the same physical host and protocol on 2026-08-31 ([L9]). The
C6/C8 K1 successor was measured on the same host, model, prompt suite, D24
budget, and complete-wall boundary through runtime `1f4687cab` on 2026-09-01
([L10]). The current-head one-group K3 overlay and C2 depth decision were
measured on the same host, model, prompt suite, D24 budget, and complete-wall
boundary through `6d6fb3ed3` on 2026-09-03 ([L13], [L14]); the full C1-C8
overlay and the prefill refresh were measured at `3d48170a7` on 2026-09-03 on
the same protocol ([L15], [L16]). Rates are total
tokens divided by summed wall across **all ten
prompts**. The M3/M4 compact headlines (15.646 and 35.618 tok/s) are
six-non-heldout arithmetic means and are not substituted into this aggregate
table.

#### Prefill-dominant throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine `b768516f2` | 147.0 | 139.8 | 169.9 | 188.9 | 211.8 | 227.7 | 237.1 | 247.3 |
| Mainline Vulkan `4e97ac86` | 111.2 | 133.6 | 127.0 | 127.0 | 137.6 | 146.8 | 148.1 | 162.3 |
| Stock HIP `9d57ce456` | 146.2 | 186.9 | 190.3 | **200.5** | **226.1** | 250.1 | 250.3 | 283.5 |
| Laurent Vulkan `c28d538df` | 149.1 | **211.9** | **192.4** | 191.9 | 222.1 | **250.4** | **252.3** | **305.8** |
| Nathan Vulkan `0eb528051` | **153.4** | 200.8 | 186.1 | 186.3 | 208.2 | 227.5 | 228.6 | 263.9 |
| `q38rocm` normal Vulkan `5d097740` | 146.9 | 186.5 | 181.3 | 184.2 | 206.1 | 229.7 | 233.0 | 273.5 |
| hipEngine current head `3d48170a7` (2026-09-03) | **201.0** | 181.9 | **207.2** | **233.1** | **258.2** | **285.1** | **291.3** | 301.8 |

#### Autoregressive complete-wall throughput

| Engine | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine `b768516f2` | 11.112 | 18.090 | **23.879** | **30.150** | **35.778** | **40.343** | **43.974** | **47.194** |
| Mainline Vulkan `4e97ac86` | 10.226 | 17.471 | 14.905 | 13.246 | 18.086 | 24.083 | 29.342 | 35.376 |
| Stock HIP `9d57ce456` | 10.635 | 17.681 | 15.266 | 13.537 | 18.271 | 23.296 | 26.544 | 30.325 |
| Laurent Vulkan `c28d538df` | 11.047 | **19.835** | 16.359 | 14.255 | 20.294 | 28.322 | 35.896 | 45.614 |
| Nathan Vulkan `0eb528051` | **11.162** | 19.662 | 16.341 | 14.321 | 20.099 | 27.575 | 34.317 | 41.511 |
| `q38rocm` normal Vulkan `5d097740` | 10.582 | 18.645 | 15.663 | 13.866 | 19.054 | 24.970 | 29.576 | 35.021 |
| hipEngine current head `3d48170a7` (2026-09-03) | **11.511** | 19.249 | **24.974** | **31.478** | **37.995** | **43.093** | **46.153** | **50.605** |

#### Explicit MTP complete-wall throughput

The first six rows are the fixed-K3 comparison. The K1 row is a
width-specific overlay from the retained successor and is not a fixed-depth
K3 result; blank cells were not re-measured under that successor protocol.
The final row combines the current-head one-group K3 overlay after the
B1/B2/B5 retentions with the active-C1 singleton-target retention at
`b58a70c82`; C1 and C3 were re-measured together at resident capacity 3
([L17]).

| Engine / route | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine K3 `b768516f2` | 15.753 | 28.441 | **30.541** | **35.474** | 27.980 | 32.807 | 33.106 | 35.423 |
| Mainline Vulkan `4e97ac86` | 21.022 | 30.840 | 27.283 | 26.955 | **32.713** | 32.307 | 38.282 | 45.458 |
| Stock HIP `9d57ce456` | 17.351 | 23.530 | 23.087 | 25.287 | 22.293 | 25.212 | **46.084** | **56.222** |
| Laurent Vulkan `c28d538df` | **21.126** | **32.221** | 28.067 | 26.184 | 31.737 | **37.154** | 43.888 | 50.837 |
| Nathan Vulkan `0eb528051` | 20.781 | 30.566 | 27.859 | 26.385 | 29.768 | 33.318 | 36.992 | 45.173 |
| `q38rocm` normal Vulkan `5d097740` | 20.357 | 27.163 | 26.178 | 26.482 | 32.297 | 31.613 | 38.314 | 45.342 |
| hipEngine K1 runtime `1f4687cab` | — | — | — | — | — | 37.074 | — | 43.421 |
| hipEngine K3 one-group, retained through `b58a70c82` (2026-09-03) | 19.428 | 30.094 | **32.919** | **37.985** | **37.280** | **41.048** | 44.492 | 50.893 |

At C6, the latest hipEngine K1 row ranks second of six at 37.074 tok/s, only
0.080 tok/s (0.22%) behind Laurent's 37.154. At C8, hipEngine improves 22.6%
over its reviewed K3 row but remains sixth at 43.421 tok/s: 3.88% behind the
nearest external route (Nathan, 45.173) and 22.77% behind stock HIP's 56.222.
This overlay compares each route's measured throughput, not equal draft depth.
The current-head one-group overlay (2026-09-03, measured after the B1
verifier-owner transfer and B5 planar-Q6 integer MMQ retentions on the same
host, suite, D24 budget, and complete-wall boundary) supersedes the K1 row as
hipEngine's best explicit route: it leads the pinned matrix at C5 (37.280
versus Mainline Vulkan 32.713, +14.0%) and C6 (41.048 versus Laurent 37.154,
+10.5%), ranks second at C7 (44.492, 3.45% behind stock HIP) and C8 (50.893,
9.48% behind stock HIP and 0.11% ahead of Laurent), and reaches 1.0057x own
AR at C8. C3/C4 one-group cells measure 32.919/37.985 tok/s (1.342x/1.207x
own AR) with 10/10 AR equality and acceptance unchanged from the pinned head;
C2 measures 30.094 (1.564x). C1 now measures 19.428 tok/s (1.687x own AR) at
resident capacity 3, with acceptance 0.7889 and 10/10 AR equality. This is a
127.47% MTP improvement over the same-capacity broken row. The fix preserves
the physical provider and dispatches only an active singleton through the
existing transactional C1 target; C2+ remains packed. C3's same-run packed
collateral is +0.15%, with acceptance and equality unchanged. The one-group
packed-verifier execution differs mechanically from the pinned subgroup route;
protocol, prompts, and boundary are identical.

All llama.cpp outputs passed the character-window and word-trigram repetition
guards. hipEngine passed 80/80 explicit-MTP generated-ID, route, and budget
cells in the pinned matrix, and the current-head refresh passes mtp_self_exact
with 10/10 engaged/budget cells at every re-measured width. The pinned K3 route
reached 1.418x/1.572x/1.279x/1.177x own AR at C1-C4 and 0.782x/0.813x/0.753x/
0.751x at C5-C8; the current-head overlay reaches 0.743x*/1.564x/1.314x/
1.207x/0.981x/0.953x/0.964x/1.006x at C1-C8. hipEngine's AR row leads the AR
matrix at C1 and C3-C8 at the current head. The
MTP rows are explicit K3 diagnostics, not automatic-admission claims;
automatic production C2-C8 remains K0.

## 1. Test method

### Comparison framework

Every route is evaluated in two separate tracks where its model support allows
it:

1. **Claim reproduction:** run the source's model, engine, and protocol. This
   track tests whether the source's claim reproduces; it does not rank engines.
2. **Standardized `mtp-bench` comparison:** run the standard Qwen3.8-27B
   `Q4_K_M` artifact on our shared ten-prompt suite. This track ranks compatible
   engines using the same model file, prompts, output length, host, and timing
   boundary, with stock llama.cpp and hipEngine as controls.

The standardized comparison model is
`/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
Custom-format routes such as `ROCmFP4_FAST`, and quant-specific Q5/Q6/Q8
recipes, cannot participate in that 1:1 table unless the implementation also
supports this standard file. Their exact-artifact results remain useful for
verifying the source claim.

### Host and common suite

| Item | Value |
| --- | --- |
| APU | AMD Ryzen AI MAX+ 395 |
| GPU | Radeon 8060S / `gfx1151` |
| Unified memory | 128 GB |
| Theoretical memory bandwidth | 256 GB/s |
| Kernel | Linux 7.1.6-1-cachyos |
| Common suite | `benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| Suite SHA-256 | `fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a` |
| Prompt coverage | 10 prompts: code, general English, general Japanese, and mixed Japanese/English; four heldouts |
| Common sampling | Greedy; prompt cache disabled |

The [reviewed matrix][L9] records the full C1-C8 hipEngine baseline, raw hashes,
commands, rates, and C=N follow-up. The [C6/C8 K1 closeout][L10] records the
latest retained runtime, ten-iteration ledger, clean endpoint, and named
blocker. The preserved [external matrix][L6] records the external commits and
shared protocol. The [source-reproduction artifact][L0] records the additional
model sizes, commands, acceptance, and route decisions.
Raw logs stay outside Git because the repository does not
retain model files, binaries, or raw server logs.

### Timing terms

- **Decode tok/s:** generated tokens divided by server-reported decode time.
- **Arithmetic decode tok/s:** the simple mean of per-request decode rates.
- **Token-weighted decode tok/s:** total generated tokens divided by total
  decode time. This prevents short, fast responses from dominating the result.
- **Complete-wall tok/s:** generated tokens divided by request wall time. It
  includes prompt evaluation and request overhead, but not model loading.
- **Prefill-dominant tok/s:** prompt tokens divided by complete wall for a
  one-output-token request. It includes one generated token and API overhead.

Compare two rates directly only when the model, backend, workload, output
length, and timing boundary match.

### Correctness checks

The campaign used the checks available for each route:

- exact task contracts, such as a complete 12-object JSON array or `1…30`;
- retained output with character-window and word-trigram repetition checks;
- fresh-server controls when request-state leakage was suspected;
- token equality between Nathan and latest mainline;
- category and heldout coverage on the common suite.

These checks establish generation validity. They do not establish equal model
quality across quantizations.

## 2. Standard comparison model

Every engine in the standardized tables used the same 17,106,775,008-byte
`Q4_K_M` file, SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
The [compact artifact][L0] retains hashes for the additional models used only
to reproduce source-specific claims.

## 3. hipEngine

**Verdict: Yes.** hipEngine runs the standard `Q4_K_M` baseline used by every
row in the standardized comparison.

The reviewed `b768516f2` baseline ran C1-C8 with production-profile BF16
arithmetic. It led AR at C3-C8, reaching 23.879-47.194 complete-wall tok/s. It
was 0.45% behind Nathan at C1 (11.112 versus 11.162) and 8.80% behind Laurent
at C2. The latest C6/C8 K1 pair measures matched AR at 39.908/47.240 tok/s.
The current retained rows ([L15], [L17]) improve AR at every width — C1
11.518 now leads Nathan's frozen 11.162, and C3-C8 reach
24.526/31.478/37.995/43.093/46.153/50.605 — leaving C2 (19.249, -3.0% versus
Laurent) as the only AR deficit.

Prefill reached 139.8-247.3 tok/s across C2-C8 (147.0 at C1) in the pinned
baseline and trailed the best engine at every width. The retained B2 input-F16
prefill route (2026-09-02, [L12]) raised matched same-host one-output
complete-wall prefill to **209.391/334.704 prompt tok/s at C2/C8**, 6.38%/
12.57% above Laurent's frozen matched rows. The standardized prefill-dominant
refresh ([L16]) measures **201.0/181.9/207.2/233.1/258.2/285.1/291.3/301.8
tok/s at C1-C8**, leading the frozen external matrix at C1 and C3-C7 and
trailing only at C2 (-14.2%) and C8 (-1.3%) versus Laurent.

Explicit K3 MTP is strong at C1-C4 and weak at C5-C8. It led C3-C4 at
30.541/35.474 tok/s and beat matched AR at every width C1-C4. The retained K1
successor raises C6 to 37.074 tok/s and C8 to 43.421, reducing their gaps to the
external leaders to 0.22% and 22.77%. All 80 reviewed K3 cells passed; the
latest K1 control/candidate pair passed all 40 cells with acceptance unchanged
at 1,540/1,610. The 2026-09-02/03 B1 verifier-owner transfer and B5 planar-Q6
integer MMQ retentions ([L11], [L14]) then lifted the one-group K3 route to
**37.280/41.048/44.492/50.893 tok/s at C5-C8** (C8 = 1.0057x own AR),
leading the pinned external matrix at C3-C6 and ranking second at C7-C8
(see the overlay row above); C2-C4 measure 30.094/32.919/37.985 tok/s
(1.564x/1.342x/1.207x). Production explicit C1 now measures 19.428 tok/s
(1.687x) at resident capacity 3 through the retained singleton target. The
C3/K3 production numerical gate previously passed 240/240
canonical and 192/192 heldout top-1 checks, with maximum KL 8.69e-4 and 8.45e-4
respectively ([L8]). Width-4 prompt streaming changed its acceptance
trajectory, so that T3 scope remains an explicit diagnostic rather than an
automatic production promotion.

Automatic serving remains narrower than the diagnostic table. Strict/BF16
C1/K3/context1-67 remains automatic at **18.191 versus 11.062 AR tok/s**
([L7]). Production C2-C8 remain automatic K0 pending their width-specific
numerical/task/serving gates; the explicit rows above do not change that policy.

### C=N MTP execution review (2026-08-31)

hipEngine does use the Qwen NextN draft module, but “NextN” does not mean one
model call predicts all K tokens. Draft depth is autoregressive: K draft steps
run serially, while each step batches every request in the physical group. For
C1-C4, verification is already flattened: one call to
`verify_target_blocks_batch()` packs each request's root plus K candidates and
runs one target-model forward. The production server caps explicit MTP groups
at four, so C5-C8 still execute serial complete subgroups.

The full-width verifier mechanism is implemented and has passed one-target-pass
correctness through C8; K3 remains slower because R20-R32 and wide proposal /
accept paths miss their best owners. The initial one-pass K1 screen identified
C6/R12 and C8/R16 as the only positive wide frontiers ([L9]). The retained
successor then removed redundant verifier-state imports and added exact Q4
owners for C6/R12 plus two C8/R16 shapes, reaching **37.074/43.421 tok/s** with
all 40 control/candidate cells passing ([L10]). C6 is now 0.22% behind the
external leader, but both widths still lose own AR and remain automatic K0.

The bounded loop initially closed on a multi-family packed-verifier dataflow
blocker; that reopen condition was then satisfied by the B1 owner transfer
plus B5 planar-Q6 integer MMQ (2026-09-02/03), which lift one-group K3 to
**37.280/41.048/44.492/50.893 tok/s at C5-C8** with C8 at 1.0057x own AR
([L11], [L14]). Remaining gaps are the 3.45%/9.48% C7/C8 deficit to stock
HIP and the C1-C2 MTP deficit to the llama.cpp forks.

## 4. `q38rocm` / ROCmFPX

**Verdict: Yes as a specialized C1 route.** Strict MTP reproduced the source
claim, but it requires a custom model and exactly one server slot.

### What we tested

- `q38rocm` source commit `5d097740` ([S3])
- Verified `q38rocm` v1.5.2 prebuilt runtime
- ROCmFPX source lineage `0fc9568e`
- Vulkan/RADV on Radeon 8060S
- Exact `ROCmFP4_FAST` target
- Built-in MTP, strict maximum depth 4 at C1
- Normal MTP K3 with the standard `Q4_K_M` at C1-C8

The repository installer contained a stale checksum. We verified the v1.5.2
binary against the GitHub release digest instead of disabling checksum
validation.

### Source-protocol results

| Metric | Published | Local |
| --- | ---: | ---: |
| AR decode tok/s | 14.02 | 14.31 |
| MTP decode tok/s | 30.56-36.04 | **38.85 mean** |
| MTP acceptance | — | 78.1% |
| Repetition guard | Reported clean | Passed |

| Prompt | Decode tok/s | Acceptance |
| --- | ---: | ---: |
| Binary search tree / code | 41.44 | 88.6% |
| Widget factory / reasoning | 38.73 | 75.7% |
| JSON entity extraction | 48.49 | 100.0% |
| Unified versus discrete memory | 26.75 | 48.0% |

### Common-suite results

| Mode | Arithmetic decode | Token-weighted decode | Complete wall | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| AR | 14.782 | 14.782 | 12.803 | — |
| Strict MTP K4 | **35.575** | **32.969** | **24.294** | 62.35% |

| Category | MTP decode tok/s |
| --- | ---: |
| Code | 44.79 |
| General English | 28.41 |
| General Japanese | 26.47 |
| Mixed Japanese/English | 33.42 |

### Concurrency limitation

Strict Qwen MTP enforces one server slot. Starting it with `-np 8` fails during
model load with:

```text
Qwen strict MTP requires a single server slot/sequence
```

The 38.85 tok/s result is therefore C1-only. It is not evidence for multi-user
throughput. Normal MTP K3 supports C1-C8, but the final standard-`Q4_K_M`
refresh measured 20.357, 27.163, 26.178, 26.482, 32.297, 31.613, 38.314, and
45.342 complete-wall tok/s. It did not lead any cell in the standardized matrix.

### What this means

- All four source-protocol repetition guards passed.
- All ten common-suite requests completed, and every category improved.
- The compact strict common-suite harness did not retain response text.
- No request failure or contamination symptom appeared at C1.
- This route uses a custom model format. Do not present its speed advantage as
  an engine-only comparison against Q4/Q5/Q6/Q8 GGUF files.

## 5. Laurent adaptive DFlash2 fork

**Verdict: No for sequential serving.** The implementation is fast in a fresh
process, but request state leaks between sequential prompts.

### What we tested

- Laurent fork commit `c28d538df`, build 10681 ([S5])
- Vulkan/RADV
- Exact `ROCmFP4_FAST` target
- Exact FP4 DFlash2 `Q4_0` sidecar
- Adaptive draft depth 3-7
- Published 300-token prose-then-JSON sequence

### Published-sequence reproduction

| Policy | Prose decode | JSON decode | JSON result |
| --- | ---: | ---: | --- |
| Bare | 14.148 | 14.128 | On-task, truncated at 300 tokens |
| Fixed K3 | 25.842 | 42.532 | On-task |
| Fixed K7 | 24.481 | 20.859 | Wrong-task prose in the JSON position |
| Adaptive K3-K7 | 25.618 | **66.838** | **Invalid: repeated prose from the previous prompt** |

The 66.838 tok/s result numerically reproduces the 65.6 claim, but it is not a
valid result. The JSON request repeated “the rhythms of the tides” from the
preceding prose request. The fork's degeneration guard still passed because it
checked output length rather than task content.

### Fresh-server controls

Restarting the server before each JSON request removed the stale prose:

| Test | Decode tok/s | Result |
| --- | ---: | --- |
| Fresh server, 300 tokens, trial 1 | 56.948 | Clean JSON, truncated at object 9 |
| Fresh server, 300 tokens, trial 2 | 56.699 | Same output hash, truncated at object 9 |
| Fresh server, 300 tokens, trial 3 | 56.991 | Same output hash, truncated at object 9 |
| Fresh server, 420 tokens, bare | 14.180 | Complete valid 12-object JSON |
| Fresh server, 420 tokens, adaptive | **56.532** | **Complete valid 12-object JSON** |

Use **56.532 tok/s** as the valid structured-output result. It is 3.99x the
matched bare route. Do not use the sequential 65.6/66.838 row.

### Fresh-process common suite

Each prompt used a new server process.

| Metric | Result |
| --- | ---: |
| Arithmetic decode | 37.752 tok/s |
| Token-weighted decode | 34.483 tok/s |
| Acceptance | 60.43% |
| Code | 51.81 tok/s |
| General English | 28.42 tok/s |
| General Japanese | 25.44 tok/s |
| Mixed Japanese/English | 31.27 tok/s |

All ten outputs were substantive and non-repetitive.

### Earlier built-in-MTP transfer test

Before the FP4+DFlash2 files were available, we tested Laurent's adaptive
controller with the older local `Q4_K_M` and its built-in MTP head ([L4]).

| Arm | Decode tok/s | Versus own AR | Acceptance |
| --- | ---: | ---: | ---: |
| Mainline AR | 11.37 | 1.000x | — |
| Mainline fixed K3 | 16.02 | 1.409x | 63.41% |
| Mainline fixed K7 | 14.83 | 1.304x | 38.43% |
| Laurent AR | 11.35 | 1.000x | — |
| Laurent fixed K3 | **18.93** | **1.668x** | 63.86% |
| Laurent fixed K7 | 14.87 | 1.310x | 38.58% |
| Laurent adaptive K3-K7 | 17.66 | 1.556x | 61.70% |

Direct conclusions from this transfer test:

- adaptive sizing recovered 18.8% over fixed K7;
- adaptive sizing was 6.7% slower than fixed K3;
- mainline `n_max=7,n_min=3` behaved like fixed K7, not Laurent adaptive;
- Laurent fixed K3 was about 18% faster than b10438 mainline, but the build
  gap prevents attributing the difference without a bisect.

Adaptive depth is therefore useful for avoiding a bad deep draft, but it is
not always the fastest policy.

### Standardized built-in-MTP result

The final standard-`Q4_K_M` matrix tested Laurent's ordinary built-in MTP K3,
not adaptive DFlash2. That route passed every repetition guard and led MTP at
C1-C2 and C6. Laurent also led prefill at C2-C3 and C6-C8 and AR at C2. This is
a broad, reusable fork result; the adaptive DFlash2 request-state failure does
not apply to this ordinary built-in-MTP path.

### What must be fixed

Laurent must reset or repair speculative state at every request boundary. The
route needs a sequential multi-prompt correctness gate before it can be used as
a reusable server. Fresh-process speed does not remove this blocker.

## 6. KyaniteLabs MTP+ngram

**Verdict: Yes, with a workload caveat.** The output is correct. The 160+ tok/s
peak measures warm replay, not novel generation.

### What we tested

- KyaniteLabs source profile `7fa3ca81` ([S4])
- llama.cpp HIP `9d57ce456`, build 10438
- Exact Unsloth `UD-Q4_K_XL`
- `HSA_ENABLE_SDMA=0`, `HSA_XNACK=1`
- 98,304 context, one slot, thinking disabled
- MTP maximum depth 12; ngram minimum 24

### Count-to-30 results

| Mode | Cold decode | Warm decode | Output |
| --- | ---: | ---: | --- |
| AR | 11.94 | 11.97 | Exact `1…30` |
| MTP K12 | 61.09 | 59.42-59.49 | Exact `1…30` |
| MTP K12 + ngram | 60.95 | **164.13-167.64** | Exact `1…30` |

MTP provides the cold speedup. Ngram provides no cold benefit. Its entire
160+ tok/s gain comes from replaying the previously generated count sequence.

### Common-suite results

| Mode | Arithmetic decode | Complete wall |
| --- | ---: | ---: |
| AR | 11.964 | 11.679 |
| MTP K12 | 24.390 | 20.450 |
| MTP K12 + ngram | **24.867** | **20.518** |

Production category rates were:

| Category | Decode tok/s |
| --- | ---: |
| Code | 35.82 |
| General English | 16.10 |
| General Japanese | 15.45 |
| Mixed Japanese/English | 21.15 |

Ngram improved arithmetic decode by 1.96% over MTP-only, but complete-wall
speed improved by only 0.33%. That difference is noise-scale for one run.

All diverse-suite outputs were substantive and non-repetitive.

### hipEngine ngram follow-up

The separate hipEngine ngram-composition route remains default-off ([L5]). In
a repetition-heavy strict C2/K3 D80 control, it improved 2.425% over MTP-only
but reached only 0.9875x true AR. D96 and D120 retained correctness or
economics blockers. Kyanite's narrow replay result does not justify enabling
ngram globally in hipEngine.

## 7. PieBru recipes and Nathanw fork

**Verdict: Yes.** The Q5/Q6/Q8 served-speed claims reproduced. Current
mainline is slightly faster in decode.

### What we tested

- PieBru recipe commit `66cfceae` ([S6])
- Nathanw fork `0eb528051a56f34567312ce63ab4e14a3fc71d89`, build 10580
- Matched mainline `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
- Vulkan/RADV
- Exact Unsloth Q5/Q6/Q8 XL targets
- Exact DFlash2 `Q8_0` sidecar
- Ten prompts, up to 128 tokens, thinking disabled

### Served-speed claims

| Quant | Published band | Nathan local | Mainline local |
| --- | ---: | ---: | ---: |
| Q5 | about 23-24 | **24.706** | **24.886** |
| Q6 | 17-21 | **20.549** | **20.343** |
| Q8 | 15-18 | **18.197** | **18.092** |

All three claims are confirmed or conservative.

### Decode results

| Quant | Engine | AR decode | DFlash decode | Acceptance |
| --- | --- | ---: | ---: | ---: |
| Q5 | Nathan | 10.695 | 30.659 | 53.19% |
| Q5 | Mainline | 10.691 | **31.119** | 53.19% |
| Q6 | Nathan | 8.778 | 26.470 | 42.92% |
| Q6 | Mainline | **8.803** | **26.867** | 42.92% |
| Q8 | Nathan | 7.275 | 23.044 | 43.94% |
| Q8 | Mainline | **7.276** | **23.374** | 43.94% |

| Quant / engine | Code | General English | General Japanese | Mixed Japanese/English |
| --- | ---: | ---: | ---: | ---: |
| Q5 Nathan | 40.11 | 25.25 | 18.08 | 29.73 |
| Q5 mainline | 40.94 | 25.79 | 18.11 | 29.82 |
| Q6 Nathan | 36.48 | 19.36 | 13.54 | 26.50 |
| Q6 mainline | 37.32 | 19.44 | 13.59 | 26.67 |
| Q8 Nathan | 31.64 | 15.33 | 13.01 | 23.59 |
| Q8 mainline | 32.37 | 15.40 | 13.06 | 23.68 |

### What this means

- Nathan and mainline produced token-exact outputs in every matched arm.
- All outputs were substantive and non-repetitive.
- Mainline was about 1.4-1.5% faster in DFlash decode.
- Nathan sometimes had faster prefill, which explains its small Q6/Q8
  complete-wall lead.
- The speedup comes from the model, sidecar, and configuration—not from a
  current Nathan decode advantage.

## 8. MikeVeerman Q8 concurrency

**Verdict: Yes, with concurrency-aware routing.** MTP is valuable at low
concurrency and harmful at C4.

### What we tested

- MikeVeerman benchmark source `cc527064` ([S2])
- Exact stock llama.cpp pin `152d337fadb93c2a099653c4072d5512c92c5bfd`
- Vulkan/RADV
- Exact Unsloth `UD-Q8_K_XL`
- 131,072 total context; four 32,768-token slots
- Greedy 256-token generations at C1-C4

The pinned build reported that `--cache-reuse 256` was unsupported for this
context and disabled it in both AR and MTP arms.

### Results

| Concurrency | Published AR | Published MTP | Published ratio | Local AR | Local MTP | Local ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 7.10 | 15.53 | 2.19x | 7.21 | **16.07** | **2.23x** |
| C2 | 13.01 | 16.63 | 1.28x | 13.35 | **17.48** | **1.31x** |
| C3 | 17.52 | 18.15 | 1.04x | 18.00 | **18.21** | **1.01x** |
| C4 | **21.75** | 16.94 | 0.78x | **21.03** | 17.58 | **0.84x** |

| Concurrency | MTP acceptance |
| ---: | ---: |
| C1 | 72.9% |
| C2 | 71.4% |
| C3 | 71.6% |
| C4 | 66.4% |

At C4, per-request throughput was 5.87 tok/s AR and 4.89 tok/s MTP.

### What this means

- MTP is a large C1 win.
- MTP is approximately neutral at C3.
- MTP loses at C4 even though acceptance remains 66.4%.
- The loss comes from saturation, not failed drafting. Batched AR uses compute
  that was otherwise available to speculative work at C1.
- Admission must account for physical concurrency. Acceptance alone is not
  enough.

## 9. Cross-route analysis

### Source-protocol results are not one leaderboard

Use the standardized `Q4_K_M` tables to rank compatible engines. The
source-protocol routes differ in:

- target format: FP4, Q4, Q5, Q6, or Q8;
- draft method: MTP, DFlash2, or ngram;
- output budget: 24, 128, 256, 300, 420, or 1,024 tokens;
- timing boundary: decode or complete wall;
- workload: code, prose, structured output, multilingual prompts, or replay.

For the one shared FP4 target, Laurent fresh-process adaptive DFlash2 was
faster than `q38rocm` strict MTP K4 on the common suite: 34.483 versus 32.969
token-weighted decode tok/s. Laurent still loses the deployment decision
because sequential requests are incorrect.

### Plain AR is memory-bound

Multiplying model-file bytes by plain-AR decode rate gives these screening
estimates:

| Target | Approximate implied bandwidth |
| --- | ---: |
| `ROCmFP4_FAST` | 208 GB/s |
| Q5 | 223 GB/s |
| Q6 | 223 GB/s |
| Q8 | 229 GB/s |

These are not hardware-counter measurements. Embeddings, metadata, and non-AR
tensors are not necessarily read for every token. The estimates show that a
claimed plain dense-AR rate above the physical 256 GB/s ceiling needs another
explanation.

### Prompt type changes speculative speed

Code and structured output generally accept more draft tokens than explanatory
prose or Japanese heldouts. Report every speculative rate with:

- prompt category;
- acceptance;
- output length;
- timing boundary.

An aggregate without those fields does not transfer to another prompt mix.

### Replay and contamination are different failures of interpretation

- **Kyanite 167.64 tok/s:** correct output on a narrow warm-replay workload.
  The number is real, but it is not novel-generation speed.
- **Laurent 66.838 tok/s:** incorrect output caused by stale request state. The
  number is not a valid result at all.

### Concurrency changes the best policy

At C1, drafting can use compute that would otherwise sit idle during weight
reads. At C4, batched AR uses that compute for real requests. This explains
Mike's 2.23x C1 gain and 0.84x C4 loss.

A scheduler must use measured physical concurrency and cycle cost. Acceptance
alone cannot decide whether to enable speculation.

### hipEngine follow-up

The reviewed matrix and C=N screen narrow the remaining work:

1. Preserve the AR path. It leads C3-C8; C1-C2 need different
   operation-complete dataflows rather than broad retuning.
2. Treat prefill as a concentrated kernel gap. C2 and C8 retain measured Q4
   prefill-owner blockers; scheduling changes were measured null.
3. Keep automatic C2-C8 on K0 until the complete profile gates pass. The
   one-group K3 route now has retained exact owners through B1/B5; the K1
   width-specific route remains a fallback diagnostic.
4. Target the remaining broad-MTP gaps directly: 3.45%/9.48% behind stock HIP
   at C7/C8 and the remaining C1-C2 deficit to the llama.cpp forks. Do not
   return to packed R4 verification or acceptance-only tuning. The retained
   active-singleton target already resolves C1 correctness and economics at
   resident capacity greater than one.

Sequential ownership, lifecycle, and contamination gates are now part of the
hipEngine correctness evidence. The external results still do not justify
copying an entire fork.

## 10. Evidence

Current reviewed matrix and successor:

- [C6/C8 K1 ten-iteration closeout][L10]
- [B1 verifier owner-transfer retention][L11]
- [B2 input-F16 prefill retention][L12]
- [B4 C2 depth screen][L13]
- [B5 planar-Q6 integer MMQ retention][L14]
- [current-head C1-C8 MTP refresh with C1 blocker][L15]
- [current-head C1-C8 prefill refresh][L16]
- [active-C1 singleton-target retention][L17]
- [full C1-C8 hipEngine refresh and C=N review][L9]
- [preserved same-host external matrix][L6]

Source-claim reproduction artifact:

- [`2026-08-28-gfx1151-qwen38-external-reproduction-survey.json`][L0]

Related hipEngine evidence uses different scopes or protocols:

- [current strict-C1 and production baseline][L7]
- [current C3/K3 production correctness gate][L8]
- [older-model fork transfer test][L4]
- [hipEngine ngram/MTP closeout][L5]

## 11. Sources

- **[S1]** hogeheer499-commits, *Qwen3.8 27B on AMD Strix Halo*, commit
  `029320fb`: [pinned guide][S1].
- **[S2]** MikeVeerman, *Qwen3.8-27B on AMD Strix Halo: what MTP speculative
  decoding gives you*, commit `cc527064`: [pinned benchmark][S2].
- **[S3]** julianmb, `q38rocm`, commit `5d097740`: [pinned report][S3].
- **[S4]** KyaniteLabs, `qwen38-27b-strix-halo`, commit `7fa3ca81`:
  [pinned report][S4].
- **[S5]** LaurentZuijdwijk adaptive DFlash2 llama.cpp fork, commit
  `c28d538df`: [pinned implementation][S5].
- **[S6]** PieBru Qwen3.8 Strix Halo evidence, commit `66cfceae`:
  [pinned repository][S6].

[L0]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-external-reproduction-survey.json
[L4]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-fork-claim-generalization.json
[L5]: ../benchmarks/results/2026-08-28-gfx1151-qwen38-ngram-mtp-composition-closeout.json
[L6]: ../benchmarks/results/2026-08-30-gfx1151-qwen38-final-six-engine-c1c8.json
[L7]: ../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json
[L8]: ../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-e5-combined-correctness.json
[L9]: ../benchmarks/results/2026-08-31-gfx1151-qwen38-reviewed-current-head-c1c8.json
[L10]: ../benchmarks/results/2026-09-01-gfx1151-qwen38-c6c8-k1-ten-iteration-closeout.json
[L11]: ../benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-full-suite.json
[L12]: ../benchmarks/results/2026-09-02-gfx1151-qwen38-b2-f16-retained.json
[L13]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-b4-c2-depth-screen.json
[L14]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-b5-planar-q6-integer-mmq-retained.json
[L15]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-current-head-mtp-c1c8-refresh.json
[L16]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-current-head-prefill-c1c8-refresh.json
[L17]: ../benchmarks/results/2026-09-03-gfx1151-qwen38-c1-singleton-target-retained.json
[S1]: https://github.com/hogeheer499-commits/strix-halo-guide/blob/029320fb/QWEN38_STRIX_HALO.md
[S2]: https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench/blob/cc52706409b0c550636ff068b06894d27079d734/README.md
[S3]: https://github.com/julianmb/q38rocm/blob/5d0977403b0dac778598b1af499bf178b46c0b35/README.md
[S4]: https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/7fa3ca810c82c38e7d5a8ef4018d1d1853cec576/README.md
[S5]: https://github.com/LaurentZuijdwijk/llama.cpp/blob/c28d538df5c02643e701a8004db84dbf1bb0ffb2/common/speculative.cpp
[S6]: https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151/tree/66cfceae5edb3dfaf049279738a6fb9cfc5638f6
