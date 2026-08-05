# hipEngine

hipEngine is a ROCm-native local LLM inference engine designed from the ground
up for AMD RDNA GPUs (starting with gfx1100, gfx1151). It pairs a small
purpose-built Python host with a complete suite of custom-tuned HIP kernels
developed through 100+ iterations of profiling and tuning.

hipEngine has lightweight dependencies with no PyTorch required for fully
supported GPUs and models.

## Core principles

- **HIP-first, not CUDA-ported.** Kernels directly target AMD hardware like 
  gfx1100/RDNA3 with wave32, vec8 FMA, and the actual cache hierarchy.
- **Torch-free runtime.** `import torch` is **not** on the hot path. The
  runtime owns a thin `hipengine.Tensor` over raw HIP/CUDA device pointers and
  drives `hipblasLt`, `hipGraph`, AOTriton, and JIT builds through `ctypes`.
  Torch appears only as an optional dlpack bridge behind the `hipengine[torch]`
  extra (~125 MiB install including the vendored AOTriton subset vs ~2 GiB with
  torch).
- **Multi-backend from day one.** Kernels live under `kernels/hip_gfx1100/`,
  `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` as
  peer trees.
- **Four-axis plugin registry.** Kernels are keyed by
  `(backend, layer, quant, variant)`. Models, quant schemes, and layers are
  plugins. No `if backend == "..."` or `if quant == "..."` branches in
  dispatch / engine / model code.
- **Fused + unfused coexist.** Every fused composite
  (`rmsnorm+rotate`, `gate_combine_residual`, …) has a numerically-equivalent
  unfused chain registered under its primitives, used as both fallback and
  correctness baseline.
- **Evidence-backed performance.** Every performance claim ships with
  model + quant + workload shape + hardware + exact command + correctness gate
  (KL ≤ 0.05, top-1 ≥ 90% vs `kernels/cpu_reference/`). See
  [`docs/BENCHMARK.md`](docs/BENCHMARK.md) and
  [`benchmarks/README.md`](benchmarks/README.md).

## Status

**v0.3.0 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are available on gfx1100 and gfx1151:
[shisa-ai/Qwen3.6-35B-A3B-PARO-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the resident GGUF path, plus native
`UD-Q3_K_M` execution with retained gfx1100 evidence. Older benchmark artifacts
may still show the historical
`Qwen3.6-35B-A3B-PARO-full4096-e5-packed` name or local MTP-BF16 assembly path;
those rows use the same packed PARO architecture and remain the evidence for the
numbers below.

- Model-aware `backend="auto"` / `quant="auto"` defaults select the registered
  PARO or GGUF route without environment-variable setup. Direct generation now
  supports exact token-id prompts, detailed outputs, logprobs, structured finish
  details, and backend execution telemetry.
- PARO and GGUF support ordinary sampling controls including top-k/min-p,
  penalties, logit bias, suppression, deterministic seeds, EOS/min-token policy,
  token stops, and multi-token stops. Covered PARO shapes use a native GPU
  sampler; unsupported shapes use an explicit host fallback.
- The OpenAI-compatible server includes capability/readiness discovery, token
  and context diagnostics, exact usage accounting, request batching, deadlines,
  cancellation, opt-in Prometheus metrics, and detailed streaming metadata.
- Local-agent support includes OpenAI-style tools, Qwen thinking controls,
  structured-output result validation, deterministic continuation handles, and
  app-local transcript sessions with fork, rollback, snapshot, and overflow
  policies.
- Qwen3.6 GGUF models with NextN tensors expose detailed MTP generation and a
  guarded explicit non-streaming server route. Dense PARO DFlash and the shared
  speculative proposal/verify/commit infrastructure are available as retained
  runtime and benchmark paths.
- The same Laguna family is supported on W7900/gfx1100 with the
  `UD-Q2_K_XL` GGUF. Its exact matrix512/attention128 prefill default combines
  role-qualified Q5/Q6 output tiling, pair16 grouped IQ gate/up/down,
  adjacent-row qrow4 SWA after a measured C256 crossover, and transient exact-F32
  Q5 expansion plus production-ordered 8x4/12x4/8x10/16x5/8x12/12x8 reduction
  on all eight roles. No dequantized weight persists. H5E reached
  **184.997/172.104/131.496 tok/s** at 512/1K/4K; H5F retained the narrow exact
  N48 micro-win. H5G's constant-80/96 tiles now publish
  **188.393/175.042/132.743 tok/s**, another **+2.192%/+2.055%/+1.329%** over
  H5F. H5H closes larger exact tiles: constant-112 loses every role and
  constant-128 reaches VGPR256 with 28–52 B scratch. The retained H5G request
  reclassifies to **2,667.034 ms / 1,720 dispatches**: Q5 **920.633 ms**, IQ
  down **560.642 ms**, attention **468.533 ms**, and Q6 **177.047 ms**. With Q5
  geometry and prior attention lanes closed, WPF-H5I's exact-Q6 F32 expansion
  plus ordered consumer clears the all-role leaf and production gates. Four roles
  select exact `16x5`/`16x4`/`8x4`; both long-K roles and the wide-N F32 role
  retain raw coltile. Q5+Q6 reuse one **150,994,944-byte** plane with no new
  allocation. Integrated tracing records **143+143** candidate launches and
  three exact fallbacks, moving Q6 **177.047 -> 110.170 ms (-37.774%)** and
  request kernel sum **2,667.034 -> 2,600.260 ms (-2.504%)**. Clean
  selector-unset production is **191.713/178.080/134.411 tok/s** at 512/1K/4K.
  H5J then promotes exact resident-segment IQ3 plus a local32 launch of the
  retained physical IQ4 body for K1024/N3072 selected down. Complete logits,
  all 48 hidden boundaries, routing prefixes, active K/V, and every
  `KVLiveSpans` field are bit-exact at KL 0; repeat and teardown match. Cached
  integrated tracing observes all **45 IQ3 + 2 IQ4** production calls, moving
  selected down **556.749 -> 497.145 ms (-10.706%)** and complete request kernel
  sum **2,600.260 -> 2,532.020 ms (-2.624%)** at unchanged **1,862** dispatches.
  Clean selector-unset production reaches **196.103/181.859/137.169 tok/s** at
  512/1K/4K, **+2.290%/+2.122%/+2.052%** over H5I and a **3.540x** matched M512
  gap. Every sample is byte-exact, deterministic, and lifecycle-clean; no
  allocation, workspace, or sidecar is added. Map/shape/registration misses and
  gfx1151 retain their preceding exact routes. H5K closes larger resident IQ3
  row ownership: rowbatch12 loses all 45 actual layers by **+6.893%/+5.771%**
  event/wall, and rowbatch16 worsens to **+10.770%/+9.870%** despite exact bytes
  and zero scratch. All temporary surfaces are removed and H5J is unchanged.
  Post-H5K attribution assigns **919.697 ms** to exact Q5, including **904.399
  ms** of ordered consumers; two roles own **741.721 ms (82.0%)**. H5L promotes
  separately registered exact weight-tile-major traversal on six material
  roles while F32 N48/N72 retain H5G. Complete M512 state is KL0/byte-exact
  across all **48** boundaries, logits, K/V/live spans, repeat, workspace, and
  lifecycle. Cached tracing observes **235** producers, **188** candidates, and
  **47** fallbacks, cutting Q5 **919.697 -> 466.986 ms (-49.224%)** and request
  kernel sum **2,532.020 -> 2,074.261 ms (-18.079%)** at unchanged **1,862**
  dispatches. H5L package-default 512/1K/4K reaches
  **237.956/217.888/157.366 tok/s (+21.342%/+19.812%/+14.725% over H5J)**.
  Post-H5L tracing ranks matched residuals attention **437.720 ms**, Q5
  **408.035 ms**, and IQ down **338.619 ms**; exact SWA qrow4 owns **268.720 ms
  / 58.49%** of attention. H5M's separately registered source-qualified qrow4
  keeps every admitted two-pass operation while skipping unused current/cache
  K/V loads. Dense starts 0/128/256/384 plus 508..515 wrap/eviction/ragged cases
  are bit-exact, and production starts 256/384 improve event/wall sums
  **6.728/6.737 -> 6.437/6.443 ms (-4.324%/-4.354%)**. Complete M512 state is
  KL0/byte-exact across all 48 boundaries, logits, K/V/live spans, repeat, and
  teardown. Cached tracing observes exactly **48 global + 72 wave32 + 72 H5M**
  calls; qrow4 falls **268.720 -> 260.500 ms (-3.059%)**, attention **459.445 ->
  450.790 ms (-1.884%)**, and request kernel sum **2,074.261 -> 2,060.485 ms
  (-0.664%)** at unchanged **1,862** dispatches. Clean package-default
  512/1K/4K promotes **238.565/218.182/158.138 tok/s
  (+0.256%/+0.135%/+0.490% over H5L)**, narrowing the matched M512 gap
  **2.91728x -> 2.90983x** with no allocation or sidecar. The production-identical
  post-H5M trace reconciles **2,060.485 ms / 1,862 dispatches** and ranks matched
  gaps attention **429.065 ms**, Q5 **406.709**, and IQ down **336.162**. Exact
  source-qualified qrow4 still owns **260.500 ms / 57.79%** of attention at starts
  256/384. H5N's separately registered exact dense-first-fill leaf derives
  identity-ring visibility without token-position/eviction reads while retaining
  cached base-offset mapping and every H5M two-pass operation. It is byte-exact to
  H5M/wave32 and improves the two slices **6.653/6.660 -> 5.744/5.762 ms
  (1.158x/1.156x event/wall)**, with both starts positive and unchanged
  local32/VGPR72/LDS0/scratch0 resources. Complete M512 state is KL0 and
  integrated qrow4/attention/request sum falls **13.918%/8.087%/1.687%**, but
  runtime ownership is rejected: clean 4K is **-0.217%**, and a seven-repeat
  adjudication confirms **7/7** H5N samples below H5M (**-0.202%** median). The
  temporary policy is removed; only the leaf remains. H5O then tests Q5's
  **465.660-ms** family/**406.709-ms** matched gap with a 320-byte factorized
  block versus 1,024 F32 bytes. Reconstructed weights and all eight role outputs
  are bit-exact and scratch-free, but every role regresses: producer-inclusive
  weighted event/wall moves **477.022/473.054 -> 606.780/614.512 ms
  (+27.202%/+29.903%)**. All H5O surfaces are removed; H5L remains production.
  H5P cross-screens H5F's exact 64-accumulator geometries under H5L's later
  weight-major traversal. Four of five roles lose at least one clock and are
  removed. The sole final-source winner is BF16 K6144/N3072 `16x4`: resources
  fall **VGPR168/LDS1536 -> VGPR136/LDS1024** at scratch0, and its 12-call
  producer-inclusive event/wall sums fall **31.306/30.890 -> 29.329/29.898 ms
  (-6.315%/-3.211%)** with byte-exact output. Complete M512 state is KL0 and
  byte-exact; tracing selects exactly **12** calls and cuts Q5/request sum
  **0.572%/0.187%**. The first clean 512 row is **-0.189%**, but its predeclared
  seven-repeat adjudication is **+0.176%**. Source-default 512/1K/4K then reaches
  **+0.093%/-0.019%/-0.054%**; the final frozen 1K/4K adjudication remains
  **-0.030%/+0.014%**, rejecting runtime ownership. The eager owner and package
  change are removed; production remains H5M/H5L and only the exact leaf stays.
  H5Q addresses the third-largest matched gap, IQ3/IQ4 down at **491.658 ms**
  versus llama.cpp HIP **155.495 ms**. Of eight exact active-expert persistent
  partitions, only P64/P128 win all **45/45** IQ3 layers on both clocks; the
  frozen robust rule retains P64. Final-source event/wall sums fall
  **492.847/491.518 -> 481.081/483.823 ms (-2.387%/-1.565%)**, every output byte
  matches H5J, and cached tracing is local128/VGPR48/LDS512/scratch0. Complete
  M512 state is KL0/byte-exact; integrated tracing selects all **45** IQ3 calls
  and cuts IQ-down/request sum **3.255%/0.491%**. Default-off clean 512/1K/4K
  improves **+0.702%/+0.278%/+0.370% (3/3 paired wins each)**; selector-unset
  source-default publication confirms **+0.663%/+0.355%/+0.267%**, again 3/3
  paired wins each. H5Q becomes the preceding production at
  **239.981/219.494/158.693 tok/s**, with a **2.89266x** matched M512 gap.
  The promoted request now reconciles **2,050.376 ms / 1,862 dispatches**
  versus llama.cpp HIP's matched **724.299 ms**. Remaining exact gaps rank
  attention **431.450 ms**, Q5 **409.559 ms**, IQ down **320.157 ms**, and
  gate/up **59.253 ms**. WPF-H5R screens exact cached-only two-pass attention
  behind the existing safe append-before-attend schedule. The global
  reconstruction loses every start at **0.636–0.926x** on both clocks and is
  removed. The retained SWA-only leaf is byte-exact at starts 0/128/256/384,
  remains local32/VGPR64/LDS0/scratch0, and includes the unchanged append cost
  while moving the actual 144-call event/wall sums **337.277/334.031 ->
  126.687/125.764 ms (2.662x/2.656x)**. Complete M512 state is KL0/byte-exact;
  corrected one-queue tracing records all **144** append-before-H5R pairs and
  cuts the SWA schedule/request sum **63.767%/9.690%** at unchanged **1,862**
  dispatches. Selector-unset one-queue 512/1K/4K improves
  **+11.340%/+4.848%/+0.746% (3/3 paired wins each)** and promotes H5R at
  **267.205/230.441/160.221 tok/s**. It adds no allocation, workspace, or
  sidecar; the matched llama.cpp HIP M512 gap narrows to **2.59795x**. Earlier
  uncapped H5R speed rows are diagnostic and superseded by this packet. The
  production-identical post-H5R trace is **1,851.695 ms / 1,862 dispatches** in
  a **1,877.998-ms** span; exact matched gaps rerank to Q5 **423.388 ms**, IQ
  down **332.278 ms**, attention **195.796 ms**, Q6 **106.386 ms**, and gate/up
  **65.602 ms**. WPF-H5S screens exact persistent row-group Q5 partitions
  **1/2/4/8/16/32**. All outputs are byte-exact and all 36 symbols are
  scratch-free, but **0/6** actual roles win both clocks; even P32 regresses
  producer-inclusive weighted event/wall **459.018/473.034 -> 565.864/566.290
  ms (+23.277%/+19.714%)**. All H5S surfaces are removed and H5L remains Q5
  production. WPF-H5T carries H5Q's four independent K256 partitions in one
  exact local32/VGPR96/LDS0/scratch0 wave. All 45 actual layers remain
  byte-exact and wall falls **485.298 -> 469.677 ms (-3.219%)**, but HIP-event
  sum regresses **474.107 -> 475.945 ms (+0.388%)** and only **12/45** layers
  win both clocks. All H5T surfaces are removed; H5Q remains IQ3 production.
  WPF-H5U's exact global cached-source leaf passes all-start byte/CPU, lifecycle,
  gfx1151 fail-closed, and cached resource gates at local256/VGPR40/SGPR128/
  dynamic-LDS16928/scratch0. Default-off M512/C4096 is KL0, physical tracing
  records **48 H5U + 144 H5R** pairs at unchanged **1,862** dispatches, and
  matched throughput improves **268.331 -> 270.610 tok/s (+0.849%, 5/5 wins)**.
  Runtime ownership is rejected: the final balanced source-default 1K gate is
  **230.181 -> 230.175 tok/s (-0.00257%, 2/8 wins)**. All temporary runtime
  plumbing is removed, the standalone leaf remains, and production stays
  **267.205/230.441/160.221 tok/s**. H5V then tests Q5's distinct exact
  local32 sequential K-partition schedule. All six H5L roles remain byte-exact;
  cached symbols are local32/SGPR128/scratch0 with unchanged LDS and +8 VGPR.
  But **0/6** roles wins both clocks, and producer-inclusive weighted event/wall
  regresses **464.968/466.267 -> 492.423/493.754 ms
  (+5.905%/+5.895%)**. All H5V code/tests are removed and H5L/H5G remain.
  H5W admits exactly three Q6 weight-major composites over already-exported
  H5L/H5P physical primitives, covering **142/143** H5I-selected calls with no
  new device body/symbol. Rows17/33 and actual M512 bytes are exact; cached
  tracing records the Q6 producer immediately before local128/VGPR136-168/
  scratch0 consumers. Final-source producer-inclusive weighted event/wall falls
  **87.859/81.559 -> 70.756/67.795 ms (-19.466%/-16.876%)**, with all three
  roles winning both clocks. Default-off runtime qualification is KL0/byte-exact
  across all 48 boundaries and complete state. Cached integration records exact
  **142 H5W + one H5I + three raw** consumers at unchanged **1,862** request /
  **289** Q6 dispatches and cuts Q6/request sum **121.306/1,851.695 ->
  92.636/1,803.036 ms (-23.635%/-2.628%)**. One-queue clean 512/1K/4K is
  **266.814/230.134/159.970 -> 271.697/233.568/161.668 tok/s
  (+1.830%/+1.492%/+1.061%)**, 3/3 wins each. Selector-unset publication confirms
  **266.763/230.491/160.091 -> 271.526/234.020/161.853 tok/s
  (+1.785%/+1.532%/+1.100%)**, again 3/3 each, promoting canonical throughput
  **+1.617%/+1.553%/+1.018%** over H5R and narrowing matched M512 to
  **2.55661x**. H5X then admits an exact standalone Q5 `[tile][k][col]` leaf:
  all six actual outputs are byte-exact, physical consumers replace **8/12/16
  scalar loads with 2/3/4 `global_load_b128`**, and four roles / **151 calls**
  win both clocks. Remove two losing surfaces and retain H5L for **37** calls.
  Six-role selected event/wall falls **1.556%/1.668%** and the four final-source
  winners fall **2.683%/3.009%**, 4/4 wins. Default-off complete state is KL0
  and byte-exact across all 48 boundaries. Four paired cached request segments
  record exact **151 H5X + 37 H5L + 47 H5G** ownership at unchanged
  **1,862/470** request/Q5 dispatches and move median Q5/request/span
  **479.776/1,826.542/1,850.682 -> 470.606/1,814.537/1,834.282 ms
  (-1.911%/-0.657%/-0.886%)**. Clean 512/1K/4K improves
  **271.744/233.742/161.579 -> 272.936/234.834/162.416 tok/s
  (+0.439%/+0.468%/+0.518%)**, 3/3 wins each. Selector-unset publication then
  confirms **271.922/234.334/162.004 -> 273.366/235.061/162.533 tok/s
  (+0.531%/+0.310%/+0.327%)**, again 3/3 each. H5X is production at canonical
  **273.366/235.061/162.533 tok/s (+0.678%/+0.445%/+0.421% over H5W)** and
  narrows the canonical-dashboard M512 gap to **2.53940x**. The corrected
  apples-to-apples C4096/direct-M512 rerun reaches **278.062 tok/s** across five
  exact token-2930/lifecycle-clean samples, **+64.03%** over the campaign-start
  **169.516 tok/s** and **2.49651x** behind llama.cpp HIP. Its five-request
  cached trace reconciles **1,831.568 ms / 1,862 dispatches** versus llama.cpp
  **724.299 ms**; residual gaps rank Q5/IQ-down/attention/Q6/gate-up at
  **407.137/326.998/234.055/77.436/59.236 ms**. WPF-H5Y now admits exact
  tile-K-row BF16 activation leaves for all six roles over unchanged H5X/H5L
  weights. Rows17/33/512 planes and outputs are byte-exact; cached ISA has the
  intended width-matched loads with identical consumer resources and scratch0.
  The **188-call** pack-inclusive event/wall aggregate falls
  **462.608/455.971 -> 263.014/274.237 ms (-43.145%/-39.856%)**, with 6/6
  both-clock wins. Default-off complete M512 state is KL0/byte-exact; paired
  tracing adds exactly 188 packs and cuts Q5/request/span
  **47.204%/9.685%/9.770%**. Default-off 512/1K/4K improves
  **272.917/234.864/162.367 -> 302.770/256.121/171.978 tok/s
  (+10.939%/+9.051%/+5.920%)**, 3/3 wins each. Selector-unset publication
  confirms **273.439/235.058/162.365 -> 303.140/256.139/171.830 tok/s
  (+10.862%/+8.969%/+5.829%)**, again 3/3 each. H5Y is production at canonical
  **303.140/256.139/171.830 tok/s (+10.892%/+8.967%/+5.720% over H5X)**.
  The binding C4096/direct-M512 row is **306.305 tok/s** from five exact
  token-2930/lifecycle-clean samples, **+80.69%** over campaign start and
  **2.26632x** behind llama.cpp HIP. Its cached trace reconciles
  **1,658.386 ms / 2,050 dispatches** versus llama.cpp **724.299 ms**; gaps now
  rank IQ-down/attention/Q5/gate-up/Q6 at
  **339.558/239.624/188.153/82.382/79.327 ms**. IQ3 alone owns
  **486.381 ms / 45 calls**. WPF-H5Z now admits a distinct exact
  activation-resident output-column P256 leaf over H5Q's local128/four-wave
  rowbatch8 arithmetic. P256/P512 alone win all **45/45** actual layers; the
  frozen max-min rule keeps P256. Final-source event/wall moves
  **478.606/486.167 -> 459.818/451.737 ms (-3.926%/-7.082%)** with byte-exact
  outputs, local128/VGPR112/LDS512/scratch0 resources, token 2930, and lifecycle
  recovery. Its bounded default-off owner reuses H5Q's active-expert ABI with no
  allocation/workspace change. Natural M512 is KL0 and byte-exact across all
  state/repeat; four paired cached requests select exact **45 H5Q or 45 H5Z**
  IQ3 calls at unchanged **2,050** dispatches and move IQ3/request/span
  **488.610/1,625.126/1,650.283 -> 477.168/1,603.812/1,624.882 ms
  (-2.342%/-1.312%/-1.539%)**. Fresh-session 512/1K/4K improves
  **302.425/256.139/171.930 -> 307.870/259.556/173.477 tok/s
  (+1.801%/+1.334%/+0.900%)**, 3/3 wins each. Selector-unset publication confirms
  **302.160/256.226/172.061 -> 307.658/259.947/173.562 tok/s
  (+1.819%/+1.452%/+0.872%)**, again 3/3 each. Promote H5Y/H5Z production at
  canonical **307.658/259.947/173.562 tok/s
  (+1.490%/+1.486%/+1.008% over H5Y/H5Q)**, narrowing the canonical M512 gap
  to llama.cpp HIP **694.184 tok/s** from **2.28998x to 2.25635x**. The binding
  H5Z C4096/direct-M512 reprofile now reaches **311.622 tok/s** from
  **312.394/312.340/311.622/311.229/311.317**, all exact token 2930 with clean
  teardown. This is **+83.83%** over campaign start, **+1.736%** over matched
  H5Y, and **2.22765x** behind llama.cpp HIP. Five production traces reconcile
  **1,628.336 ms / 2,050 dispatches** in a **1,651.364-ms** median span; current
  gaps rank IQ-down/attention/Q5/gate-up/Q6 at
  **325.570/235.310/182.882/78.514/77.504 ms**. IQ3 remains first at
  **472.416 ms / 45 calls**, but all immediate exact IQ ownership/geometries are
  already screened. **WPF-H6A exact dense-initial cached-only attention metadata
  elision** now qualifies a bounded default-off owner for both gfx1100 leaves.
  Natural M512 is KL0 and byte-exact across all 48 boundaries, complete logits/
  KV/`KVLiveSpans`, and repeat at unchanged **161,120,256-byte** workspace. Four
  paired cached requests preserve **2,050** dispatches and select exact **48 H6A
  global + 144 H6A SWA** calls, moving attention schedule/request-sum/span
  **254.976/1,627.696/1,653.806 -> 170.086/1,560.817/1,581.621 ms
  (-33.294%/-4.109%/-4.365%)** with unchanged resources and scratch0. Clean
  default-off 512/1K/4K improves **307.071/259.710/173.388 ->
  312.331/261.467/173.954 tok/s (+1.713%/+0.677%/+0.326%)**, 3/3 wins each.
  Selector-unset publication confirms **307.158/260.161/173.375 ->
  312.781/261.591/173.997 tok/s (+1.831%/+0.550%/+0.359%)**, again 3/3.
  Promote H6A source production at canonical **312.781/261.591/173.997 tok/s
  (+1.665%/+0.633%/+0.251% over H5R/H5Y/H5Z)**. The binding post-H6A
  C4096/direct-M512 row is now **326.174 tok/s** from five exact token-2930,
  lifecycle-clean samples: **+92.414%** over campaign start and **+4.670%** over
  matched H5Z. A provenance audit found the prior llama.cpp row bound only its
  launcher, not the plain implementation library, so it is superseded as
  synthetic evidence. A clean c0bc8591 patched rebuild with both hashes and
  **5/5 top-1 2930** markers measures exact natural/C4096/BF16 llama.cpp HIP at
  **696.342 tok/s**; separate synthetic pp512 is **711.410 tok/s**, consistent
  with the user's **714.07**. The post-H6A matched gap is **2.13488x** and
  kernel sum is **1,568.190 ms** versus llama.cpp **718.241 ms**; residuals rank
  IQ-down/Q5/attention/gate-up/Q6 at
  **336.609/187.223/147.249/93.203/79.112 ms**. **WPF-H6B** screens a
  genuinely new active-IQ3 signed-magnitude/scale plane. Its 16-byte records
  and all **45/45** actual-layer outputs are exact, but producer-inclusive H5Z
  -> H6B event/wall regresses **462.301/450.204 -> 575.804/587.342 ms
  (+24.552%/+30.461%)** with **0/45** both-clock wins. The consumer also
  compiles one b96 rather than the required b128 record load. Remove every H6B
  source/key/test surface, retain H6A/H5Y/H5Z production, and treat immediate
  IQ3 ownership/tile/source-MMQ/segment-plane premises as closed
  ([H6B rejection](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-signed-magnitude-segment-plane-rejected.json) ·
  [post-H6A matched residual / H6B target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6a-matched-residual.json) ·
  [H6A production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-production.json) ·
  [H6A candidate](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-candidate.json)). The post-H6B audit shows Q5 is already **87.2%** exact H5Y consumer time while producers are only **10.3%**, so H6C does not repeat Q5 producer or geometry work. **WPF-H6C exact special-IQ3 expert-major fused-SiLU rowbatch4** instead instantiates the existing RT1-compatible grouped template for the one special layer. On actual layer-47 IQ3 gate/up weights and natural M512 routing, the fair route-major+post-gather -> pre-gather+H6C path is complete-byte exact and moves event/wall **32.691/32.724 -> 15.458/15.438 ms (-52.716%/-52.825%, 2.115x/2.120x)**. Its bounded owner is KL0/byte-exact across complete M512 state and repeat at unchanged **600,141,856-byte** total scratch; four cached requests preserve **2,050** dispatches and exact **46 IQ2 + one H6C**, **45 H5Z + two H5J down**, and **48+144 H6A attention** topology. The gather-inclusive special path falls **32.127 -> 15.030 ms (-53.215%)**. Selector-unset 512/1K/4K improves **311.969/261.519/173.987 -> 316.106/263.864/174.840 tok/s (+1.326%/+0.897%/+0.490%)**, 3/3 wins each, promoting H6C at canonical **+1.063%/+0.869%/+0.484%** over H6A/H5Y/H5Z. Fixed natural-M512/C4096 improves **325.211 -> 328.863 tok/s (+1.123%, 5/5 wins)** and narrows exact llama.cpp HIP **696.342** to **2.11742x** ([H6C production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-production.json) · [H6C runtime candidate](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-runtime-candidate.json) · [H6C leaf](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-candidate.json) · [H6C target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-target.json)). A clean source-default refresh reaches **329.563 tok/s** from five exact token-2930 samples, **+94.413%** over campaign start and **2.11293x** behind llama.cpp. Its representative cached request is **1,546.351 ms / 2,050 dispatches**; IQ-down/Q5/attention/Q6/gate-up gaps are **334.482/186.766/146.896/79.035/74.403 ms**. **WPF-H6D exact row-interleaved IQ3 VOPD** is now the retained IQ3 source default: it preserves all 72 useful FMAs and load/reduction/barrier counts, forms **17** FMA/FMA VOPD pairs, cuts issue slots **72 -> 55**, metadata VGPR **107 -> 99**, and runtime VGPR **112 -> 104** at LDS512/scratch0. Complete natural-M512 is KL0/byte-exact across all 48 boundaries, logits, K/V/`KVLiveSpans`, and repeat at unchanged workspace/scratch. Four cached requests preserve **2,050** dispatches and exact **45 H5Z or 45 H6D + two H5J** topology, cutting IQ3/request/span **2.564%/0.251%/0.861%**. Selector-unset 512/1K/4K improves **315.267/264.136/175.276 -> 319.072/265.872/176.138 tok/s (+1.207%/+0.657%/+0.492%)**, 3/3 wins each. Fixed C4096/M512 improves **329.327 -> 332.308 tok/s (+0.905%, 5/5 wins)** and narrows exact llama.cpp HIP **696.342** to **2.09547x**; **92/92** guards pass and H5Z/H5Q remain rollback ([H6D production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-production.json) · [H6D candidate/runtime](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-candidate.json) · [post-H6C matched residual / H6D target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6c-matched-residual.json)). A clean promoted-H6D refresh now reaches **332.992 tok/s** from five exact token-2930/lifecycle-clean samples, **+96.436%** over campaign start and **2.09117x** behind exact llama.cpp HIP. Its representative trace reconciles **1,530.211 ms / 2,050 dispatches**; current/llama IQ-down, Q5, attention, Q6, and gate/up gaps are **320.874/186.614/146.882/78.701/72.663 ms**. **WPF-H6E exact Q6 activation-tile-K-row transfer** is now admitted as a standalone gfx1100 leaf. All three H5W roles across rows 17/33/M512 preserve complete H5W output bytes, the full H5Y activation plane, sampled independent CPU values, package maps, and the existing **161,120,256-byte** owner. On actual Q6 weights, pack+exact-producer+consumer-inclusive weighted event/wall moves **65.969/66.187 -> 58.085/58.217 ms (-11.952%/-12.042%, 1.136x/1.137x)**; every role wins both clocks. Cached ISA realizes b64 for rowbatch4 and b64+u16 for rowbatch5 while runtime resources stay **VGPR136/168, LDS1024/1536, scratch0** with matching H5W grids and no compiler under profile. A bounded default-off owner is KL0/byte-exact across all 48 boundaries, logits, K/V/`KVLiveSpans`, repeat, and teardown at unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch. Four cached requests substitute exact **142 H5W -> 142 H6E** while adding 142 existing pack launches; Q6/request-span moves **92.867/1,572.498 -> 84.000/1,563.696 ms (-9.549%/-8.802 ms)**. Clean 512/1K/4K improves **318.224/265.944/176.173 -> 320.081/266.778/176.529 tok/s (+0.584%/+0.313%/+0.202%)**, 3/3 exact wins. Selector-unset H5W rollback -> H6E source 512/1K/4K improves **318.215/266.225/176.015 -> 319.854/267.357/176.470 tok/s (+0.515%/+0.425%/+0.259%)**, 3/3 exact wins each; fixed C4096/M512 improves **332.443 -> 333.329 tok/s (+0.266%, 5/5 wins)** and is **2.08905x** behind llama.cpp HIP. H6E is now Q6 source production with H5W/H5I rollback and no new allocation/workspace/sidecar/selector. Its clean refresh reaches **334.512 tok/s / 1,519.289-ms** kernel sum, **+97.333%** over campaign start and **2.08166x** behind llama.cpp HIP; residual gaps are IQ-down/Q5/attention/gate-up/Q6 **320.074/186.357/146.489/71.686/70.012 ms**. **WPF-H6F exact IQ3 paired-output reduction amortization** is now the retained gfx1100 IQ3 source default, with H6D/H5Z/H5Q registered rollback. Physical output stride **0x100 -> 0x200** proves barriers **24 -> 12 per rowbatch**; runtime resources remain VGPR152/LDS512/scratch0 at unchanged grid. All **45/45** actual layers are exact and win both clocks, moving event/wall **445.316/436.801 -> 352.255/360.918 ms (-20.898%/-17.372%, 1.264x/1.210x)**. Complete natural-M512 is KL0/byte-exact across all 48 boundaries, logits, K/V/`KVLiveSpans`, repeat, and teardown at unchanged workspace/scratch. Four cached requests substitute exact **45 H6D -> 45 H6F**, cutting request kernel sum **1,540.306 -> 1,458.072 ms (-5.339%)**. Selector-unset 512/1K/4K improves **320.079/267.093/176.521 -> 336.830/278.753/181.563 tok/s (+5.234%/+4.365%/+2.856%)**, 3/3 exact wins each. Fixed C4096/M512 improves **333.248 -> 352.761 tok/s (+5.856%, 5/5 wins)** and narrows llama.cpp HIP **696.342** to **1.97397x** ([H6F production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-production.json) · [H6F candidate](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-candidate.json) · [post-H6E residual / H6F target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6e-matched-residual.json) · [H6E production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-k-activation-tile-k-row-production.json) · [H6E candidate/runtime](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-k-activation-tile-k-row-candidate.json) · [post-H6D target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6d-matched-residual.json)). Clean promoted H6F now measures **353.798 tok/s** from **354.182/354.022/353.553/353.798/353.034**, all exact token 2930 with clean teardown. Its representative cached request is **1,435.431 ms / 2,192 dispatches** in a **1,460.237-ms** median span versus llama.cpp HIP **696.342 tok/s / 718.241 ms**. Current/llama components are IQ down **376.170/154.434 ms**, Q5 **250.665/58.737**, attention **171.168/21.624**, gate/up **476.822/401.393**, Q6 **85.704/14.455**, and remaining **74.902/67.598**; the wall gap is **1.96819x**. **WPF-H6G exact Q5 one-step K-record prefetch is rejected** on the two dominant H5Y consumers: complete output/activation/weight-plane bytes pass, but ISA immediately waits after next-record loads with zero useful overlap, and actual-weight direct event/wall regresses **+4.443%/+4.906%** while producer/pack-inclusive regresses **+3.774%/+4.095%**. Remove every H6G surface and retain H5Y/H6F production ([H6G rejection](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-prefetch-rejected.json) · [post-H6F residual / H6G target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6f-matched-residual.json)). **WPF-H6I exact IQ3 triple-output reduction** is now the retained gfx1100 IQ3 source default. Complete M512 is KL0/byte-exact across all 48 hidden boundaries, logits, K/V/`KVLiveSpans`, and repeat at unchanged scratch. Four cached requests substitute exact **45 H6F -> 45 H6I** at unchanged **2,192 dispatches**, cutting IQ3/request-sum/span **9.559%/1.906%/2.200%**. Selector-unset 512/1K/4K gains **2.304%/1.650%/0.719%**, 3/3 exact wins each; fixed C4096/M512 improves **352.966 -> 360.154 tok/s (+2.036%, 5/5 wins)** and narrows llama.cpp HIP **696.342** to **1.93346x**. H6F/H6D/H5Z/H5Q remain rollback ([H6I production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-production.json) · [H6I candidate/runtime](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-candidate.json) · [target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-target.json) · [post-H6I residual / H6J target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6i-matched-residual.json)). Clean promoted H6I reaches **359.963 tok/s** from five exact token-2930/lifecycle-clean samples, **+112.347%** over campaign start and **1.93448x** behind llama.cpp HIP. Its representative cached request is **1,409.540 ms / 2,192 dispatches**; current/llama components are IQ down **342.209/154.434 ms**, Q5 **253.606/58.737**, attention **172.347/21.624**, gate/up **479.738/401.393**, Q6 **86.361/14.455**, and remaining **75.280/67.598**. Q5's compiler-prefetch premise is physically closed and IQ-down was just promoted, so **WPF-H6J exact dense-initial SWA qrow4 unscaled-dot replay** screens H6A SWA's **115.555 ms / 144 calls**. It is complete-byte/CPU/span exact and physically removes the duplicate QK pass at metadata VGPR54/LDS8192/private0, but rocprof reports runtime VGPR248 and every start loses both clocks. Weighted H6A -> H6J moves **95.924 -> 133.542 ms event (0.718x)** and **97.607 -> 139.600 ms wall (0.699x)**. Remove every candidate/test surface, skip runtime ownership, and retain H6A ([H6J rejection](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-swa-dot-replay-rejected.json)). **WPF-H6K exact IQ3 quadruple-output reduction** then realizes stride **0x400** and **4 -> 3 epochs / 8 -> 6 barriers** with exact bytes and no spill/scratch, but runtime VGPR rises **168 -> 200**. All **45/45** layers regress: event **329.061 -> 339.509 ms (+3.175%)** and wall **332.027 -> 337.538 ms (+1.660%)**. Remove every H6K source/key/exclusion/test surface, skip runtime, and retain H6I ([H6K rejection](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-quadruple-output-reduction-rejected.json)). **WPF-H6L exact IQ2 pair16 grouped rowbatch16** is now the retained gfx1100 IQ2 source default. Frozen boundary/CPU checks, all **46/46** actual-layer both-clock gates, and complete natural-M512 KL0/byte-exact state pass at runtime VGPR112/LDS512/local64/grid65536x256/scratch0 with unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch. Four cached requests preserve **2,192 dispatches** and substitute exact **46 rowbatch8 -> 46 H6L**, moving IQ2/request-sum/span **460.772/1,424.447/1,452.975 -> 377.540/1,351.047/1,372.593 ms (-18.064%/-5.153%/-5.532%)**. Selector-unset rowbatch8 rollback -> H6L 512/1K/4K improves **343.370/282.905/182.706 -> 362.826/295.544/188.636 tok/s (+5.666%/+4.468%/+3.246%)**, 3/3 exact wins each; fixed natural C4096/M512 improves **360.451 -> 381.893 tok/s (+5.949%, 5/5 wins)** and narrows llama.cpp HIP **696.342** to **1.82340x**. **212/212** guards pass and rowbatch8 remains same-ABI rollback ([H6L production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-production.json) · [candidate/runtime](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-candidate.json) · [target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-target.json)).

The clean promoted H6L refresh reaches **381.977 tok/s**, **+125.334%** over campaign start and **1.82299x** behind matched llama.cpp HIP **696.342**. Its exact **1,326.062-ms / 2,192-dispatch** request leaves Q5/IQ-down/attention/Q6 gaps **194.004/189.827/151.442/72.392 ms**, while gate/up is already **7.498 ms faster** than llama.cpp. **WPF-H6M exact explicit wait-split Q5 pipelining is rejected**: both roles are byte-exact and physically realize **13/4 loads -> 32 current FMAs -> one wait**, but the 70-call direct event/wall aggregate regresses **194.618/195.249 -> 205.367/205.331 ms (+5.523%/+5.164%)** and inclusive regresses **215.590/216.860 -> 227.873/227.347 (+5.697%/+4.836%)**. Remove every H6M surface and retain H5Y/H6L production ([H6M rejection](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-wait-split-rejected.json) · [post-H6L residual / H6M target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6l-matched-residual.json)).

**WPF-H6N exact global dense-initial fixed-512 score storage is now the retained
gfx1100 source default.** Complete natural M512 remains KL0/byte-exact across
logits, all **48/48** hidden boundaries, K/V/`KVLiveSpans`, repeat, and teardown
at unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch.
Four cached requests preserve **2,192 dispatches** and substitute exactly **48
H6A global -> 48 H6N** while keeping 144 H6A SWA; global/attention/kernel-sum/
span move **57.126/169.556/1,320.178/1,346.667 -> 31.969/148.140/1,305.325/
1,327.300 ms**. Fresh selector-unset fixed C4096/M512 improves H6A rollback ->
H6N source **381.772 -> 387.571 tok/s (+1.519%, 5/5 wins)**, **+128.633%** over
campaign start and **1.79668x** behind matched llama.cpp HIP **696.342**.
Selector-unset 512/1K/4K is **-0.054%/+0.199%/+0.054%**, exact/finite and
lifecycle-clean; 4K wins **3/3**. H6A global remains rollback, H6A SWA is
unchanged, gfx1151 stays excluded, and **81/81** guards pass
([H6N production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-production.json) ·
[candidate/runtime](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-candidate.json) ·
[target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-target.json)).

**WPF-H6P exact staged-wave-publication triple-output IQ3 is the retained
same-ABI rollback under H6Q source production.** Sequential A/B/C publication
preserves H6I
bytes/arithmetic while reducing runtime VGPR **168 -> 112**. Complete natural
M512 is KL0/byte-exact across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown; selector-unset 512/1K/4K improves
**363.420/295.760/188.704 -> 365.186/296.589/189.108 tok/s
(+0.486%/+0.280%/+0.214%)**, **3/3** exact wins each, and fixed paired M512 is
**387.780 -> 388.320 (+0.139%, 4/5 wins)**. The independent clean post-commit
refresh is **389.145 tok/s** from **389.145/389.310/390.187/388.384/388.354**,
all token 2930 and lifecycle-clean. Fresh matched llama.cpp HIP is **690.791
tok/s** from **694.342/692.318/690.152/690.791/690.480**, all **5/5** top-1
2930; the user-reported **714.07** is synthetic/random default `pp512`.

Campaign-start/pre-H6Q H6P/current H6Q/llama.cpp component milliseconds are Q5
**1,270.458/254.839/258.472/58.314**, IQ-down
**557.091/337.353/322.866/153.860**, attention
**488.304/148.927/151.408/21.512**, Q6 **157.073/87.346/88.160/14.668**,
gate/up **460.143/397.616/403.242/397.805**, and remaining
**68.623/76.410/77.086/67.849**. Kernel sum is
**3,001.692/1,302.492/1,301.236/714.008 ms**. The independent clean H6Q wall is
**390.947 tok/s** from **390.947/391.717/390.762/391.127/390.571**, all token
2930 and lifecycle-clean, leaving **1.76697x** to matched llama.cpp HIP.

Q5 stays numerically first but its exact mechanisms are closed. **WPF-H6Q exact
compact-shuffle-loop staged-wave IQ3 is the retained gfx1100 source default;
H6P is the explicit same-ABI rollback.** Its physical leaf reduces static
bpermutes **120 -> 24**, code **8,360 -> 6,620 bytes**, and metadata/runtime
VGPR **107/112 -> 95/96** while preserving H6P bytes/arithmetic/topology.
Complete natural M512 is KL0/byte-exact across logits, all **48/48** hidden
boundaries, K/V/`KVLiveSpans`, repeat, and teardown. Four cached requests preserve
**2,192 dispatches** and substitute exact **45 H6P -> 45 H6Q**, moving IQ3/
request-sum/span **325.508/1,341.698/1,371.705 -> 310.128/1,335.166/1,356.944
ms (-4.725%/-0.487%/-1.076%)**. Fresh selector-unset 512/1K/4K improves
**365.029/296.601/189.169 -> 367.696/298.295/189.848 tok/s
(+0.730%/+0.571%/+0.359%)**, **3/3** wins each; fixed C4096/M512 improves
**389.072 -> 390.887 (+0.467%, 5/5 wins)** and is **1.76724x** behind fresh
matched llama.cpp HIP. Workspace/scratch remain unchanged and **156/156** guards
pass.

**WPF-H6R exact DPP peer-exchange staged-wave IQ3 is now the retained gfx1100
source default; H6Q remains the explicit same-ABI rollback.** Its physical leaf
stays exact on all **45/45** actual layers and emits zero bpermutes, exact **24
permlanex16 + 96 DPP**, unchanged arithmetic/topology, and
private0/spill0/scratch0 at metadata/runtime VGPR **101/104**. Complete natural
M512 is KL0/byte-exact across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown. Four production-identical cached requests
preserve **2,192 dispatches** and replace only **45 H6Q -> 45 H6R** calls,
moving IQ3/request-sum/span **310.159/1,332.893/1,362.094 ->
267.241/1,285.199/1,307.416 ms (-13.837%/-3.578%/-4.014%)**. Fresh
selector-unset 512/1K/4K improves **367.777/299.019/190.144 ->
381.726/308.807/193.931 tok/s (+3.793%/+3.274%/+1.992%)**, with **3/3** exact
wins each. Fixed natural C4096/M512 improves **391.307 -> 407.780 tok/s
(+4.210%, 5/5 wins)** and remains **1.69403x** behind matched llama.cpp HIP
**690.791 tok/s**. Workspace/scratch remain **161,120,256/600,141,856 bytes**,
gfx1151 fails closed, and **219/219** guards pass. Clean committed H6R
reprofiling reaches **407.091 tok/s** from
**406.301/407.165/407.091/407.141/406.318**, all token 2930 and
lifecycle-clean, with **1,247.252 ms / 2,192 dispatches** versus matched
llama.cpp HIP **690.791 tok/s / 714.008 ms**. Campaign-start/current/llama.cpp
component milliseconds are Q5 **1,270.458/255.672/58.314**, attention
**488.304/149.392/21.512**, IQ-down **557.091/279.045/153.860**, Q6
**157.073/87.437/14.668**, gate/up **460.143/399.483/397.805**, and remaining
**68.623/76.224/67.849**. The next target-only leaf is **WPF-H6S exact DPP
peer-exchange dense-initial SWA qrow4**: transfer only H6R's proven
permlanex16+DPP 8/4/2/1 peer operation into H6A SWA's **117.506 ms / 144-call**
exact reduction, preserving attention ownership, arithmetic, K/V traffic, and
`KVLiveSpans` behavior
([post-H6R residual / H6S target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6r-matched-residual.json) ·
[H6R production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-production.json) ·
[H6R candidate/runtime](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-candidate.json) ·
[post-H6Q residual / target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6q-matched-residual.json) ·
[H6Q production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-compact-shuffle-loop-production.json)).

**WPF-H6S exact DPP peer-exchange dense-initial SWA qrow4 is rejected and all
candidate surfaces are removed.** The leaf is complete-byte exact, finite,
span-immutable, and lifecycle-clean at starts 0/128/256/384. ISA realizes the
intended **52 -> 12 bpermutes + 8 permlanex16 + 32 DPP**, code
**7,044 -> 6,676 bytes**, and metadata/runtime VGPR **64/64 -> 59/64** with
private0/spill0/scratch0. Nevertheless, every start loses both clocks; weighted
144-call H6A -> H6S moves event **94.696 -> 108.850 ms (+14.946%, 0.870x)** and
wall **96.707 -> 112.761 ms (+16.601%, 0.858x)**. The one-shot gate therefore
keeps H6A SWA and H6N global production unchanged and closes DPP attention peer
exchange without runtime qualification or follow-up tuning
([H6S rejection](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-dpp-peer-rejected.json)).

**WPF-H6T exact fused-DPP-add staged-wave IQ3 is now the retained gfx1100 IQ3
source default; H6R remains explicit same-ABI rollback.** The exact leaf passes
**9/9** and **45/45** actual layers on both clocks while converting **72 DPP adds
+ 24 row-shift-1 moves -> 96 DPP adds + zero moves**, cutting slots/code **1,399
-> 1,384 / 8,016 -> 7,920 bytes** at unchanged metadata/runtime VGPR **101/104**
and scratch0. Complete natural M512 is KL0 and byte-exact across all **48/48**
hidden boundaries, K/V/`KVLiveSpans`, repeat, and teardown. Four cached requests
preserve **2,192** dispatches and substitute exact **45 H6R -> 45 H6T**;
IQ3/request-sum/span move **267.433/1,284.605/1,313.165 ->
261.844/1,283.120/1,304.737 ms (-2.090%/-0.116%/-0.642%)**. Fresh selector-
unset fixed C4096/M512 improves H6R rollback -> H6T source **407.600 -> 408.900
tok/s (+0.319%, 5/5 wins)** and is **1.68939x** behind matched llama.cpp HIP
**690.791 tok/s**. Fresh 512/1K/4K publication improves **381.821/307.478/193.289
-> 383.162/308.780/193.629 tok/s (+0.351%/+0.423%/+0.176%)**, all **3/3** exact
wins. Allocation/workspace/dispatch remain unchanged, gfx1151 fails closed, and
**144/144** source-policy/runner guards pass
([H6T production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-production.json) ·
[H6T candidate/runtime](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-candidate.json) ·
[H6T target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-target.json)).

**WPF-H6U exact DPP-add wave reduction is now the retained gfx1100 Q6 source
default; H6E remains explicit rollback.** The frozen **11/11** leaf replaces
H6E's **320/400/400 bpermutes** with exact **64+256 / 80+320 / 80+320
permlanex16+DPP-add**, cutting runtime VGPR **136/168/168 -> 112/144/144** at
unchanged LDS/scratch0. Complete natural M512 is KL0 and byte-exact across all
**48/48** hidden boundaries, complete logits, K/V/`KVLiveSpans`, repeat, and
teardown. Production reuses the qualified exact **2/46/94 H6E -> H6U**
substitution at **2,192** dispatches; consumer/Q6/request-sum/span move
**54.144/86.958/1,276.589/1,305.317 -> 48.443/81.029/1,274.060/1,295.123 ms
(-10.529%/-6.817%/-0.198%/-0.781%)**. Fresh selector-unset fixed C4096/M512
improves H6E rollback -> H6U source **409.485 -> 411.704 tok/s (+0.542%, 5/5
wins)** and is **1.67788x** behind matched llama.cpp HIP **690.791 tok/s**.
Fresh 512/1K/4K improves **382.632/308.496/193.767 ->
384.637/309.813/194.321 tok/s (+0.524%/+0.427%/+0.286%)**, all **3/3** exact
wins. Promotion changes only three selected-map values; allocation, workspace,
scratch, dispatches, F32 N72 fallback, and gfx1151 remain unchanged, and
**153/153** source/kernel/backend/runner guards pass
([H6U production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-production.json) ·
[H6U candidate/runtime](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-candidate.json) ·
[post-H6T residual / H6U target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6t-matched-residual.json)).

The clean committed H6U matched refresh is **410.220 tok/s** from
**409.861/411.224/410.765/410.220/410.191**, all token 2930, deterministic, and
allocation-clean. It is **+141.994%** over campaign start, **+0.318%** over the
clean H6T checkpoint, and **1.68395x** behind matched llama.cpp HIP **690.791
tok/s**. Its reconciled cached request is **1,232.836 ms / 2,192 dispatches** in
a **1,255.013-ms** span:

| Matched M512 component | Campaign start | Current H6U | llama.cpp HIP exact | Remaining gap |
| --- | ---: | ---: | ---: | ---: |
| Q5 projections | 1,270.458 ms | **255.137 ms** | **58.314 ms** | **196.823 ms** |
| Attention | 488.304 | **148.882** | **21.512** | **127.370** |
| IQ3/IQ4 down | 557.091 | **272.131** | **153.860** | **118.271** |
| Q6 projections | 157.073 | **81.744** | **14.668** | **67.076** |
| IQ2/special-IQ3 gate/up | 460.143 | **398.743** | **397.805** | **0.938** |
| Remaining | 68.623 | **76.200** | **67.849** | **8.351** |
| **Kernel sum** | **3,001.692** | **1,232.836** | **714.008** | **518.827** |

**WPF-H6V exact DPP-add Q5 wave reduction is rejected; all candidate surfaces
are removed and production remains H5Y/H6U/H6T.** All six roles are byte-exact
and physical codegen realizes exact **32/96/80/96/80/80 permlanex16 +
128/384/320/384/320/320 DPP adds**, zero bpermutes/moves, fewer code slots and
VGPR, unchanged FMA/load/LDS/barrier/store counts, and scratch0. The 188-call
weighted consumer improves event/wall **269.681/271.908 -> 267.729/267.342 ms
(-0.724%/-1.679%)**, but only **3/6** roles win both clocks. BF16 K3072/N1024
regresses **+12.795%/+13.346%**, BF16 K6144/N3072 misses event by **0.560%**,
and F32 K3072/N6144 regresses **+4.137%/+1.757%**. The frozen all-role gate
therefore fails: skip runtime qualification, remove HIP/Python/key/export/test/
gfx1151-exclusion surfaces without tuning, and retain clean H6U **410.220
tok/s**
([H6V rejection](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q5-dpp-wave-reduction-rejected.json) ·
[post-H6U target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6u-matched-residual.json)).

**WPF-H6W exact late-start dense-initial SWA qrow4 aligned global-score-record
replay is the retained gfx1100 SWA source default; H6A is explicit rollback.**
H6W borrows the first aligned **18,874,368 bytes** of the existing Q5 F32 plane
with no allocation/workspace growth and same-stream projection → attention →
FFN lifetime. Complete natural M512 remains KL0/byte-exact across logits, all
**48/48** hidden boundaries, K/V/`KVLiveSpans`, repeat, and teardown. Four
production-identical cached requests preserve **2,192** dispatches and exact
**48 H6N + 72 H6A + 72 H6W** topology; selected late SWA/attention/kernel-sum/
span improve **81.990/144.957/1,224.048/1,254.740 →
62.470/127.063/1,207.903/1,229.421 ms**. Fresh selector-unset fixed natural
C4096/M512 improves H6A rollback **411.192→417.421 tok/s (+1.515%, 5/5)** and
is **1.65490×** behind matched llama.cpp HIP **690.791**. Fresh 512/1K/4K
improves **385.356/309.745/194.411→390.382/312.026/194.709 tok/s
(+1.304%/+0.736%/+0.153%)**, all **3/3** exact wins. Workspace/scratch remain
**161,120,256/600,141,856 bytes**, gfx1151 fails closed, and **115/115** guards
pass
([H6W production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-production.json) ·
[candidate/runtime](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-candidate.json)).

**The current committed W7900 source adds H6Z exact late-global score/weight
replay and reaches 423.233 tok/s** on the matched natural C4096/direct-M512
protocol, versus **690.791 tok/s** for matched llama.cpp HIP (**1.63218x
behind**). The cache-only H6Z trace is **1,195.702 ms / 2,192 dispatches** with
exact **24 H6N + 24 H6Z + 72 H6A + 72 H6W** topology and zero compiler. The
remaining Q5/IQ-down/attention/Q6 gaps are **198.740/116.810/93.654/66.495
ms**; gate/up is already **1.929 ms faster** than llama.cpp.

**WPF-H7A exact late-SWA scaled-score replay is rejected at complete-byte
exactness.** Structure/preflight and one cached build pass, but start256 differs
from H6W at **80,469/1,179,648** output elements (max **4.656613e-9**) and
start384 at **100,075/1,179,648** (max **3.7252903e-9**). H6W compiles replay
`dot*scale-max` as fused FMA; recording a scaled score rounds before subtraction.
Per the frozen gate, H7A is removed without resource profiling, timing, tuning,
or rerun. H6W/H6Z production stays **423.233 tok/s**
([H7A rejection](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-scaled-score-replay-rejected.json) ·
[target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6z-matched-residual.json) ·
[H6Z production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-score-weight-replay-production.json)).

A fresh post-rejection check reaches **422.602 tok/s** and a compiler-free trace
records **1,200.759 ms / 2,192 dispatches**; the unchanged source remains within
**0.149%** of the retained 423.233-tok/s checkpoint. Next target **WPF-H7B exact
lane-parallel IQ3 final-row publication** attacks H6T's **263.748 ms / 45
calls** without changing arithmetic: lanes0..7 each publish one row using the
same wave0→1→2→3 sum. It models static/dynamic LDS-load and global-store wave
instructions **24 / 824,451,072 → 3 / 103,056,384 each (-87.5%)** at unchanged
logical bytes. RED-first complete bytes, exact code-object resources, cached
trace, and **45/45 plus aggregate** both-clock wins are binding
([post-H7A residual / H7B target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7a-rejection-matched-residual.json)).

**WPF-H7B is rejected at its first compiled resource gate.** Exactness passes
**10/10**, and ISA realizes the intended **24→3** b128 LDS-load and d16-store
sites with unchanged 23 global loads/12 LDS stores/two barriers/216 FMAs/24
permlanex16/96 DPP while shrinking code **7,920→5,916 B**. Metadata VGPR,
however, rises **101→108** beyond the frozen **≤101** ceiling. Per contract,
skip rocprof/timing, do not tune or recompile, remove every H7B surface, and
retain H6T/H6Z production **422.602 tok/s / 1,200.759 ms**
([H7B rejection](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-lane-parallel-final-rows-rejected.json)).

Fresh committed production is **422.947 tok/s** and a compiler-free trace is
**1,199.578 ms / 2,192 dispatches**. The residual Q5/IQ-down/attention/Q6 gaps
are **197.783/118.305/93.890/66.748 ms**. Standalone **WPF-H7C exact raw-Q6
DPP-add wave reduction** is admitted for the three remaining raw source-GGUF
calls, which own **28.474 ms (34.904% of Q6)**. Complete correctness passes
**22/22**. BF16/F32 code/slots fall **4,840/843→4,228/681** and
**5,040/909→4,452/749**, with exact **0 bpermutes + 32 permlanex16 + 128 DPP
adds**, metadata/runtime VGPR **60/64** and **55/56**, LDS512, and scratch0.
The one-shot actual-weight screen improves all three roles on both clocks and
the aggregate **37.248/37.303→36.983/36.998 ms event/wall
(-0.712%/-0.817%)**. Bounded default-off runtime ownership is now qualified:
complete M512 state is KL0/byte-exact across all 48 hidden boundaries and full
KV/spans; four cache-only requests preserve **2,192 dispatches** while replacing
exactly two BF16 plus one F32 generic calls, cutting the selected subwindow
**28.543→28.220 ms (-1.132%)** and span **1,280.898→1,279.005 ms**. Matched
C4096/M512 improves **420.701→420.914 tok/s (+0.0505%, 4/5)**, and 512/1K/4K
medians improve **+0.0552%/+0.0274%/+0.0179%** with exact state and unchanged
scratch. Source promotion now selects the qualified H7C map while preserving
the named empty generic rollback. Fresh source-selected M512 state remains KL0
and byte-exact across **48/48** boundaries and full KV/spans. A fresh four-run
trace again replaces exactly **2 BF16 + 1 F32** calls and improves selected
raw-Q6/Q6/span **28.583/81.639/1,283.417→28.376/81.470/1,280.788 ms** with zero
compiler. Fresh source aggregate timing is mixed: fixed C4096/M512 is
**419.433→418.487 tok/s (-0.225%, 2/5)**, while 512/1K/4K is
**+0.0925%/+0.0372%/-0.0488%**. Retain source under the cycle-wall policy based
on repeatable selected-subwindow/span wins and the immutable all-role leaf
screen; do not claim the noisy aggregate rows as wins. The last clean committed
checkpoint remains **422.947 tok/s / 1,199.578 ms** pending post-commit reprofile
([H7C production](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-production.json) ·
[candidate/runtime](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-candidate.json) ·
[target](benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7b-rejection-matched-residual.json)).

Fresh post-H7C production is **422.786 tok/s**, **1.63390x behind** matched
llama.cpp HIP **690.791 tok/s**. The representative cache-only request is
**1,197.499 ms** in a **1,219.043-ms** span across **2,192 dispatches** with
zero compiler. Q5/IQ-down/attention/Q6 gaps are
**196.915/117.620/93.693/66.653 ms** and explain **98.219%** of the remaining
kernel gap. **WPF-H7D closes the latest exact Q5 VOPD scheduling premise**:
naive row interleaving leaves control/candidate at the same **52 paired FMAs**,
and forced pairing fails compilation with gfx1100's `src0` VGPR-bank rule.

Standalone **WPF-H7E IQ3 two-plane residual-D4 source-MMQ remains diagnostic,
but its temporary runtime owner is rejected and removed**. The leaf itself
passes **9/9**, is spill/scratch-free at metadata/runtime VGPR **148/152**, and
wins the immutable all-45 producer-inclusive screen
**247.297/260.672→186.732/180.752 ms**. Natural-M512 state and cached tracing
also looked promising at KL **0.000224**, top-1 **100%**, and diagnostic
IQ-down **269.921→208.298 ms**, with zero scratch growth.

Those narrow gates do not waive complete quality. All 18 committed prompts were
independently extended to M512 and all **576/576** teacher-forced steps exercised
changed arithmetic. H7E fails at max KL **5.630805 > 0.05** and general-Japanese
top-1 **115/128 = 89.844% < 90%**; suite top-1 is **531/576 = 92.188%**. Same-
mode repeats are deterministic, but control/candidate free-running equality is
only **21/54** at h16 and **6/54** at h32. Poolside off-shape fallback and
lifecycle pass. No promotion timing was run. Production remains H6T/IQ4 at
**422.786 tok/s**, and prompt/layer-conditioned salvage is forbidden
([H7E rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-complete-quality-rejected.json) ·
[candidate](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-source-mmq-candidate.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7c-matched-residual-iq3-d4x2-target.json)).

**WPF-H7H exact full-group Q5 compute is now the retained gfx1100 Q5 source**
for both divisible natural-M512 roles (`c8r4` **92 calls**, `c12r8` **35**).
The complete H7G map remains named rollback for all eight roles, including its
four padded-tail roles; H5Y remains the exact primitive fallback. RED-first leaf
correctness passes **13/13** and the production object remains private/spill/
scratch0 at metadata/runtime VGPR **72/72 and 194/200**, LDS **512/1,536**.

Fresh selector-unset H7G -> H7H source qualification is KL0/byte-exact across
all **48/48** hidden boundaries, complete state, and repeat at unchanged
**161,120,256-byte** workspace / **600,141,856-byte** scratch. Fixed C4096/M512
improves **423.045 -> 426.745 tok/s (+0.874%, 5/5 wins)**; clean 512/1K/4K
improves **392.829/312.307/194.847 -> 396.922/315.105/195.775 tok/s
(+1.042%/+0.896%/+0.477%)**, all **3/3** exact wins. Clean H7H production is
**427.407 tok/s** from **426.886/428.531/428.010/427.407/426.065**, **+0.603%**
over H7G and **1.61624x** behind matched llama.cpp HIP **690.791 tok/s**.

Source tracing records exact **61 H7G + 127 H7H** calls among **2,925**
dispatches on one queue/stream; every one of five clean profiled requests has
that same Q5 topology and exactly **2,192 dispatches**. The representative
request is **1,185.096 ms** in a **1,206.456-ms** span, with zero compiler.

| Matched M512 component | Campaign start | Before H7H | Current H7H | llama.cpp HIP | Current gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q5 projections | 1,270.458 ms | 248.888 ms | **237.185 ms** | 58.314 ms | **178.871 ms** |
| IQ3/IQ4 down | 557.091 | 272.226 | **273.577** | 153.860 | **119.717** |
| Attention | 488.304 | 115.472 | **116.227** | 21.512 | **94.715** |
| Q6 projections | 157.073 | 81.541 | **81.900** | 14.668 | **67.233** |
| IQ2/special-IQ3 gate/up | 460.143 | 398.098 | **399.683** | 397.805 | **1.878** |
| Remaining | 68.623 | 76.199 | **76.524** | 67.849 | **8.674** |
| **Kernel sum** | **3,001.692** | **1,192.424** | **1,185.096** | **714.008** | **471.088** |

Only the Q5 reduction is attributable to H7H; unchanged-family deltas are
profile noise. Q5/IQ-down/attention/Q6 still account for **97.760%** of the
matched kernel gap
([H7H production](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-production.json) ·
[candidate/runtime](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-candidate.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7g-matched-full-group-q5-target.json) ·
[H7G rollback production](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-padded-compute-production.json)).

The post-H7H rerank selects target-only **WPF-H7I exact raw-Q6 full-group
compute**. H7C's three natural-M512 raw-Q6 roles are exactly divisible by their
rowbatch8/16 geometries yet still evaluate `row < rows` inside every unrolled
FMA group. They own **28.482 ms / 34.776%** of current Q6. A frozen first-and-
only actual-weight 5/15/5 screen requires all three roles and their aggregate
to win both clocks; H7C -> H7I improves weighted event
**35.840 -> 20.323 ms (-43.295%, 1.764x)** and synchronized wall
**34.854 -> 21.974 ms (-36.954%, 1.586x)**. Every output is byte-exact, finite,
and allocation-clean; no role subset is admissible.

The first out-of-tree object reduces BF16/F32 code **4,228 -> 4,060 / 4,452 ->
4,032 bytes**, slots **681 -> 623 / 749 -> 631**, scalar row comparisons
**9 -> 2 / 17 -> 2**, and raises dual-FMAC sites **1 -> 10 / 1 -> 11**. It
preserves 24 global loads, one store, the exact DPP/LDS reduction, and
private/spill/scratch0; metadata VGPR **69/64** stays within the frozen 72-VGPR
ceiling. Production remains H7H/H7C at **427.407 tok/s / 1,185.096 ms**. Freeze
a separate three-role RED before adding any named H7I surface
([post-H7H residual / H7I target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

Standalone **WPF-H7I is admitted**, and its exact three-role package capability
is now a qualified bounded default-off runtime owner; live source remains H7C.
Its RED-first **22/22** matrix proves exact-M512 H7I bytes against H7C plus
sampled CPU values while rows1/7/8/9 retain complete H7C fallback; rows511/513
and every wrong role fail before HIP loading. The first repository object
reproduces BF16/F32 code/slots **4,060/623** and **4,032/631**, metadata VGPR
**69/64**, LDS512, and spill/scratch0 exactly. A non-adjudicative actual-weight
replay remains byte-exact and improves weighted event/wall
**35.432/34.617 -> 20.089/21.762 ms (1.764x/1.591x)**.

Bounded H7C -> H7I qualification is KL0/byte-exact across all **48/48** hidden
boundaries, complete logits/KV/`KVLiveSpans`, repeat, scratch, and teardown.
Fixed C4096/M512 improves **426.583 -> 429.000 tok/s (+0.567%, 5/5 wins)**;
clean 512/1K/4K improves **396.104/315.021/195.729 ->
399.127/316.409/196.109 tok/s (+0.763%/+0.441%/+0.194%)**, all **3/3** exact
wins. Cache-only integration records exactly **2 BF16 + 1 F32 H7I**, zero H7C,
and **2,925** total dispatches at runtime VGPR72/64 with zero compiler.
Production remains H7H/H7C **427.407 tok/s** pending the separate source gate
([H7I candidate/runtime](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json)).

**WPF-H7I exact raw-Q6 full-group compute is now the retained gfx1100 raw-Q6
source**, with complete named H7C rollback. Fresh selector-unset H7C -> H7I
qualification is KL0/byte-exact across all **48/48** hidden boundaries,
complete state, repeat, unchanged **161,120,256/600,141,856-byte** workspace/
scratch, and teardown. Fixed C4096/M512 improves **427.903 -> 429.434 tok/s
(+0.358%, 5/5 wins)**; clean 512/1K/4K improves
**396.414/315.253/195.754 -> 398.219/316.228/196.385 tok/s
(+0.455%/+0.309%/+0.322%)** with positive medians and exact state throughout.

Clean production reaches **431.310 tok/s** from
**431.143/431.948/431.310/431.479/430.607**, **+0.913%** over H7H/H7C and
**1.60161x** behind matched llama.cpp HIP **690.791 tok/s**. Every one of five
profiled requests preserves **2,192 dispatches** and exact **2 BF16 + 1 F32
H7I** raw-Q6 topology; the representative kernel sum/span is
**1,172.241/1,193.552 ms**, with zero compiler.

| Matched M512 component | Campaign start | Before H7I | Current H7I | llama.cpp HIP | Current gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q5 projections | 1,270.458 ms | 237.185 ms | **235.199 ms** | 58.314 ms | **176.885 ms** |
| IQ3/IQ4 down | 557.091 | 273.577 | **272.309** | 153.860 | **118.449** |
| Attention | 488.304 | 116.227 | **115.317** | 21.512 | **93.805** |
| Q6 projections | 157.073 | 81.900 | **74.409** | 14.668 | **59.742** |
| IQ2/special-IQ3 gate/up | 460.143 | 399.683 | **398.590** | 397.805 | **0.785** |
| Remaining | 68.623 | 76.524 | **76.417** | 67.849 | **8.567** |
| **Kernel sum** | **3,001.692** | **1,185.096** | **1,172.241** | **714.008** | **458.233** |

Only Q6's reduction is attributable to H7I; unchanged-family deltas are profile
noise. Q5/IQ-down/attention/Q6 own **97.959%** of the matched kernel gap
([H7I production](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-production.json) ·
[candidate/runtime](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

**WPF-H7J exact Q5 full-grid bounds specialization is rejected.** The frozen
actual-weight 5/15/5 gate predeclared both H7H roles as inseparable. Outputs are
byte-exact/finite and all allocations recover, but the dominant `c8r4` role
(**92 calls**) regresses to **0.99954x event / 0.99127x wall**. The `c12r8`
role (**35 calls**) and weighted aggregate improve, but selecting that favorable
subset after timing is forbidden. Production therefore remains H7H/H7I at
**431.310 tok/s**, **1.60161x** behind matched llama.cpp HIP
([H7J rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-grid-bounds-rejected.json)).

The post-H7J rerank selects target-only **WPF-H7K exact late-start SWA
score-to-weight publication**. H6W's starts256/384 score-replay owner accounts
for **72 calls / 62.627 ms**, or **54.309%** of current attention. H7K keeps
H6W's first-pass unscaled dots/maxima, fused `dot*scale-max`, token-order
denominator, token-order unnormalized PV, and final divide exactly; it only
publishes four weights per aligned record between denominator and PV passes.
The operation removes **255,135,744** dynamic lane-0 weight broadcasts while
adding **128,065,536** aligned record operations (**2.049 GB** logical record
traffic; rationale only, not a speed claim). Starts256/384 are inseparable, and
the first object must pass complete-byte/CPU, physical-resource, named-trace,
and both-clock per-start plus weighted gates with no salvage. Production stays
H7H/H7I **431.310 tok/s**, **1.60161x** behind matched llama.cpp HIP; no H7K
repository surface or performance result exists before RED
([post-H7J residual / H7K target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7j-matched-swa-weight-publication-target.json)).

**WPF-H7K is rejected at its frozen first-object physical gate.** The first and
only repository object keeps H6W unchanged and passes code/slots/VGPR
**5,048 B / 875 / 54**, 28 bpermutes, four exponentials, 56 FMAs, two b128
record stores, and spill/scratch0. However, both required aligned `float4`
record reads scalarize to **0 b128 + 2 b32 sites**, failing the predeclared
**2 b128-load** premise. No candidate correctness, trace, or timing screen is
consumed; recompilation/tuning is forbidden, and every H7K source/test/key/
export/exclusion surface is removed. Production remains H7H/H7I
**431.310 tok/s**, **1.60161x** behind matched llama.cpp HIP
([H7K physical rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-weight-publication-physical-rejected.json)).

The post-H7K rerank selects target-only **WPF-H7L exact IQ3 full-batch/live-tail
split**. An actual-routing replay over all **45** IQ3 layers finds **230,400**
live compact rows represented by **33,547** rowbatch8 iterations: **24,650
(73.479%)** are complete and cover **197,200 rows (85.590%)**, while **8,897**
tails contain **33,200** live plus **37,976** inactive compute slots. H7L keeps
H6T's complete-batch interleaved VOPD math and fused-DPP publication unchanged,
then computes and publishes only the final tail's 1..7 live rows in the same
per-row FMA/reduction/store order. The modeled inactive work is **4.200B FMA +
2.333B exchange wave operations**; this is source-operation rationale, not a
speed claim.

Freeze strict K1024/N3072/E256/P256/P64/local128/rowbatch8 ownership, complete
rows1/7/8/9/M512 plus all tail sizes and reversed-P64 controls, unchanged
compaction/ABI/workspace/maps/gfx1151, first-object VGPR/LDS/spill/code bounds,
named cache-only execution, and one immutable all-45-layer 5/15/5 screen. Every
layer and the aggregate must win HIP-event and synchronized-wall clocks; any
miss removes H7L without subset salvage, tuning, recompile, or favorable rerun.
Production remains **431.310 tok/s / 1,172.241 ms**, **1.60161x** behind matched
llama.cpp HIP, and no H7L implementation or speed result exists before RED
([post-H7K residual / H7L target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7k-matched-iq3-live-tail-target.json)).

**WPF-H7L is rejected at its frozen first-object physical gate.** The single
repository object preserves H6T at **7,920 B / 1,384 slots / VGPR101 / spill0**,
but the full-batch/live-tail sibling expands to **49,592 B / 9,082 slots /
VGPR133 / 270 SGPR spills**. It therefore fails the declared **14,000-byte**,
**2,400-slot**, **VGPR<=101**, and spill0 bounds before candidate correctness,
named tracing, or timing. No rewrite/recompile or tail/layer subset is allowed;
every H7L body/export/wrapper/key/RED/gfx1151 surface is removed. Production
remains H7H/H7I **431.310 tok/s**, **1.60161x** behind matched llama.cpp HIP
([H7L physical rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-live-tail-physical-rejected.json)).

**WPF-H7M exact two-wave/two-K256-partition IQ3 is rejected.** Physical-only
preselection chooses a no-spill LDS-activation form at **12,744 B / 2,171 slots
/ VGPR113 / LDS16,768** over the no-spill VGPR166 register form before timing.
The selected local64 body remains byte-exact on all **45/45** actual IQ3 layers,
but loses every layer on both clocks: H6T -> H7M aggregate event is **246.763 ->
392.180 ms (+58.929%, 0.629x)** and wall is **261.551 -> 377.358 ms (+44.277%,
0.693x)**. No repository/runtime/source surface is added and production stays
**431.310 tok/s**
([H7M rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-two-wave-k-partition-rejected.json)).

**WPF-H7N exact raw-Q6 c16r4 direct ordered consumption is rejected.** The
immutable first object is physically clean at **8,900/8,872 B**, **1,393/1,390
slots**, **VGPR112**, **LDS1,024**, and spill/scratch0 for BF16/F32. All three
actual roles remain byte-exact, and one launch replaces inclusive H6U
activation-pack + Q6-to-F32 producer + ordered-consumer triples, but each role
is **3.95–5.46x slower**. The 142-call event aggregate regresses **48.267 ->
233.861 ms (+384.516%, 0.206x)** and wall regresses **48.520 -> 231.238 ms
(+376.583%, 0.210x)**. Add no repository surface; production stays **431.310
tok/s**
([H7N rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-c16r4-direct-rejected.json)).

**WPF-H7O exact H7I raw-Q6 geometry crossover is rejected.** The immutable
constant-32 object is physically clean: crossed BF16 c2r16 is **4,060 B / 634
slots / VGPR64**, and crossed F32 c4r8 is **4,032 B / 620 / VGPR69**, both
LDS512/spill0. Outputs remain byte-exact. Both BF16 roles win, but the F32 role
regresses to **0.912x event / 0.913x wall**. The three-role aggregate improves
**21.909 -> 21.314 ms event (1.028x)** and **21.905 -> 21.488 ms wall
(1.019x)**, but the predeclared all-role gate forbids favorable BF16-only
salvage. No repository surface is added and production stays **431.310 tok/s**
([H7O rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-geometry-crossover-rejected.json)).

**WPF-H7P candidate-distance-only IQ3 D4x2 boundary repair is rejected before
repository code or candidate timing.** A prompt-independent audit compares H7E's
pre-BF16 FP32 accumulators with exact H6T over all **45** natural-M512 IQ3
layers: **16,306,295 / 707,788,800 values (2.30384%)** round differently. A
1/16-cell guard repairs **6.234%** of outputs but catches only **43.799%** of
mismatches; a 1/4-cell guard repairs **24.931%** yet leaves **5,206,620** wrong
values. Even the 1.0-cell guard selects **99.719%** of outputs, still misses
**14,702** mismatches, and is only **0.592x** exact in the ideal zero-overhead
linear model. No tested threshold is complete, so do not implement or time this
distance-only guard. A materially different prompt-independent error-size
certificate remains a separate hypothesis; production stays **431.310 tok/s**
([H7P rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-boundary-repair-rejected.json)).

**WPF-H7Q/H7R third-plane residual certificates are also rejected before code
or candidate timing.** H7Q's sparse D4x2/D4x3 disagreement set selects
**2.30385%** of outputs and catches **99.7364%** of D4x2 mismatches, but leaves
**42,981** wrong; making its boundary union complete selects **99.7205%** of
all outputs. H7R's outward-rounded producer-residual × exact-IQ3-L1 bound
captures every observed mismatch, but even its tightest K64 form flags
**74.5071%** of **707,788,800** outputs. Only **30.6591%** repair density could
break even before guard cost; the best zero-guard-cost model is **0.695x**
exact, and the declared read-ceiling models peak at **0.610x**. Add no guard,
sidecar, queue, repair kernel, RED, runtime, or source owner; rerank outside IQ3
residual repair and retain **431.310 tok/s**
([H7Q/H7R rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-residual-certificates-rejected.json)).

The post-H7R rerank selects target-only **WPF-H7S exact raw-Q6 c2r32 packed-
activation cross-row reuse**. Current H6U executes **142** exact activation-pack
+ Q6-to-F32 producer + ordered-consumer triples at weighted event/wall
**48.267/48.520 ms**. H7S keeps the pack ABI but uses rowbatch32 and a raw-Q6
c2r32 consumer, reducing each 64-accumulator K step from H7N's **16 decodes / 68
load sites** to **2 decodes / 8 Q6-field + 4 aligned b128 activation loads**.
The static model removes **142 producer launches/request** (**2,192 -> 2,050**
dispatches if ultimately selected) and models **0.937x** current input bytes;
these are source-level rationale, not a physical or speed claim.

Freeze all three 2/46/94-call roles together, exact M512 bytes/CPU/pack/poison/
finite/lifecycle and rows511/513 fail-closed behavior, unchanged H6U fallback/
workspace/maps/gfx1151, first-object vector-load/opcode/VGPR/LDS/spill bounds,
and named cache-only pack+consumer execution. One immutable actual-weight 5/15/5
screen must improve every role and the weighted aggregate on HIP-event and
synchronized-wall clocks. No role, geometry, prompt, tuning, recompilation, or
favorable-rerun salvage is admissible. Production remains **431.310 tok/s /
1,172.241 ms**, **1.60161x** behind matched llama.cpp HIP, and no H7S code or
speed result exists before RED
([post-H7R residual / H7S target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7r-matched-raw-q6-cross-row-reuse-target.json)).

**WPF-H7S is rejected and fully removed after the immutable all-role screen.**
Its first object passes every frozen physical gate at **5,912/5,884 bytes**,
**864/860 slots**, **VGPR112 / SGPR24 / LDS1,024**, spill/scratch0, with exact
**4 b128 + 8 raw-Q6 loads, 64 FMAs, 64 permlanex16, and 256 DPP adds**.
Complete bytes, CPU samples, rowbatch32 packs, poison, finiteness, lifecycle,
and the compiler-free pack→consumer/no-producer trace all pass. Nevertheless,
every role loses both clocks; the weighted H6U→H7S aggregate regresses
**49.193→149.544 ms event (0.329x)** and **49.721→146.161 ms wall (0.340x)**.
Remove all H7S code/keys/RED/exclusion surfaces, forbid role/geometry/rerun
salvage, and retain production **431.310 tok/s**
([H7S rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-cross-row-reuse-rejected.json)).

**WPF-H7T late-start QK-only tensorized score replay is rejected at the
complete quality gate.** The immutable first object and named trace pass:
global/SWA consumers are **3,968/4,224 bytes**, **683/744 slots**, metadata
**VGPR49/53**, local/wave32, and spill/scratch0; all four cache-only chains are
key-widen→query-pack→one-QK→consumer with no PV/value-widen/standalone-softmax
or compiler. Standalone correctness is **10/10**.

The binding **18-prompt / 576-step** four-category lane executes exactly
**7,008/7,008** H7T calls but reaches maximum KL **0.393845 > 0.05**. Top-1 is
**562/576 (97.569%)**, every category remains above 90% top-1, deterministic
repeats/oracle/lifecycle pass, but all four categories exceed the KL ceiling.
Run no H7T 5/15/5 admission timing, remove every implementation/RED/gfx1151
surface without subset or rerun salvage, and retain production **431.310
tok/s**
([H7T quality rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-qk-only-score-replay-quality-rejected.json) ·
[H7T target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7s-qk-only-score-replay-target.json)).

The clean rerank selects target-only **WPF-H7U exact stable parallel MoE active
compaction**. The current one-workgroup scheduler scans all **5,120 routed
lanes** across 256 experts and owns **25.187 ms / 47 calls**, **32.960%** of
the 76.417-ms remaining bucket; the following **7.717-ms** packed-hidden gather
is explicitly unchanged. H7U transfers the already registered stable
count→Blelloch-prefix→ballot-scatter sibling from gfx1151 to gfx1100, replacing
47 serial launches with 141 parallel stages at zero allocation/workspace or
model-arithmetic change. Prior gfx1151 exact production is rationale only, not
a W7900 result. Production remains **431.310 tok/s / 1,172.241 ms / 2,192
dispatches**; no gfx1100 candidate has run. Freeze RED first, then require all
metadata/full-state bytes, named **47+47+47** cache-only tracing, and one
inseparable all-47-layer both-clock gate without layer/expert/length/rerun
salvage
([H7U target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7t-parallel-moe-compaction-target.json)).

**WPF-H7U is now admitted as a bounded default-off gfx1100 capability; the
production source remains serial.** The one package-only constant exposes the
unchanged registered sibling. GREEN is **9/9**; all **47** natural M512
metadata records, **47** packed-hidden gathers, **48/48** hidden boundaries,
logits, KV/`KVLiveSpans`, token **2930**, repeat, and lifecycle are exact.
Physical inspection shows local256/wave32 count/prefix/scatter at metadata
VGPR **10/17/31**, private/spill/scratch0. Named tracing records exact
**47+47+47**, zero serial, unchanged 47 gather, and **2,286 application
dispatches** on one queue/stream.

The first immutable all-layer 5/15/5 screen wins **47/47** on both clocks:
serial→parallel aggregate is **20.508→1.297 ms event (15.813x)** and
**20.701→1.445 ms synchronized wall (14.331x)**; minimum layer speedups are
**14.012x/12.690x**. This is standalone leaf evidence only. Production remains
**431.310 tok/s / 1,172.241 ms / 2,192 dispatches** until separate bounded
runtime and clean source-default gates pass
([H7U candidate](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-candidate.json)).

**WPF-H7U stable parallel MoE compaction is now the retained gfx1100 source
owner**, with the serial scheduler preserved as an explicit rollback. The
bounded fixed C4096/M512 gate is byte-exact and improves **430.412→436.602
tok/s (+1.438%, 5/5 paired wins)**. Fresh source-default 512/1K/4K gates remain
exact, finite, and lifecycle-clean while improving
**398.781→404.250 (+1.371%) / 316.758→320.700 (+1.245%) /
196.636→197.866 tok/s (+0.626%)**, with **3/3** paired wins at every length.

Clean matched C4096/direct-M512 production reaches **437.189 tok/s** from
**436.223/437.801/437.337/436.568/437.189**, **+1.363%** over H7I and
**1.58007x behind** matched llama.cpp HIP **690.791 tok/s**. Selected-region
tracing records exact **47 count + 47 prefix + 47 scatter**, zero serial,
unchanged 47 gather, **2,286 dispatches**, one queue, and zero compiler in every
one of five requests. Parallel compaction is **1.155 ms** versus the prior
serial **25.187 ms**; representative kernel sum falls
**1,172.241→1,160.833 ms** despite unrelated-family trace noise.

| Matched M512 component | Campaign start | Before H7U | Current H7U | llama.cpp HIP | Current gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q5 projections | 1,270.458 ms | 235.199 ms | **236.539 ms** | 58.314 ms | **178.225 ms** |
| IQ3/IQ4 down | 557.091 | 272.309 | **275.424** | 153.860 | **121.564** |
| Attention | 488.304 | 115.317 | **116.296** | 21.512 | **94.784** |
| Q6 projections | 157.073 | 74.409 | **74.982** | 14.668 | **60.314** |
| IQ2/special-IQ3 gate/up | 460.143 | 398.590 | **404.375** | 397.805 | **6.570** |
| Remaining | 68.623 | 76.417 | **53.218** | 67.849 | **-14.632 ms** |
| **Kernel sum** | **3,001.692** | **1,172.241** | **1,160.833** | **714.008** | **446.825 ms** |

Only the compaction/remaining reduction is attributable to H7U; deltas in
unchanged families are profiler noise. The still-actionable gap is concentrated
in Q5, IQ-down, attention, and Q6
([H7U production](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-production.json) ·
[bounded candidate](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-candidate.json)).

The post-H7U rerank selects target-only **WPF-H7V exact dequantized-Q6 full-
batch/live-tail predicate elimination**. The existing 142 H6U consumers own
**49.191 ms**, **65.604%** of current Q6. Across the 2/46/94-call rowbatch5/4/5
roles, **1,757,184 / 1,763,328 workgroups (99.652%)** are complete groups yet
retain two dynamic row predicates. H7V keeps the activation pack, exact
Q6-to-F32 producer, every ordered FMA/DPP/LDS/store operation, and H6U itself as
fallback. Rowbatch4 uses one full launch; rowbatch5 uses one full-prefix launch
plus one exact remainder-2 H6U tail, modeling **142→238 consumer launches** and
**2,286→2,382 request dispatches** at zero allocation/workspace growth.

This is not H7I/H7N/H7S raw-Q6 replacement, H7J Q5 full-grid salvage, or H7L
IQ3 live-tail retry. Freeze RED first, require one clean physical object, exact
full+tail recomposition, named producer/pack/full/tail topology, and one
inseparable all-three-role plus weighted-aggregate both-clock 5/15/5 gate. No
candidate has run and no speed claim exists
([post-H7U / H7V target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7u-q6-full-batch-live-tail-target.json)).

**WPF-H7V is rejected after its first immutable all-role screen.** The sole
object passes every physical gate and shrinks H6U: BF16 r4 is **5,808 B / 873
slots / VGPR108**, BF16 r5 **6,960 / 1,001 / 139**, and F32 r5 **6,928 / 996 /
139**, with exact FMA/permlanex16/DPP/barrier/LDS structure and no spills.
GREEN is **9/9**; full-request tracing proves **142 packs + 143 producers + 142
H7V full + 96 exact H6U tail consumers / 2,382 dispatches**, one queue, and zero
compiler. Outputs are byte-exact and lifecycle-clean, but both rowbatch5 roles
lose both clocks. The 142-call H6U→H7V aggregate regresses **47.949→48.680 ms
event (+1.524%, 0.985x)** and **48.522→49.162 ms wall (+1.318%, 0.987x)**.
Remove every H7V implementation/RED/key surface, retain H6U and production
**437.189 tok/s**, and do not salvage rowbatch4 or rerun
([H7V rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q6-full-batch-live-tail-rejected.json)).

**WPF-H7W is rejected after its sole immutable all-45-layer screen.** The one
**469,056-byte** object passes every frozen physical gate: P128 and retained
P256 are both **7,920 B / 1,384 slots / VGPR101 / SGPR78 / LDS384 / spill0**
with exact **216 FMA, 24 permlanex16, 96 DPP adds, 24 LDS-b128 loads, 12 LDS
stores, and two barriers**. GREEN passes **12/12**. Cache-only tracing names
exact **45 H7W P128 + two unchanged IQ4 / 2,286 dispatches**, local128,
grid16,384×64, runtime VGPR104/LDS512/scratch0, one queue, and zero compiler.

All 45 outputs are byte-exact and lifecycle-clean, but only **16/45** layers
win both clocks. H6T P256→H7W P128 moves the all-layer event sum
**260.663→261.392 ms (+0.280%, 0.99721x)** and synchronized wall
**260.731→262.135 ms (+0.538%, 0.99464x)**. This fails both the per-layer and
aggregate gates. Remove every H7W export/wrapper/key/RED/backend-exclusion
surface, retain H6T P256 and production **437.189 tok/s**, and forbid
layer/expert/routing/prompt/length subset, partition-retune, rewrite,
recompile, or favorable-rerun salvage
([H7W rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-output-p128-rejected.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7v-iq3-output-p128-target.json)).

The clean post-H7W rerank selects target-only **WPF-H7X exact H6W one-slot
BF16 K/V software pipeline**. Matched status is:

| Component | Campaign start (ms) | Current (ms) | llama.cpp HIP (ms) |
| --- | ---: | ---: | ---: |
| Q5 projections | 1,270.458 | 235.987 | 58.314 |
| IQ3/IQ4 down | 557.091 | 273.063 | 153.860 |
| Attention | 488.304 | 115.607 | 21.512 |
| Q6 projections | 157.073 | 74.719 | 14.668 |
| IQ2/special-IQ3 gate/up | 460.143 | 400.672 | 397.805 |
| Remaining | 68.623 | 53.299 | 67.849 |
| **Kernel sum** | **3,001.692** | **1,153.347** | **714.008** |

Retained wall throughput is **169.516→437.189 tok/s** versus matched llama.cpp
HIP **690.791 tok/s**; hipEngine remains **1.58007× behind**. H6W alone owns
**72 calls / 62.656 ms**, **54.198%** of attention. Its current local32/wave32
body is **4,984 B / 871 slots / metadata VGPR54 / runtime VGPR56 / LDS0 /
spill0**. In each steady K and V path it issues four BF16 loads and immediately
drains `vmcnt(3→0)` before current-slot arithmetic. At natural M512,
**63,866,880 / 64,032,768 slots (99.7409%)** have a next slot available.

H7X adds one separately named gfx1100 H6W-equivalent sibling: preload slot 0,
issue slot *n+1* K before complete slot-*n* QK/reduction/max/store work, and
independently issue slot *n+1* V before slot-*n* exp/denominator/PV work. It
preserves bytes, arithmetic order, `KVLiveSpans`, allocation/workspace,
dispatches, H6W source/default, and every fallback. Freeze RED first; then
require one object at VGPR≤64/spill0, physical next-slot load overlap, complete
starts256/384 H6W+CPU identity, exact **72 H7X / 2,286-dispatch** named tracing,
and one inseparable starts256/384 plus aggregate dual-clock 5/15/5 screen. No
candidate exists and no H7X speed claim has been made
([post-H7W / H7X target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

**WPF-H7X is rejected at its sole first-object physical gate, before candidate
execution.** The object is otherwise clean: H7X is **5,320 B / 931 slots /
VGPR54 / SGPR44 / LDS0 / spill0**, versus unchanged H6W **4,984 B / 871 slots
/ VGPR54 / SGPR40**, and preserves the declared bpermute/record/exp/FMA/FMAC/
store structure. But each steady next-slot four-u16 K and V clause is followed
immediately by `vmcnt(3→0)`: there are **zero current-slot instructions**
between its final load and first wait. Thus neither intended overlap exists.
Remove all H7X code/RED/backend-exclusion surfaces, run no correctness/trace/
timing/runtime/source gate, forbid rewrite/recompile/subset/rerun salvage, and
retain H6W plus production **437.189 tok/s**
([H7X physical rejection](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-kv-prefetch-physical-rejected.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

The clean post-H7X rerank keeps the component table above unchanged and selects
target-only **WPF-H7Y exact H6W lane-major BF16 K/V mirror loads**. H6W's
natural cache presents each wave as `[part4][lane32]`, forcing **8 static
`global_load_u16` + 8 staged `vmcnt` waits** across its K and V passes. A
separately named H7Y leaf consumes caller-provided `[lane32][part4]` mirrors,
so every lane reads its same four BF16 values with one aligned `b64` per pass
before the unchanged QK/score-replay/PV arithmetic.

Across the exact 72-call starts256/384 route, the operation model changes
**512,262,144 → 128,065,536 global-load issue slots (-75%)** and removes the
same **384,196,608 wait-issue slots**, while the **32,784,777,216-byte** K/V
payload is unchanged. This is selection arithmetic, not speed evidence. Freeze
RED first, then require one object with exactly **2 b64 / 0 u16 / ≤2 waits**,
unchanged H6W opcodes/resources, complete transpose/H6W/CPU/record/span bytes,
and one inseparable all-72 per-start+aggregate dual-clock 5/15/5 screen.
H7Y is now retained as an explicit standalone leaf. Its sole object is **4,900
B / 855 slots / metadata VGPR54 / SGPR40 / spill0**, with exactly **2 b64 / 0
u16 / 2 waits**; named execution is runtime VGPR56/LDS0/scratch0. All **72
actual-layer** outputs and score planes are byte-exact. The immutable screen
moves H6W→H7Y **56.607→56.259 ms event (-0.616%)** and **56.559→56.317 ms
wall (-0.428%)**, with both starts positive.

The separate bounded runtime owner now also qualifies default-off. It adds exact
**72 MiB / 72 allocations** of SWA K/V mirrors and one fused natural+lane-major
writer without changing the **2,286-dispatch** request topology. Complete M512
state is KL0 and byte-exact across all **48/48** boundaries, logits, K/V/spans,
repeat, and teardown. The named request contains exact **144 fused writers + 72
H7Y + 72 H6A + 24 H6N + 24 H6Z** on one queue with zero compiler; H7Y is
VGPR56/LDS0/scratch0 and the writer is VGPR24/LDS0/scratch0. Writer-inclusive
fixed C4096/M512 improves **436.120→436.785 tok/s (+0.152%)**. Clean 512/1K/4K
medians improve **+0.0530%/+0.1217%/+0.0043%**, all exact and lifecycle-clean.

The separate selector-unset source gate is exact but rejects promotion at its
first binding median: H6Z/H6W rollback **436.403 tok/s** versus H7Y source
**436.275 tok/s (-0.0294%, 0.99971×; 2/5 paired wins)**. Per the frozen rule,
no source 512/1K/4K, trace, or post-commit gate runs. The active H6Z/H6W map and
published production **437.189 tok/s** remain unchanged; the qualified H7Y
owner stays default-off
([source rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-source-rejected.json) ·
[runtime candidate](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-runtime-candidate.json) ·
[standalone](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-candidate.json) ·
[target](benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7x-swa-lane-major-cache-target.json)).

Two actual-state, no-timing repair audits close the next changed-arithmetic
premises before repository implementation. H5A exact-value Q5 SGEMM differs on
only **123,111 / 134,742,016 BF16 outputs (0.0914%)**, but mismatches reach the
center of the candidate BF16 cell: complete candidate-distance recall requires
selecting **100%** of BF16 outputs, while **96.737%** of its F32 outputs differ.
The retained H2 source-FlashAttention leaf differs after the real softplus gate
on **44,171,810 / 207,618,048 BF16 outputs (21.276%)** and touches **405,132 /
405,504 exact qrow4/head workgroups (99.908%)**. Even an omniscient, zero-
overhead linear repair model costs **20.971 + 115.285 = 136.255 ms**, versus
current exact attention **115.385 ms (0.8468×)**. All 48 candidate calls leave
exact query/KV/span/gated state unchanged, final token 2930 and lifecycle pass,
and no compiler runs. Reject both repairs, add no runtime/source surface, and
retain H6Z/H6W production
([repair-audit rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-attention-repair-audits-rejected.json)).

The next exact target is **WPF-H8A**, a bounded resident F32 cache for all 12
full-attention `attn_q` and 12 `attn_output` Q5 tensors. The existing exact
coltile16 producer builds each immutable **75,497,472-byte** plane once at
session setup; requests retain the same activation pack and H7G consumer while
skipping **24 producer launches / 5.596 ms**, modeling **2,286→2,262**
dispatches. The cache is exactly **1,811,939,328 bytes (1.6875 GiB)**. A live
owner+child+24-buffer feasibility run completed exact M512 with token 2930,
**4.167 GB** free, zero compiler, and full lifecycle recovery. This is target
selection, not a speed claim. Freeze an all-or-nothing RED owner and complete
24-plane/state/topology/both-clock gates before implementation; the full Q5
family and every layer/prompt subset remain forbidden
([H8A target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A now qualifies as a bounded default-off owner. Its Python-only implementation
keeps the HIP source/object byte-identical and publishes one immutable **24-plane
/ 1,811,939,328-byte** raw-pointer map. Every real plane matches a fresh retained
producer over all **75,497,472 bytes**; complete M512 is KL0/top-1 100% and
byte-exact across all 48 hidden boundaries, logits, final/post hidden, KV/spans,
repeat, and teardown. The named request proves **24 setup / 0 request producers
+ 24 target packs + 24 H7G consumers / 2,262 application dispatches** on one
queue/stream with zero compiler. Fixed C4096/M512 improves **436.765→438.368
tok/s (+0.367%, 5/5 paired wins)**; clean 512/1K/4K medians improve
**+0.748%/+0.332%/+0.257%**, each with 3/3 paired wins and exact state
([H8A runtime candidate](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-runtime-candidate.json) ·
[H8A target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A is now retained gfx1100 source production at clean commit `c4ea62347`.
Selector-unset source qualification repeats complete 24-plane/state exactness,
improves fixed transient-H7G rollback **435.272→437.286 tok/s (+0.463%, 5/5)**,
and wins clean 512/1K/4K by **+0.290%/+0.142%/+0.215%**. The frozen clean
post-commit wall is **440.353 tok/s** from **440.298/441.248/440.550/440.034/
440.353**, all exact token 2930 and lifecycle-clean. Five-request tracing records
**2,262 dispatches**, **1,151.215-ms** representative kernel sum, and
**1,174.598-ms** median span with exact **24 setup / 0 request coltile16
producers**, one queue/stream, and zero compiler activity. This is **+0.724%**
over retained H7U/H6Z/H6W **437.189 tok/s**, **+159.771%** over campaign start,
and **1.56872×** behind matched llama.cpp HIP **690.791 tok/s / 714.008 ms**.
The remaining kernel gaps are Q5 **173.395 ms**, IQ-down **120.186**, attention
**94.231**, and Q6 **59.985**
([H8A production](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-production.json)).

The post-H8A exact audit selects **WPF-H8B scoped activation-pack reuse** without
changing a device body, arithmetic, plane, allocation, or source owner. One
cache-only natural-M512 request records **330** H5Y/H6U tile-K-row packs and
proves **107** are consecutive byte-identical recomputations inside complete
immutable projection groups: 12 full-attention Q/K/V triples remove 24, 35 SWA
K/V pairs remove 35, 46 shared-Q5 gate/up pairs remove 46, and the dense-Q5 plus
layer-47 shared-Q6 pairs remove two. The complete target models **330→223 packs
/ 2,262→2,155 dispatches** and **2.342 ms** removable profile time; its
zero-overhead **441.242 tok/s (+0.202%)** ceiling is not a candidate claim. The
audit remains exact at token2930/position511 with finite state, lifecycle
recovery, and zero compiler. Freeze scope/key/failure semantics and all 95
recurrence runs together before implementation; no layer, role, prompt,
length, or favorable-rerun subset is admissible
([H8B target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B now qualifies as a bounded default-off owner with no device, arithmetic,
allocation, workspace, or H8A ownership change. One generic scope-local cache
publishes only a successfully produced exact pointer/geometry/stream key and
owns the complete attention-Q/K/V, dense-gate/up, and shared-expert-gate/up
class set. Complete M512 remains exact at token2930/position511 and executes
**223 packs (24 resident + 199 transient)**. The named trace proves exact
**330→223 packs / 2,262→2,155 application dispatches**, unchanged non-pack
kernel names/counts, one queue/stream, and zero compiler. Fixed C4096/M512
improves **438.412→438.919 tok/s (+0.116%, 4/5 paired wins)**; clean
512/1K/4K medians improve **+0.148%/+0.175%/+0.152%**, all exact. Source
remains H8A **440.353 tok/s** pending a separately frozen H8B source-default
gate
([H8B runtime candidate](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-runtime-candidate.json) ·
[H8B target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B is now retained gfx1100 source production at clean commit `6b9411b15`.
Selector-unset qualification preserves exact M512 state and **223 packs / 2,155
dispatches**, improves fixed disabled rollback **438.114→439.243 tok/s
(+0.258%, 5/5)**, and wins clean 512/1K/4K by
**+0.109%/+0.0097%/+0.055%**. The frozen post-commit wall is **440.893
tok/s** from **440.893/441.722/441.411/440.829/439.543**, all exact and
lifecycle-clean. Five-request tracing records **1,146.420-ms** median kernel
sum / **1,166.621-ms** span, **2,155 dispatches**, one queue/stream, and zero
compiler activity. This is **+0.122%** over H8A **440.353 tok/s**, **+160.089%**
over campaign start, and **1.56680×** behind matched llama.cpp HIP **690.791
tok/s / 714.008 ms**. The remaining kernel gaps are Q5 **172.115 ms**,
IQ-down **119.303**, attention **93.837**, and Q6 **58.652**
([H8B production](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json)).

WPF-H8C's exact dual-weight shared-Q5 gate/up leaf passes its first-object and
correctness gates: local128/runtime-VGPR136/LDS1KiB/scratch0, one physical
activation load for two weight streams, and byte-exact gate/up outputs at
rows17/33/M512. The binding complete-class screen nevertheless rejects it.
All **46/46** real layer pairs are byte-exact and finite, but only **14/46**
win both clocks; summed H7H→H8C event time is **27.8051→27.8323 ms
(0.9990×)** and synchronized wall is **28.0210→28.0053 ms (1.0006×)**.
The frozen no-salvage rule therefore removes the leaf, capabilities, and RED
surface before runtime wiring. Production remains H8B at **440.893 tok/s /
2,155 dispatches**, **1.56680×** behind matched llama.cpp HIP **690.791
tok/s**
([H8C rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-shared-q5-dual-consumer-rejected.json) ·
[H8C target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8b-shared-q5-dual-consumer-target.json)).

**WPF-H8D complete-class exact-value Q6 F32 SGEMM is rejected before target
publication.** A cached all-six-shape/**144-call** M512 screen compares the
current H6U/H7I/H5I controls with exact Q6-to-F32 expansion, exact BF16-to-F32
widening, rocBLAS SGEMM, and BF16 result casting where required. Five shapes
win both clocks and the diagnostic aggregate improves **74.099→40.969 ms
(1.809×)** by HIP events and **74.469→41.232 ms (1.806×)** by synchronized
wall. F32 K3072×N72 nevertheless regresses **0.03965→0.09167 ms (0.4325×)**
event and **0.04260→0.09559 ms (0.4456×)** wall. The frozen complete-class
rule forbids a post-screen 143-call subset, so no H8D target, RED, runtime, or
576-step quality gate follows. Production remains H8B **440.893 tok/s / 2,155
dispatches**
([H8D rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-k-f32-sgemm-complete-class-rejected.json)).

**WPF-H8E quality-selected F32 hipBLASLt attention algorithms are rejected at
the complete quality gate.** A source-unchanged synthetic screen enumerates all
**4×4 QK/PV heuristics** over eight global/SWA shapes: all **128** combinations
are finite and each shape has four numerical classes. A fixed closest-output
rule changes five of six H5B mappings and models exact→candidate
**109.065→85.822 ms** by HIP events. Matched C4096/M512 then reaches token2930,
KL **0.000231**, deterministic repeat, and warmed **440.193→445.968 tok/s**.
The binding 18-prompt/**576-step** gate executes all **10,512** expected
candidate stacks but reaches maximum KL **0.391103 > 0.05** at **563/576**
top-1. Diagnostic suite prefill improves **328.443→429.801 tok/s (1.3086×)**,
but is not retainable. Add no target, RED, runtime map, or source policy; do not
favorably rerun another numerical class. Production remains H8B **440.893
tok/s / 2,155 dispatches**
([H8E rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-quality-selected-f32-attention-algorithms-rejected.json)).

**WPF-H8F targets exact resident shared-Q5 F32-plane publication.** The clean
H8B trace contains exactly **92** K3072×N1024 coltile8 producers per request,
**3.439745 ms** median across five requests, before the unchanged 92 H7H
consumers; H8B already reduces their activation packs to 46. A source-unchanged
live audit holds all **92 × 12,582,912 = 1,157,627,904 bytes (1.078125 GiB)**
alongside H8A, completes M512 at token2930, and leaves **2.802734 GiB** free.
The target extends the immutable resident map **24→116** planes and models
**2,155→2,063** request dispatches without changing consumer arithmetic. This
is target/ceiling evidence only, not a speed claim: freeze the complete
92-tensor class before implementation and forbid partial-plane, layer, role,
prompt, token, route, length, or favorable-rerun salvage
([H8F target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8e-resident-shared-q5-f32-cache-target.json)).

**WPF-H8F is rejected by its immutable clean-length gate.** The Python-only
owner passes all **116/116** complete-plane comparisons, KL0 and exact
logits/**48/48** hidden boundaries/KV/repeat, and named tracing proves **24+92**
setup producers followed by **0** shared request producers, unchanged **46**
shared packs/**92** H7H consumers, and **2,155→2,063** dispatches. Fixed
C4096/M512 improves **439.301→439.811 tok/s (+0.1162%, 5/5)**. The binding
first 512/1K/4K run improves 512 **406.734→408.125 (+0.3421%)** but regresses
1K **322.929→322.700 (-0.0710%)** and 4K **198.6149→198.6115 tok/s
(-0.00171%)**. The no-length-subset/no-rerun rule removes the owner, registry,
capabilities, backend exclusion, and RED test; H8B remains production at
**440.893 tok/s / 2,155 dispatches**, **1.56680×** behind matched llama.cpp HIP
([H8F rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-shared-q5-f32-cache-rejected.json)).

**WPF-H8G existing exact global-qrow6 transfer is rejected before target
publication.** A cached complete W7900 screen compares the already-registered
sibling-RDNA3 qrow6 body with H6N at start128 and H6Z at starts256/384. Qrow6
wins both clocks only at start128 (**1.0492× event / 1.0362× wall**), differs
from the current exact output at every start (**730,971–749,888 F32-bit
mismatches**), and loses both late starts. The 36-call aggregate regresses
**15.869→21.545 ms event (0.7365×)** and **16.078→21.803 ms wall (0.7374×)**.
The complete-class/no-subset rule forbids retaining start128 alone; add no
H8G target, RED, runtime owner, or source policy. H8B remains production at
**440.893 tok/s / 2,155 dispatches**, **1.56680×** behind matched llama.cpp HIP
([H8G rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-global-qrow6-transfer-rejected.json)).

**WPF-H8H exact prefill attention+softplus dual publication is rejected at
its first-object physical gate.** All four separately named kernels execute
with F32-bit-exact context, BF16-byte-exact gated output, unchanged complete
`KVLiveSpans`, clean lifecycle, and zero compiler processes. H6N stays within
its ceiling at runtime VGPR **40≤48**, but H6Z is **88>56**, H6A **80>72**, and
H6W **80>64**. The frozen no-resource-rewrite/no-subset rule therefore skips
leaf timing and removes all four kernels, wrappers, keys, gfx1151 exclusions,
and the RED test. The registered H6N/H6Z/H6A/H6W plus standalone-gate fallback
remains production at H8B **440.893 tok/s / 2,155 dispatches**, **1.56680×**
behind matched llama.cpp HIP
([H8H target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8g-prefill-attention-softplus-dual-publication-target.json) ·
[H8H rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-prefill-attention-softplus-dual-publication-physical-rejected.json)).

**WPF-H8I exact stream-ordered Q5 partition accumulation is rejected and
removed.** Its **24** stage specializations pass complete partition/final-bit
identity and physical admission at local32, zero LDS/scratch/spills, and
runtime VGPR maxima **80/200/168/200/168/168** within all six ceilings. The
binding actual-weight screen nevertheless loses both clocks for every role;
the weighted 188-call aggregate regresses **222.555→289.013 ms event
(+29.861%)** and **225.438→286.922 ms wall (+27.273%)**. The no-subset/no-rerun
rule removes all HIP, wrapper, registry, gfx1151-exclusion, and RED surfaces.
H8B remains production at **440.893 tok/s / 2,155 dispatches**, **1.56680×**
behind matched llama.cpp HIP
([H8I target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8h-streamed-q5-partitions-target.json) ·
[H8I rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-stream-ordered-q5-partitions-rejected.json)).

**WPF-H8J exact IQ3 four-workgroup occupancy is the next target-only leaf.**
H6T owns **45 calls / 264.377 ms**, or **96.783%** of current IQ-down. At
local128/runtime-VGPR104, its four-wave workgroups fit only **3 complete
workgroups / 12 resident waves per SIMD**. One separately named, otherwise
identical `launch_bounds(128,4)` sibling must compile at **≤96 VGPR** and zero
scratch to admit **4 workgroups / 16 waves**. This **+33.333% resident-wave**
model is not a speed result. Freeze RED before source changes, preserve every
H6T operation and byte, and require all **45/45** actual layers plus the
aggregate to win both clocks in one immutable screen; no launch-bound sweep,
layer subset, rewrite, recompile, or favorable rerun is admissible
([H8J target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8i-iq3-four-workgroup-occupancy-target.json)).

**WPF-H8J is rejected at its first-object physical gate.** AMD clang preserves
H6T's exact **7,920-byte / 1,384-slot** instruction stream (apart from two
PC-relative table relocations) and its **VGPR101 / SGPR78 / LDS384 / spill0**
metadata despite `launch_bounds(128,4)`. Because **101 > 96**, the object still
fits only **3 complete four-wave workgroups per SIMD**, not the required four.
The no-rewrite rule skips correctness, runtime tracing, and timing; every H8J
source/RED surface is removed and H8B production remains **440.893 tok/s / 2,155
dispatches**, **1.5668×** behind matched llama.cpp HIP
([H8J rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-four-workgroup-occupancy-physical-rejected.json)).

**WPF-H8K exact IQ3 uniform-rowbatch4 triple-output ownership is the next
target-only leaf.** Frozen natural-M512 routing across all **45** IQ3 layers has
**230,400 useful rows**. H6T rowbatch8 executes **33,547 epochs / 268,376 row
slots**, including **37,976 padded slots (14.150%)**; fixed rowbatch4 executes
**61,546 epochs / 246,184 slots**, including **15,784 padded slots (6.411%)**.
Thus the exact operation removes **22,192 compute slots (−8.269%)** and halves
the explicit activation-plus-accumulator liveness model **40→20 dwords**, while
openly adding **83.462% row epochs** and **57,341,952 modeled barrier epochs**.
This is not a speed result. Freeze RED before source changes; require ≤96 VGPR,
complete H6T/CPU bytes, and every **45/45** layer plus aggregate to win both
clocks with no rowbatch/layer/routing/rerun salvage
([H8K target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8j-iq3-rowbatch4-triple-output-target.json)).

**WPF-H8K is rejected at its frozen named-trace resource gate.** The exact
rowbatch4 body passes complete edge/CPU bytes and first-object physical
admission at metadata **VGPR70 / SGPR58 / LDS192 / spill0**, with code/slots
**4,916/882**, **19 loads / 108 FMAs / 12 permlanex16 / 48 DPP adds**, and zero
scratch. A cache-only natural-M512 trace returns exact production state and
names all **45 H8K + 2 IQ4** calls at **2,155 dispatches**, local128,
runtime-VGPR72, and scratch0, but reports runtime LDS **512 > frozen 256 B**.
Honor the no-resource-gate-rewrite rule: skip all-layer timing and runtime/source
work, remove candidate plus RED, and retain H8B **440.893 tok/s**, **1.5668×**
behind matched llama.cpp HIP
([H8K rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-rowbatch4-triple-output-runtime-lds-rejected.json)).

**WPF-H8L exact IQ3 lossless 12-bit codebook packing is the next target-only
leaf.** H6T's 256-entry table stores four magnitudes in each uint32. Every
coordinate is one of **4/12/20/28/36/44/52/62**, so one uint16 can carry four
3-bit codes and recover each exact magnitude with
`4 + 8*code + 2*(code == 7)`. All **256/256** entries reconstruct exactly and
remain unique while storage falls **1,024→512 bytes (−50%)**. Across all **45**
IQ3 layers, the immutable operation model changes **105,529,737,216→52,764,868,608
logical codebook bytes** while preserving H6T's raw weights, 216 FMAs, exchange/
DPP reductions, LDS publication, ownership, and outputs. This is neither cache-
traffic nor speed evidence. Freeze RED before source changes; require exact
H6T/CPU/all-45 bytes and every layer plus aggregate to win both clocks with no
representation/layer/routing/rerun salvage
([H8L target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8k-iq3-codebook12-target.json)).

**WPF-H8L is rejected by its sole immutable all-45 screen.** All 256 entries,
edge/CPU outputs, and **45/45** actual layers are byte-exact. The first object
passes at **8,740 B / 1,523 slots / VGPR111 / LDS384 / spill0** and realizes the
frozen **8 b128 + 3 b32 + 12 d16** load shape; an exact-state trace names **45
H8L / 0 H6T / 2 IQ4** calls at runtime **VGPR112/LDS512/scratch0**. Timing is
decisively negative: **0/45** layers win both clocks and aggregate event/wall
regresses **260.044/260.757→290.496/290.437 ms (+11.710%/+11.382%)**. Honor the
no-width/layer/rerun rule, remove candidate plus RED, and retain H8B **440.893
tok/s**, **1.5668×** behind matched llama.cpp HIP
([H8L rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-codebook12-all45-timing-rejected.json)).

**WPF-H8M exact IQ3 sign-folded BF16 codebook is the next target-only leaf.**
The current H6T object contains **24 `v_cmp_eq_u32` + 24 `v_cndmask_b32`**
sign-application sites across its triple-output body. Independent enumeration
builds **4,096/4,096 exact and unique** uint64 records indexed by one sign
nibble plus one grid byte; each record carries four signed magnitudes as BF16.
This expands the table **1,024→32,768 bytes** and the all-45 logical table-byte
model **105,529,737,216→211,059,474,432 (+100%)** at unchanged wave-load count,
while targeting removal of all 48 sign sites. It is an explicit byte-for-ALU
trade, not cache-traffic or speed evidence. Freeze RED before source changes;
require the fixed **six-b64** load shape, no sign cmp/select sites, non-growing
VGPR/resources, exact H6T/CPU/all-45 bytes, and every layer plus aggregate to
win both clocks without dtype/layout/cache/layer/rerun salvage
([H8M target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8l-iq3-signed-bf16-codebook-target.json)).

**WPF-H8M is rejected at its frozen first-object metadata-VGPR gate.** The
complete four-node matrix is exact and the sole object realizes the prescribed
**8 b128 + 3 b32 + 6 b64 + 6 d16_b16** loads, eliminates all **24 compare +
24 select** sign sites, and preserves 216 FMAs, reductions, LDS, barriers, and
stores. Code/slots improve **7,920/1,384→7,768/1,321**, but metadata VGPR grows
**101→102**, violating the non-growing **≤101** contract. Honor the no-rewrite/
recompile/resource-relaxation rule: skip trace and timing, remove candidate plus
RED, and retain H8B **440.893 tok/s**, **1.5668×** behind matched llama.cpp HIP
([H8M rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-signed-bf16-codebook-physical-rejected.json)).

**WPF-H8N exact Q5 twin-team weight staging is the next target-only leaf.** Q5
remains the largest matched gap at **230.429 vs 58.314 ms (+172.115 ms)**; its
six H7G/H7H consumers own **188 calls / 203.861 ms**. One fixed local256 body
pairs two independent logical local128 teams: team 0 loads each exact F32
K128×COL slab into ping-pong LDS, then both teams consume it for adjacent row
groups while preserving every scalar FMA and reduction/store bit. Across the
complete role class, modeled logical F32-plane bytes fall
**807,571,292,160→407,862,509,568 (−49.495%)** and workgroups fall
**5,021,440→2,689,792 (−46.434%)**, at unchanged **1,433,445,335,040 FMAs**.
The cost is explicit: dynamic barrier epochs grow **5,021,440→90,764,032
(18.075×)** and fixed LDS is **9,216–18,944 bytes**. This is no cache-traffic or
speed result. Freeze RED before source work; all six roles and their weighted
aggregate must win both clocks without role/geometry/buffer/recompile/rerun
salvage
([H8N target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8m-q5-twin-team-weight-staging-target.json)).

**WPF-H8N is rejected by its sole immutable six-role timing screen.** Complete
edge/control/CPU bytes pass **5/5**; the five object instances meet all frozen
VGPR/LDS/spill/scratch limits, and a compiler-free trace names all six local256
roles. The intended traffic model is physically realized but loses decisively:
**0/6** roles win both clocks. The weighted 188-call H7G/H7H→H8N aggregate
regresses event **212.742→370.566 ms (+74.186%)** and synchronized wall
**224.095→365.407 ms (+63.059%)**. Honor the no-role/geometry/buffer/K-tile/
rewrite/recompile/rerun rule, remove candidate plus RED, and retain H8B
**440.893 tok/s / 2,155 dispatches**, **1.5668×** behind matched llama.cpp HIP
([H8N rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-twin-team-weight-staging-rejected.json)).

**WPF-H8O exact gfx1100 after-router least-priority MoE branch concurrency is
rejected at its binding fixed gate.** The exact two-queue schedule remains
byte-identical across all 48 hidden boundaries, complete logits, final/post
hidden, K/V plus `KVLiveSpans`, token 2930, and lifecycle, with zero compiler
activity. It nevertheless loses every matched pair: serial control→candidate is
**438.604→436.514 tok/s (-0.4765%, 0/7 wins)** versus the required positive
median and ≥5/7. Honor the frozen no-schedule/priority/queue/boundary/length/
rerun salvage rule: skip trace and 512/1K/4K gates, remove the temporary
descriptor and RED, leave all three gfx1100 production capabilities false, and
retain one-queue H8B **440.893 tok/s / 2,155 dispatches**, **1.5668×** behind
matched llama.cpp HIP
([H8O rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-after-router-low-priority-moe-concurrency-rejected.json) ·
[H8O target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8n-moe-shared-after-router-low-priority-target.json)).

**WPF-H8P lossless Q5 signed-int16 power-of-two plane is rejected before a
GPU target.** The proposed 520-byte block would replace 1,024 exact F32 bytes,
but a real `blk.0.attn_q.weight` value is `0x3d72fd00 = 62205 × 2^-20`.
Because 62,205 is odd, no larger binary exponent is exact; because it exceeds
32,767, signed int16 cannot hold it even with one exponent per value. A
16,777,216-value audit confirms **46.132%** of individual values fail, so
subblock splitting cannot repair the invariant. No source, RED, compiler, GPU,
or timing path was created; retain H8B **440.893 tok/s**
([H8P analytical rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-fixed16-power2-plane-analytical-rejected.json)).

**WPF-H8Q exact Q6 int16-product plus tiled-F32-scale plane is rejected at
the first-object physical gate.** The transient implementation passes **15/15**
correctness tests across all three roles, rows 17/33/512, sampled CPU values,
and exact **−4,064/+4,096** product planes. Its local64 producer passes at
**VGPR14/LDS0/spill0**, and each consumer is spill/scratch-free with the intended
uniform scale load, but consumer metadata VGPR is **169/136/169** versus frozen
runtime ceilings **160/128/160**. Metadata already exceeds every ceiling, so no
named trace or timing screen is admissible. Honor the no-resource-rewrite/
recompile/rerun salvage rule; remove candidate and RED, retain H8B
**440.893 tok/s / 2,155 dispatches**, and make no speed claim
([H8Q rejection](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-int16-product-plane-physical-rejected.json) ·
[H8Q target](benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8p-q6-int16-product-plane-target.json)).

**Future W7900 Laguna parity implementation is tabled.** The canonical
[status and resumption handoff](docs/LAGUNA-PARITY-STATUS.md) records the final
campaign/current/llama.cpp table, retained gain timeline, closed H8C-H8Q ladder,
and mandatory high-leverage restart policy. No H8R target is selected.

Both short
  rows exceed 150 tok/s and H6E production 4K remains positive; 16K+ stays closed below
  the 800/700 stretch target
  ([post-H5Z matched residual / H6A target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5z-matched-residual.json) ·
  [H5Z production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-production.json) ·
  [H5Z candidate](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-candidate.json) ·
  [post-H5Y matched residual / H5Z target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5y-matched-residual.json) ·
  [H5Y production](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-production.json) ·
  [H5Y candidate](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-candidate.json) ·
  [post-H5X matched residual / H5Y target](benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5x-matched-residual.json) ·
  [preceding H5X production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-production.json) ·
  [H5X candidate](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-candidate.json) ·
  [H5X target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5w-residual.json) ·
  [H5W production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-production.json) ·
  [H5W candidate](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-candidate.json) ·
  [H5W target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-target.json) ·
  [H5V rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-rejected.json) ·
  [H5V target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-target.json) ·
  [H5U runtime rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-runtime-rejected.json) ·
  [H5U global cached-source leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-candidate.json) ·
  [H5U target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-target.json) ·
  [H5T IQ3 rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-rejected.json) ·
  [H5T target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-target.json) ·
  [H5S rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-persistent-row-group-rejected.json) ·
  [post-H5R residual / H5S target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5r-residual.json) ·
  [H5Q production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-production.json) ·
  [H5R SWA leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-candidate.json) ·
  [post-H5Q residual / H5R target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5q-residual.json) ·
  [H5Q leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-candidate.json) ·
  [H5Q target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-target.json) ·
  [H5P rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-runtime-rejected.json) ·
  [H5P leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-retune-candidate.json) ·
  [H5P target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-retune-target.json) ·
  [H5O rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-factorized-exact-plane-rejected.json) ·
  [H5O target](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-factorized-exact-plane-target.json) ·
  [H5N runtime rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-dense-first-fill-runtime-rejected.json) ·
  [H5N leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-dense-first-fill-exact-candidate.json) ·
  [post-H5M residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5m-residual.json) ·
  [H5M leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-sourcequal-exact-candidate.json) ·
  [post-H5L residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5l-residual.json) ·
  [H5L production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) ·
  [H5L leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-candidate.json) ·
  [H5J production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) ·
  [post-H5K residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5k-residual.json) ·
  [H5K rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json)).
  A separately registered WPF-H2 F16-WMMA FlashAttention leaf keeps BF16 K/V
  and complete `KVLiveSpans` while moving the standalone 12-global/36-SWA M512
  family **490.919 -> 21.719 ms (22.603x)**, nominally matching llama.cpp's
  **21.725-ms** trace. Runtime promotion is rejected: the complete
  18-prompt/576-step gate reaches max KL **1.804860 > 0.05** despite **564/576**
  top-1 and **1.027x** diagnostic prefill. The temporary runtime path is removed
  and production stays exact
  ([rejection](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-source-flash-attention-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-source-flash-attention-candidate.json)).
  The separately registered WPF-H3 IQ3/IQ4 source-MMQ leaf moves all 47 actual
  M512 selected-down layers **565.437 -> 115.951 ms (4.877x)**; IQ3 alone is
  **27.145% below** llama.cpp's matched family trace. Runtime promotion is
  rejected: complete quality reaches max KL **0.373028 > 0.05** despite
  **567/576** top-1 and **1.192x** diagnostic prefill, while an IQ3-only source
  followup still reaches **0.372917**. The temporary owner is removed and
  production stays exact
  ([rejection](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-candidate.json)).
  The separately registered WPF-H4 Q6 F16/rocBLAS leaf moves the six-shape,
  144-call M512 family **174.351 -> 14.349 ms (12.151x)**, **3.825% below**
  llama.cpp's matched **14.919865-ms** stack. Runtime promotion is rejected:
  complete changed-arithmetic quality reaches max KL **0.338657 > 0.05** despite
  **567/576** top-1 and **1.042x** diagnostic prefill. The temporary
  97,517,568-byte owner is removed and production stays exact
  ([rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-candidate.json)).
  A clean post-H4 apples-to-apples refresh measures exact hipEngine
  **169.516 tok/s** versus llama.cpp HIP **694.184 tok/s (4.095x)**. Its cached
  trace is **3,001.692/3,016.780 ms** kernel sum/span with only **0.500%**
  outside kernels; Q5 exact coltile alone owns **1,270.458 ms / 42.325%**.
  WPF-H5A's separately registered exact-value F32 Q5 producer/SGEMM leaf now
  moves the role-qualified 235-call family **1,256.936 -> 221.137 ms
  (5.684x)** by HIP events, corroborated by **5.273x** synchronized wall. The
  regressive N48 gate remains exact fallback; all candidate outputs are finite
  at max mean KL **1.59e-9** and top-1 **100%**. Its default-off owner passes
  natural M512 at KL **0.0003742**, but the binding 18-prompt/576-step lane
  rejects SGEMM reassociation at maximum KL **1.143627 > 0.05** despite
  **564/576 (97.917%)** top-1 and diagnostic prefill **152.359 -> 202.707 tok/s
  (1.330x)**. The owner/workspace/selector are removed and exact production is
  unchanged. H5B's existing packed F32 dense-initial hipBLASLt attention route
  clears its W7900 transfer screen: tuned selected-context leaf timing is
  **109.897 -> 62.655 ms (1.754x)**, natural M512 passes KL **0.000429** / top-1
  **100%**, and cached request tracing cuts attention **488.304 -> 60.669 ms
  (8.049x)** plus complete kernel sum **3,001.692 -> 2,603.520 ms (-13.265%)**.
  The binding extension preserves all 18 natural prompts as M512 suffixes and
  observes all **10,512** expected package-mapped candidate launches. It rejects
  QK/PV reassociation at maximum KL **0.444675 > 0.05** despite **564/576
  (97.917%)** top-1, deterministic repeats, lifecycle recovery, and diagnostic
  prefill **165.555 -> 190.103 tok/s (1.148x)** with every category positive.
  The gfx1100 capability/map/owner seam is removed; exact production remains.
  H5C/H5D then returns to exact Q5 arithmetic: a transient exact-value weight
  expansion feeds local128 ordered **8x4/4x8** consumers that preserve coltile
  K/FMA/wave/store order byte-for-byte. H5E extends that invariant to
  **4x16/8x8/16x4**, owns all eight roles, and removes universally regressive
  1x64/2x32. The final-source 235-call gate moves H5D weighted event/wall
  **1,085.630/1,040.166 -> 951.876/961.993 ms (-12.320%/-7.515%)** with the same
  bounded 150,994,944-byte plane and no persistent sidecar. H5F's 12x4 N48
  micro-policy saves another **4.224/1.989 us** per M512 request. H5G retains
  exact 8x10/16x5/8x12/12x8 on five roles; its strong changed-role gate cuts
  H5F **8.639%/7.479%** by event/wall and traces at VGPR168/200 with zero
  scratch. The H5G package-default route remains KL0/byte-exact across all 48
  boundaries, logits, K/V, repeats, and lifecycle; H5I reuses that plane for
  exact Q6 and publishes **191.713/178.080/134.411 tok/s** at 512/1K/4K. H5H
  removes all larger Q5 candidates after universal regressions and the
  constant-128 spill cliff
  ([current H5I production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-production.json) ·
  [H5G Q5 production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-production.json) ·
  [H5H boundary rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-register-boundary-rejected.json) ·
  [post-H5G residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5g-residual.json) ·
  [post-H5I residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5i-residual.json) ·
  [H5I leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-candidate.json) ·
  [H5C leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-candidate.json) ·
  [reprofile](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-exact-residual-reprofile.json) ·
  [H5A rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-rejected.json) ·
  [H5B rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-rejected.json) ·
  [H5B leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-candidate.json)).
- The pinned Poolside Laguna S 2.1 Q4_K_M target is supported on gfx1151 for
  torch-free c=1 blocking/streaming generation, Poolside-v1 reasoning/tool
  parsing, and exact source-bound cached loading. Its quality-admitted
  selector-unset production prefill reaches **354.820 tok/s** at pp512
  (**353.421/355.584/354.820** across three clean repetitions), up
  **4.655x** from the preceding 76.226 tok/s default. The complete category
  lane passes at maximum KL **0.040725**, **317/320 (99.0625%)** top-1,
  neutral decode, deterministic repeats, and exact lifecycle recovery. The
  gfx1151 package combines D8/D4 resident-T16 integer-dot expert tiles,
  row-scaled hipBLASLt source-F16 projections, Q4/Q6 WMMA dense/shared
  projections, and online-softmax qrow2 global/sliding attention; exact routes
  remain rollback paths
  ([production evidence](benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json)).
  Its matched BF16 DFlash B4
  drafter is supported only as an explicit library/OpenAI opt-in; true AR stays
  default because the canonical full-suite DFlash economics are `0.9469x` with
  heldout and non-code regressions.
- PARO BF16 KV has retained W7900 evidence through 128K, and **208 Ki is the
  recommended safe BF16 cap** on a physical 24 GB XTX. The all-layer 256K INT8
  layout fits its tracked-memory gate but fails Qwen3.6 fidelity. The milder
  six-BF16/four-INT8 native layout passes the tested GGUF accuracy gates, but
  fails PARO accuracy and quality-preserving 256 Ki request-scratch capacity.
  On gfx1151, explicit short-context uniform `int8_per_token_head` GGUF requests
  now support continuous c1/c2/c4/c8 ownership through rounded context 8192 by
  retaining bounded BF16 attention mirrors. That route is not default or
  memory-saving; tail4, direct/no-mirror INT8 attention, longer c>N INT8, and
  PARO INT8 remain unsupported for continuous serving. Current capacity,
  throughput, speculative-decode, and concurrency evidence is reported below
  with separate gfx1100/gfx1151 provenance and correctness gates.

This remains an alpha, single-GPU release. Production PARO native `c>1` decode
is retained only for the certified gfx1151 profile; gfx1100 remains direct-c2
only and broader PARO shapes are still gated. General app-local sessions do not
reuse resident KV, structured outputs are not grammar-constrained decoding, and
the server MTP route is explicit-only. See [the API limitations](docs/API.md#current-limitations)
and [concurrency status](docs/CONCURRENCY.md#current-answer) for the exact
boundaries.


## Hardware targets

| Backend | Hardware | Status |
| --- | --- | --- |
| `cpu_reference` | Any CPU, numpy | Correctness oracle; CI without GPU |
| `hip_gfx1100` | AMD Radeon Pro W7900 / RX 7900 XTX (RDNA3) | Active backend |
| `hip_gfx1151` | AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, RDNA3.5) | Active backend |
| `cuda_sm86` | NVIDIA Ampere consumer (3090-class) | Planned peer backend |

`backend="auto"` is the public API/server default. It maps exact `gfx1100` and
`gfx1151` detections to the matching HIP backend; unknown ROCm targets warn and
select `cpu_reference` where a CPU implementation exists. Users on nearby targets
such as `gfx1101`/`gfx1102` can force a backend with `backend="hip_gfx1100"`,
`--backend hip_gfx1100`, or `HIPENGINE_BACKEND=hip_gfx1100` after validating
correctness/performance.

On gfx1151, hipEngine sets `GPU_MAX_HW_QUEUES=1` before HIP loads because it
reduces a retained gfx11 low-power queue failure and is non-regressive at short
context. It is not a repeated-128K lifecycle guarantee: current production can
still hit the firmware/scheduler stall. A matched follow-up reproduces the stall
under both HIP 7.15 and HIP 7.13, so downgrading ROCm is not a safe workaround.
Explicit values are preserved; set `GPU_MAX_HW_QUEUES=4` before process start to
restore ROCm's documented default for diagnosis. gfx1100 is unchanged. See
[`docs/ENVS.md`](docs/ENVS.md) and the
[cross-stack lifecycle artifact](benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json).

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Memory Usage

The clean 2026-07-13 profile-aware BF16 frontier (`5a49b16d`) directly tests
the current Qwen3.6 packed PARO model on a physical 24 GB gfx1100 card. The
automatic low-memory prefill profile makes **208 Ki the recommended safe BF16
cap** with 0.361 GiB observed headroom; 220 Ki completes but leaves only about
78 MiB and is edge-only. Separately, compact 256K all-layer INT8 (`d6504544`)
fits its tracked layout gate but fails fidelity. Native tail-four mixed KV saves
18.75% of K/V and passes the tested GGUF accuracy gates, but PARO accuracy and
the quality-preserving 256 Ki XTX request-scratch allocation reject. Accordingly,
256K remains diagnostic allocation capacity—not a supported route.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route / profile | Hardware | Context/decode | Tracked peak | Observed device peak | Device/card margin | Capacity / quality status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PARO BF16 KV reference | W7900, default chunks | 128K/128 | **22.124 GiB** | 21.107 GiB phase sample | n/a | Reference path |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | **208 Ki (212,992)/128** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended practical safe cap** |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | 220 Ki (225,280)/128 | 23.369 GiB | **23.908 GiB** | **+0.076 GiB (~78 MiB)** | Physical pass, but **edge only—not safe cap** |
| PARO BF16 KV, default 48 GB-card profile | W7900 | 220 Ki (225,280)/128 | 24.090 GiB | at least 24.832 GiB | at most -0.848 GiB vs 24 GB card | Rejected for this larger-chunk profile |
| PARO INT8 per-token/head KV, FP16 scales | W7900 | 256K/128 | **22.971 GiB** | 21.041 GiB phase sample | +1.029 GiB tracked | **Rejected** by Qwen3.6 matched-context and task gates |
| PARO tail-four Hadamard-group32 mixed KV, BF16-oracle prefill | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.469 GiB before failed allocation** | 22.566 GiB after clean OOM | 1.418 GiB free before request scratch; insufficient | **Rejected:** `HIP error 2` OOM and PARO fidelity failure; no segfault |
| PARO tail-four Hadamard-group32 mixed KV, direct-streaming control | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.290 GiB** | **23.590 GiB** live sample | **+0.394 GiB** live | Allocation passes, but direct packed prefill is **correctness-rejected** |

The native explicit `tail4_hadamard_group32` layout keeps K/V for
full-attention layers `3,7,11,15,19,23` in BF16 and stores only layers
`27,31,35,39` as Hadamard-group32 INT8 with FP16 scales. At 262,400 retained
rows it uses `4,366,336,000` K/V bytes—**18.75% below BF16**—with no persistent
BF16 shadow. PARO's quality-preserving prefill uses a temporary BF16 oracle;
GGUF's post-quality layout audit reports zero persistent oracle/mirror buffers.
Native PARO still fails 1/11 prompts at 512/8 and 2/11 at 4K/16 (58.82%
worst-prompt top-1), and its 256 Ki quality-preserving request scratch OOMs.

The clean `c971262f` therock-7.15 GGUF-only closure passes all 11 prompts at
512/8 (max KL `0.007455`, top-1 100%) and 4K/16 (mean/max KL
`0.0001369/0.009926`, aggregate/minimum-prompt top-1 `99.47%/94.12%`) plus
bounded `mixed_v1` at 128K/16 (max KL `5.19e-5`, top-1 100%). At 128K,
persistent K/V is `2,185,297,920` bytes versus BF16 `2,689,597,440` bytes and
live owned memory falls `24.168 -> 23.698 GiB`. It still rejects promotion:
production 4K prefill/decode regress `0.67%/0.75%`, one-shot 128K decode
regresses `3.82%`, and production prefill allocates then frees
`1,075,838,976` bytes—byte-exact to four BF16 layer caches—raising allocator
high water `24.168 -> 24.700 GiB`. The transient attribution is inferred from
the exact bytes; it is not a persistent shadow. The policy remains explicit
and non-default. Evidence:
`benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json` and
`benchmarks/results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json`.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The INT8 layout retains 2,686,976,000 payload bytes plus 20,992,000 FP16 scale
bytes across ten full-attention layers and no BF16 K/V shadow. Its final
BF16-reference-token matched 128K/16 gate rejects at mean/max KL
`0.85128/4.97382` and 41.18% top-1 agreement. Format and mixed-policy screens
did not find a candidate that transferred through 4K.

The former llama.cpp Q8_0 pass is now a repeated-token saturation control, not
representative quality evidence. On identical Q4_K_M weights at exact mixed
4K/16, native Q8_0 rejects at mean/max KL `0.075654/1.26009` despite 94.12%
top-1; F16/F16 is exactly zero. K-only and V-only Q8 reach `0.096682` and
`0.243219` mean KL, while full Q8 benefits from non-additive K/V cancellation.
The repeated full-Q8 control is only `0.00000619` KL, confirming prompt content
as the dominant difference.

hipEngine shows the same protocol effect. Host per-head/group32/Hadamard all
pass repeated 4K/16 near `0.000002` KL but reject mixed at
`0.12779/0.28106/0.25180`. Pure native per-head INT8 rejects mixed at
`0.19038/2.99555`, 88.24% top-1, with all ten layers INT8 and no BF16 mirror.
Direct arithmetic is therefore not a universal fidelity repair. The separate
same-weight hipEngine-GGUF-BF16 versus llama.cpp-F16 bridge preserves 100%
top-1; its `0.26606` all-position mean KL is prompt-final dominated, while 16
decode rows average `0.000510`.

The original five-category free-generation reference is unscorable. In the
replacement restricted-choice diagnostic, INT8 flips one of two
BF16-qualified 4K answers (multihop `D -> C`) but retains all three qualified
32K answers. This shows that large KL can change a bounded functional decision
without implying every answer changes; it remains partial evidence, not support
for 256K INT8. Memory was measured once; timing is diagnostic.

See the
[`capacity/fidelity outcome`](benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json),
[llama.cpp repeated-token Q8_0 control](benchmarks/results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json),
[repeated/mixed prompt and native arithmetic isolation](benchmarks/results/2026-07-13-w7900-gguf-q8-kv-protocol-arithmetic-isolation.json),
[same-weight GGUF bridge](benchmarks/results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json),
[bounded functional check](benchmarks/results/2026-07-13-w7900-paro-int8-kv-functional-mc.json),
[format screen](benchmarks/results/2026-07-13-w7900-paro-kv-format-ablation.json),
and [policy screen](benchmarks/results/2026-07-13-w7900-paro-kv-policy-ablation.json).

### llama.cpp configuration note

The repository has no compact artifact or source revision for the former
llama.cpp Q8_0 memory tables, so those numbers are not toplines. The tested
configuration was:

```bash
--flash-attn on -ctk q8_0 -ctv q8_0 -c 262144 -b 128 -ub 128
```

A replacement capacity table must record the GGUF fingerprint, llama.cpp
commit/build, GPU, full command, and whole-card sampling artifact.

## Model Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

**Status: retained.** The GGUF column is the clean 2026-07-16 final
selector-unset BF16-KV sweep at `28b37356` on therock HIP 7.15: independent
right-sized sessions, one discarded warmup, three measured runs, and production
graph decode. All 18 GGUF final IDs are exact and maximum prefill/decode stdev
over median is `0.658%/0.223%`. PARO and llama.cpp retain their clean July 12
protocols. PARO is W4 PARO/BF16 KV; the other columns use Q4_K_M with BF16/F16
KV, so bold values are descriptive rather than same-quant wins. GGUF prefill now
beats llama.cpp HIP at every shape and Vulkan through 64K; GGUF decode beats HIP
everywhere and is closest to Vulkan at 4K (`-2.47%`).

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 2716.648 | 2412.320 | 2627.990 |
| 1K/128 | 2995.876 | **3052.541** | 2389.670 | 2631.750 |
| 4K/128 | 2943.038 | **2953.101** | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 2078.038 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 1559.878 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 1037.378 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 92.833 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 98.148 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 100.522 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 88.240 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 76.691 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 62.669 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.228 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.295 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 21.670 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 22.234 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 22.879 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 24.168 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

W7900 row sources: [final hipEngine GGUF sweep](benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[July 12 accepted summary](benchmarks/results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[superseded July 12 hipEngine GGUF](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 correctness oracle](benchmarks/results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

> Thanks to Framework for sending a dedicated Framework Desktop Strix Halo motherboard for this profiling and tuning work.

**Status: current IOMMU-off refresh retained through 64K; repeated GGUF 128K
blocked.** The clean 2026-07-17 table at `2edbb2ee` refreshes PARO, GGUF, and
both llama.cpp backends under `amd_iommu=off`. GGUF 512-64K passes clean
provenance, finite logits, exact final IDs, and the 5% variance gate; maximum
prefill/decode stdev over median is **0.122%/0.028%**, and all 15 IDs are
`9707`.

Relative to the previous published IOMMU-on rows, the arithmetic mean change
across 11 eligible hipEngine cells is **+4.60% prefill / +6.20% decode**; GGUF
alone averages **+8.84% / +5.84%**. This is directional, not causal, because
the hipEngine revision/routing also changed; a same-commit reboot A/B remains
necessary. The setting leaves zero IOMMU groups and disables the XDNA/NPU
driver. GGUF 128K still times out after a 584.059 tok/s warmup and 583.464 tok/s
measured pass, so no stale 128K number is carried forward. Bold values remain
descriptive because quant/KV types and memory scopes differ.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1298.259 | **1395.379** | 1184.628 | 1161.498 |
| 1K/128 | 1332.199 | **1481.943** | 1192.768 | 1154.327 |
| 4K/128 | 977.252 | **1444.733** | 1148.155 | 1114.081 |
| 32K/128 | 827.350 | **1132.215** | 843.252 | 873.573 |
| 64K/128 | 690.642 | **892.663** | 632.774 | 702.742 |
| 128K/128 | 498.101 | — (blocked) | 432.033 | **499.728** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **70.750** | 52.761 | 53.222 | 63.795 |
| 1K/128 | **65.905** | 54.658 | 53.044 | 63.391 |
| 4K/128 | **66.728** | 55.297 | 52.338 | 61.863 |
| 32K/128 | **53.458** | 45.983 | 45.946 | 52.286 |
| 64K/128 | 44.793 | 39.388 | 40.353 | **45.160** |
| 128K/128 | 32.615 | — (blocked) | 32.728 | **35.569** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.716** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.344** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **21.881** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The memory columns have different scopes: hipEngine reports tracked allocator
high-water, while llama.cpp reports absolute whole-device amdgpu GTT used,
sampled every 10 ms. Use them for within-column context growth, not small
cross-column allocator comparisons. Row source: [`current IOMMU-off refresh and
128K blocker`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json).
Exact settings and gates are in the canonical
[`benchmarks/README.md`](benchmarks/README.md#gfx1151-model-throughput).

### Current gfx1151 GGUF decode baselines

These are separate exact repeated-token SOL-G4/G5 controls. The model sweep
above excludes graph capture from steady decode throughput; SOL-G5 charges one
capture/instantiate and destroy to each 128-token window.

<!-- BEGIN TOPLINE:GFX1151_GGUF_EAGER -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| GGUF eager c1 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B UD-Q4_K_M; BF16 KV; `[9707] * 512`; TheRock HIP 7.15; TuneD accelerator-performance; clean scalar/candidate/scalar, 1 discarded + 4 measured runs per leg; 128 eager steps; graph off | **48.850 tok/s** (`20.471 ms/token`), **+0.309%** vs clean scalar control | Retained for this exact repeated-token protocol; control/candidate ranges do not overlap, every output ID is 9707, and the G1 hidden/state/KV oracle is linked |
| GGUF state-bound graph c1 | Radeon 8060S/gfx1151; same current model/KV/prompt/stack; 1 warmup + 4 measured rotating same-session runs; 128 steps; capture and destroy charged | **48.704 tok/s** (`20.532 ms/token`), **-0.293%** vs same-run eager; **+0.201%** vs scalar graph | Exact 128/128 state/KV/token replay, but current G5 rejects a graph-over-eager speed claim; graph default policy is tracked separately |
<!-- END TOPLINE:GFX1151_GGUF_EAGER -->

Artifacts: [`SOL-G4 eager audit`](benchmarks/results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json)
and [`SOL-G5 production graph audit`](benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

See [`benchmarks/README.md`](benchmarks/README.md) for the platform freshness
index, exact settings, run commands, and evidence status.

## Speculative decode (DFlash / MTP)

Every displayed route has its own same-protocol AR control. The exact/default
and `llama-compat` columns are separate because only `llama-compat` shares the
B2 natural24 structure used by the llama.cpp comparison.

<!-- BEGIN TOPLINE:SPECULATIVE -->
#### Qwen3.6-27B Q4_K_M llama.cpp Vulkan campaign floor, W7900/gfx1100

Latest tracked-clean llama.cpp `ee0445c99` (build 10250), Release Vulkan,
RADV/Mesa 26.1.4, `Vulkan0`, full GPU offload, flash attention, and F16 K/V on
the exact 17,106,773,120-byte GGUF (`a7cbd3ec...29e0f`):

| AR shape | Result | Samples / boundary |
| --- | ---: | --- |
| `pp512` | **792.308 tok/s** | 5 samples, stddev 1.300 tok/s; llama-bench default warmup |
| `pp4096` | **754.093 tok/s** | 5 samples, stddev 4.584 tok/s; llama-bench default warmup |
| `tg128` | **12.61795 tok/s** | 5 samples, stddev 0.00126 tok/s; standalone llama-bench generation, not context-matched |

The separate stateful server boundary uses raw `[9707] * N` prompts, prompt
cache off, one discarded warmup, five samples, and 127 timed transitions per
128 returned tokens:

| Context-matched shape | Prefill | AR decode | Samples / correctness |
| --- | ---: | ---: | --- |
| `512/128` | **79.805 tok/s** | **12.57431 tok/s** | 5; captured output is `[9707] * 128` |
| `4096/128` | **81.792 tok/s** | **12.48779 tok/s** | 5; two captured trajectories exactly match 512/128 |

The roughly 10x llama-bench/server prefill difference is real for their
respective benchmark-versus-stateful-request boundaries; neither number is
substituted for the other.

The full 10-prompt natural25 category suite selects B3 after a same-run B1-B4
sweep. Transition-normalized decode is **12.546 AR -> 68.082 MTP tok/s
(5.4265x)**; native llama.cpp generation accounting is **13.069 -> 70.919
tok/s**, while complete client/request accounting is **9.607 -> 36.122 tok/s
(3.7600x)**. Draft acceptance is **77.25%** and accepted/output is **65.20%**.
B3 beats B4 on weighted engine and complete client throughput even though B4
accepts more per output. All four categories improve; base/MTP content hashes
match for 6/10 prompts, with the four differences classified as external
llama.cpp batching/numerical diagnostics rather than a hipEngine correctness
gate.

The non-concurrent Vulkan query logger also reconciles both sampled walls.
For base 512/8, 5.783 s of query intervals is +1.89% versus 5.676 s client
wall; dense FFN is 49.04%, elementwise/transport 26.08%, other projections
14.07%, and GDN/activations 9.37%. For one natural B3 request, the separate
draft/target contexts total 69.562/553.562 ms = 623.124 ms, -2.94% versus
642.030 ms client wall; combined dense FFN is 46.52%, other projections
27.34%, elementwise/transport 13.19%, and LM head 7.53%. These synchronized
profiles rank work only and are not toplines.

This is the external floor for the new dense campaign, not a hipEngine result.
Artifact:
[`2026-08-04-qwen36-27b-llamacpp-vulkan-baseline.json`](results/2026-08-04-qwen36-27b-llamacpp-vulkan-baseline.json).

#### Qwen3.6-27B Q4_K_M hipEngine dense baseline, W7900/gfx1100

Clean hipEngine `da6865f74`, TheRock HIP 7.15, BF16 K/V, resident bulk
prefill, and one-step state-bound graph decode on the same GGUF:

| Context-matched shape | Prefill | AR decode | Vulkan delta | Samples / correctness |
| --- | ---: | ---: | ---: | --- |
| `512/128` | **50.515 tok/s** | **19.556 tok/s** | -36.70% prefill / **+55.52% decode** | 3 measured resets; all final IDs `9707` |
| `4096/128` | **50.473 tok/s** | **18.649 tok/s** | -38.29% prefill / **+49.34% decode** | 3 measured resets; all final IDs `9707` |

The graph capture is about 49 ms and excluded only from the steady-state decode
windows on both shapes; subsequent measured resets rearm the same graph after
restoring state. Tracked resident peaks are 26.123/28.947 GiB and sampled HIP
peaks are 27.339/30.184 GiB. hipEngine already beats the stateful Vulkan AR
decode floor, but misses both matched prefill gates, so this is an optimization
baseline rather than a parity row.

The true-AR plus exact transactional-MTP gate uses all ten committed natural25
prompts, 24 timed transitions per request, a fixed six/four train/heldout split,
and complete proposal/verify/commit/residual wall:

| Route | Decode | MTP / true AR | Draft acceptance | Accepted/output | Heldout ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | **20.361 tok/s** | 1.0000x | n/a | 0% | 1.0000x |
| Exact B1 | **17.128 tok/s** | **0.8412x** | 89.84% | 46.00% | 0.8143x |
| Exact B2 | 16.005 tok/s | 0.7861x | 82.07% | 60.40% | 0.7433x |
| Exact B3 | 14.858 tok/s | 0.7297x | 75.68% | 67.20% | 0.6580x |

All 250 visible IDs at every budget match true AR, all GPU accept summaries
match CPU, and every stage ledger reconciles exactly. B1 is only the least-slow
control: no exact MTP row is promoted, and Vulkan B3 remains 68.082 tok/s.

Cached-only rocprofv3 traces rank the work. The 512 prefill profile reconciles
9,911.076 ms kernels to 9,944.618 ms host wall: Q4_K pack8 projections are
78.86% and dense BF16 GEMM is 20.05%. One natural B3 leaf reconciles a
1,590.971-ms host wall to a 1,590.457-ms device-activity span; target verify is
92.58% of host wall, while Q4_K row kernels plus dense BF16 GEMV consume 82.73%
of kernel time. Target row batching/projection routing is therefore the first
optimization target. Artifact:
[`2026-08-04-qwen36-27b-hipengine-baseline.json`](results/2026-08-04-qwen36-27b-hipengine-baseline.json).

#### Qwen3.6-27B exact native target rowtile, W7900/gfx1100

Clean hipEngine `14bcea43b` keeps the same GGUF, BF16 K/V, ten natural25
prompts, 24-transition denominator, true-AR control, candidate streams, and
acceptance semantics as `da6865f74`. Native row-serial attention plus block FFN
uses the exact 2-4-row dense-BF16 rowtile; `serial-exact` remains the rollback
control.

| Route | Serial-exact baseline | Native rowtile | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.361 tok/s | **20.362 tok/s** | +0.009% | 1.0000x | n/a |
| B1 | 17.128 tok/s | **18.751 tok/s** | **+9.48%** | 0.9209x | **-9.28%** |
| B2 | 16.005 tok/s | **18.752 tok/s** | **+17.17%** | **0.9209x** | **-15.78%** |
| B3 | 14.858 tok/s | **17.983 tok/s** | **+21.03%** | 0.8831x | **-18.70%** |

Every full/train/heldout row improves. All four categories improve at every
budget (decode range **+9.14% to +21.21%**), all 250 visible IDs per budget
match true AR, GPU accept summaries match CPU, and stage ledgers reconcile.
B1/B2 are numerically tied, so this promotes the native target path rather than
a budget winner. Exact per-row state capture raises tracked peak **28.392 ->
28.995 GiB (+0.603 GiB)**, accepted for the 9-21% complete-wall gain; ownership
returns to zero. hipEngine MTP remains below own AR and Vulkan B3, so D27-O3
continues from a refreshed profile. Artifact:
[`2026-08-04-qwen36-27b-native-target-rowtile-retained.json`](results/2026-08-04-qwen36-27b-native-target-rowtile-retained.json).

#### Qwen3.6-27B exact resident-Q4 verifier rowtile, W7900/gfx1100

Clean hipEngine `4181b85fb` keeps the prior native transaction/state route and
reuses each resident Q4_K pack8 gate/up weight across exactly 2-4 verifier rows.
The local32 kernel preserves every pack8 FMA, shuffle, and BF16 output bit;
serial-exact, row-one, rows above four, registry misses, and the explicit
rowtile opt-out remain exact fallbacks.

| Route | Prior native rowtile | Native + resident Q4 rowtile | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.362 tok/s | **20.372 tok/s** | +0.047% | 1.0000x | n/a |
| B1 | 18.751 tok/s | **20.634 tok/s** | **+10.04%** | **1.0129x** | **-10.16%** |
| B2 | 18.752 tok/s | **21.752 tok/s** | **+16.00%** | **1.0678x** | **-15.14%** |
| B3 | 17.983 tok/s | **21.467 tok/s** | **+19.38%** | **1.0538x** | **-17.91%** |

Every full/train/heldout row improves, as does every category at every budget
(**+9.78% to +19.94%**). All 250 IDs per budget, acceptance ledgers, GPU/CPU
accept summaries, and state/KV/hidden/provider oracles remain exact. The real
transaction gate observes rowtile rows `{2,3,4}`. Peak stays **28.995 GiB**
with zero allocations after close, so this optimization adds no memory beyond
the prior native-state tradeoff. B2 is now a clear exact winner at **1.0678x**
own AR, although it remains far below Vulkan B3 at 68.082 tok/s.

The refreshed retained B3 profile assigns **89.57%** of wall to target verify
and selected the still-serial exact dense-BF16 GEMV at **238.766 ms / 19.63%**
of complete wall. Its exact local128 successor is promoted below. Artifact:
[`2026-08-04-qwen36-27b-resident-q4-rowtile-retained.json`](results/2026-08-04-qwen36-27b-resident-q4-rowtile-retained.json).

#### Qwen3.6-27B exact dense local128 verifier GEMV, W7900/gfx1100

Clean hipEngine `03ba1c479` maps the original 256 independent dense-BF16
arithmetic partitions onto 128 physical threads. The register/LDS/wave tree
reproduces every local256 FMA and reduction boundary. Native-verifier c1 shapes
through K=10,240 select local128; larger K, non-native rows, registry misses,
and WMMA retain exact established paths.

| Route | Prior resident-Q4 rowtile | + exact dense local128 | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.372 tok/s | **20.362 tok/s** | -0.048% | 1.0000x | n/a |
| B1 | 20.634 tok/s | **20.846 tok/s** | **+1.03%** | **1.0238x** | **-1.10%** |
| B2 | 21.752 tok/s | **22.102 tok/s** | **+1.61%** | **1.0854x** | **-1.81%** |
| B3 | 21.467 tok/s | **21.840 tok/s** | **+1.74%** | **1.0726x** | **-1.78%** |

All 30 prompt/budget rows improve (**+0.12% to +2.72%**), and every full,
train, heldout, and category rollup improves (**+0.54% to +2.06%**). All 250
IDs per budget, acceptance ledgers, GPU/CPU summaries, state/KV/hidden/provider
oracles, and stage reconciliations remain exact. Peak is unchanged at
**28.995 GiB** with zero live allocations after close. Local32 and local64
siblings were exact but slower and were removed/rejected; the K=17,408
local256 fallback avoids a measured regression. B2 remains selected at
**1.0854x** own AR and 22.102 tok/s, still 67.54% below Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-dense-local128-retained.json`](results/2026-08-04-qwen36-27b-dense-local128-retained.json).

#### Qwen3.6-27B exact reusable native-verifier graph, W7900/gfx1100

Clean hipEngine `ea33e6405` preserves native row-serial attention/Conv/GDN and
the retained row-bulk FFN arithmetic, but submits each B1/B2/B3 target bucket
through one reusable HIP graph. Device-live token/position/context/
`KVLiveSpans` metadata and graph-owned FP32 trunk rows keep the transaction exact;
N2 device accept/commit remains B1/B2.

| Route | Exact dense local128 | + reusable native graph | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.362 tok/s | **20.379 tok/s** | +0.083% | 1.0000x | n/a |
| B1 | 20.846 tok/s | **23.225 tok/s** | **+11.41%** | **1.1396x** | **-11.10%** |
| B2 | 22.102 tok/s | **24.820 tok/s** | **+12.30%** | **1.2179x** | **-12.11%** |
| B3 | 21.840 tok/s | **25.193 tok/s** | **+15.35%** | **1.2362x** | **-14.82%** |

All 30 prompt/budget rows improve (**+2.91% to +16.97%**), and every full,
train, heldout, and category rollup improves (**+9.92% to +16.15%**). All 250
IDs per budget, acceptance/cycle ledgers, GPU/CPU summaries, stage
reconciliations, and reject/partial/full state/KV/hidden oracles remain exact;
the same B3 executable also matches scalar target rows at positions 6, 7, and
9. Every measured cycle submits without fallback. The prescribed gate includes
one measured first capture for B1/B2; B3 is warm after the common warmup.
Tracked peak rises only **0.0007 GiB** to **28.996 GiB** and returns to zero.
B3 becomes the selected exact route at **1.2362x** own AR, but remains 63.00%
below Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-native-verifier-graph-retained.json`](results/2026-08-04-qwen36-27b-native-verifier-graph-retained.json).

#### Qwen3.6-27B exact staged linear projections, W7900/gfx1100

Clean hipEngine `b94810438` keeps each linear-attention Conv/GDN transition and
post-row state journal token-serial, but stages independent norm/QKV/gate/
alpha/beta projections across the verifier rows and runs one exact row-bulk
`ssm_out`. MoE and c1 retain the prior scalar fallback; no new kernel or buffer
is introduced.

| Route | Reusable native graph | + staged linear projections | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.379 tok/s | **20.309 tok/s** | -0.342% | 1.0000x | n/a |
| B1 | 23.225 tok/s | **27.734 tok/s** | **+19.42%** | **1.3656x** | **-18.22%** |
| B2 | 24.820 tok/s | **33.544 tok/s** | **+35.15%** | **1.6517x** | **-29.49%** |
| B3 | 25.193 tok/s | **36.652 tok/s** | **+45.49%** | **1.8047x** | **-35.64%** |

All 30 prompt/budget rows improve (**+19.08% to +47.62%**), and every full,
train, heldout, and category rollup improves (**+19.23% to +45.90%**). All 250
IDs per budget, acceptance/cycle ledgers, GPU/CPU summaries, graph submissions,
stage reconciliations, and reject/partial/full plus changing-position state/KV/
hidden oracles remain exact. The six Q4 QKV/gate rows2-4 component checks are
also bit-exact on GPU1 and improve 1.17-1.67x same-GPU. Tracked peak is
byte-identical at **28.996 GiB** and returns to zero. B3 remains selected at
**1.8047x** own AR, but is still 46.17% below Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-staged-linear-projections-retained.json`](results/2026-08-04-qwen36-27b-staged-linear-projections-retained.json).

#### Qwen3.6-27B exact staged full attention, W7900/gfx1100

Clean hipEngine `a67d59ac9` stages independent full-attention norm/Q/V,
split/key-cast/head-norm-RoPE, and output work across verifier rows. The narrow
Q4 K projection remains scalar after a measured rowtile regression, and every
row preserves the prior `KV write -> attention -> gate` sequence. Eager, MoE,
c1, split-decode, and scaled/non-BF16 paths retain the scalar fallback.

| Route | Staged linear projections | + staged full attention | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.309 tok/s | **20.361 tok/s** | +0.256% | 1.0000x | n/a |
| B1 | 27.734 tok/s | **28.348 tok/s** | **+2.21%** | **1.3923x** | **-2.54%** |
| B2 | 33.544 tok/s | **34.818 tok/s** | **+3.80%** | **1.7100x** | **-4.37%** |
| B3 | 36.652 tok/s | **38.322 tok/s** | **+4.56%** | **1.8821x** | **-5.61%** |

All 30 prompt/budget rows improve (**+1.66% to +5.32%**), and every full,
train, heldout, and category rollup improves (**+1.99% to +4.83%**). All 250
IDs per budget, acceptance/cycle ledgers, GPU/CPU summaries, graph submissions,
stage reconciliations, and reject/partial/full plus changing-position state/KV/
hidden oracles remain exact. GPU1 component checks are bit-exact and improve
full-Q/full-output 1.35-1.67x; the 0.83-0.90x narrow-K rowtile is rejected and
not dispatched. Tracked peak is byte-identical at **28.996 GiB** and returns to
zero. B3 remains selected at **1.8821x** own AR, but is still 43.71% below
Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-staged-full-attention-retained.json`](results/2026-08-04-qwen36-27b-staged-full-attention-retained.json).

#### Qwen3.6-27B exact compact proposal scoring, W7900/gfx1100

Clean hipEngine `39509df39` reuses the registered exact BF16/raw-Q6 pack-winner
and final-top1 primitive for production NextN proposals. Supported heads copy
only the 8-byte token/value result to the host; explicit diagnostics and
unsupported layouts retain full-vocabulary FP32 logits. No kernel arithmetic,
sampling rule, prompt branch, or target-verification path changes.

| Route | Full proposal logits | Compact exact top-1 | Decode delta | MTP / true AR | Proposal-wall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.361 tok/s | **20.394 tok/s** | +0.160% | 1.0000x | n/a |
| B1 | 28.348 tok/s | **28.878 tok/s** | **+1.87%** | **1.4160x** | **-16.41%** |
| B2 | 34.818 tok/s | **35.712 tok/s** | **+2.57%** | **1.7511x** | **-16.74%** |
| B3 | 38.322 tok/s | **39.610 tok/s** | **+3.36%** | **1.9422x** | **-17.87%** |

All 30 prompt/budget rows improve (**+1.56% to +3.74%**), and every full,
train, heldout, and category rollup improves (**+1.72% to +3.47%**). Draft
IDs, target top-1 rows, acceptance/commit ledgers, all 250 visible IDs per
budget, GPU/CPU summaries, graph submissions, stage reconciliations, and
reject/partial/full plus changing-position state/KV/hidden oracles remain exact.
The compact workspace raises tracked peak by exactly **248,328 bytes** to
**28.996 GiB** and frees completely. B3 remains selected at **1.9422x** own AR,
but is still 41.82% below Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-compact-proposal-scoring-retained.json`](results/2026-08-04-qwen36-27b-compact-proposal-scoring-retained.json).

#### Qwen3.6-27B exact dense `ssm_out` local128 rowtile, W7900/gfx1100

Clean hipEngine `4a4a615a8` selects the registered virtual256/local128
rowtile only for native dense-BF16 rows 2-4 at K6,144/N5,120. Every other
shape/session and registry miss retains local256. The arithmetic partitions,
s=128 register add, s=64/32 LDS tree, first-wave 16..1 tree, BF16 boundary,
weights, and workspace are unchanged.

| Route | Compact-proposal baseline | Qualified `ssm_out` rowtile | Decode delta | MTP / true AR | Complete-wall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR (c1 route unchanged) | 20.394 tok/s | 20.355 tok/s | -0.192% noise | 1.0000x | +0.192% |
| B1 | 28.878 tok/s | **28.979 tok/s** | **+0.349%** | **1.4237x** | **-0.348%** |
| B2 | 35.712 tok/s | **35.826 tok/s** | **+0.318%** | **1.7601x** | **-0.317%** |
| B3 | 39.610 tok/s | **39.714 tok/s** | **+0.263%** | **1.9511x** | **-0.262%** |

Every full/train/heldout/category aggregate improves (**+0.117% to +0.551%**).
Twenty-six of 30 individual prompt-budget timing rows improve; the other four
are noise-negative by at most **0.145%**. All 250 IDs per budget, acceptance/
cycle ledgers, GPU/CPU summaries, stage reconciliations, and reject/partial/full
plus changing-position state/KV/hidden oracles remain exact. The W7900 oracle
observes candidate rows exactly `{2,3,4}` at only K6,144/N5,120.

The predeclared same-GPU XTX component screen is BF16-bit exact and improves
that leaf **1.2095x/1.1654x/1.0761x** for rows 2/3/4; broad local128 selection
is rejected. Tracked peak remains byte-identical at **28.996 GiB** and frees to
zero. B3 remains selected at **1.9511x** own AR, but is still 41.67% below
Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-dense-virtual256-rowtile-retained.json`](results/2026-08-04-qwen36-27b-dense-virtual256-rowtile-retained.json).

#### Qwen3.6-27B exact Q4 dual-rowtile SiLU cycle wall, W7900/gfx1100

Clean hipEngine `98114138e` replaces two native-verifier K5,120/N17,408
resident-pack8 rowtiles plus separate SiLU with one registered local64 leaf at
rows 2-4. Separate waves reproduce the gate and up FMA/shuffle trees, publish
the same BF16 projection values through 128 B LDS, and execute the unchanged
BF16 SiLU expression. Every other row, shape, session, opt-out, or registry miss
retains the unfused chain.

A balanced production-shape W7900 leaf screen is byte-exact and improves
chain -> fused event time by **1.1060x/1.1093x/1.1128x** at rows 2/3/4. The
production B3 trace observes 448 fused launches, no legacy N17,408 rowtiles,
and exactly **896 fewer graph dispatches**. Versus the pre-`ssm_out` compact
profile, complete cycle wall falls **676.596 -> 662.705 ms (-2.053%)** and the
queue-gap/copy-overlap bucket falls **192.822 -> 174.794 ms (-9.350%)**. The
complete-window delta is explicitly compounded with the separately retained
`ssm_out` specialization; the -896 dispatch delta is uniquely attributable to
this fusion.

The one-run natural25 packet is mixed: B1/B2/B3 measure
**29.282/35.850/39.614 tok/s** (**+1.046%/+0.066%/-0.250%** versus the clean
`ssm_out` baseline). Every selected-B3 aggregate scope is noise-negative by
only **0.060-0.368%**, while IDs, acceptance, state, tracked peak
(**28.996 GiB**), and teardown remain exact. Retain the exact default for its
same-device leaf, launch-count, queue-gap, and cycle-wall evidence, but do not
advance the canonical **39.714 tok/s / 1.9511x own-AR** headline. Artifact:
[`2026-08-04-qwen36-27b-q4-dual-rowtile-silu-retained.json`](results/2026-08-04-qwen36-27b-q4-dual-rowtile-silu-retained.json).

#### Qwen3.6-27B exact dense verifier chain journals, W7900/gfx1100

Clean hipEngine `d4616f120` reuses the registered exact Conv and c1-association
GDN t-loops for dense native rows 2-4. One launch per family writes every
post-row journal directly while leaving resident initial state immutable;
deferred verification performs no resident write. Missing either registry key,
c1, MoE, absent journals, and ordinary prefill retain the complete scalar
producer/copy chain.

| Route | Immediate predecessor | Exact chain journals | Decode delta | MTP / true AR | Complete-wall delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR (c1 route unchanged) | 20.363 tok/s | 20.366 tok/s | +0.013% noise | 1.0000x | -0.013% |
| B1 | 29.282 tok/s | **30.130 tok/s** | **+2.895%** | **1.4794x** | **-2.813%** |
| B2 | 35.850 tok/s | **37.357 tok/s** | **+4.205%** | **1.8343x** | **-4.035%** |
| B3 | 39.614 tok/s | **41.440 tok/s** | **+4.608%** | **2.0348x** | **-4.405%** |

The hermetic one-prompt B3 trace confirms the predeclared ownership exactly:
**13,767 -> 9,063 dispatches (-4,704 / -34.17%)**. State-sized copy kernels
fall **3,360 -> 672**, scalar Conv/GDN producers are replaced by **672** chain
launches, and complete cycle wall falls **662.664 -> 635.545 ms (-4.09%)**;
target-verify host wall falls **5.52%** and queue-gap/copy-overlap wall falls
**8.94%** with an unchanged acceptance ledger.

All 30 prompt/budget rows improve against the immediate predecessor
(**+2.53% to +5.75%**), and every full/train/heldout/category rollup improves
(**+2.71% to +4.79%**). IDs, acceptance/cycle ledgers, transaction state,
tracked peak (**28.996 GiB**), and teardown remain exact. Versus the prior
canonical row, B3 advances **39.714 -> 41.440 tok/s (+4.346%)** and crosses
2x own AR; it remains **39.13%** below llama.cpp Vulkan B3. Artifact:
[`2026-08-04-qwen36-27b-chain-journals-retained.json`](results/2026-08-04-qwen36-27b-chain-journals-retained.json).

#### Qwen3.6-27B exact shared-page full-attention batch, W7900/gfx1100

Clean hipEngine `9c2f33b99` batches only dense N1 native full-attention rows
whose BF16 K/V cache and physical page table are shared. All row K/V writes
retire first; per-row device live counts preserve causality; then one registered
exact 256-thread attention owner and one whole-batch gate replace four scalar
attention/gate pairs. Metadata mismatch or registry absence retains the complete
scalar chain, and B2 rows=3 remains with the separate N2 bulk graph.

| Route | Exact chain journals | Shared-page attention batch | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR (unchanged c1 route) | 20.366 tok/s | 20.370 tok/s | +0.024% noise | 1.0000x | n/a |
| B1 | 30.130 tok/s | 30.116 tok/s | -0.045% noise | 1.4784x | -0.001% |
| B2 (N2 owner; diagnostic) | 37.357 tok/s | 37.447 tok/s | +0.241% | 1.8383x | -0.562% |
| **B3 (retained)** | 41.440 tok/s | **41.705 tok/s** | **+0.641%** | **2.0474x** | **-0.967%** |

The hermetic one-prompt B3 trace confirms **9,063 -> 8,391 dispatches
(-672 / -7.41%)**. Target scalar attention falls **448 -> 0** and is replaced
by 112 shared-table batches; 448 row gates become 112 whole-batch gates.
Attention+gate kernel sum falls **7.930 -> 4.478 ms**. Complete marker wall
measures **635.545 -> 608.946 ms (-4.19%)**, but unchanged phases also moved
about 4%, so the full profiled ratio is not uniquely attributed to this route.

All ten selected-B3 prompt rows improve (**+0.097% to +0.839%**), as does every
B3 full/train/heldout/category rollup (**+0.490% to +0.761%**). B1 and B2 are
reported only as diagnostics; there is no all-budget speed claim. IDs,
acceptance/cycle ledgers, transaction state, tracked peak (**28.996 GiB**), and
teardown remain exact. B3 is now **38.74%** below llama.cpp Vulkan. Artifact:
[`2026-08-04-qwen36-27b-full-attention-shared-batch-retained.json`](results/2026-08-04-qwen36-27b-full-attention-shared-batch-retained.json).

#### Qwen3.6-27B exact full-attention K grid-y batch, W7900/gfx1100

Clean hipEngine `d0cee5efa` resolves a quant-specific alias of the existing
exact generic pack8 wrapper for dense N1 full-attention K at K5,120/N1,024.
Each row retains its scalar K traversal, FP32 FMA/shuffle tree, and BF16 store;
one grid-y launch merely exposes the independent verifier rows. A missing key
keeps the complete scalar loop. Production ownership is N1 rows `{2,4}`; B2
rows=3 remains in the separate N2 bulk graph.

| Route | Shared-page attention batch | K grid-y batch | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR (unchanged c1 route) | 20.370 tok/s | 20.350 tok/s | -0.099% noise | 1.0000x | n/a |
| B1 | 30.116 tok/s | **30.154 tok/s** | **+0.126%** | **1.4818x** | **-0.234%** |
| B2 (N2 owner; diagnostic) | 37.447 tok/s | 37.663 tok/s | +0.577% | 1.8508x | -0.559% |
| **B3 (retained)** | 41.705 tok/s | **41.890 tok/s** | **+0.442%** | **2.0584x** | **-0.595%** |

The hermetic one-prompt B3 trace confirms **8,391 -> 8,055 dispatches
(-336 / -4.00%)**. Target K changes exactly from **448 scalar launches / 4.919
ms** to **112 grid-y launches / 2.252 ms (-54.23%, 2.185x)**; target-verify
kernel sum falls **399.277 -> 395.971 ms (-0.828%)**. The profiled complete
marker wall is noise-negative at **608.946 -> 611.399 ms (+0.403%)** because
queue-gap/copy-overlap grows 5.708 ms, so that single-prompt profile wall is not
a speed claim. Physical ownership, selected-family time, and the unprofiled
full-suite gate are the binding evidence.

Every B1-B3 full/train/heldout/category aggregate improves (**+0.016% to
+0.992%**); all selected-B3 scopes improve **+0.379% to +0.479%**. One B3
prompt and several other individual rows move slightly negative within noise,
so there is no 30/30-row claim. IDs, acceptance/cycle ledgers, complete
transaction state, tracked peak (**28.996 GiB**), and teardown remain exact.
B3 is now **38.47%** below llama.cpp Vulkan. Artifact:
[`2026-08-04-qwen36-27b-full-attention-k-grid-y-retained.json`](results/2026-08-04-qwen36-27b-full-attention-k-grid-y-retained.json).

#### Qwen3.6-27B exact target-resident Q6T16 proposal scoring, W7900/gfx1100

Clean hipEngine `b888de605` keeps the target-attached NextN provider inside the
live target session and borrows its source-compatible token embedding and
Q6T16 lm-head without taking ownership. The NextN-specific shared-head norm
remains independently materialized. A quant-neutral four-axis
`linear+argmax/.../proposal_top1_exact_bf16` contract retains the prior exact
raw-Q6 scorer and composes the existing exact T16 producer/reducer when the
borrowed head is Q6T16; standalone/full-logit fallbacks remain intact.

| Route | K grid-y baseline | Resident Q6T16 proposal | Decode delta | MTP / true AR | Proposal delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR (unchanged c1 route) | 20.350 tok/s | 20.310 tok/s | -0.196% noise | 1.0000x | n/a |
| B1 | 30.154 tok/s | **30.719 tok/s** | **+1.873%** | **1.5125x** | **-15.375%** |
| B2 | 37.663 tok/s | **38.627 tok/s** | **+2.560%** | **1.9019x** | **-16.546%** |
| **B3 (retained)** | 41.890 tok/s | **43.240 tok/s** | **+3.224%** | **2.1290x** | **-17.468%** |

The hermetic one-prompt B3 trace replaces exactly **25 raw-Q6 stage1 calls /
49.460 ms** with **25 half-vocabulary T16 calls / 37.925 ms**. Including the
unchanged reducer contract, exact proposal scoring falls **50.439 -> 38.445 ms
(-23.78%, 1.312x)**; proposal stage wall falls **15.14%**, complete marker wall
falls **611.399 -> 585.481 ms (-4.24%)**, and kernel sum falls **2.86%** with
unchanged 8,055 dispatches. IDs and acceptance ledger `[3,3,2,3,3,0,3]` match.

Every one of the 30 prompt-budget rows and every full/train/heldout/category
aggregate improves (**+1.357% to +3.801%** prompts; **+1.510% to +3.300%**
scopes). IDs, acceptance, transaction state, and stage reconciliation remain
exact. Borrowing removes exactly **1,758,105,600 bytes** of duplicate residents;
tracked peak falls **28.996 -> 27.359 GiB** and teardown returns to zero. B3 is
now **36.49%** below llama.cpp Vulkan. Artifact:
[`2026-08-04-qwen36-27b-resident-t16-proposal-retained.json`](results/2026-08-04-qwen36-27b-resident-t16-proposal-retained.json).

#### Qwen3.6-27B selective source-Q6 T16 projection residency, W7900/gfx1100

Clean hipEngine `c44e32ff6` keeps 32 wide Q6 `ffn_down` and 24 wide Q6
`attn_qkv` tensors solely in source-preserving T16 form instead of expanding
them to dense BF16. Their resident footprint falls **8,220,835,840 ->
3,371,827,200 bytes**, saving exactly **4,849,008,640 bytes / 4.516 GiB**. The
eight narrow K5,120/N1,024 `attn_v` tensors remain dense because actual-weight
c1 T16 is only 0.40x as fast. `decode_repack=false`, missing registry siblings,
and the unfused chains retain the prior fallbacks.

| Route | Resident-T16 proposal baseline | Selective wide Q6T16 | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 20.310 tok/s | **21.735 tok/s** | **+7.016%** | 1.0000x | n/a |
| B1 | 30.719 tok/s | **33.456 tok/s** | **+8.910%** | **1.5393x** | **-9.493%** |
| B2 | 38.627 tok/s | **40.067 tok/s** | **+3.727%** | 1.8434x | **-4.636%** |
| **B3 (retained)** | 43.240 tok/s | **44.886 tok/s** | **+3.807%** | **2.0652x** | **-5.053%** |

Every one of the 30 prompt-budget rows and every full/train/heldout/category
aggregate improves (**+3.362% to +15.563%** prompts; **+3.596% to +11.303%**
scopes). MTP is exactly greedy to candidate AR and GPU acceptance remains
CPU-exact. The new source-preserving arithmetic is quality-gated, not claimed
BF16-byte-identical to the old once-materialized dense route: **249/250**
natural tokens agree, with only fluent Japanese `計画案` -> `計画書`. B1 keeps
115 accepted tokens while proposals fall 128 -> 126; B2/B3 accepted/proposed
totals are unchanged. B2/B3 own-AR ratios decline about 3% only because the
shared AR denominator improves 7.02%, not because absolute MTP or quality
regresses.

The hermetic B3 trace proves exact physical ownership: **224 `ffn_down` + 168
`attn_qkv`** dense launches become Q6T16, while all **56 `attn_v`** calls stay
dense. The wide family falls **104.936 -> 79.719 ms (-24.03%, 1.316x)**,
target-verify wall falls **5.12%**, complete marker wall falls **585.481 ->
561.644 ms (-4.07%)**, and dispatches remain 8,055. GPU1 actual-weight gates
measure maximum KL **4.73e-5**, minimum top-1 **98.4375%**, and wide M512 WMMA
speedups **3.332x/3.251x**; cached tracing names the intended local32,
VGPR72, scratch-free T16 WMMA symbol.

| Shape | Dense-resident baseline | Selective Q6T16 | Prefill delta | Graph AR decode | Tracked peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 152.910 tok/s | **202.011 tok/s** | **+32.11% / 1.321x** | 19.565 -> **20.896 tok/s (+6.80%)** | 26.123 -> **21.607 GiB** |
| 4096/128 | 144.308 tok/s | **188.765 tok/s** | **+30.81% / 1.308x** | 18.701 -> **19.784 tok/s (+5.79%)** | 28.947 -> **24.431 GiB** |

Each populated row has three measured resets after warmup; all six final IDs
remain `9707`, and prefill/decode variation is at most 1.29%/0.26%. Selected
B3 is now **34.07%** below llama.cpp Vulkan. Artifact:
[`2026-08-04-qwen36-27b-selective-q6t16-projections-retained.json`](results/2026-08-04-qwen36-27b-selective-q6t16-projections-retained.json).

#### Qwen3.6-27B exact compact-Q4T16 dense FFN sidecars, W7900/gfx1100

Clean hipEngine `0439ecc3c` retains the exact pack8 prefill/fallback residents
and adds compact-T16 decode sidecars to exactly 64 dense `ffn_gate` plus 64
`ffn_up` weights. Existing exact T16 fusion owns c1; the new registered
weight-reuse fusion owns native rows 2-4. Missing sidecars/keys, unsupported
shapes, non-native sessions, and `decode_repack=false` retain pack8.

| Route | Selective-Q6 baseline | Compact Q4T16 FFN | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 21.735 tok/s | 21.700 tok/s | -0.161% noise | 1.0000x | n/a |
| B1 | 33.456 tok/s | **36.078 tok/s** | **+7.837%** | **1.6626x** | **-8.419%** |
| B2 | 40.067 tok/s | **42.879 tok/s** | **+7.017%** | **1.9760x** | **-7.879%** |
| **B3 (retained)** | 44.886 tok/s | **47.496 tok/s** | **+5.814%** | **2.1887x** | **-6.630%** |

Every one of 30 prompt-budget rows improves **+5.333% to +8.261%**, and every
full/train/heldout/category aggregate improves **+5.668% to +8.001%**. IDs,
acceptance, cycle ledgers, GPU/CPU checks, and complete transaction state remain
exact. True AR is byte-identical and its one-run movement is disclosed as
noise.

The hermetic B3 trace confirms exact ownership: **448 pack8 fused-FFN calls /
106.201 ms** become **448 compact-T16 calls / 76.440 ms (-28.02%, 1.389x)**.
Target verification falls **469.671 -> 452.948 ms (-3.56%)**, kernel sum falls
**436.018 -> 407.085 ms (-6.64%)**, and complete marker wall falls **561.644 ->
548.050 ms (-2.42%)** with the same 8,055 dispatches.

| Shape | Selective-Q6 baseline | Compact Q4T16 FFN | Prefill delta | Graph AR decode | Tracked peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 202.011 tok/s | 201.698 tok/s | -0.155% noise | 20.896 -> **20.910 tok/s (+0.065%)** | 21.607 -> **27.749 GiB** |
| 4096/128 | 188.765 tok/s | 188.580 tok/s | -0.098% noise | 19.784 -> **19.804 tok/s (+0.101%)** | 24.431 -> **30.574 GiB** |

All six populated final IDs remain `9707`. The speed win adds exactly
**6,595,543,040 bytes / 6.143 GiB** because pack8 remains required for
prefill/fallback; teardown returns to zero. This is retained for the 48-GiB
W7900 campaign target and is not claimed to fit the 24-GiB component board.
Selected B3 is now **30.24%** below llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-q4t16-ffn-sidecars-retained.json`](results/2026-08-05-qwen36-27b-q4t16-ffn-sidecars-retained.json).

#### Qwen3.6-27B row-selective compact-Q4T16 projections, W7900/gfx1100

Clean hipEngine `2dbf6abdd` extends compact-T16 decode sidecars to all **288**
screened rank-2 Q4 weights while retaining pack8 for populated prefill and
fail-closed fallback. Seven families use T16 at native rows 2-4; `attn_qkv` and
`attn_v` retain pack8 at row 2 because their actual-weight component results
were flat at **0.9977x/0.9969x**, then select T16 at rows 3-4. No prompt/token
condition or new environment selector is introduced.

| Route | FFN-only Q4T16 baseline | Row-selective Q4T16 | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 21.700 tok/s | 21.644 tok/s | -0.259% noise | 1.0000x | n/a |
| B1 | 36.078 tok/s | **37.544 tok/s** | **+4.064%** | **1.7346x** | **-4.506%** |
| B2 | 42.879 tok/s | **45.634 tok/s** | **+6.426%** | **2.1084x** | **-7.217%** |
| **B3 (retained)** | 47.496 tok/s | **50.344 tok/s** | **+5.996%** | **2.3260x** | **-6.986%** |

Every one of 30 prompt-budget rows improves **+3.726% to +7.092%**, and every
full/train/heldout/category aggregate improves **+3.971% to +6.623%**. IDs,
acceptance, cycle ledgers, GPU/CPU checks, complete transaction state, and
teardown remain exact. The binding transaction census sees compact-T16 rows
`{2,3,4}` across every production shape and pack8 only at the two declared
row-2 exceptions.

The hermetic B3 trace replaces exactly **1,008 pack8 single-Q4 calls / 81.461
ms** with **1,008 compact-T16 calls / 51.370 ms (-36.94%, 1.586x)**. Target
verification falls **452.948 -> 423.244 ms (-6.56%)**, kernel sum falls
**407.085 -> 378.591 ms (-7.00%)**, and complete marker wall falls **548.103
-> 515.594 ms (-5.93%)** with unchanged 8,055 dispatches.

| Shape | FFN-only Q4T16 baseline | Row-selective Q4T16 | Prefill delta | Graph AR decode | Tracked peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 201.698 tok/s | **202.550 tok/s** | **+0.422%** | 20.910 -> **20.940 tok/s (+0.144%)** | 27.749 -> **31.656 GiB** |
| 4096/128 | 188.580 tok/s | **188.637 tok/s** | **+0.030%** | 19.804 -> 19.768 tok/s (-0.184% noise) | 30.574 -> **34.481 GiB** |

All six populated final IDs remain `9707`. The extension adds exactly
**4,194,959,360 bytes / 3.907 GiB**, bringing total compact-Q4T16 sidecars to
**10.049 GiB** because pack8 remains required. Teardown returns to zero. This
is a 48-GiB W7900 route and is not claimed to fit the 24-GiB component board.
Selected B3 is now **26.05%** below llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-q4t16-row-selective-sidecars-retained.json`](results/2026-08-05-qwen36-27b-q4t16-row-selective-sidecars-retained.json).

#### Qwen3.6-27B exact Q6T16 col8 verifier rowtiles, W7900/gfx1100

Retained implementation `a821d571b` splits each exact local128 Q6T16 verifier
slab from sixteen to eight output columns while preserving all 128 K
partitions, each FP32 FMA, the wave32 trees, the ordered four-wave sum, and the
BF16/F32 store. Production selects col8 only for BF16 rows 2-5 and FP32 rows
3-4; screened flat or losing rows retain col16. No allocation, environment
flag, or prompt/token condition is added.

| Route | Prior col16 rowtile | Exact col8 rowtile | Decode delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 21.644 tok/s | 21.676 tok/s | +0.145% noise | 1.0000x | n/a |
| B1 | 37.544 tok/s | **38.581 tok/s** | **+2.762%** | **1.7799x** | **-3.734%** |
| B2 | 45.634 tok/s | **48.712 tok/s** | **+6.745%** | **2.2473x** | **-8.319%** |
| **B3 (retained)** | 50.344 tok/s | **52.652 tok/s** | **+4.585%** | **2.4291x** | **-6.277%** |

Every one of 30 prompt-budget rows improves **+1.839% to +7.321%**, and every
full/train/heldout/category aggregate improves **+2.588% to +7.049%**. All 750
MTP IDs, 250 AR IDs, acceptance ledgers, GPU/CPU summaries, stage accounting,
B1-B3 transaction state, ownership, and teardown remain exact. Peak is
byte-identical at **35,317,648,936 bytes / 32.892 GiB**.

Hermetic B3 tracing observes exactly **399** col8 launches and unchanged
**8,055** total dispatches. Q6 rowtiles fall **99.135 -> 75.244 ms (-24.10%,
1.318x)**: BF16 output falls **80.906 -> 60.253 ms (-25.53%)** and FP32 output
falls **18.229 -> 14.991 ms (-17.76%)**. Target-verify host wall falls
**423.244 -> 399.282 ms (-5.66%)**, kernel sum falls **378.591 -> 356.972 ms
(-5.71%)**, and complete wall falls **515.594 -> 492.172 ms (-4.54%)**. The
candidate uses 80 VGPR / 512 B LDS / zero scratch versus 136 / 1,024 / zero.

The rows-2-5-only selector cannot execute in populated 512/4096 prefill or c1
graph decode, so those expensive controls were not repeated; unchanged
natural25 true AR and the complete transaction gate are the binding controls.
The measured candidate was the exact five-file patch committed unchanged as
`a821d571b`. Selected B3 is now **22.66%** below llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-q6t16-col8-rowtiles-retained.json`](results/2026-08-05-qwen36-27b-q6t16-col8-rowtiles-retained.json).

#### Qwen3.6-27B narrow Q4T16 col4 verifier rowtile, W7900/gfx1100

The exact col4 pressure split lowers row-4 VGPR **144 -> 96**, but the binding
profile rejects its initial two-shape policy. K5,120/N10,240 rises **8.914 ->
9.489 ms (+6.44%)**, driving kernel sum **+0.588%**, target verify **+1.426%**,
and complete B3 wall **492.172 -> 496.956 ms (+0.972%)**. That shape returns
to its retained row-2 pack8 and rows-3/4 compact-T16 tile8 owners.

Only K5,120/N1,024 rows 2-4 keep col4. The complete trace improves that exact
family **0.869 -> 0.777 ms (1.118x)**, and an 11-sample W7900 actual-weight
leaf confirms **16.883 -> 12.201 us (1.384x, 11/11 wins)** with zero BF16
mismatches. The final focused selector test and complete B1-B3 transaction
pass. This **0.0916-ms / 0.0186%-of-window** mechanical win is below natural25
resolution, so the canonical headline remains **52.652 tok/s / 2.4291x own
AR / 22.66% below Vulkan**. Artifact:
[`2026-08-05-qwen36-27b-q4t16-col4-shape-policy.json`](results/2026-08-05-qwen36-27b-q4t16-col4-shape-policy.json).

#### Qwen3.6-27B byte-neutral planar-Q6T16 replacement, W7900/gfx1100

Clean implementation `3728531ba` replaces the same 56 wide `ffn_down` and
`attn_qkv` Q6T16 residents with exact 3,360-byte planar-qmicro records. Root
lm-head, narrow V, gfx1151, explicit legacy plans, and registry misses keep the
legacy layout. No sidecar, allocation, flag, or prompt/token condition is added.

| Route | Legacy Q6T16 | Planar qmicro | Delta | MTP / true AR | Target-verify delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 21.676 tok/s | **23.031 tok/s** | **+6.252%** | 1.0000x | n/a |
| B1 | 38.581 tok/s | **39.830 tok/s** | **+3.239%** | 1.7294x | **-3.416%** |
| B2 | 48.712 tok/s | **49.848 tok/s** | **+2.331%** | 2.1644x | **-2.651%** |
| **B3 (retained)** | 52.652 tok/s | **54.547 tok/s** | **+3.599%** | **2.3685x** | **-4.223%** |

All 30 prompt-budget rows improve **+1.501% to +4.408%**, every
full/train/heldout/category scope improves, and IDs/acceptance remain exact.
Because c1 AR improves faster than verifier rows, B3/own-AR declines **2.4291x
-> 2.3685x (-2.497%)**; only the absolute MTP win is claimed.

A clean contemporaneous B3 trace records unchanged **8,055** dispatches. The
392 wide-Q6 calls fall **59.876 -> 46.714 ms (-21.98%, 1.282x)**, total kernel
sum falls **356.003 -> 343.678 ms (-3.46%)**, and complete wall falls **504.331
-> 500.205 ms (-0.818%)**. The no-warmup first target pass rises, while steady
passes 2-7 improve **47.742 -> 45.840 ms (-3.98%)**.

Populated 512/4096 prefill improves **202.550/188.637 -> 207.022/192.528 tok/s
(+2.208%/+2.062%)** and graph AR improves **20.940/19.768 -> 22.107/20.926
(+5.572%/+5.859%)**. All final IDs are `9707`; peaks remain byte-identical at
**31.656/34.481 GiB** and teardown returns to zero. Canonical B3 is now
**19.88% below** llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-q6t16-qmicro-planar-retained.json`](results/2026-08-05-qwen36-27b-q6t16-qmicro-planar-retained.json).

#### Qwen3.6-27B byte-neutral planar-Q6T16 root head, W7900/gfx1100

Clean implementation `094941175` extends the retained planar-qmicro layout to
the distinct K5,120/N248,320 root head. Proposal top-1, generic FP32 c1, and
native target FP32 rows use registered same-quant consumers. The K2,048 MoE
head, gfx1151, X8 requests, explicit legacy plans, and registry misses retain
legacy T16. The allocation remains exactly **1,042,944,000 bytes**.

| Route | Wide-planar baseline | + planar root head | Delta | MTP / true AR | Primary stage delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 23.031 tok/s | **23.069 tok/s** | **+0.166%** | 1.0000x | n/a |
| B1 | 39.830 tok/s | **40.179 tok/s** | **+0.875%** | 1.7417x | proposal **-5.116%** |
| B2 | 49.848 tok/s | **50.813 tok/s** | **+1.935%** | 2.2026x | proposal **-6.612%** |
| **B3 (retained)** | 54.547 tok/s | **55.899 tok/s** | **+2.478%** | **2.4231x** | proposal **-7.424%** |

Every full/train/heldout/category scope improves, all B2/B3 prompt rows improve,
and IDs, acceptance, GPU/CPU summaries, and stage ledgers remain exact. One B1
`code_merge_intervals` cell is **-0.653%** in the one-run per-prompt timing, so
there is no 30/30-row claim.

The tracked-clean one-prompt B3 trace cuts intended root work **55.638 ->
48.851 ms (-12.20%)**: proposal top-1 stage1 **40.606 -> 37.127 ms (-8.57%)**
and target FP32 rows **15.032 -> 11.724 ms (-22.00%)**. Kernel sum falls
**1.729%** and complete no-warmup wall falls **4.367%** at unchanged **8,055**
dispatches. First-use graph capture also moves; steady target passes 2-7 improve
only **45.840 -> 45.553 ms (-0.627%)**, so the physical root family and
unprofiled suite bind attribution.

Populated 512/4096 prefill improves **207.022/192.528 -> 207.864/193.001 tok/s
(+0.407%/+0.246%)** and graph AR improves **22.107/20.926 -> 22.208/21.017
(+0.456%/+0.438%)**. All six final IDs are `9707`; peaks remain byte-identical
at **31.656/34.481 GiB** and teardown returns to zero. Canonical B3 is now
**17.90% below** llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-q6t16-qmicro-root-head-retained.json`](results/2026-08-05-qwen36-27b-q6t16-qmicro-root-head-retained.json).

#### Qwen3.6-27B exact NextN state-only full-accept tail, W7900/gfx1100

Clean implementation `e15d85d1c` keeps the same embedding/fusion and complete
full-attention+dense-FFN NextN block, including its draft K/V write, in a
non-final full-accept catch-up, but omits its discarded final RMSNorm and
248,320-row lm-head score. Scored proposals, partial accepts, target
verification, transactions, candidates, and acceptance are unchanged.

| Route | Planar-root baseline | + state-only tail | Delta | MTP / true AR | Proposal delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR control | 23.069 tok/s | **23.217 tok/s** | +0.642% noise | 1.0000x | path cannot execute |
| B1 | 40.179 tok/s | **41.512 tok/s** | **+3.317%** | 1.7880x | **-24.99%** |
| B2 | 50.813 tok/s | **51.974 tok/s** | **+2.286%** | 2.2386x | **-13.56%** |
| **B3 (retained)** | 55.899 tok/s | **56.802 tok/s** | **+1.616%** | **2.4466x** | **-7.35%** |

All 30 prompt-budget rows and every full/train/heldout/category scope improve;
IDs, acceptance, GPU/CPU summaries, complete transaction state, the
**32.892-GiB** peak, and teardown remain exact. Populated AR/prefill are not
repeated because this path is NextN-only and true AR plus the transaction gate
bind non-regression.

The tracked-clean one-prompt B3 trace removes exactly four unused stage-1 heads,
four reducers, and **16 dispatches**. Proposal-update kernels fall **9.731 ->
3.615 ms (-62.85%)**, update host wall **12.747 -> 7.190 ms (-43.60%)**, and
complete marker wall **478.319 -> 464.238 ms (-2.94%)**. The ordinary scored
proposal body is flat at **48.671 -> 48.658 ms** of kernels. Canonical B3 is now
**16.57% below** llama.cpp Vulkan. Artifact:
[`2026-08-05-qwen36-27b-nextn-state-only-tail-retained.json`](results/2026-08-05-qwen36-27b-nextn-state-only-tail-retained.json).

#### Qwen3.6-27B quality-gated Q5T16 dense `ssm_out`, W7900/gfx1100

Clean implementation `24fef47da` replaces exactly 48 gfx1100
K6,144/N5,120 dense-BF16 `ssm_out` residents with source-faithful Q5T16.
Direct c1, exact local128/col4 rows 2-4, larger-row fallback, and dense WMMA
prefill are registered siblings; all other shapes/roles, decode-repack-off,
registry misses, and gfx1151 retain dense BF16. Residency falls exactly
**1,958,215,680 bytes / 1.824 GiB**.

| Route | State-only-tail baseline | + Q5T16 `ssm_out` | Delta | MTP / true AR | Acceptance / work |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 23.217 tok/s | **24.049 tok/s** | **+3.585%** | 1.0000x | candidate trajectory control |
| B1 | 41.512 tok/s | **43.170 tok/s** | **+3.994%** | 1.7951x | 115 / 127 |
| B2 | 51.974 tok/s | **54.621 tok/s** | **+5.094%** | 2.2712x | 151 / 182 |
| **B3 (retained)** | 56.802 tok/s | **59.551 tok/s** | **+4.839%** | **2.4762x** | **169 / 219** |

Every full/train/heldout/category scope and every B2/B3 prompt improves. One B1
prompt is **-2.116%**. Candidate MTP matches its own AR on every row. Nine of ten
prior-route outputs remain byte-identical; only `general_ja_explain` diverges
after token 18 into a fluent memory-bandwidth explanation rather than the prior
fluent FLOPS explanation. Aggregate component quality is **7.38e-5 max KL /
99.79% top-1**, and the complete B1-B3 transaction/state/provider/lifecycle gate
passes, so this is retained as a disclosed quality-gated route rather than a
bit-identical dense-BF16 replacement.

The tracked-clean one-prompt B3 trace cuts the 336-call physical family
**37.004 -> 22.911 ms (-38.09%)**, target-verify host wall **380.843 -> 361.440
ms (-5.095%)**, and complete marker wall **464.238 -> 444.023 ms (-4.354%)** at
unchanged **8,039** dispatches.

Populated 512/4096 prefill advances **207.864/193.001 -> 234.014/215.771 tok/s
(+12.58%/+11.80%)** and graph AR advances **22.208/21.017 -> 23.241/21.841
(+4.65%/+3.92%)**. All six final IDs are `9707`, each peak falls exactly 1.824
GiB, and teardown returns to zero. Prefill and AR remain well above matched
stateful Vulkan; natural25 B3 is now **12.53% below** Vulkan's 68.082 tok/s.
Artifact:
[`2026-08-05-qwen36-27b-q5t16-ssm-out-retained.json`](results/2026-08-05-qwen36-27b-q5t16-ssm-out-retained.json).

#### Qwen3.6-27B exact GDN-to-Q5T16 BF16 handoff, W7900/gfx1100

Clean implementation `a5f25c9ad` writes the existing round-to-nearest-even BF16
`ssm_out` input from the scalar or chain GDN producer while preserving its FP32
output and every recurrent-state bit. The resident Q5T16 weight plugin owns the
two registered boundaries; registry misses and gfx1151 retain ordinary GDN plus
standalone cast plus BF16 Q5T16.

| Route | Q5T16 baseline | + producer handoff | Delta | MTP / true AR | Acceptance / work |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 24.049 tok/s | 23.750 tok/s | -1.247% timing variance | 1.0000x | all tokens exact |
| B1 | 43.170 tok/s | **43.357 tok/s** | **+0.433%** | 1.8256x | 115 / 127 |
| B2 | 54.621 tok/s | **54.678 tok/s** | **+0.104%** | 2.3023x | 151 / 182 |
| **B3 (retained)** | 59.551 tok/s | **59.790 tok/s** | **+0.402%** | **2.5175x** | **169 / 219** |

All 750 MTP tokens, AR tokens, acceptance counts, and GPU/CPU decisions are
exact. Target-verify wall improves **0.466%/0.390%/0.479%** at B1/B2/B3. Timing
noise leaves the worst prompt/category at **-0.561%/-0.287%**; B3 has one
negative prompt and one category at only **-0.466%/-0.094%**.

The corrected one-prompt W7900 trace removes **336** casts and dispatches
(**8,039 -> 7,703**), changes GDN+cast **15.522 + 0.606 -> 15.465 ms**, cuts
target kernels **0.809%** and target host wall **0.332%**, and cuts complete
kernel sum **0.707%**. Complete marker wall is disclosed at **+0.209%** queue
variance; an independent first exact run measures complete/target wall
**-1.875%/-1.494%**. GPU1 complete GDN-handoff medians improve c1-c4
**1.264x/1.153x/1.112x/1.088x**.

Populated graph AR improves **23.241 -> 23.284 tok/s (+0.184%)** at 512 and
**21.841 -> 21.903 (+0.281%)** at 4K. Prefill is noise-flat
(**+0.242%/-0.299%**), peaks are byte-identical, all IDs are `9707`, and
teardown returns to zero. Natural25 B3 is now **12.18% below** Vulkan's 68.082
tok/s. Artifact:
[`2026-08-05-qwen36-27b-gdn-bf16-handoff-retained.json`](results/2026-08-05-qwen36-27b-gdn-bf16-handoff-retained.json).

#### Qwen3.6-27B exact one-launch rollback snapshot, W7900/gfx1100

Clean implementation `01291b066` corrects the prior copy attribution: the
**672** target-window copies are seven pre-verify snapshots of 48 Conv plus 48
GDN resident states, not final-row commits. `_StateJournal` now owns immutable
live/snapshot pointer tables and replaces each 96-copy snapshot with one
registered 64-KiB-chunked pair-copy launch. Registry, shape, layer-pair, and
backend misses retain the original copies; per-row journals are unchanged and
gfx1151 deliberately misses.

| Route | GDN-handoff baseline | + one-launch snapshot | Delta | MTP / true AR | Acceptance / work |
| --- | ---: | ---: | ---: | ---: | ---: |
| True AR | 23.750 tok/s | 24.114 tok/s | +1.533% timing variance | 1.0000x | MTP-only route is unreachable |
| B1 | 43.357 tok/s | **43.792 tok/s** | **+1.004%** | 1.8161x | 115 / 127 |
| B2 | 54.678 tok/s | **55.254 tok/s** | **+1.053%** | 2.2914x | 151 / 182 |
| **B3 (retained)** | 59.790 tok/s | **60.262 tok/s** | **+0.789%** | **2.4991x** | **169 / 219** |

All 30 prompt-budget rows and every full/train/heldout/category scope improve;
the smallest prompt and scope wins are **+0.149%/+0.235%**. All tokens,
acceptance, GPU/CPU decisions, transaction state, and teardown are exact. The
same-session AR rise is timing variance and numerically lowers MTP/AR despite
every absolute MTP row improving. Populated AR was not repeated because
`_StateJournal` cannot execute outside MTP.

The corrected one-prompt W7900 trace changes **672 copies / 4.528 ms** into
**7 launches / 3.458 ms**, removes **665 dispatches**, cuts target host wall
**0.683%**, and cuts complete marker wall **444.949 -> 438.031 ms (-1.555%)**.
An independent exact no-marker profile also improves **53.932 -> 54.315 tok/s**.
The pointer plan adds only **1,540 bytes**. GPU1 production-shape synchronized
wall improves **656.504 -> 436.108 us (1.505x)** with byte-exact forward and
reverse copies. Natural25 B3 is now **11.49% below** Vulkan's 68.082 tok/s.
Artifact:
[`2026-08-05-qwen36-27b-journal-snapshot-copy-retained.json`](results/2026-08-05-qwen36-27b-journal-snapshot-copy-retained.json).

#### Qwen3.6-27B exact producer-folded rollback snapshot, W7900/gfx1100

Clean implementation `82a7f8691` removes the native verifier's separate
pre-verify Conv/GDN snapshot without weakening rollback. Exact chain producers
store each immutable resident input into one session-owned rollback row while
writing the existing output-row journals. The snapshot becomes restorable only
after the complete target block retires; a failure after selected-state commit
still restores all 96 original buffers. Serial mode and every paired-owner,
registry, shape, or backend miss retain the five-row journal and pointer-copy
fallback; gfx1151 deliberately misses.

GPU1 rows 2/3/4 preserve every initial-state, output-row, FP32-output, and fused
BF16-handoff bit. Added producer-store cost is only **2.478/2.922/3.714 us per
layer**, projecting **317.167/295.850/257.839 us saved per B1/B2/B3 cycle**
against the retained one-launch copy. The binding W7900 B1-B3 transaction adds
a post-commit forced-failure gate and restores every state byte exactly.

Tracked-clean B3 tracing removes all **7 snapshot launches / 3.458 ms**. Conv
plus GDN producers rise only **17.079 -> 18.943 ms**, a direct **1.593 ms**
target-family saving; target and complete kernel sums improve **3.025/2.977
ms**. Peak tracked allocation falls exactly **635,437,056 bytes**
(**606.0 MiB**), from five state rows to one.

The ten-prompt gate improves target verify at every budget:
**4.8487 -> 4.8419 s (-0.140%)**, **3.6468 -> 3.6360 s (-0.296%)**, and
**3.2299 -> 3.2136 s (-0.506%)**. Unrelated proposal/commit/host timing variance
leaves complete B1/B2/B3 **0.524%/0.318%/0.303% lower**, so this is retained as
an exact physical target-window and memory win, not a new broad topline. The
canonical headline remains **60.262 tok/s / 2.4991x own AR / 11.49% below
Vulkan**. Artifact:
[`2026-08-05-qwen36-27b-producer-folded-rollback-snapshot-retained.json`](results/2026-08-05-qwen36-27b-producer-folded-rollback-snapshot-retained.json).

#### Qwen3.6-27B selective FFN-down residual fusion, W7900/gfx1100

Clean implementation `8e281f115` adds exact gfx1100 `linear+residual`
composites for the retained planar-Q6 and compact-Q4T16 FFN-down rowtiles. Each
producer first materializes the ordinary BF16 projection value in-register,
then performs the same BF16 residual read, FP32 add, and final BF16 rounding as
the standalone add. Projection-plus-add remains the registered fallback; c1,
FP32 residuals, unsupported shapes/keys/backends, and gfx1151 fail closed.

GPU1 actual Q6/Q4 K17,408/N5,120 weights are bit exact at rows 2-4 and save
**2.372-3.684 / 2.136-3.924 us per layer**. The all-row W7900 gate rejects
planar-Q6 row4: its 224-call family rises **32.831 -> 38.892 ms (+18.46%)** and
natural B3 falls **60.079 -> 59.274 tok/s (-1.341%)** with every prompt and
scope negative. Selective commit `4bbe8eef8` therefore keeps Q6 rows 2-3 and Q4
rows 2-4, with all other rows on the exact fallback.

The selective tracked-clean B3 profile removes **224 dispatches**, cuts complete
marker **451.442 -> 448.386 ms (-0.677%)**, and cuts target host **365.954 ->
364.985 ms (-0.265%)**. Target/complete kernel sums rise **0.555%/0.453%**, so
this is a launch/queue contraction, not an arithmetic claim. The ten-prompt gate
moves B1/B2 **43.563 -> 43.922 (+0.825%)** and **55.079 -> 55.196 tok/s
(+0.213%)**, with every aggregate scope positive; B3 is mixed at **60.079 ->
59.951 tok/s (-0.214%)**. All IDs, acceptance, state, memory, and teardown are
exact. Retain the selective default without replacing the canonical **60.262
tok/s / 2.4991x own AR / 11.49% below Vulkan** headline. Artifact:
[`2026-08-05-qwen36-27b-ffn-down-residual-fusion-retained.json`](results/2026-08-05-qwen36-27b-ffn-down-residual-fusion-retained.json).

#### Qwen3.6-27B rounded residual plus next-RMSNorm, W7900/gfx1100

Clean implementation `fda35418e` replaces each native N1 inter-layer
FFN-down -> rounded BF16 residual add -> next attention RMSNorm boundary with
ordinary FFN-down plus one exact rounded `add+rmsnorm` owner. The composite
publishes the identical BF16 residual bits and runs the unchanged RMS reduction
on those rounded values. It owns B1/B3 rows2/4 only; B2/N2, c1, prefill, F32
residuals, unsupported policies/keys/backends, the final layer, and gfx1151 keep
the registered unfused fallback.

GPU1 actual planar-Q6 and compact-Q4 boundaries preserve all rows2-4 residual
and normalized BF16 surfaces and save **2.543-4.320 us/layer**, with
**22-30/31** paired wins. The tracked-clean W7900 B3 trace executes **441**
rounded leaves, removes **217 dispatches**, cuts target host/kernel
**364.985/257.289 -> 349.141/256.721 ms (-4.341%/-0.221%)**, and cuts complete
marker/kernel **448.386/313.141 -> 431.236/312.507 ms
(-3.825%/-0.202%)**. Acceptance remains exact at **17/21** and peak bytes are
unchanged.

Ten-prompt natural25 is intentionally disclosed as mixed: immediate B1/B2/B3
moves **43.922/55.196/59.951 -> 43.741/55.332/60.012 tok/s
(-0.412%/+0.247%/+0.103%)**. B2 cannot execute this N1 route, B1 scopes are
negative, and B3 prompt/scope timing straddles zero. Tokens, acceptance, state,
memory, and teardown remain exact. Retain the exact physical/launch contraction
without replacing canonical **60.262 tok/s / 2.4991x own AR / 11.49% below
Vulkan**. Artifact:
[`2026-08-05-qwen36-27b-rounded-next-rmsnorm-retained.json`](results/2026-08-05-qwen36-27b-rounded-next-rmsnorm-retained.json).

#### Qwen3.6-27B shared-cache KV batch primitive, runtime rejected

Correctness commit `e3adc89fa` adds a gfx1100-only rows2/4 shared-page BF16 KV
writer. Each grid-Y row preserves the scalar FP32-K/BF16-V conversion and store
arithmetic while indexing one common physical page table. The reversed-page
positions-255..258 fixture is byte-exact to scalar launches and the direct CPU
cache oracle; the routed W7900 B1-B3 transaction is also exact. GPU1 component
graph medians improve **1.762x/3.172x**, and cache-only tracing records the
rows4 body at **6.240 us**, local256, VGPR8, scratch0.

The production route fails its predeclared W7900 complete-path gate. Target
writers improve **448 / 1.083574 ms -> 112 / 0.378642 ms (-65.05%)**,
target/complete dispatches each fall **336**, and target/complete kernel sums
improve **0.832%/0.675%**. Target host and complete marked wall nevertheless
regress **1.405%/2.786%**. The runner is restored to scalar writes and natural25
is not run; retain only the measured exact primitive. The canonical **60.262
tok/s / 2.4991x own AR / 11.49% below Vulkan** headline is unchanged. Artifact:
[`2026-08-05-qwen36-27b-shared-kv-write-runtime-rejected.json`](results/2026-08-05-qwen36-27b-shared-kv-write-runtime-rejected.json).

#### Qwen3.6-27B dense-F32 alpha/beta pair primitive, runtime rejected

Correctness commit `7e274241a` adds a gfx1100-only same-input pair for the 48
linear-attention `ssm_alpha`/`ssm_beta` F32 weight pairs. Rows1-3 use one flat
pair grid and row4 uses two rows/block; two ordinary dense-F32 GEMVs remain the
fallback. Every rows1-4 output is scalar-BF16-bit exact and passes the CPU
KL/top-1 gate. All 48 real pairs improve GPU1 event medians
**1.771x/1.786x/1.837x/1.560x**, with **21/21** wins per selected row count.
The routed W7900 B1-B3 transaction is exact; cache-only row4 tracing records
local256, VGPR24, LDS2,048 B, scratch0, and **12.080 us**.

The production route fails its frozen W7900 complete-path gate. The pair family
improves **672 / 3.572950 ms -> 336 / 2.802259 ms (-21.57%)**, target/complete
dispatches each fall **336**, and target/complete kernel sums improve
**0.140%/0.086%**. Target host and complete marked wall nevertheless regress
**0.201%/0.189%**. Generic pair dispatch is removed before natural25/populated
AR; retain only the measured primitive. Canonical **60.262 tok/s / 2.4991x own
AR / 11.49% below Vulkan** is unchanged. Artifact:
[`2026-08-05-qwen36-27b-dense-f32-alpha-beta-pair-runtime-rejected.json`](results/2026-08-05-qwen36-27b-dense-f32-alpha-beta-pair-runtime-rejected.json).

#### Qwen3.6-27B alpha/beta plus snapshot-Conv primitive, runtime rejected

Correctness commit `95fff80d3` adds a gfx1100-only one-grid owner for both
K5,120/N48 dense-F32 projections and the independent C10,240/K4 rollback-safe
chain-Conv. The pair blocks preserve the scalar local256 arithmetic trees and
the final 40 blocks preserve the registered snapshot-Conv body. All 48 real
layers at rows1-4 are bit-exact for alpha, beta, Conv output/journal, and initial
snapshot; CPU KL/top-1 gates pass. GPU1 event medians improve
**2.028x/1.949x/1.892x/1.731x**, with synchronized wall agreement and
**21/21** wins at every row count. The routed W7900 B1-B3 transaction is exact,
and cache-only tracing records local256, VGPR32, LDS2,048 B, scratch0, with
steady **6.920/9.960 us** durations.

The production route fails its frozen W7900 complete-path gate. The combined
family improves **1,008 / 5.469564 ms -> 336 / 3.221938 ms (-41.09%)**,
target/complete dispatches each fall **672**, and target/complete kernel sums
improve **1.016%/0.884%**. Target host and complete marked wall nevertheless
regress **0.546%/1.629%**. Generic dispatch and runner ownership are removed
before natural25/populated AR; production again uses two scalar projections plus
snapshot Conv. Retain only the measured primitive. Canonical **60.262 tok/s /
2.4991x own AR / 11.49% below Vulkan** is unchanged. Artifact:
[`2026-08-05-qwen36-27b-dense-f32-pair-chain-conv-runtime-rejected.json`](results/2026-08-05-qwen36-27b-dense-f32-pair-chain-conv-runtime-rejected.json).

#### Qwen3.6-27B dependent alpha/beta-to-GDN primitive, runtime rejected

Correctness commit `4e6c8e21d` and producer-locality repair `648cd4bf0` add a
gfx1100-only per-V-head owner for both K5,120/N48 dense-F32 projections plus
the rollback-safe chain GDN and Q5T16 BF16 handoff. The repaired local256 body
stages four per-head Conv Q/K/V slices in 17,920 B of dynamic LDS before the
projection scan. Rows1-4 preserve alpha, beta, every recurrent journal and
initial snapshot bit, FP32 output, and BF16 handoff; CPU KL/top-1 gates pass.
GPU1 event screens improve **1.2958x/1.1657x/1.0893x/1.0360x**, with **21/21**
wins per row count, and the complete routed W7900 B1-B3 transaction is exact.

The final tracked-clean current-clock W7900 profile fails its frozen compound
gate. The three-leaf family moves **1,008 / 20.613994 ms -> 336 / 21.480201 ms
(+4.202%)** despite removing **672** target/complete dispatches. Target host and
complete marked wall improve **3.179%/2.440%**, but target/complete kernel sums
regress **0.351%/0.268%**. Generic dispatch and runner ownership are removed
before natural25/populated AR; production again uses two scalar projections,
snapshot Conv, and snapshot+cast GDN. Post-unroute coverage passes **129/129**.
Retain only the measured primitive; canonical **60.262 tok/s / 2.4991x own AR /
11.49% below Vulkan** is unchanged. Artifact:
[`2026-08-05-qwen36-27b-dense-f32-pair-gdn-runtime-rejected.json`](results/2026-08-05-qwen36-27b-dense-f32-pair-gdn-runtime-rejected.json).

#### Qwen3.6-27B native target-graph split, runtime rejected

The restored B3 profile contains three apparent **5.736-5.799 ms** idle gaps
exactly 256 dispatches apart, but unprofiled graph experiments show they are
profiler instrumentation. Explicit graph upload changes first launch only. An
exact cloned 836-node GPU1 graph is slightly slower when split four ways.

The decisive W7900 test topologically partitions the actual one-stream B3
target DAG from **816 nodes** into four exact **204-node** executables and
submits them on the same stream with one synchronization. Tokens, hidden and
pre-output rows, and every captured Conv/GDN state-row byte remain identical.
Across 21 alternating same-state pairs, submit medians move only **40.563484 ->
40.500983 ms (1.00154x)** and complete call medians **40.983989 -> 40.918268
ms (1.00161x)**. The paired median saving is **0.089741 ms** inside a ~3.2-ms
range—not the profiler-implied ~17.2 ms. No source route or natural25 run is
admitted. Canonical **60.262 tok/s / 2.4991x own AR / 11.49% below Vulkan** is
unchanged. Artifact:
[`2026-08-06-qwen36-27b-native-target-graph-split-rejected.json`](results/2026-08-06-qwen36-27b-native-target-graph-split-rejected.json).

#### Qwen3.6-27B exact B5 higher budget, runtime rejected

A completion audit found that the shared MTP ABI and trailing NextN provider
support budgets `(1,2,3,5)`, while this dense GGUF route had only measured
B1-B3. B4 remains unsupported. The initial exact ten-prompt B3/B5 run confirms
that B5 improves acceptance (**169 -> 184**) and reduces cycles (**76 -> 61**),
but missing six-row owners collapse throughput **59.942 -> 23.523 tok/s**.

Adding temporary row-6 instantiations of the existing exact Q4/Q5/Q6 templates
removes every direct fallback. On the same favorable code prompt, B5 target
verify falls **724.097 -> 306.675 ms (-57.65%)** and throughput rises **30.103
-> 63.511 tok/s (2.110x)**, temporarily exceeding same-process B3 **59.597
tok/s**. That single prompt is not a promotion gate. B5 still costs **61.335
ms/pass**, above the frozen **<=50.3 ms** full-suite break-even and roughly
**42.5 ms** Vulkan break-even.

The decisive trace shows steady B5 at **923 nodes / 47.774 ms kernels / 53.279
ms span**, versus B3 **819 / 35.163 / 39.718 ms**. The residual is the real
cost of six exact rows—primarily Q4 **+7.599 ms/pass** and Q6 **+2.275 ms**—not
another missing leaf. No second optimized full-suite run is spent after the
predeclared screen fails; all row-6/runtime changes are removed and B1-B3 stays
production. Canonical **60.262 tok/s / 2.4991x own AR / 11.49% below Vulkan**
is unchanged. Artifact:
[`2026-08-06-qwen36-27b-dense-b5-budget-rejected.json`](results/2026-08-06-qwen36-27b-dense-b5-budget-rejected.json).

#### Qwen3.6-27B exact populated pack8 prefill tile8x8, W7900/gfx1100

Clean hipEngine `68e8c10c5` reuses each resident Q4_K output-pack8 weight
stream across eight populated token rows while preserving every scalar FP32
FMA, wave32 reduction, zero-seeded final sum, and BF16 boundary. The existing
`use_wmma_prefill` contract selects tile8x8 only from 512 rows; smaller rows,
opt-out, c1, native verifier rows 2-4, and registry misses retain their exact
owners. Wide gate/up pairs become two exact singleton leaves.

| Shape | Scalar pack8 prefill | Exact tile8x8 prefill | Prefill delta | Graph AR decode | Matched stateful Vulkan | Tracked peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 50.515 tok/s | **152.910 tok/s** | **+202.70% / 3.027x** | 19.556 -> **19.565 tok/s (+0.05%)** | **+91.60%** vs 79.805 | 26.123 GiB (unchanged) |
| 4096/128 | 50.473 tok/s | **144.308 tok/s** | **+185.91% / 2.859x** | 18.649 -> **18.701 tok/s (+0.28%)** | **+76.43%** vs 81.792 | 28.947 GiB (unchanged) |

Each row has three measured resets after warmup; all six final IDs are `9707`.
A separate same-session W7900 oracle compares scalar and tile routes at both
contexts: first/next full logits have zero mismatches and KL zero, and all live
Conv/GDN/full-attention KV state hashes match after prefill and the next c1
step. The full GPU1 exact-pack8 file passes 21/21 and adds no weights or
workspace.

Cached selected-region production tracing executes tile8x8 **288** times at
local32/VGPR144/SGPR128/LDS512/scratch0. The Q4 family falls
**7,815.869 -> 1,496.173 ms (-80.86%, 5.224x)** and complete kernel sum falls
**9,911.076 -> 3,241.171 ms (-67.30%)**. Dense BF16 prefill GEMM is now the
largest p512 family at **1,654.462 ms / 51.05%** and is queued for D27-L1 after
D27-O3. All natural-suite prompts are only 39-71 rows, so this >=512 policy
cannot alter the retained exact B3 MTP route and that suite was not rerun.
Artifact:
[`2026-08-04-qwen36-27b-exact-pack8-prefill-tile8x8-retained.json`](results/2026-08-04-qwen36-27b-exact-pack8-prefill-tile8x8-retained.json).

#### GGUF MTP comparison, Radeon Pro W7900/gfx1100

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR | llama.cpp HIP bundled MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2 natural24, reusable B1/B2 target graphs | Natural25 request / 24 timed transitions | B2, natural25 request / 24 timed transitions |
| Decode | **98.75 tok/s fixed / 96.75 tok/s natural24** | 68.50 tok/s | **122.67 tok/s** | 78.05 tok/s transition-normalized | 115.44 tok/s transition-normalized |
| Own true AR | same route | 98.75 tok/s | 96.75 tok/s | same route | 78.05 tok/s |
| MTP / own AR | 1.0000x | **0.6936x** | **1.2679x** | n/a | **1.4791x** |
| Draft acceptance | n/a | 73.53% | 80.45% | n/a | 81.56% |
| Accepted draft/output | n/a | 50.00% | 60.00% | n/a | 58.40% |
| Complete wall per output/transition | 10.336 ms natural24 | 14.696 ms | **8.186 ms** | 12.812 ms | 8.662 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp autoregressive | native llama.cpp compatibility target |

The W7900 route now reuses one fixed-address target graph per B1/B2 shape
bucket. Live token, position, context, and cursor metadata are staged on device;
the five two-row output-cap tails use B1 and the four true one-row/no-draft
cycles stay on AR. Unsupported configurations fall back before launch, while a
post-launch failure never re-executes a possibly mutating verifier.

Two clean full-suite processes at `0d7b86e7` measure **123.33 and 122.67 tok/s**
(0.54% spread). The conservative run is **1.2679x** its true graph AR and
**6.26% faster** than llama.cpp's **115.44 tok/s / 8.662 ms-transition** floor,
while complete wall is **5.50% lower** at **8.186 ms/output**. Draft acceptance
and accepted/output remain exactly **80.45% / 60.00%**. All 240 output IDs and
all 96 cycle semantics in both runs match the prior eager-target
`llama-compat` baseline. hipEngine uses BF16 KV versus llama.cpp F16 KV, and the
external row remains `performance_claim=false` because its preserved
instrumentation checkout is dirty.

The target graph also passes the real 35B oracle at two B2 positions plus B1:
target top-1, 16,384 FP32 hidden values, each set of 60 captured and 60 resident
Conv/GDN buffers, all 20 K/V buffers, and cursors are byte-exact. A cached
six-step trace records zero measured recaptures, **18.67 ms host / 13.67 ms
kernels / 5.00 ms residual**, and the expected dynamic-metadata, cursor-advance,
and top-1 widening leaves. The prior eager profiler residual was 38.41 ms.

Exact/default remains the semantic control. `llama-compat` remains explicit-only
because direct partial commit is not serial-prefix-equivalent; this retained
speed result does not make it the automatic exact route. The fixed-cycle exact
and natural24 compatibility rows remain different protocols.

##### W7900 reusable-native `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 96.75 | **122.67** | **1.2679x** | 80.45% | 60.00% | 8.186 ms |
| Train | 6 | 96.13 | **124.70** | **1.2973x** | **87.25%** | 61.81% | 8.052 ms |
| Heldout | 4 | 97.68 | **119.73** | **1.2257x** | **71.43%** | 57.29% | 8.388 ms |
| `code` | 4 | 97.06 | **127.81** | **1.3168x** | 93.94% | 64.58% | 7.854 ms |
| `general_en` | 2 | 94.25 | **123.37** | **1.3091x** | 75.68% | 58.33% | 8.138 ms |
| `general_ja` | 2 | 98.04 | **118.42** | **1.2079x** | 69.23% | 56.25% | 8.480 ms |
| `mixed_ja_en` | 2 | 97.40 | **116.78** | **1.1990x** | 72.97% | 56.25% | 8.604 ms |

Every category and the heldout split beat their true same-protocol AR control;
even the slowest category remains above the aggregate external floor in the
conservative run. The corrected 54.88 tok/s eager-target row remains the
optimization baseline, not the current route. Artifacts:
[`retained reusable route`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json),
[`N2 ownership diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n2.json),
[`N3 complete-cycle diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3.json),
[`N3P proposal-submission diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json),
[`prior baseline`](results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json),
and [`llama.cpp floor`](results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json).

#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2 natural24, NativeSpecCycle N3 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 56.39 tok/s (0.9895x own AR) | **80.10 tok/s (1.4282x own AR)** | 70.99 tok/s native (1.3530x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **80.10 tok/s** | 68.15 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **56.09 tok/s** | 50.37 tok/s |
| Cross-engine MTP / own AR | n/a | 1.4282x | 1.3530x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 17.808 ms/output | 12.551 ms/output | 14.673 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | N3 complete public cycle; accuracy-traded | native llama.cpp compatibility target |

The IOMMU-off exact/default B5 route remains the current semantic control at
**56.39 vs 56.98 true-AR tok/s (0.9895x)**. `llama-compat` is separate,
explicit-only, and not serial-prefix-equivalent. On current main, registering
the reusable gfx1151 target graph moves the clean direct-commit control
**70.020 -> 80.132 tok/s (+14.44%)**; N3 public complete-cycle ownership retains
**80.099 tok/s (+14.39%)** and cuts complete wall **14.314 -> 12.551 ms/output
(-12.32%)**. N3 is only **0.042%** below target-only N1.

All **240 output IDs / 97 cycle semantics** match across clean control, N1, and
N3, with unchanged **77.72% draft acceptance / 59.58% accepted-output**. The
prior clean `2edbb2ee` direct-commit row remains slightly higher at **81.90
tok/s** (-2.20% versus current N3), but it is a different revision/run and no
source regression is attributed. Against the preserved transition-normalized
llama.cpp context, current N3 is **80.10 vs 68.15 tok/s (+17.53%)**; BF16 versus
F16 KV and the dirty preserved llama.cpp source remain disclosed.

##### gfx1151 NativeSpecCycle N3 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | N3 tok/s | N3 / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 56.09 | **80.10** | **1.4282x** | 77.72% | 59.58% | 12.551 ms |
| Train | 6 | 55.97 | **80.91** | **1.4457x** | **82.08%** | 60.42% | 12.429 ms |
| Heldout | 4 | 56.26 | **78.91** | **1.4025x** | **71.79%** | 58.33% | 12.733 ms |
| `code` | 4 | 56.12 | **86.08** | **1.5338x** | 91.04% | 63.54% | 11.684 ms |
| `general_en` | 2 | 57.26 | **78.98** | **1.3795x** | 71.79% | 58.33% | 12.716 ms |
| `general_ja` | 2 | 55.61 | **75.12** | **1.3509x** | 69.23% | 56.25% | 13.388 ms |
| `mixed_ja_en` | 2 | 55.35 | **75.66** | **1.3669x** | 69.23% | 56.25% | 13.282 ms |

Every category and the heldout split beats its true same-protocol AR control and
improves versus the clean current-main direct-commit route by **9.91% to
19.45%**. The real 35B N1/N2 oracle passes target IDs, FP32 hidden rows, all 60
Conv/GDN and 20 full-KV buffers, selected commits, and cursors. The six-step
cached trace records zero recaptures, **24.891 ms host / 21.674 ms kernels /
3.218 ms residual**, 940 calls/step, and the expected zero-scratch metadata
leaf. N3P remains unregistered on gfx1151 because it is not needed for this win
and was not the gfx1100 topline. Artifact:
[`gfx1151 NativeSpecCycle transfer`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json).

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`W7900 GGUF MTP transfer`](benchmarks/results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
and [`current gfx1151 IOMMU-off MTP refresh`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json).
Historical gfx1151 controls remain linked from the canonical benchmark record.
Historical hipEngine OpenAI MTP server rows are excluded. The current raw-ID
route counts exact completion IDs across every choice and owns batch timing
once. The
corrected 2026-07-11 server matrix finds that compatibility MTP changes true-AR
IDs even at c1, so it must remain explicit-only despite diagnostic c1/c2 speed
gains; SOL-S1 routes automatic requests to exact/default AR while keeping the
compatibility hook explicit-only. See the
[`route-gate artifact`](benchmarks/results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json)
and canonical [`benchmarks/README.md`](benchmarks/README.md#gfx1151-gguf-server-automatic-route-gate-2026-07-11).

The clean gfx1151 PARO DFlash S4 profile is exact but not competitive:
`9.68` versus `65.27 tok/s` AR (`0.148x`) at B4/32 tokens. Branch-copy is
faster but diverges at generated token 1, and fused target LM-head is 5.16%
slower than unfused. See the
[`compact profile`](benchmarks/results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json)
and the canonical
[`benchmark analysis`](benchmarks/README.md#gfx1151-paro-dflash-s4-profile-2026-07-11).

## Concurrency

Current GGUF direct-model-step tables are retained separately for gfx1100 and
gfx1151. Both have exact native c2/c4/c8 graph routes, direct throughput, and
live OpenAI membership. gfx1151 additionally retains occupancy-adaptive GGUF
serving, explicit short mirrored-INT8 c1/c2/c4/c8, and production PARO
c2/c4/c8; gfx1100 PARO remains direct-c2 only. See
[`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) for the exact boundaries.

The linked records keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c4/c8 model-step throughput and retained real
OpenAI SSE arbitrary-C server scaling.** All rows use `UD-Q4_K_M`, BF16 KV,
greedy top-1, W7900/gfx1100, and TheRock HIP 7.15. Timing scopes stay separate:
direct rows time synchronized graph steps; server rows time complete concurrent
SSE cycles including admission, prompt work, decode, delivery, and completion.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 85.469 | 85.469 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.693 / 11.955 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 127.427 | 63.714 | 1.491x | 1.504x | 0.951 / 0.954 s | 15.765 / 16.023 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 184.575 | 46.144 | 2.160x | 2.178x | 2.020 / 2.023 s | 21.715 / 22.021 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **246.872** | **30.859** | **2.888x** | **2.913x** | **3.475 / 3.479 s** | **32.414 / 32.749 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 183.020 | 22.878 | 2.141x | 2.160x | 3.055 / 4.084 s | 43.767 / 44.281 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 84.738 | 21.185 | 0.991x | 1.000x | 0.548 / 0.877 s | 47.225 / 48.142 ms | 26.985 GiB* |

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical-c1 control | 1 | masked physical c8 | 25.583 | 25.583 | 1.000x | 0.807x | 5.003 s | 0.271 / 0.276 s | 36.499 / 39.809 ms | 29.312 GiB |
| physical c8 | 8 | 1x c8 | **136.122** | 17.015 | **5.321x** | 4.293x | 7.523 s | 1.751 / 2.088 s | 41.712 / 45.068 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + sparse c8 | 88.592 | 9.844 | 3.463x | 2.794x | 13.003 s | 1.709 / 2.235 s | 81.087 / 88.112 ms | 31.969 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **111.380** | **8.568** | **4.354x** | **3.513x** | **14.940 s** | **1.886 / 3.323 s** | **87.502 / 93.631 ms** | **32.869 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 31.708 | 2.439 | 1.239x | 1.000x | 52.479 s | 2.424 / 3.390 s | 382.821 / 396.004 ms | 32.869 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and median
of three; one physical c8 is **2.888x** c1 and **+34.89%** over c4+c4, with a
**748 packed-native / 0 row-local / 0 copy** trace. Server protocol uses 512
exact prompt IDs and 128 generated outputs/request, a 20 ms admission window,
one discarded plus three measured bursts, and scheduler latency. Logical c1 is
honestly a masked physical-c8 production control; C9/C13 are multiple declared
buckets, never wider native widths. All **189/189** server requests match
resident prompt IDs, direct-c1 outputs, usage, and finish metadata. Grouped C13
is **4.354x** logical-c1 and **3.513x** serial; one exact c8→c13 live trace emits
**1,664/1,664** IDs at **107.284 aggregate tok/s** and drains ownership to zero.
Starred server memory is cumulative in one prepared process.

Artifacts: [`C4`](benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json),
[`E2 native c8`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json).
Historical mixed-quant/mixed-scope results remain in
[`benchmarks/HISTORY.md`](benchmarks/HISTORY.md).

### gfx1151 / Radeon 8060S PARO direct c2/c4/c8 and production shape catalog (Qwen3.6 35B-A3B, 512/128)

**Status: direct and resident OpenAI c2/c4/c8 are retained; gfx1151 selects
them by package default.** Equivalent clean tree `e175e28f` (pushed as
`8c8cc15e`) generalizes the exact c2 route into true physical c4/c8 without c2
stacking. G5 attaches those identity-matched widths to one shared stable-slot
owner for public `LLM`, blocking OpenAI, and concurrent SSE. Explicit legacy
`=0` flags remain rollback opt-outs.

Three p512/d128 processes per width pass all **5,754/5,754** recorded IDs plus
all-layer state/KV, sparse lifecycle, ten-prompt category/heldout, primitive,
and cached-profiler gates.

| Explicit direct route | Median aggregate decode | Per-request decode | Classification |
| --- | ---: | ---: | --- |
| c1 graph | **70.810 tok/s** | 70.810 tok/s | independent reference |
| serial c2 bridge | **65.574 tok/s** | 32.787 tok/s | exact fallback control |
| native selected-batch c2 | **79.237 tok/s** | 39.619 tok/s | **retained direct c2; 1.1190x c1 / 1.2084x serial** |
| true physical c4 | **100.209 tok/s** | 25.052 tok/s | **retained direct c4; 1.4152x c1** |
| true physical c8 | **99.943 tok/s** | 12.493 tok/s | **retained direct c8; 1.4114x c1** |

The production table below uses the clean blocking F1 wall: 512 raw prompt IDs,
128 generated IDs/request, one warmup plus three measured bursts, and a fresh
server per width. All **68/68** warmup/measured/live rows are exact; c1/c2/c4/c8
scale to **47.124/51.962/60.323/61.253 aggregate tok/s** with <=0.994% variance.
The complementary exact-roundtrip SSE packet keeps all **100/100** rows exact at
**36.327/38.666/42.471/41.487/35.633 tok/s** for c1/c2/c4/c8/serial-c8; native
c8 is **1.164x** serial and live c4->c8 admission is **38.191 tok/s**. A 1+7 c8
stress adds **72/72** exact rows, and a no-native-flag OpenAI c4 gate run from
`/tmp` loads the packaged profile, observes physical widths 2/4, and records no
fallback.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained OpenAI aggregate |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **47.124 tok/s** |
| 2 | native `2` | retained physical width | **51.962 tok/s** |
| 3 | `2+1` | exact partition; not native c3 | no separate claim |
| 4 | native `4` | retained physical width | **60.323 tok/s** |
| 5 | `4+1` | exact partition; not native c5 | no separate claim |
| 6 | `4+2` | exact partition; not native c6 | no separate claim |
| 7 | `4+2+1` | exact partition; not native c7 | no separate claim |
| 8 | native `8` | retained physical width | **61.253 tok/s** |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: W4 PARO/BF16 KV, 40 layers, 8 warmup decode steps, 128 measured
decode steps, greedy sampling, TheRock HIP 7.15, one hardware queue, TuneD
`accelerator-performance`, and `amd_iommu=off`. All direct-width process
variance is <=0.054%. c4/c8 are **+41.52%/+41.14% vs c1**; c8 aggregate is
**0.265% below c4**, while its median model-step time is **0.183% faster than
two c4 steps**. c3/c5/c6/c7 retain no native-width claim. See the
[retained direct-c2/c4/c8 artifact](benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json),
[G5 blocking F1](benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json),
[G5 SSE](benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json),
[c8 repeatability](benchmarks/results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json),
[package-default OpenAI c4](benchmarks/results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json),
and the [canonical run record](benchmarks/README.md#paro-concurrency-and-production-routing).

### gfx1151 / Radeon 8060S direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c2/c4/c8 model steps and c1-preserving
occupancy-adaptive OpenAI SSE serving.** F2 maps only ephemeral execution rows
into c1/c2/c4/c8 while stable scheduler, state, and KV ownership stays fixed.
F3 adds exact singleton-indexed packed-AR GDN: direct c2/c4/c8 improve
**+8.71%/+5.25%/+4.04%**, while c1 is structurally unchanged. The prior F2
server packet remains retained but was not remeasured for this direct-only F3
refresh. All rows use `UD-Q4_K_M`, BF16 KV, greedy top-1, one HIP queue, and the
active `amd_iommu=off` boot. Direct/category and E3 gates retain **188,080** and
**134,160** exact hidden comparisons; the c8 trace is **748 packed-native / 0
row-local / 0 copies**, with diagnostic Conv/GDN time down **50.94%**.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / retained serial-c4† | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 50.335 | 50.335 | 1.000x | 1.002x | 0.370 / 0.372 s | 19.874 / 20.160 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 78.552 | 39.276 | 1.561x | 1.564x | 2.177 / 2.177 s | 25.459 / 25.820 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 108.050 | 27.013 | 2.147x | 2.151x | 3.394 / 3.403 s | 37.026 / 37.399 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **133.251** | **16.656** | **2.647x** | **2.653x** | **6.841 / 6.841 s** | **60.004 / 60.641 ms** | **25.401 GiB** |
| chunked c8 control† | 8 | 2x c4, serialized | 102.724 | 12.841 | 2.043x | 2.045x | 5.089 / 6.787 s | 77.902 / 78.467 ms | 26.069 GiB* |
| serial-c4 rate control† | 4 | 4x c1, serialized | 50.235 | 12.559 | 0.999x | 1.000x | 0.927 / 1.485 s | 79.643 / 80.637 ms | 26.985 GiB* |

† Controls are retained from clean pre-F3 `ef46ee8c`. The serial-c4 c1 path is
structurally unchanged and remains the ratio reference; chunked c8 is historical
because the F3 candidate would also change each physical c4 group.

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| occupancy-adaptive c1 | 1 | 1x native c1 | 43.033 | 43.033 | 1.000x | 0.999x | 2.974 s | 0.417 / 0.417 s | 19.759 / 20.057 ms | 29.046 GiB |
| physical c8 | 8 | 1x c8 | **86.942** | 10.868 | **2.020x** | 2.019x | 11.778 s | 2.614 / 3.347 s | 64.949 / 68.260 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + c1 | 77.302 | 8.589 | 1.796x | 1.795x | 14.903 s | 2.734 / 3.783 s | 85.747 / 91.143 ms | 31.035 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **73.235** | **5.633** | **1.702x** | **1.701x** | **22.721 s** | **3.624 / 5.526 s** | **132.547 / 141.787 ms** | **32.386 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 43.066 | 3.313 | 1.001x | 1.000x | 38.638 s | 3.386 / 5.390 s | 257.799 / 271.455 ms | 32.386 GiB* |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and the
median of three. F3 direct c1/c2/c4/c8 is
**50.335/78.552/108.050/133.251 aggregate tok/s** with maximum rate
stdev/median **0.096%**; one physical c8 is **2.647x** c1 and **+23.32%** over
the current direct-c4 rate. Server protocol remains the clean F2 packet: 512
exact prompt IDs and 128 generated outputs/request, a 20 ms admission window,
one discarded plus three measured bursts, and scheduler latency. C9/C13 are
multiple declared buckets, never wider native widths. All **189/189** requests
match resident prompt IDs, direct-c1 outputs, usage, and finish metadata.
Grouped C13 is **1.702x** logical-c1 and **1.701x** serial; one exact c8→c13
live trace emits **1,664/1,664** IDs at **71.891 aggregate tok/s** and drains
ownership. Clean C2→C8 and C4→C8 traces preserve **256/256** IDs each. Starred
server memory is cumulative.

Artifacts: [`F3 singleton-indexed GDN`](benchmarks/results/2026-07-19-gfx1151-gguf-f3-singleton-gdn-retained.json),
[`F2 occupancy-adaptive serving`](benchmarks/results/2026-07-19-gfx1151-gguf-f2-occupancy-adaptive-serving.json),
[`E1 direct correctness`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json),
[`E1 direct scaling`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json).

The explicit short mirrored-INT8 server packet separately records blocking
c1/c2/c4/c8 **40.467/57.211/72.037/72.514 tok/s** and exact SSE
**39.665/52.225/68.665/79.789 tok/s**. All **117** server rows and the full
11-prompt/99-position KL/top-1 gate pass; C8 drains ownership and packed
workspace. Bounded BF16 mirrors mean this is not a memory-saving default, and
strict high-C SLO plus tail4/direct/long INT8 remain open. Evidence:
[`mirrored INT8 continuous concurrency`](benchmarks/results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json).

### gfx1151 historical cross-engine concurrency (2026-06-15)

**Status: stale diagnostic.** hipEngine used PARO W4/BF16 KV; llama.cpp used
Vulkan Q4_K_S/f16 KV, and vLLM did not produce a healthy server. The summary
lacks the measured hipEngine commit and has incomplete device provenance. No
eligible historical row is inferred from the current direct GGUF table.

Source artifacts: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[`vLLM blocked`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

A 2026-06-13 RX 7900 XTX rerun reached c1/c2/c4 but c8 blocked with HIP OOM;
see [`XTX partial`](benchmarks/results/2026-06-13-hipengine-qwen35-concurrency-decode-latest-xtx-blocked-c8.json).
Replicate the W7900 hipEngine, llama.cpp Vulkan, and vLLM concurrency rows with:

```bash
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

The exact settings and gfx1151 runner gap are recorded in
[`benchmarks/README.md`](benchmarks/README.md#readme-sweep-test-procedure).

## GGUF Support

As of v0.2.0, hipEngine includes resident Qwen3.6 GGUF support for `Q4_K_M` and
`Q4_K_S` model files (with more formats planned). This is a major runtime path,
not just a loader shim: GGUF has its own quant readers, bulk-prefill path,
decode-repacked T16 layouts, and fast-path controls.

The 40-layer `UD-Q3_K_M` target keeps resident `IQ3_XXS`/`IQ4_XS`
selected-expert weights compressed and executes native gate/up and down kernels,
bulk prefill, and graph decode. The merged W7900 branch preserves its first
correctness-oriented baseline: **614.089/92.285**, **623.583/97.373**, and
**616.135/98.111** prefill/decode tok/s at 512/128, 1K/128, and 4K/128. Those
2026-07-19 measurements describe the historical branch implementation; the
newer optimized direct/native Q3 route and current records are documented in
[`benchmarks/README.md`](benchmarks/README.md#merged-ud-q3_k_m-gpu1-and-w7900-records).

Current caveats:

- PARO models take ~22s to load on the W7900 test host in the current refresh;
  GGUF Q4_K_M currently takes about 74s because decode-repack happens on load.
  On-disk caching could reduce startup time later, but would require additional
  storage for repacked layouts.
- GGUF has higher base weight residency than packed PARO before KV cache is the
  deciding factor. The full-attention KV slope is the same 10-layer Qwen3.6
  shape; the 24 GiB long-context gap is mostly the loaded-weight baseline.
  Packed PARO is ~19.07 GiB on disk, while the local GGUF tensor payloads are:

  | GGUF tensor family | Q4_K_M GiB | Q4_K_M mix | Q4_K_S GiB | Q4_K_S mix | Q4_K_S - Q4_K_M |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Q4_K | 11.531 | 54.7% | 16.875 | 84.8% | +5.344 |
  | Q5_K | 6.531 | 31.0% | 0.000 | 0.0% | -6.531 |
  | Q8_0 | 1.932 | 9.2% | 1.932 | 9.7% | +0.000 |
  | Q6_K | 1.004 | 4.8% | 1.004 | 5.0% | +0.000 |
  | F32/BF16 metadata | 0.098 | 0.5% | 0.098 | 0.5% | +0.000 |
  | **Total tensor payload** | **21.097** | **100.0%** | **19.909** | **100.0%** | **-1.188** |

  In other words, `Q4_K_S` saves ~1.19 GiB versus `Q4_K_M` by replacing the
  selected-MoE `Q5_K` expert-down payload with `Q4_K`; it still starts above
  packed PARO, and hipEngine's resident T16/pack8 decode layouts add their own
  allocator shape. On 24 GiB cards, current `Q4_K_M` BF16-KV support is a
  mid-context path unless a lower-memory KV/weight policy is explicitly enabled.
  The current clean gfx1151 p512/d128 census is **21.478 GiB** owned/tracked:
  **20.461 GiB** replacement weights, **0.503 GiB** required raw token embedding,
  **0.097 GiB** dense weights/metadata, and **0.417 GiB** scratch/session buffers.
  It confirms the default T16 path replaces source layouts rather than retaining
  raw+packed copies; this short-context margin is not a 128K capacity claim.
- GGUF is close enough to PARO to share some high-level scheduling ideas, but in
  practice it needs substantial GGUF-only kernels and dispatch. The goal for
  future releases is to keep closing the remaining PARO/GGUF speed gap.


## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine serve                    OpenAI-compatible server    │
├─────────────────────────────────────────────────────────────────┤
│  LOADING (torch-free)                                           │
│  safetensors mmap + hipMemcpyAsync / HF config / jinja2 chat    │
│  templates / HF tokenizers (Rust)                               │
├─────────────────────────────────────────────────────────────────┤
│  DISPATCH                                                       │
│  Scheduler / Block Manager (KVPolicy) / Prefix Cache            │
│  Fusion Planner (chain → kernel plan, fused preferred)          │
│  Model / Quant / Layer plugins / Engine loop (hipGraph replay)  │
├─────────────────────────────────────────────────────────────────┤
│  CORE (torch-free primitives)                                   │
│  hipengine.Tensor / device / memory / stream / graph / blas     │
│  build (hipcc subprocess + ctypes.CDLL + .so cache)             │
├─────────────────────────────────────────────────────────────────┤
│  KERNELS (backend-keyed custom HIP implementations)             │
│  kernels/hip_gfx1100/  attention / linear_attn / moe / quant    │
│                        wmma / norm / rotary / fused             │
│  kernels/hip_gfx1151/  native target-arch peer backend          │
│  kernels/cuda_sm86/    (future)                                 │
│  kernels/cpu_reference/ correctness oracle, no GPU required     │
└─────────────────────────────────────────────────────────────────┘
```

Full layer diagram, plugin axes, KV cache ABI, and roadmap are in
[`docs/PLAN.md`](docs/PLAN.md).

## Installation

```bash
# PyPI wheel: runtime, JIT kernel sources, vendored AOTriton, and server
pip install hipengine

# Source checkout: fetch Git LFS payloads before an editable install
git lfs install
git lfs pull
pip install -e .

# with the optional dlpack torch bridge for user-boundary interop
pip install "hipengine[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.10+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

### ROCm / TheRock setup for retained benchmark rows

For retained gfx1100 benchmark rows, use the pinned AMD TheRock environment in
[`docs/THEROCK.md`](docs/THEROCK.md), not an ad-hoc mixed `/opt/rocm` runtime.
Current retained rows use TheRock ROCm `7.13.0a20260423` with:

```text
HIP version: 7.13.26162-1140233ffe
```

On this host (`Linux 7.0.10-1-cachyos`, W7900 VBIOS `113-D7070100-138`, RX 7900
XTX VBIOS `113-EXT89622-001`), ROCm 7.14 nightly diagnostics showed GGUF prefill
and MTP wall-time regressions, so 7.13 remains the canonical stack until a newer
ROCm release beats the same gates. See `docs/THEROCK.md` for the exact `pip
install`/repair commands, clean process wrapper, and the upstream TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) reference.

The installed app exposes a small command group:

```bash
hipengine --help
hipengine serve --help
hipengine bench list
```

## Quickstart

Model loading does not start network downloads. Populate the Hugging Face cache
before using a repository ID:

```bash
hf download shisa-ai/Qwen3.6-35B-A3B-PARO-packed
```

Then construct `LLM` with the same repository ID:

```python
from hipengine import LLM, SamplingParams

llm = LLM("shisa-ai/Qwen3.6-35B-A3B-PARO-packed")
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

`LLM(model)` auto-detects `gfx1100` or `gfx1151` and selects the model plugin's
quantization. The Qwen3.6 GGUF path also selects T16 decode-repack plus the
retained WMMA-prefill/GEMV-decode session profile. Explicit `backend=` and
`quant=` arguments are overrides; supported PARO and GGUF models do not require
hipEngine environment variables. Unsupported registry combinations fail instead
of falling back to a torch path.

## OpenAI-compatible server

The OpenAI-compatible FastAPI layer is installed by default:

```bash
pip install hipengine
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --served-model-name qwen-paro
```

`--model` accepts either a local filesystem path or a Hugging Face model ID
already present in the local HF cache; hipEngine resolves IDs locally and does
not download weights during startup.

Core endpoints are `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions`, with token-level SSE streaming, logprobs,
OpenAI-style tools, structured-output validation, and Qwen thinking controls.
hipEngine extensions provide readiness/capability discovery, token and context
diagnostics, and app-local session management. Chat responses separate
`<think>` reasoning into `reasoning_content`. The server eagerly warms the model
on startup, caps omitted chat `max_tokens` with
`--chat-default-max-tokens` (default 4096), and has an explicit `--debug` mode
for full request/response payload logging. See the complete
[`docs/API.md` endpoint table](docs/API.md#endpoints) for bearer-token auth,
request examples, feature contracts, diagnostics, and current limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, phase roadmap, LoC budgets |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols, baselines, correctness gate, artifact format |
| [`docs/TESTING.md`](docs/TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, source-lineage drift workflow, JIT cache gotchas, build profiles |
| [`docs/ENVS.md`](docs/ENVS.md) | Environment variables, TheRock setup, benchmark/profiling profiles |
| [`docs/ROOFLINE.md`](docs/ROOFLINE.md) | RDNA3 / W7900 performance model and decision tree |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Implementation status and concrete milestones |
| [`docs/API.md`](docs/API.md) | OpenAI-compatible server usage and endpoint support |
| [`docs/LAGUNA.md`](docs/LAGUNA.md) | Laguna S 2.1 gfx1151 support contract, implementation record, DFlash boundary, and evidence index |
| [`docs/LAGUNA-prefill.md`](docs/LAGUNA-prefill.md) | Laguna prefill plans: W7900 exact WPF-1T coltile/grouped-IQ/qrow4 production through 4K, plus the gfx1151 campaign record |
| [`docs/LAGUNA-PARITY-STATUS.md`](docs/LAGUNA-PARITY-STATUS.md) | Tabled W7900 Laguna parity status, apples-to-apples component table, gain timeline, and high-leverage resumption gates |
| [`docs/PREFILL.md`](docs/PREFILL.md) | Native prefill implementation spec |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Normal sampling parameter support plan |
| [`docs/MTP.md`](docs/MTP.md) | Multi-token prediction plan |
| [`docs/NATIVE_SPEC_CYCLE.md`](docs/NATIVE_SPEC_CYCLE.md) | Canonical N0-N5 milestone glossary, ownership distinctions, current speculative performance scorecard, and evidence index |
| [`docs/DFLASH.md`](docs/DFLASH.md) | DFlash draft-model speculative decode plan |
| [`docs/SOL-OPTIMIZATION.md`](docs/SOL-OPTIMIZATION.md) | gfx1151 PARO/GGUF optimization ledger and completion gates |
| [`docs/MTP-LLAMACPP-PARITY.md`](docs/MTP-LLAMACPP-PARITY.md) | Current GGUF MTP parity results and open reruns |
| [`docs/PARO-GGUF-MTP-TRANSFER.md`](docs/PARO-GGUF-MTP-TRANSFER.md) | PARO follow-up queue from GGUF/MTP server and verifier work |
| [`docs/HIP-vs-VULKAN.md`](docs/HIP-vs-VULKAN.md) | Current timing-contract v2 backend conclusions and portability gates |
| [`benchmarks/README.md`](benchmarks/README.md) | Canonical topline scoreboard, platform freshness, protocols, and refresh commands |
| [`AGENTS.md`](AGENTS.md) | Ground rules for every coding / review / benchmarking task |
| [`WORKLOG.md`](WORKLOG.md) | Append-only cross-session journal of decisions and measurements |

## Development

```bash
# narrowest test suite (CPU-only paths run without a GPU)
pytest -q

# kernel source-lineage drift check before any port
python3 scripts/check_lineage.py --kind kernel --diff stat
```

See [`AGENTS.md`](AGENTS.md) for the full workflow: when to run the
CPU-reference correctness gate, when to add a `rocprofv3 --kernel-trace` smoke,
and what a retained benchmark row requires.

## References & lineage

hipEngine is not a fork of any project; it is a brand new codebase with from-scratch
code and kernels. Of course it builds on the work of many others:

- [ROCm](https://github.com/ROCm/rocm) - of course this all sits on AMD's open-source
  compute stack, notably on [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - most of the original
  kernel tuning iteration loops used this as a host-layer. Some of the performance 
  limitations of the architecture motivated the hipEngine rewrite, but we remain
  grateful and deeply appreciative of nano-vllm as a great research platform.
- [ParoQuant](https://github.com/z-lab/paroquant) - after reviewing the current SOTA on model
  quantization, we chose ParoQuant as the first target due to both its excellent accuracy
  *and* its efficiency (QTIP/[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is 
  very cool but proved challenging to implement performant RDNA3 kernels)
- [FastDMS](https://github.com/shisa-ai/FastDMS) - our KVCache ABI is shaped by the lessons 
   learned from building our DMS reference implementation.

Greetz: [ROCmFPX](https://github.com/charlie12345/ROCmFPX), [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
