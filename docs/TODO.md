Agentic
- soft thinking stop
- still do outputs after limit
- autoclear instead of fail?
- maxtokens to remaining context by default, =auto

- guidance/grammars - json, etc

- TP
- Routing
- Multiple models loaded


 Current state of agentic support in hipEngine:

 ### What's already implemented:

 1. auto max_tokens - Server already defaults chat max_tokens to max_context - prompt - 1 when omitted (v0.2.2)
 2. Reasoning content splitting - _ReasoningSplitter separates `

 from  response` blocks in streaming/non-streaming server responses
 3. Resident session reuse - Session buckets prevent per-turn reallocation
 4. Context preallocation - Server preallocates KV up to max_context_tokens
 5. Stop sequences - Per-request stop sequences (not deeply integrated though)
 6. Eager warmup - Model + warmup generation before server reports ready

 ### What's missing for agentic use:

 1. Thinking soft-close (the user's idea) - When approaching budget, boost the </think> logit probability to force the model out of
    thinking mode. Currently there's zero logit manipulation capability.
 2. Separate thinking vs output budgets - max_tokens is a single pool. A model spending 8000 tokens thinking and 0 answering gets no
    output. Need ability to say "think for ≤ X tokens, then answer with up to Y tokens" or "ensure at least Z tokens remain for the
    answer."
 3. Graceful budget exhaustion - When max_tokens is hit:
     - No way to detect we're still in think mode and inject </think>\n
     - No way to append a forced closing token
     - Answer just truncates mid-stream
 4. Logit bias/manipulation API - No mechanism exists for:
     - Boosting specific tokens (e.g.,   response)
     - Suppressing tokens
     - Dynamic bias based on position in generation
 5. Generation-aware context management - The server does auto max_tokens but there's no:
     - Per-generation progress tracking for budget management
     - "Soft" vs "hard" token limits
     - Last-chance budget handler
 6. Stop token awareness in the engine - No mechanism to detect thinking-close tokens mid-generation and adjust budget
 7. Streaming hooks for budget management - The stream path has no budget-awareness

 Let me now write up the comprehensive analysis.</think>Now I have a complete picture. Here's the analysis:

 ────────────────────────────────────────────────────────────────────────────────

 Agentic Inference Features for hipEngine — Gap Analysis

 ### What's Already There

 ┌──────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────┬───────────────────┐
 │ Feature                                                                  │ Where                                │ Status            │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ auto max_tokens — chat requests with no max_tokens fill remaining        │ api.py:_request_max_tokens           │ ✅ v0.2.2         │
 │ context                                                                  │                                      │                   │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ Reasoning split — separates think/response blocks into reasoning_content │ api.py:_ReasoningSplitter            │ ✅ v0.2.1         │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ Resident session reuse — KV survives across chat turns (bucketed)        │ qwen35_paro.py:_session_capacity_for │ ✅                │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ Context preallocation — --max-context-tokens / auto-sizing on startup    │ api.py:ensure_resident_context       │ ✅                │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ Stop sequences — per-row stop tokens in the batch scheduler              │ batch_scheduler.py:stop_tokens       │ ✅ (plumbing      │
 │                                                                          │                                      │ only)             │
 ├──────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┼───────────────────┤
 │ Eager warmup — model loaded before server reports ready                  │ api.py:eager_load_model              │ ✅                │
 └──────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────┴───────────────────┘

 ### What's Missing for Agentic Harnesses

 #### 1. Thinking soft-close via logit bias (your idea)

 Right now the decode loop is purely greedy-argmax with zero logit manipulation hooks. When the model is in think mode and budget is
 approaching exhaustion, we need to boost the </think> token logit to force transition to answer mode.

 Plumbing needed:
 - Token ID lookup for the \<think>\n and \<response>\n string sequences (they tokenize to the literal text in Qwen — the model is not
   using special tokens)
 - A logit-bias vector that the decode kernel applies before argmax, indexed by remaining budget
 - A budget-tracking callback that lives between the HIP graph replay and the token stream consumer

 Architecture fit:
 - Logit bias sits at the LM-head/sampler layer — not a per-layer kernel concern
 - Best placed as a sample-time hook just after lm-head produces logits, before argmax
 - Could be registered as a SamplingParams field: logit_bias: dict[int, float] with the thinking tokens
 - Or as a new ThinkingBudget concept with built-in ramp logic

 Kernel impact: Minimal. The bias application is an element-wise vector add that can be fused into the existing argmax kernel or done as
 a tiny separate kernel. For the graph-captured decode path, the bias values would need to be baked into the graph per step (or we'd
 need dynamic-bias support, which is the harder version).

 #### 2. Separate thinking vs. output budgets

 Current max_tokens is a single undifferentiated pool. For Qwen thinking models, the model produces:

 ```
   thinking
   ...chain-of-thought...
   </think>
   ...actual answer...
   <|endoftext|>
 ```

 If max_tokens=8192 and the model thinks for 8100 tokens, the answer gets 92 tokens — or zero if the  response tag itself is the 8193rd
 token.

 What we need:
 - max_think_tokens — soft or hard cap on thinking length
 - min_answer_tokens — guaranteed reserve budget for the answer
 - A policy that says "when think_budget_remaining ≤ 0, boost the \<response> logit to effectively force transition"

 Implementation path:
 A ThinkingBudget class that wraps the generation lifecycle:

 ```python
   @dataclass
   class ThinkingBudget:
       max_think_tokens: int | None = None       # soft limit: boost close-tag at this point
       hard_think_cap: int | None = None          # hard limit: force-inject "</think>"
       min_answer_tokens: int | None = None      # reserve this many from total budget

       def logit_bias(self, tokens_generated: int, in_think: bool) -> dict[int, float]:
           """Return per-token bias based on budget state."""
           ...

       def last_chance_action(self, tokens_generated: int, in_think: bool) -> LastChanceAction:
           """When total budget is fully exhausted, what to do."""
           ...
 ```

 The _ReasoningSplitter already tracks _in_reasoning state — this is the same mechanic pushed downstream to the generation loop itself.

 #### 3. Graceful budget exhaustion

 When the decode graph finishes its remaining replays and we read the generated token ids, three things can go wrong:

 1. Model is still in think mode → answer is empty (worst case)
 2. Model has started  response but it got truncated mid-tag → corrupted output
 3. Model produced a partial answer → at least something

 What we need:
 - After graph replay, decode the generated text and check _ReasoningSplitter state
 - If still _in_reasoning and budget is exhausted, inject  response\n as forced tokens
 - These forced tokens need either another graph replay or a "short tail" host-step path

 Architecture note: This is hard with the graph-captured decode because the graph replays all remaining steps atomically. For the
 soft-close path, we'd need either:
 - Graph-replay only up to soft_limit steps, then switch to host-stepped decode for the remainder (loses graph perf for last few tokens)
 - Bake the bias ramp into per-step graphs (many graph variants)
 - Use graph replay for bulk, then host-step for the "controlled close" tail — simplest, acceptable since the tail is short

 #### 4. Default to max available context for agentic harnesses

 The server already does auto max_tokens for chat when omitted. But for agentic harnesses (like pi, the coding agent we're in), the
 requirement is different:

 - Agents often call the API with explicit max_tokens set low (e.g., 4096)
 - Agents typically want max_context_tokens to be the model maximum, not a server-constrained value
 - Agents need the server to advertise its model capabilities accurately

 Current gap: The server caps max_context_tokens to what KV memory allows — correct for single-turn safety but may be unnecessarily
 conservative for agentic use. An agent that sends short prompts can use the full model context.

 Fix: Add --force-model-max-context server flag that preallocates for the model's advertised max (or a user-provided value) and fails
 fast if KV can't fit, rather than silently capping.

 #### 5. Token-level visibility for harnesses

 Agent harnesses need to know exactly how many tokens were consumed by thinking vs. answer. The current _ReasoningSplitter lives in the
 server layer and separates after the fact. Moving this into the GenerationRequest/GenerationResult layer would let:

 - The agent adjust its behavior based on think:answer ratio
 - Truncation logic operate on the actual token counts
 - Benchmarks track think vs answer throughput separately

 #### 6. Practical priority order

 For the specific agentic coding use case (pi-style harness):

 ┌──────────┬───────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
 │ Priority │ Feature                                       │ Rationale                                                       │
 ├──────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
 │ P0       │ Thinking soft-close via  response logit boost │ Prevents the "empty answer" failure mode; minimal kernel work   │
 ├──────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
 │ P1       │ Separate think/answer budget tracking         │ Core to the user's ask; requires plumbing but no kernel changes │
 ├──────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
 │ P2       │ Graceful exhaustion (inject closing tags)     │ Last-resort safety net; host-only change                        │
 ├──────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
 │ P3       │ Default to model max context                  │ Configuration change, already partially supported               │
 ├──────────┼───────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
 │ P4       │ Token-level think/answer accounting           │ Nice to have for observability, enables smarter truncation      │
 └──────────┴───────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘

 #### 7. How the logit boost would work technically

 For Qwen thinking models, the  response tag is literally tokenized as:

 ```python
   # In the tokenizer:
   " response"  # starts with a space/newline, is 3-4 tokens typically
 ```

 The boost mechanism:
 1. Before each decode step, compute budget_remaining and in_think state
 2. If budget_remaining < soft_limit and in_think, compute a ramp — bias increases linearly as budget approaches zero
 3. The bias is applied to the FP32 logits array on-device (an element-wise add of a sparse vector)
 4. Then argmax as usual

 For the graph-captured path, the complication is that the bias ramp changes per-step, so we can't bake it into a single graph. Options:
 - A: Host-step the tail (last ~100 tokens) — loses some graph perf but that's fine since it's the minority of steps
 - B: Capture one graph with a constant "soft close" bias, and another without — 2 graph variants
 - C: Add a device-side "budget counter" kernel that auto-boosts  response logits — most elegant but most kernel work

 Option A is the obvious Phase 1: graph replay for bulk, host step for the controlled tail with dynamic bias. The tail is < 200 tokens,
 so the perf impact is negligible.

 ────────────────────────────────────────────────────────────────────────────────

 ### Summary

 The highest-leverage improvement for agentic coding use is the thinking soft-close via \<response> logit boost — it directly addresses
 the "model thinks forever, produces empty answer" failure mode. Combined with separate think/answer budgets, this gives agentic
 harnesses predictable, useful output. All of this is host-side plumbing with minimal kernel changes (a small element-wise bias kernel
 that can be fused into the existing argmax path).

 Strong agree with the direction. My main opinion: don’t build “thinking soft-close” as a one-off feature. Build a tokenizer-aware
 decode controller with state, budgets, logit processors, and forced-token queues. Then thinking budgets, stop sequences, JSON/tool
 constraints, logit bias, “don’t EOS before answer,” etc. are all policies on the same primitive.

 Tweaks to your priority list

 ### 1. Promote token-level state/accounting to P0

 Soft-close depends on knowing:

 - are we inside reasoning?
 - how many reasoning tokens have been generated?
 - how many visible answer tokens remain guaranteed?
 - are we halfway through a multi-token delimiter?

 So I’d make a core DecodeState / GenerationTelemetry layer first. The API splitter is useful, but budget control needs this inside
 generation, not only after text streaming.

 ### 2. Soft-close should target </think>, not “response”

 For Qwen-style visible thinking, the robust state machine is roughly:

 ```text
   THINK -> CLOSING_THINK -> ANSWER -> DONE
 ```

 When budget pressure starts, bias or force the closing delimiter sequence such as </think>\n, then let the model answer normally.

 Boosting “response”/answer-start tokens is model-template-specific and risks weird partial transitions. Better:

 1. Bias the first token(s) of accepted close sequences.
 2. Once the model starts one close sequence, force the remaining tokens of that sequence.
 3. Then switch to answer budget accounting.

 ### 3. Implement general logit processors

 Instead of a special ThinkingBudget.logit_bias(), I’d add a sampler/logit processor stack:

 - static OpenAI-style logit_bias
 - suppress token IDs
 - force next token / forced token queue
 - stop-sequence DFA
 - min-tokens / suppress EOS before answer starts
 - dynamic budget processor
 - later: JSON schema / grammar constraints

 Then thinking control is just one LogitsProcessor.

 ### 4. Graceful exhaustion should be honest

 If we inject </think> or append text synthetically, the response should say so in metadata.

 Example:

 ```json
   "finish_details": {
     "reason": "thinking_budget_exhausted",
     "forced_close": true,
     "synthetic_tokens": 3,
     "reasoning_tokens": 2048,
     "answer_tokens": 312
   }
 ```

 Best case: force tokens through the normal decode path so KV state remains consistent. If we append text outside the model, mark it
 synthetic and avoid committing it silently to session state.

 Big thing I’d add: selective session commit

 For agentic harnesses this may be huge.

 If the server keeps resident KV across turns and the model generated:

 ```text
   <think> huge hidden reasoning </think>
   final answer
 ```

 then the next turn must not accidentally include hidden reasoning in the persistent session unless the client explicitly wants that.
 Coding agents usually want to retain only the visible assistant answer/tool call, not the whole chain of thought.

 So session reuse needs a commit policy:

 ```text
   append_all          # raw generated tokens
   append_visible_only # final answer/tool call only
   append_none         # stateless response, keep prompt cache only
 ```

 append_visible_only may require recomputing/prefilling the visible answer after generation, but correctness beats a poisoned resident
 context.

 Other high-value agentic features

 ### 1. Forkable prefix/session cache

 Coding harnesses reuse giant system prompts, repo summaries, tool instructions, etc. Expose handles like:

 ```json
   "session": {
     "cache_key": "pi-system-v7",
     "fork_from": "conversation-123",
     "commit": "visible_only"
   }
 ```

 Support:

 - prefix cache
 - fork/rollback
 - continue from generation handle
 - pinned prefix vs evictable turn history

 This is probably as valuable as thinking control for real agent workloads.

 ### 2. Strict structured output / tool-call decoding

 Agent harnesses fail expensively on malformed JSON/tool calls. Since hipEngine owns decode, it can do better than post-hoc parsing:

 - strict JSON mode
 - JSON schema / grammar constraints
 - tool-call-only mode
 - constrained unified diff / patch mode
 - force valid close braces before EOS

 This is a major differentiator versus generic OpenAI-compatible serving.

 ### 3. Continuation handles

 If generation hits length mid-answer, return a resumable handle:

 ```json
   "finish_reason": "length",
   "continuation_id": "gen_abc123"
 ```

 Then the harness can request:

 ```json
   {"continue_generation": "gen_abc123", "max_tokens": 2048}
 ```

 without reprefilling the whole prompt.

 ### 4. Better streaming metadata

 Stream usage deltas, not only text:

 ```json
   {
     "delta": {"content": "..."},
     "usage_delta": {
       "reasoning_tokens": 1,
       "answer_tokens": 0,
       "state": "thinking"
     }
   }
 ```

 Also useful:

 - TTFT
 - prefill ms
 - decode tok/s
 - cache hit/miss
 - KV bytes used
 - stop reason
 - budget pressure state

 ### 5. Request cancellation and deadlines

 Agents often abandon a request when a tool result arrives, user interrupts, or another branch wins. Need cheap cancellation checked
 every decode step/batch, plus deadline support.

 ### 6. Prompt/token diagnostics endpoint

 Add /tokenize, /detokenize, /count_tokens, and maybe /fit_context.

 Agent harnesses need to know before sending:

 - prompt tokens
 - available output tokens
 - whether context will fit
 - what got truncated
 - server max vs model max

 Suggested revised priority

 My ordering would be:

 1. DecodeState + token-level accounting
 2. General logit processor / forced-token / stop-DFA framework
 3. Thinking budget policy on top of that
 4. Graceful exhaustion + explicit finish metadata
 5. Selective session commit / visible-only KV behavior
 6. Structured tool/JSON/diff constrained decoding
 7. Forkable prefix cache + continuation handles

 So yes: your list is right. I’d just make the foundation more general, because the same machinery that fixes “thinks forever, no
 answer” also unlocks most of the really valuable agentic/coding harness features.
