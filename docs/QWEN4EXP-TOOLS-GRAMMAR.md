# Qwen4Exp Tools And Grammar

The Qwen4Exp generator now owns its chat protocol. It renders the embedded
GGUF Jinja template in an immutable sandbox, including the native XML
function/parameter instructions, grouped tool responses, assistant tool
history and thinking prefix. OpenAI JSON-string history arguments are parsed
into objects before rendering; caller messages are not mutated.

The existing generic JSON tool parser remains in place for other models.
Qwen4Exp accepts native XML calls and JSON tool envelopes. Bare JSON tool
compatibility requires declared tools and is disabled when parsing a structured
answer, so an ordinary JSON object containing `name` is not mistaken for a call.
Reasoning examples are not lifted into tool calls. Malformed XML and duplicate
parameters are rejected; strings retain their values while explicitly typed
non-string parameters are decoded as JSON. Streaming emits normalized OpenAI
tool-call deltas after validation.

## Constrained Decoding

Qwen4Exp greedy text requests support:

- `response_format` JSON object and JSON Schema.
- `guided_json`, `guided_regex` and `guided_choice`.
- `grammar` / `guided_grammar` strings in GBNF or Lark form.
- Declared XML tool names/parameters, required/named tool choices, and the
  single-call boundary unless `parallel_tool_calls` is enabled.
- Tools plus a structured answer schema: the model may emit a valid tool call
  or a schema-conforming answer. Tool responses are not validated as JSON answers.

`llguidance` 1.8 supplies the grammar engine. Its tokenizer is built directly from
the existing Rust-tokenizers JSON, without Transformers or Torch. Each request
owns a separate matcher. Full F32 logits are copied to the host for constrained
requests; the token mask is applied **before argmax**, and the selected token is
explicitly supplied to the next decode step. Unconstrained requests retain the
compact device-token path.

Grammar acceptance without an EOS token is a stop boundary, not permission to
strip the last content token. This distinction preserves final JSON braces.
Grammar schemas are part of request grouping, and matcher state is not shared
between requests. Lazy chat requests resolve their model protocol before the
first render. Capability reporting is model-scoped.

The direct API accepts `SamplingParams(grammar={"format": "json_schema",
"value": schema})`, or equivalent format/value specs for regex, choice and
GBNF/Lark. Models without a mask implementation reject direct grammar requests.
Speculative and multimodal grammar paths are explicitly unsupported rather than
silently running without masks.

## Scope And Limits

- This is greedy text support for Qwen4Exp, not a new sampler for every model.
- The XML tool grammar uses explicit root object properties in declaration
  order. Nested non-string schemas use llguidance JSON compilation. Raw strings
  support enum/const, length, and pattern constraints; unsupported combinations
  such as pattern plus length are rejected rather than weakened.
- XML header names cannot contain whitespace or angle brackets. The native
  parameter closing delimiter is reserved; this is the model's XML-like format,
  not arbitrary XML with entity expansion.
- Result validation remains in place. Token budgets can truncate a response;
  grammar support does not guarantee that every request finishes within its budget.
- Syntax/schema compliance does not guarantee a correct tool choice or argument
  meaning. No benchmark prompt, tool name or token ID is special-cased.
- This is not new qualification of arbitrary thinking-budget controls, MTP,
  multimodal generation, or a fast GPU grammar-mask kernel.

## Validation

The fixture `tests/fixtures/qwen4exp_chat_template.jinja` was extracted from the
binding UD-Q4_K_XL GGUF; SHA256
`12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`.

Focused tests cover template/history rendering, XML/JSON compatibility,
typed/malformed calls, open reasoning, schema masks before argmax, special-token
terminals, request isolation, selected-token handoff, EOS/content boundaries,
single-call constraints and unsupported speculation.

Real Framework gfx1151 HTTP checks passed required tools, a multi-turn tool
response, concurrent JSON schemas, streamed calls and GBNF output.

An isolated installation of `SeraphimSerapis/tool-eval-bench` at
`cf54b4bfe705f12f71e8866f10730572497c8105` measured:

| Development sanity run | Result |
| --- | --- |
| Short suite, 15 scenarios | 93/100: 14 pass, TC-03 implicit selection fails |
| Category O, 6 scenarios | 83/100: 4 pass, TC-68/69 partial on unnecessary/unrelated tool choices |

TC-68 intentionally omits API `response_format`; its prompt schema was not
extracted or specially constrained. TC-69's final answer was valid JSON; its
partial score came from an unrelated tool call.

These are development-tree functional checks, not a frozen-revision throughput
baseline or a before/after comparison to the externally reported 49 failures.
The CLI requires a known adapter label, so `vllm` selected its OpenAI adapter;
the actual server was hipEngine, BF16 KV, c2, 16K context, MTP off, temperature0,
thinking disabled. Final compatibility-only guards have focused CPU coverage.
The full69-scenario suite and hard mode were not run.

Commands, run configuration, scenario summaries, source caveats and raw hashes:
[validation packet](../benchmarks/results/2026-09-06-framework-qwen4exp-tools-grammar.json).
