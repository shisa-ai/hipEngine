# Token logprob bytes and text alignment

Flash-Next text autoregressive generation records the original tokenizer bytes
for selected tokens and top-logprob candidates. Probabilities describe the
processed distribution used for selection; byte metadata does not change the
sampler or combine probabilities across tokens.

A token may contain only part of a UTF-8 character. Its individual `token` text
can therefore contain `�` even when the generated response is valid Unicode.
Concatenate the byte arrays before decoding to reconstruct the text. Do not
concatenate the individually decoded token strings for an alignment check.

## Chat responses

`choices[].logprobs.content[]` includes `bytes` for each selected token and each
`top_logprobs[]` candidate when the backend provides them. Unknown byte metadata
remains `null`; the server does not substitute the UTF-8 encoding of `�`.
Private reasoning logprob entries use the same representation.

For example, the bytes `[232, 136]` and `[144]` belong to two tokens, but together
encode `舐`. Both token probabilities remain separate. A following `。` token
keeps its own entry rather than being removed by a character-count mismatch.

## Legacy completion responses

`choices[].logprobs.text_offset[]` counts Unicode characters, including any echoed
prompt. Tokens sharing one character can have the same offset.

When selected-token bytes are available, `choices[].logprobs.hipengine` adds:

- `token_bytes`: arrays aligned with `tokens`; an echoed prompt has `null`.
- `top_logprob_tokens`: candidate lists aligned with `tokens`. Each candidate
  has `token_id`, `token`, `logprob`, and `bytes`. An echoed prompt has `null`.

Use `top_logprob_tokens` when candidate identity matters. The legacy
`top_logprobs` field is a text-keyed map and cannot distinguish two byte-fragment
candidates that both decode individually to `�`. The candidate lists preserve
both IDs, byte sequences, and probabilities.

## Visibility and scope

Logprobs belong to whole tokens. Stop trimming and reasoning removal do not
invent a probability for a substring of a token. Chat content logprobs exclude
hidden reasoning; prompt-opened reasoning uses the same split as response text.

Flash-Next logprob SSE uses the buffered detailed-generation route. These byte
fields do not claim live incremental logprob streaming or sampled MTP support.
Other backends may omit bytes. See [the HTTP API](API.md) for logprob admission,
omission reporting, and streaming capability flags.
