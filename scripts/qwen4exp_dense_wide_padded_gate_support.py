"""Declared HC-down padded-tail diagnostic scope and engagement accounting."""

PARENT = "dense_wide256_f16in_f32_f32_out"
CANDIDATE = "dense_wide256_padded_f16in_f32_f32_out"


def admitted_shape(rows, k, n):
    return 512 <= rows <= 4096 and k == 10240 and n == 320


def call_in_scope(args):
    if len(args) != 6:
        raise ValueError("expected raw linear ABI: pointers,rows,K,N")
    return admitted_shape(*map(int, args[3:6]))


def expected_calls(tokens, chunk):
    if tokens < 1 or not 512 <= chunk <= 4096:
        raise ValueError("positive tokens and chunk512..4096 required")
    full, tail = divmod(tokens, chunk)
    return 70 * (full + int(tail >= 512))


def heldout_tokens(tokens):
    """Use at least512 tokens to exercise the scoped sibling on every heldout.

    Repeat/truncate each short tokenized prompt to512 independent of its text,
    category or outputs. This is a state/isolation fixture, not a task score.
    """
    if not tokens:
        raise ValueError("empty heldout")
    return list(tokens) if len(tokens) >= 512 else (list(tokens) * ((511 // len(tokens)) + 1))[:512]
