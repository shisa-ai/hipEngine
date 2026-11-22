"""Inkling GGUF byte-level BPE or the tokenizer sidecar used by Loom packages."""

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers


# The Inkling checkpoint uses the GPT-4o pre-tokenization pattern. This is the
# checkpoint's tokenizer.json pattern, expressed as a single raw string.
_INKLING_SPLIT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def inkling2_gguf_tokenizer(info, vocab_size):
    meta = info.metadata
    tokens = meta.get("tokenizer.ggml.tokens")
    eos = meta.get("tokenizer.ggml.eos_token_id")
    if tokens is None:
        # A standalone GGUF can carry an adjacent tokenizer.json; a Loom
        # package carries it one directory above its parameter files.
        for directory in (info.path.parent, info.path.parent.parent):
            sidecar = directory / "tokenizer.json"
            if sidecar.is_file():
                tokenizer = Tokenizer.from_file(str(sidecar))
                break
        else:
            raise ValueError("Inkling 2 GGUF requires an embedded tokenizer or tokenizer.json sidecar")
    else:
        if (meta.get("tokenizer.ggml.model"), meta.get("tokenizer.ggml.pre")) != ("gpt2", "inkling2"):
            raise ValueError("unsupported Inkling 2 GGUF embedded tokenizer model or pre-tokenization")
        merges = [tuple(pair.split(" ")) for pair in meta["tokenizer.ggml.merges"]]
        tokenizer = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=merges))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(_INKLING_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = decoders.ByteLevel()
        types = meta.get("tokenizer.ggml.token_type", [1] * len(tokens))
        tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False)
                                      for token, kind in zip(tokens, types, strict=True) if kind in (3, 4)])
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"Inkling 2 GGUF tokenizer size differs from vocabulary size {vocab_size}")
    if type(eos) is not int or not 0 <= eos < vocab_size or tokenizer.id_to_token(eos) is None:
        raise ValueError(f"invalid Inkling 2 GGUF EOS token ID: {eos!r}")
    return tokenizer, eos
