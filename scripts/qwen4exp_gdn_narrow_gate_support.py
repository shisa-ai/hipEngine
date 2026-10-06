"""Shared admission/count contract for checked-index GDN qualification."""


def validate_call(args, *, max_rows):
    if (len(args) != 14 or tuple(args[10:14]) != (16, 48, 128, 128)
            or not 2 <= args[9] <= max_rows):
        raise ValueError(f"requires Hk16/Hv48/D128,2..{max_rows} rows")


def expected_calls(tokens, chunk):
    """Recovered UD production uses the serial wave-norm leaf on all 36 GDN layers."""
    if tokens < 1 or not 1 <= chunk <= 4096:
        raise ValueError("positive tokens and chunk1..4096 required")
    full, tail = divmod(tokens, chunk)
    return 36 * (full * (chunk >= 2) + (tail >= 2))
