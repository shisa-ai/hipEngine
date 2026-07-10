"""Canonical, tokenizer-independent token-row identity helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    encoded = ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
