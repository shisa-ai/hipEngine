"""Resolve published result citations without flattening nested artifacts."""

from pathlib import PurePosixPath
import re

_JSON_PATH = re.compile(r"((?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.json)(?!l)")


def cited_result_paths(text):
    result = set()
    for token in _JSON_PATH.findall(text):
        parts = PurePosixPath(token).parts
        if "results" in parts:
            parts = parts[parts.index("results") + 1:]
        elif len(parts) > 1:
            # Keep legacy flat artifact names, but a path outside results/ is
            # documentation or an external source, not a benchmark citation.
            continue
        if not parts or ".." in parts:
            raise ValueError(f"invalid result citation: {token}")
        result.add(PurePosixPath(*parts).as_posix())
    return sorted(result)
