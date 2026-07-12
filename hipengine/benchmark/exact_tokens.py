"""Shared exact-token fixtures and direct/HTTP generation oracles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.prompts import file_sha256, token_ids_sha256


EXACT_TOKEN_ORACLE_KIND = "hipengine_exact_token_oracle"
EXACT_TOKEN_ORACLE_SCHEMA_VERSION = 1
DEFAULT_EXACT_TOKEN_FIXTURE = Path("fixtures/qwen35_paro/parent_512_32_seed1234.json")


def _token_row(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a token-ID sequence")
    if not value:
        raise ValueError(f"{label} must not be empty")
    row: list[int] = []
    for index, token in enumerate(value):
        if not isinstance(token, int) or isinstance(token, bool):
            raise ValueError(f"{label}[{index}] must be an integer")
        token_id = int(token)
        if token_id < 0:
            raise ValueError(f"{label}[{index}] must be non-negative")
        row.append(token_id)
    return tuple(row)


def _nonnegative_generated_row(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a token-ID sequence")
    if not value:
        return ()
    return _token_row(value, label=label)


@dataclass(frozen=True)
class ExactTokenFixture:
    path: Path
    name: str
    prompt_length: int
    prompt_rows: tuple[tuple[int, ...], ...]
    file_sha256: str
    source: str | None = None

    @property
    def prompt_count(self) -> int:
        return len(self.prompt_rows)

    @property
    def row_sha256(self) -> tuple[str, ...]:
        return tuple(token_ids_sha256(row) for row in self.prompt_rows)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "source": self.source,
            "file_sha256": self.file_sha256,
            "prompt_length": self.prompt_length,
            "prompt_count": self.prompt_count,
            "prompt_token_ids_sha256": list(self.row_sha256),
        }


def load_exact_token_fixture(
    path: str | Path = DEFAULT_EXACT_TOKEN_FIXTURE,
    *,
    prompt_length: int | None = None,
    prompt_count: int | None = None,
) -> ExactTokenFixture:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{fixture_path} must contain a JSON object")

    configured_length = prompt_length
    if configured_length is None:
        configured_length = payload.get("prompt_length", payload.get("prompt_len"))
    if not isinstance(configured_length, int) or isinstance(configured_length, bool) or configured_length <= 0:
        raise ValueError("prompt_length must be a positive integer")
    length = int(configured_length)

    raw_rows = payload.get("prompt_rows")
    rows: list[tuple[int, ...]] = []
    if raw_rows is not None:
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
            raise ValueError("prompt_rows must be a sequence")
        rows = [_token_row(row, label=f"prompt_rows[{index}]") for index, row in enumerate(raw_rows)]
    else:
        flat = _token_row(payload.get("prompt_ids"), label="prompt_ids")
        if len(flat) % length != 0:
            raise ValueError("flat prompt_ids length must be divisible by prompt_length")
        rows = [
            flat[offset : offset + length]
            for offset in range(0, len(flat), length)
            if len(flat[offset : offset + length]) == length
        ]
    if not rows:
        raise ValueError("fixture contains no complete prompt rows")
    if any(len(row) != length for row in rows):
        raise ValueError("every prompt row must match prompt_length")

    requested_count = prompt_count
    if requested_count is None:
        candidate = payload.get("prompt_count")
        requested_count = int(candidate) if isinstance(candidate, int) and not isinstance(candidate, bool) else len(rows)
    if not isinstance(requested_count, int) or isinstance(requested_count, bool) or requested_count <= 0:
        raise ValueError("prompt_count must be a positive integer")
    selected = tuple(rows[index % len(rows)] for index in range(int(requested_count)))
    return ExactTokenFixture(
        path=fixture_path,
        name=str(payload.get("name") or fixture_path.stem),
        prompt_length=length,
        prompt_rows=selected,
        file_sha256=file_sha256(fixture_path),
        source=None if payload.get("source") is None else str(payload.get("source")),
    )


@dataclass(frozen=True)
class ExactTokenOracle:
    mode: str
    prompt_rows: tuple[tuple[int, ...], ...]
    generated_rows: tuple[tuple[int, ...], ...]
    max_tokens: int

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"direct", "http"}:
            raise ValueError("exact-token oracle mode must be 'direct' or 'http'")
        prompts = tuple(_token_row(row, label=f"prompt_token_ids[{index}]") for index, row in enumerate(self.prompt_rows))
        generated = tuple(
            _nonnegative_generated_row(row, label=f"generated_token_ids[{index}]")
            for index, row in enumerate(self.generated_rows)
        )
        max_tokens = int(self.max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if not prompts:
            raise ValueError("exact-token oracle must contain at least one prompt")
        if len(prompts) != len(generated):
            raise ValueError("prompt and generated oracle row counts differ")
        if len({len(row) for row in prompts}) != 1:
            raise ValueError("exact-token oracle prompt rows must have equal length")
        for index, row in enumerate(generated):
            if len(row) != max_tokens:
                raise ValueError(
                    f"generated_token_ids[{index}] contains {len(row)} IDs; expected max_tokens={max_tokens}"
                )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "prompt_rows", prompts)
        object.__setattr__(self, "generated_rows", generated)
        object.__setattr__(self, "max_tokens", max_tokens)

    @classmethod
    def from_rows(
        cls,
        *,
        mode: str,
        prompt_rows: Sequence[Sequence[int]],
        generated_rows: Sequence[Sequence[int]],
        max_tokens: int,
    ) -> "ExactTokenOracle":
        return cls(
            mode=mode,
            prompt_rows=tuple(tuple(row) for row in prompt_rows),
            generated_rows=tuple(tuple(row) for row in generated_rows),
            max_tokens=max_tokens,
        )

    @classmethod
    def from_json_path(cls, path: str | Path) -> "ExactTokenOracle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("exact-token oracle must be a JSON object")
        if payload.get("kind") != EXACT_TOKEN_ORACLE_KIND:
            raise ValueError("exact-token oracle kind is invalid")
        if payload.get("schema_version") != EXACT_TOKEN_ORACLE_SCHEMA_VERSION:
            raise ValueError("exact-token oracle schema_version is unsupported")
        shape = payload.get("shape")
        if not isinstance(shape, Mapping):
            raise ValueError("exact-token oracle shape is missing")
        return cls.from_rows(
            mode=str(payload.get("mode") or ""),
            prompt_rows=payload.get("prompt_token_ids") or (),
            generated_rows=payload.get("generated_token_ids") or (),
            max_tokens=int(shape.get("max_tokens", -1)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        prompt_length = len(self.prompt_rows[0])
        return {
            "kind": EXACT_TOKEN_ORACLE_KIND,
            "schema_version": EXACT_TOKEN_ORACLE_SCHEMA_VERSION,
            "mode": self.mode,
            "shape": {
                "prompt_count": len(self.prompt_rows),
                "prompt_length": prompt_length,
                "max_tokens": self.max_tokens,
            },
            "prompt_token_ids": [list(row) for row in self.prompt_rows],
            "prompt_token_ids_sha256": [token_ids_sha256(row) for row in self.prompt_rows],
            "generated_token_ids": [list(row) for row in self.generated_rows],
            "generated_token_ids_sha256": [token_ids_sha256(row) for row in self.generated_rows],
        }


def validate_exact_token_parity(
    oracle: ExactTokenOracle,
    *,
    mode: str,
    prompt_rows: Sequence[Sequence[int]],
    generated_rows: Sequence[Sequence[int]],
    max_tokens: int,
) -> dict[str, Any]:
    candidate = ExactTokenOracle.from_rows(
        mode=mode,
        prompt_rows=prompt_rows,
        generated_rows=generated_rows,
        max_tokens=max_tokens,
    )
    prompt_equal = candidate.prompt_rows == oracle.prompt_rows
    generated_equal = candidate.generated_rows == oracle.generated_rows
    if not prompt_equal:
        raise ValueError("prompt token IDs differ from exact-token oracle")
    if not generated_equal:
        raise ValueError("generated token IDs differ from exact-token oracle")
    return {
        "passed": True,
        "oracle_mode": oracle.mode,
        "candidate_mode": candidate.mode,
        "prompt_ids_equal": True,
        "generated_ids_equal": True,
        "prompt_count": len(candidate.prompt_rows),
        "prompt_length": len(candidate.prompt_rows[0]),
        "max_tokens": candidate.max_tokens,
    }
